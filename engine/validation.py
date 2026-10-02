from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from jiwer import wer
from tqdm import tqdm

from utils.text import normalize_text_for_wer
from engine.stats import safe_corr


def _pair_rank_accuracy(pred: list[float], target: list[float], min_delta: float) -> Dict[str, float]:
    pv = np.asarray(pred, dtype=np.float64)
    tv = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pv) & np.isfinite(tv)
    pv = pv[mask]
    tv = tv[mask]
    if pv.size < 2:
        return {"acc": float("nan"), "npairs": 0.0}

    tdiff = tv[:, None] - tv[None, :]
    valid = tdiff > float(min_delta)
    npairs = int(np.sum(valid))
    if npairs < 1:
        return {"acc": float("nan"), "npairs": 0.0}

    pdiff = pv[:, None] - pv[None, :]
    sel = pdiff[valid]
    # Crédito 0.5 para empates exactos en predicción.
    acc = float(np.mean((sel > 0.0).astype(np.float64) + 0.5 * (sel == 0.0).astype(np.float64)))
    return {"acc": acc, "npairs": float(npairs)}


def _triangular_window(length: int, overlap: int, is_first: bool, is_last: bool, device: torch.device) -> torch.Tensor:
    w = torch.ones(length, dtype=torch.float32, device=device)
    ov = int(max(0, min(overlap, length // 2)))
    if ov <= 0:
        return w
    if not is_first:
        w[:ov] = torch.linspace(0.0, 1.0, steps=ov, device=device, dtype=torch.float32)
    if not is_last:
        w[-ov:] = torch.minimum(w[-ov:], torch.linspace(1.0, 0.0, steps=ov, device=device, dtype=torch.float32))
    return w


def _enhance_waveform_chunked(
    generator: torch.nn.Module,
    waveform_1d: torch.Tensor,
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    direct_max_seconds: float,
) -> torch.Tensor:
    if waveform_1d.ndim == 2:
        waveform_1d = waveform_1d.squeeze(0)
    wav = waveform_1d.detach().to(torch.float32).cpu()
    total = int(wav.shape[-1])
    if total < 1:
        return wav

    chunk = int(round(float(chunk_seconds) * float(sample_rate)))
    direct_max = int(round(float(direct_max_seconds) * float(sample_rate)))
    if chunk <= 0:
        chunk = total
    if direct_max <= 0:
        direct_max = chunk
    run_direct = (total <= direct_max) or (chunk >= total)
    if run_direct:
        with torch.inference_mode():
            inp = wav.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = generator(inp)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(0).detach().float().cpu()
        return out[:total]

    overlap = int(round(float(overlap_seconds) * float(sample_rate)))
    overlap = max(0, min(overlap, chunk - 1))
    hop = max(1, chunk - overlap)

    acc = torch.zeros(total, dtype=torch.float32)
    wsum = torch.zeros(total, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, total, hop):
            end = min(total, start + chunk)
            seg = wav[start:end]
            seg_len = int(end - start)
            if seg_len < chunk:
                seg = F.pad(seg, (0, chunk - seg_len))

            seg_dev = seg.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = generator(seg_dev)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(0).detach().float().cpu()[:seg_len]

            win = _triangular_window(
                length=seg_len,
                overlap=overlap,
                is_first=(start == 0),
                is_last=(end >= total),
                device=out.device,
            )
            acc[start:end] += out * win
            wsum[start:end] += win
            if end >= total:
                break
    enh = acc / wsum.clamp_min(1e-6)
    return enh.clamp(min=-1.0, max=1.0)


def _enhance_with_fallback(
    generator: torch.nn.Module,
    waveform_1d: torch.Tensor,
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    min_chunk_seconds: float,
    direct_max_seconds: float,
) -> torch.Tensor:
    cur = float(max(chunk_seconds, min_chunk_seconds))
    min_chunk = float(max(1.0, min_chunk_seconds))
    while True:
        try:
            return _enhance_waveform_chunked(
                generator=generator,
                waveform_1d=waveform_1d,
                device=device,
                sample_rate=sample_rate,
                chunk_seconds=cur,
                overlap_seconds=overlap_seconds,
                direct_max_seconds=direct_max_seconds,
            )
        except (torch.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            is_oom = isinstance(e, torch.OutOfMemoryError) or ("out of memory" in msg)
            if (not is_oom) or (cur <= min_chunk + 1e-6):
                raise
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            cur = max(min_chunk, cur * 0.7)


@torch.no_grad()
def whisper_transcribe_longform_chunked(
    whisper_model,
    whisper_processor,
    audio_1d: torch.Tensor,
    sr: int,
    device: torch.device,
    chunk_seconds: float,
    overlap_seconds: float,
    condition_on_prev_tokens: bool = False,
    language: str = "es",
    task: str = "transcribe",
) -> str:
    if audio_1d.ndim == 2:
        audio_1d = audio_1d.squeeze(0)

    total = int(audio_1d.shape[-1])
    if total <= 0:
        return ""

    chunk = max(1, int(round(chunk_seconds * sr)))
    overlap = max(0, min(int(round(overlap_seconds * sr)), chunk - 1))
    hop = max(1, chunk - overlap)

    texts = []
    model_dtype = next(whisper_model.parameters()).dtype

    for start in range(0, total, hop):
        end = min(total, start + chunk)
        seg = audio_1d[start:end].detach().cpu().float().numpy()

        feats = whisper_processor(
            seg,
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feats.input_features.to(device=device, dtype=model_dtype)
        attention_mask = getattr(feats, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        kwargs = {
            "language": language,
            "task": task,
            "condition_on_prev_tokens": bool(condition_on_prev_tokens),
            # Evita warning de logits processors duplicados en algunos checkpoints.
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
        }
        if attention_mask is not None:
            pred_ids = whisper_model.generate(input_features, attention_mask=attention_mask, **kwargs)
        else:
            pred_ids = whisper_model.generate(input_features, **kwargs)

        txt = whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        if txt:
            texts.append(txt)

        if end >= total:
            break

    return " ".join(texts).strip()


@torch.no_grad()
def validate_wer_ecu911(
    generator: torch.nn.Module,
    dataloader,
    whisper_model,
    whisper_processor,
    train_device: torch.device,
    whisper_device: torch.device,
    sample_rate: int,
    max_samples: Optional[int],
    chunk_seconds: float,
    overlap_seconds: float,
    condition_on_prev_tokens: bool,
    use_precomputed_noisy_wer: bool = True,
    precomputed_noisy_wer_max: float | None = 5.0,
    per_sample_wer_cap: float | None = None,
    generator_chunk_seconds: float = 24.0,
    generator_overlap_seconds: float = 1.0,
    generator_min_chunk_seconds: float = 8.0,
    generator_direct_max_seconds: float = 12.0,
) -> Dict[str, float]:
    generator.eval()
    whisper_model.eval()

    refs = []
    hyps_enh = []
    noisy_wers = []
    enh_wers = []
    hyps_noisy = []
    noisy_source = "asr"
    noisy_precomputed_used = 0
    noisy_precomputed_rejected = 0
    skipped_empty_ref = 0
    skipped_enhance_oom = 0
    seen = 0

    pbar = tqdm(dataloader, desc="[val] WER", leave=False)
    for batch in pbar:
        wave = batch["waveform"]  # mantener en CPU; mover sample a sample para evitar OOM
        durations = batch.get("durations")
        texts = batch["transcripts"]

        for i in range(wave.shape[0]):
            if max_samples is not None and seen >= max_samples:
                break

            noisy_i = wave[i]
            try:
                enh_i = _enhance_with_fallback(
                    generator=generator,
                    waveform_1d=noisy_i,
                    device=train_device,
                    sample_rate=sample_rate,
                    chunk_seconds=float(generator_chunk_seconds),
                    overlap_seconds=float(generator_overlap_seconds),
                    min_chunk_seconds=float(generator_min_chunk_seconds),
                    direct_max_seconds=float(generator_direct_max_seconds),
                )
            except (torch.OutOfMemoryError, RuntimeError) as e:
                msg = str(e).lower()
                is_oom = isinstance(e, torch.OutOfMemoryError) or ("out of memory" in msg)
                if not is_oom:
                    raise
                skipped_enhance_oom += 1
                if train_device.type == "cuda":
                    with torch.cuda.device(train_device):
                        torch.cuda.empty_cache()
                continue

            if durations is not None:
                true_len = int(float(durations[i]) * sample_rate)
                true_len = max(1, min(true_len, noisy_i.shape[-1]))
                noisy_i = noisy_i[:true_len]
                enh_i = enh_i[:true_len]

            ref = normalize_text_for_wer(texts[i])
            # Evita WER patológico por muestras sin referencia válida.
            if not ref:
                skipped_empty_ref += 1
                continue
            used_precomputed = False
            if use_precomputed_noisy_wer and ("wer" in batch):
                try:
                    w_pre = float(batch["wer"][i])
                except Exception:
                    w_pre = float("nan")
                is_valid_pre = np.isfinite(w_pre) and (w_pre >= 0.0)
                if is_valid_pre and (precomputed_noisy_wer_max is not None):
                    is_valid_pre = (w_pre <= float(precomputed_noisy_wer_max))
                if is_valid_pre:
                    noisy_source = "precomputed"
                    noisy_wers.append(w_pre)
                    noisy_precomputed_used += 1
                    used_precomputed = True
                else:
                    noisy_precomputed_rejected += 1

            if not used_precomputed:
                hyp_n = whisper_transcribe_longform_chunked(
                    whisper_model,
                    whisper_processor,
                    noisy_i.to(whisper_device),
                    sr=sample_rate,
                    device=whisper_device,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    condition_on_prev_tokens=condition_on_prev_tokens,
                )
                hyp_n = normalize_text_for_wer(hyp_n) or ""
                try:
                    w_n = float(wer([ref], [hyp_n]))
                except Exception:
                    w_n = float("nan")
                if (per_sample_wer_cap is not None) and np.isfinite(w_n):
                    w_n = min(w_n, float(per_sample_wer_cap))
                noisy_wers.append(w_n)
                hyps_noisy.append(hyp_n)

            hyp_e = whisper_transcribe_longform_chunked(
                whisper_model,
                whisper_processor,
                enh_i.to(whisper_device),
                sr=sample_rate,
                device=whisper_device,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                condition_on_prev_tokens=condition_on_prev_tokens,
            )

            h_e = normalize_text_for_wer(hyp_e) or ""
            refs.append(ref)
            hyps_enh.append(h_e)
            try:
                w_e = float(wer([ref], [h_e]))
            except Exception:
                w_e = float("nan")
            if (per_sample_wer_cap is not None) and np.isfinite(w_e):
                w_e = min(w_e, float(per_sample_wer_cap))
            enh_wers.append(w_e)
            seen += 1

            del enh_i
            if train_device.type == "cuda" and ((seen % 2) == 0):
                with torch.cuda.device(train_device):
                    torch.cuda.empty_cache()

        if max_samples is not None and seen >= max_samples:
            break

    if (not refs) or (not noisy_wers) or (not enh_wers):
        return {
            "wer_noisy": float("nan"),
            "wer_orig": float("nan"),
            "wer_enh": float("nan"),
            "wer_gain": float("nan"),
            "n": 0.0,
            "skipped_enhance_oom": float(skipped_enhance_oom),
        }

    noisy_arr = np.asarray(noisy_wers, dtype=np.float64)
    enh_arr = np.asarray(enh_wers, dtype=np.float64)
    noisy_arr = noisy_arr[np.isfinite(noisy_arr)]
    enh_arr = enh_arr[np.isfinite(enh_arr)]
    if (noisy_arr.size < 1) or (enh_arr.size < 1):
        return {
            "wer_noisy": float("nan"),
            "wer_orig": float("nan"),
            "wer_enh": float("nan"),
            "wer_gain": float("nan"),
            "n": 0.0,
            "skipped_enhance_oom": float(skipped_enhance_oom),
        }

    wer_noisy = float(np.mean(noisy_arr))
    wer_enh = float(np.mean(enh_arr))
    # Métrica corpus para diagnóstico (no usada en gain principal).
    # Si no tenemos hipótesis para todos los samples, dejamos NaN.
    if len(hyps_noisy) == len(refs) and len(hyps_noisy) > 0:
        try:
            wer_noisy_corpus = float(wer(refs, hyps_noisy))
        except Exception:
            wer_noisy_corpus = float("nan")
    else:
        wer_noisy_corpus = float("nan")
    try:
        wer_enh_corpus = float(wer(refs, hyps_enh))
    except Exception:
        wer_enh_corpus = float("nan")
    return {
        "wer_noisy": wer_noisy,
        "wer_orig": wer_noisy,  # alias explícito para "audio original/noisy"
        "wer_enh": wer_enh,
        "wer_gain": wer_noisy - wer_enh,
        "n": float(len(refs)),
        "wer_noisy_corpus": wer_noisy_corpus,
        "wer_orig_corpus": wer_noisy_corpus,  # alias explícito para "audio original/noisy"
        "wer_enh_corpus": wer_enh_corpus,
        "wer_gain_corpus": (wer_noisy_corpus - wer_enh_corpus)
        if np.isfinite(wer_noisy_corpus) and np.isfinite(wer_enh_corpus)
        else float("nan"),
        "noisy_source_precomputed": 1.0 if noisy_source == "precomputed" else 0.0,
        "noisy_precomputed_used": float(noisy_precomputed_used),
        "noisy_precomputed_rejected": float(noisy_precomputed_rejected),
        "skipped_empty_ref": float(skipped_empty_ref),
        "skipped_enhance_oom": float(skipped_enhance_oom),
    }


@torch.no_grad()
def validate_discriminator_correlation(
    wer_discriminator: torch.nn.Module,
    dataloader,
    device: torch.device,
    output_head: str = "abs",
    rank_min_delta: float = 1e-3,
    decoder_metric_key: str | None = None,
    decoder_metric_higher_worse: bool = True,
) -> Dict[str, float]:
    wer_discriminator.eval()
    output_head = str(output_head).strip().lower()
    if output_head not in ("abs", "rel"):
        raise ValueError(f"output_head inválido: {output_head}. Usa 'abs' o 'rel'.")
    preds_quality = []
    targets_quality = []
    preds_logwer = []
    targets_logwer = []
    preds_decoder = []
    targets_decoder = []

    for batch in tqdm(dataloader, desc="[val] D_WER corr", leave=False):
        wave = batch["waveform"].to(device)
        dur = batch.get("durations")
        if dur is not None:
            dur = dur.to(device)

        logits = wer_discriminator(
            wave,
            durations=dur,
            audio_paths=batch.get("audio_paths", None),
            output=output_head,
        ).detach().float().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        neg_logits = -logits
        mask_np = None
        if "quality_mask" in batch:
            mask_np = batch["quality_mask"].detach().cpu().numpy().astype(bool)

        if "quality_scores" in batch:
            target_q = batch["quality_scores"].detach().float().cpu().numpy()
            if mask_np is not None:
                target_q = target_q[mask_np]
                probs = probs[mask_np]
                neg_logits = neg_logits[mask_np]
            preds_quality.extend(probs.tolist())
            targets_quality.extend(target_q.tolist())

            # Si también hay WER, calcula correlación log-WER en paralelo.
            if "wer" in batch:
                wer_v = batch["wer"].detach().float().cpu().numpy()
                if mask_np is not None:
                    wer_v = wer_v[mask_np]
                target_lw = np.log1p(np.maximum(0.0, wer_v))
                preds_logwer.extend(neg_logits.tolist())
                targets_logwer.extend(target_lw.tolist())
        elif "wer" in batch:
            wer_v = batch["wer"].detach().float().cpu().numpy()
            if mask_np is not None:
                wer_v = wer_v[mask_np]
                probs = probs[mask_np]
                neg_logits = neg_logits[mask_np]
            target_q = 1.0 / (1.0 + np.maximum(0.0, wer_v))
            target_lw = np.log1p(np.maximum(0.0, wer_v))
            preds_quality.extend(probs.tolist())
            targets_quality.extend(target_q.tolist())
            preds_logwer.extend(neg_logits.tolist())
            targets_logwer.extend(target_lw.tolist())

        if decoder_metric_key and (decoder_metric_key in batch):
            dec_v = batch[decoder_metric_key].detach().float().cpu().numpy()
            dec_mask = batch.get(f"{decoder_metric_key}_mask")
            if dec_mask is not None:
                dec_mask = dec_mask.detach().cpu().numpy().astype(bool)
            else:
                dec_mask = np.ones_like(dec_v, dtype=bool)
            if mask_np is not None:
                dec_v = dec_v[mask_np]
                dec_mask = dec_mask[mask_np]
            if dec_v.size > 0:
                pred_dec = (-logits if mask_np is None else (-logits)[mask_np])
                if dec_mask is not None:
                    pred_dec = pred_dec[dec_mask]
                    dec_v = dec_v[dec_mask]
                if dec_v.size > 0:
                    target_dec = dec_v if decoder_metric_higher_worse else (-dec_v)
                    preds_decoder.extend(pred_dec.tolist())
                    targets_decoder.extend(target_dec.tolist())

    c_quality = safe_corr(preds_quality, targets_quality)
    c_logwer = safe_corr(preds_logwer, targets_logwer)
    c_decoder = safe_corr(preds_decoder, targets_decoder)
    ra_quality = _pair_rank_accuracy(preds_quality, targets_quality, min_delta=rank_min_delta)
    ra_logwer = _pair_rank_accuracy(preds_logwer, targets_logwer, min_delta=rank_min_delta)
    ra_decoder = _pair_rank_accuracy(preds_decoder, targets_decoder, min_delta=rank_min_delta)

    return {
        "pearson_quality": float(c_quality["pearson"]),
        "spearman_quality": float(c_quality["spearman"]),
        "n_quality": float(c_quality["n"]),
        "rank_accuracy_quality": float(ra_quality["acc"]),
        "rank_pairs_quality": float(ra_quality["npairs"]),
        "pearson_logwer": float(c_logwer["pearson"]),
        "spearman_logwer": float(c_logwer["spearman"]),
        "n_logwer": float(c_logwer["n"]),
        "rank_accuracy_logwer": float(ra_logwer["acc"]),
        "rank_pairs_logwer": float(ra_logwer["npairs"]),
        "pearson_decoder_metric": float(c_decoder["pearson"]),
        "spearman_decoder_metric": float(c_decoder["spearman"]),
        "n_decoder_metric": float(c_decoder["n"]),
        "rank_accuracy_decoder_metric": float(ra_decoder["acc"]),
        "rank_pairs_decoder_metric": float(ra_decoder["npairs"]),
        # Alias de conveniencia para selección/monitoring
        "rank_accuracy": float(ra_logwer["acc"]),
        # Compat backward con métricas antiguas
        "pearson": float(c_quality["pearson"]),
        "spearman": float(c_quality["spearman"]),
        "n": float(c_quality["n"]),
    }

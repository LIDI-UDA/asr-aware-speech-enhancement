from __future__ import annotations

import argparse
import pickle
import sys
import zlib
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Permite ejecutar este script como "python scripts/..." sin depender de PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import load_config
from utils.audio import load_audio


@torch.no_grad()
def _decoder_stats_longform(
    whisper_model: WhisperForConditionalGeneration,
    whisper_processor: WhisperProcessor,
    waveform_1d: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    chunk_seconds: float,
    condition_on_prev_tokens: bool,
) -> Dict[str, float]:
    if waveform_1d.ndim == 2:
        waveform_1d = waveform_1d.squeeze(0)
    waveform_1d = waveform_1d.detach().to(torch.float32).cpu()
    if waveform_1d.numel() < 1:
        return {
            "asr_avg_entropy": float("nan"),
            "asr_avg_logprob": float("nan"),
            "asr_no_speech_prob": float("nan"),
            "asr_compression_ratio": float("nan"),
            "asr_token_count": 0.0,
        }

    chunk = max(1, int(round(float(chunk_seconds) * float(sample_rate))))
    total = int(waveform_1d.shape[-1])
    model_dtype = next(whisper_model.parameters()).dtype
    tokenizer = whisper_processor.tokenizer
    nospeech_token_id = None
    try:
        tid = tokenizer.convert_tokens_to_ids("<|nospeech|>")
        if isinstance(tid, int) and tid >= 0:
            nospeech_token_id = int(tid)
    except Exception:
        nospeech_token_id = None

    entropy_sum = 0.0
    logprob_sum = 0.0
    token_count = 0
    nospeech_prob_sum = 0.0
    nospeech_count = 0
    hyps = []

    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        w_np = waveform_1d[start:end].numpy()
        feats = whisper_processor(
            w_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feats.input_features.to(device=device, dtype=model_dtype)
        attention_mask = getattr(feats, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        kwargs = {
            "language": "es",
            "task": "transcribe",
            "do_sample": False,
            "num_beams": 1,
            "return_dict_in_generate": True,
            "condition_on_prev_tokens": bool(condition_on_prev_tokens),
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
        }
        if attention_mask is not None:
            out = whisper_model.generate(input_features, attention_mask=attention_mask, **kwargs)
        else:
            out = whisper_model.generate(input_features, **kwargs)

        seq = out.sequences
        if seq.shape[1] <= 1:
            continue
        with torch.no_grad():
            fwd = whisper_model(
                input_features=input_features,
                attention_mask=attention_mask,
                decoder_input_ids=seq[:, :-1],
                use_cache=False,
                return_dict=True,
            )
        scores = [fwd.logits[:, i, :] for i in range(fwd.logits.shape[1])]

        txt = whisper_processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if txt:
            hyps.append(txt)
        seq0 = seq[0]
        gen_tokens = seq0[1:1 + len(scores)]

        if nospeech_token_id is not None:
            step0 = scores[0].float()[0]
            p0 = torch.softmax(step0, dim=-1)
            if nospeech_token_id < int(p0.numel()):
                nospeech_prob_sum += float(p0[nospeech_token_id].item())
                nospeech_count += 1

        for step, step_logits in enumerate(scores):
            logits = step_logits.float()[0]
            logp = torch.log_softmax(logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            ent = float(-(p * logp).sum().item())
            tok = int(gen_tokens[step].item())
            tok_logp = float(logp[tok].item())
            entropy_sum += ent
            logprob_sum += tok_logp
            token_count += 1

    hyp_text = " ".join(hyps).strip()
    if hyp_text:
        raw_b = hyp_text.encode("utf-8", errors="ignore")
        comp_b = zlib.compress(raw_b)
        comp_ratio = float(len(raw_b) / max(1, len(comp_b)))
    else:
        comp_ratio = float("nan")
    if token_count < 1:
        return {
            "asr_avg_entropy": float("nan"),
            "asr_avg_logprob": float("nan"),
            "asr_no_speech_prob": float("nan"),
            "asr_compression_ratio": comp_ratio,
            "asr_token_count": 0.0,
        }
    return {
        "asr_avg_entropy": float(entropy_sum / float(token_count)),
        "asr_avg_logprob": float(logprob_sum / float(token_count)),
        "asr_no_speech_prob": (
            float(nospeech_prob_sum / float(nospeech_count))
            if nospeech_count > 0 else float("nan")
        ),
        "asr_compression_ratio": comp_ratio,
        "asr_token_count": float(token_count),
    }


def _iter_splits(data: Dict, splits: Iterable[str]) -> List[str]:
    out = []
    for s in splits:
        if s not in data:
            continue
        if not isinstance(data[s], list):
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Agrega métricas de decoder-ASR al ecu911_prepared.pkl")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--pkl", type=str, default=None, help="Ruta al ecu911_prepared.pkl (si no, usa config.paths.processed_data)")
    ap.add_argument("--output", type=str, default=None, help="Salida .pkl (si no, sobreescribe --pkl)")
    ap.add_argument("--model-name", type=str, default=None, help="Whisper model para stats (default: evaluation.whisper_model_name)")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="Splits a procesar (default: todos los splits tipo lista)")
    ap.add_argument("--limit", type=int, default=0, help="Limitar samples por split (0=sin límite)")
    ap.add_argument("--overwrite-existing", action="store_true", help="Recalcular aunque ya exista asr_avg_entropy")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pkl_path = Path(args.pkl) if args.pkl else (Path(cfg["paths"]["processed_data"]) / "ecu911_prepared.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    requested_splits = args.splits if args.splits else [k for k, v in data.items() if isinstance(v, list)]
    splits = _iter_splits(data, requested_splits)
    if not splits:
        raise RuntimeError(f"Sin splits válidos en {requested_splits}. Disponibles: {list(data.keys())}")

    model_name = args.model_name or cfg.get("evaluation", {}).get("whisper_model_name", "openai/whisper-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[decoder_stats] model={model_name} device={device} dtype={dtype}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    whisper_processor = WhisperProcessor.from_pretrained(model_name)

    sr = int(cfg["audio"]["target_sr"])
    chunk_seconds = float(cfg.get("evaluation", {}).get("whisper_chunk_seconds", 30.0))
    condition_on_prev = bool(cfg.get("evaluation", {}).get("condition_on_prev_tokens", False))

    total_done = 0
    total_skip = 0
    for split in splits:
        samples = data[split]
        n = len(samples) if args.limit <= 0 else min(len(samples), int(args.limit))
        pbar = tqdm(range(n), desc=f"[decoder_stats] {split}")
        done_split = 0
        skip_split = 0
        for i in pbar:
            s = samples[i]
            if (not args.overwrite_existing) and ("asr_avg_entropy" in s) and ("asr_avg_logprob" in s):
                skip_split += 1
                continue
            audio_path = s.get("audio_path", None)
            if (audio_path is None) or (not Path(audio_path).exists()):
                skip_split += 1
                continue
            try:
                wav, _ = load_audio(str(audio_path), target_sr=sr, normalize=bool(cfg["audio"]["normalize_rms"]))
                stats = _decoder_stats_longform(
                    whisper_model=whisper_model,
                    whisper_processor=whisper_processor,
                    waveform_1d=wav,
                    sample_rate=sr,
                    device=device,
                    chunk_seconds=chunk_seconds,
                    condition_on_prev_tokens=condition_on_prev,
                )
                s.update(stats)
                done_split += 1
            except Exception:
                skip_split += 1
                continue
        total_done += done_split
        total_skip += skip_split
        print(f"[decoder_stats] split={split} done={done_split} skipped={skip_split}")

    out_path = Path(args.output) if args.output else pkl_path
    with out_path.open("wb") as f:
        pickle.dump(data, f)
    print(f"[decoder_stats] guardado: {out_path}")
    print(f"[decoder_stats] total_done={total_done} total_skipped={total_skip}")


if __name__ == "__main__":
    main()

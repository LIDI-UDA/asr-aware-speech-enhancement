from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.ecu911_dataset import create_ecu911_dataloader, create_spc_dataloader
from engine.logging import MetricAccumulator
from engine.losses import multi_resolution_stft_loss, si_sdr_loss
from engine.validation import validate_wer_ecu911
from models.generator import create_generator
from trainers.common import build_state, create_optimizer, load_resume_state, load_whisper_asr


def _batch_noisy_clean(batch):
    if "waveform_noisy" in batch and "waveform_clean" in batch:
        return batch["waveform_noisy"], batch["waveform_clean"]
    return batch["waveform"], batch["waveform"]


def _train_chunk_paired_batch(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    durations: torch.Tensor | None,
    sample_rate: int,
    chunk_seconds: float,
    random_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Recorta pares noisy/clean con el mismo offset para controlar VRAM en pretrain.
    """
    if chunk_seconds <= 0:
        return noisy, clean, durations

    target_len = int(sample_rate * chunk_seconds)
    if target_len <= 0:
        return noisy, clean, durations

    if durations is None:
        if noisy.shape[-1] <= target_len:
            return noisy, clean, None
        if random_chunk:
            max_start = noisy.shape[-1] - target_len
            start = int(torch.randint(0, max_start + 1, (1,), device=noisy.device).item())
        else:
            start = 0
        end = start + target_len
        dur_out = torch.full(
            (noisy.shape[0],),
            float(target_len) / float(sample_rate),
            device=noisy.device,
            dtype=torch.float32,
        )
        return noisy[..., start:end], clean[..., start:end], dur_out

    durations_cpu = durations.detach().float().cpu()
    noisy_chunks = []
    clean_chunks = []
    chunk_durations = []

    for i in range(noisy.shape[0]):
        true_len = int(round(float(durations_cpu[i].item()) * float(sample_rate)))
        true_len = max(1, min(true_len, int(noisy.shape[-1]), int(clean.shape[-1])))
        noisy_wav = noisy[i, :true_len]
        clean_wav = clean[i, :true_len]

        if true_len > target_len:
            if random_chunk:
                max_start = true_len - target_len
                start = int(torch.randint(0, max_start + 1, (1,), device=noisy.device).item())
            else:
                start = 0
            end = start + target_len
            noisy_wav = noisy_wav[start:end]
            clean_wav = clean_wav[start:end]
            cur_len = target_len
        else:
            cur_len = true_len
            if noisy_wav.shape[-1] < target_len:
                pad = target_len - noisy_wav.shape[-1]
                noisy_wav = F.pad(noisy_wav, (0, pad))
                clean_wav = F.pad(clean_wav, (0, pad))

        noisy_chunks.append(noisy_wav)
        clean_chunks.append(clean_wav)
        chunk_durations.append(float(cur_len) / float(sample_rate))

    noisy_out = torch.stack(noisy_chunks, dim=0)
    clean_out = torch.stack(clean_chunks, dim=0)
    dur_out = torch.tensor(chunk_durations, device=noisy.device, dtype=torch.float32)
    return noisy_out, clean_out, dur_out


def _build_pretrain_loader(cfg):
    spc_dir = Path(cfg["paths"].get("spc_corpus_dir", ""))
    if spc_dir.exists():
        try:
            return create_spc_dataloader(cfg, corpus_dir=str(spc_dir), stage="train", use_degradation=True)
        except Exception as e:
            print(f"[pretrain] SPC no usable ({type(e).__name__}: {e}). Fallback a ECU911.")
    return create_ecu911_dataloader(cfg, stage="train", purpose="default")


def _clear_cuda_memory(*devices: torch.device) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    seen: set[int] = set()
    for dev in devices:
        if dev.type != "cuda":
            continue
        idx = int(dev.index if dev.index is not None else torch.cuda.current_device())
        if idx in seen:
            continue
        seen.add(idx)
        with torch.cuda.device(idx):
            torch.cuda.empty_cache()


def run_pretrain(ctx, resume: str | None = None):
    cfg = ctx.config
    device = ctx.devices.train_device

    generator = create_generator(cfg).to(device)
    opt_g = create_optimizer(generator, cfg["optimizer"]["generator"])
    ctx.amp.register_optimizer("G", opt_g)

    train_loader = _build_pretrain_loader(cfg)
    val_loader = create_ecu911_dataloader(cfg, stage="val", purpose="default")

    whisper_model, whisper_proc = load_whisper_asr(cfg, ctx.devices.validate_device, for_evaluation=True)

    epoch0, global_step = load_resume_state(
        ctx.ckpt,
        resume,
        models={"generator": generator},
        optimizers={"G": opt_g},
        amp=ctx.amp,
    )

    lcfg = cfg.get("losses", {})
    fft_sizes = [int(x) for x in lcfg.get("mrstft_fft_sizes", [256, 512, 1024])]
    hop_sizes = [int(x) for x in lcfg.get("mrstft_hop_sizes", [64, 128, 256])]
    w_mrstft = float(lcfg.get("mrstft_weight", 1.0))
    w_sisdr = float(lcfg.get("si_sdr_weight", 0.4))
    w_l1 = float(lcfg.get("l1_weight", 0.1))
    pcfg = cfg.get("pretrain", {})
    train_chunk_seconds = float(pcfg.get("train_chunk_seconds", 0.0))
    train_random_chunk = bool(pcfg.get("train_random_chunk", True))
    oom_chunk_backoff = float(pcfg.get("oom_chunk_backoff", 1.0))
    oom_chunk_backoff = float(max(0.1, min(1.0, oom_chunk_backoff)))
    oom_min_chunk_seconds = float(pcfg.get("oom_min_chunk_seconds", train_chunk_seconds))
    oom_min_chunk_seconds = max(1.0, oom_min_chunk_seconds)
    sample_rate = int(cfg["audio"]["target_sr"])

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    eval_cfg = cfg.get("evaluation", {})
    full_val_samples = int(eval_cfg.get("max_val_samples", 128))
    quick_val_samples = int(eval_cfg.get("quick_val_max_samples", min(32, full_val_samples)))
    full_val_every = max(1, int(eval_cfg.get("full_val_every_epochs", 5)))
    force_val_samples = eval_cfg.get("force_val_samples", None)
    force_val_samples = None if force_val_samples is None else max(1, int(force_val_samples))
    adaptive_chunk_seconds = float(train_chunk_seconds)
    skipped_oom = 0

    for epoch in range(epoch0, epochs):
        _clear_cuda_memory(device, ctx.devices.validate_device)
        generator.train()
        meter = MetricAccumulator()
        epoch_meter = MetricAccumulator()
        epoch_optimizer_steps = 0
        oom_log_budget = 3
        current_chunk_seconds = float(adaptive_chunk_seconds)
        pbar = tqdm(train_loader, desc=f"[pretrain] epoch {epoch+1}/{epochs}")
        ctx.amp.zero_grad("G")
        print(
            f"[pretrain][epoch {epoch+1}] train_chunk_seconds={train_chunk_seconds:.2f} "
            f"active_chunk_seconds={current_chunk_seconds:.2f} "
            f"random_chunk={train_random_chunk} "
            f"oom_backoff={oom_chunk_backoff:.2f} oom_min_chunk={oom_min_chunk_seconds:.2f}"
        )

        for it, batch in enumerate(pbar):
            noisy = None
            clean = None
            enh = None
            loss_mrstft = None
            loss_sisdr = None
            loss_l1 = None
            loss_full = None
            loss = None
            try:
                noisy, clean = _batch_noisy_clean(batch)
                durations = batch.get("durations", None)
                noisy, clean, _ = _train_chunk_paired_batch(
                    noisy=noisy,
                    clean=clean,
                    durations=durations,
                    sample_rate=sample_rate,
                    chunk_seconds=current_chunk_seconds,
                    random_chunk=train_random_chunk,
                )
                noisy = noisy.to(device)
                clean = clean.to(device)

                with ctx.amp.autocast():
                    enh = generator(noisy)
                    loss_mrstft = multi_resolution_stft_loss(enh, clean, fft_sizes=fft_sizes, hop_sizes=hop_sizes)
                    loss_sisdr = si_sdr_loss(enh, clean)
                    loss_l1 = F.l1_loss(enh, clean)
                    loss_full = w_mrstft * loss_mrstft + w_sisdr * loss_sisdr + w_l1 * loss_l1
                    loss = loss_full / accum_steps

                ctx.amp.backward("G", loss)
                do_step = ((it + 1) % accum_steps == 0) or ((it + 1) == len(train_loader))
                if do_step:
                    grad_norm = ctx.amp.clip_grad_norm_("G", generator.parameters(), grad_clip)
                    ctx.amp.step("G")
                    ctx.amp.update("G")
                    ctx.amp.zero_grad("G")
                    global_step += 1
                    epoch_optimizer_steps += 1
                else:
                    grad_norm = 0.0
            except (torch.OutOfMemoryError, RuntimeError) as e:
                msg = str(e).lower()
                is_oom = isinstance(e, torch.OutOfMemoryError) or ("out of memory" in msg)
                if not is_oom:
                    raise
                if oom_log_budget > 0:
                    short_msg = str(e).strip().replace("\n", " ")
                    if len(short_msg) > 240:
                        short_msg = short_msg[:240] + "..."
                    print(
                        f"[pretrain][epoch {epoch+1}] OOM(it={it+1}, chunk={current_chunk_seconds:.2f}s): "
                        f"{short_msg}"
                    )
                    oom_log_budget -= 1
                skipped_oom += 1
                ctx.amp.zero_grad("G")
                noisy = None
                clean = None
                enh = None
                loss_mrstft = None
                loss_sisdr = None
                loss_l1 = None
                loss_full = None
                loss = None
                _clear_cuda_memory(device, ctx.devices.validate_device)
                if oom_chunk_backoff < 0.999 and current_chunk_seconds > oom_min_chunk_seconds:
                    new_chunk = max(oom_min_chunk_seconds, current_chunk_seconds * oom_chunk_backoff)
                    if new_chunk < current_chunk_seconds:
                        print(
                            "[pretrain] OOM detectado; reduciendo train_chunk_seconds "
                            f"{current_chunk_seconds:.2f} -> {new_chunk:.2f} (epoch {epoch+1}, it {it+1})"
                        )
                        current_chunk_seconds = new_chunk
                        adaptive_chunk_seconds = min(float(adaptive_chunk_seconds), float(new_chunk))
                continue

            meter.add(
                loss=float(loss_full.detach()),
                loss_mrstft=float(loss_mrstft.detach()),
                loss_sisdr=float(loss_sisdr.detach()),
                loss_l1=float(loss_l1.detach()),
                grad_norm=float(grad_norm),
            )
            epoch_meter.add(
                loss=float(loss_full.detach()),
                loss_mrstft=float(loss_mrstft.detach()),
                loss_sisdr=float(loss_sisdr.detach()),
                loss_l1=float(loss_l1.detach()),
                grad_norm=float(grad_norm),
            )

            if do_step and global_step % int(cfg["training"].get("log_every", 50)) == 0:
                logs = meter.mean_dict(clear=True)
                ctx.logger.log_scalars("train", global_step, logs)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in logs.items() if k.startswith("loss")})

        epoch_logs = epoch_meter.mean_dict(clear=True)
        if epoch_logs:
            print(
                "[pretrain][epoch "
                f"{epoch+1}] "
                + " | ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_logs.items()))
                + f" | optimizer_steps={epoch_optimizer_steps}"
                + f" | skipped_oom={skipped_oom}"
            )
            ctx.logger.log_scalars("epoch_train", epoch + 1, epoch_logs)
        _clear_cuda_memory(device, ctx.devices.validate_device)

        if (epoch + 1) % int(cfg["training"].get("validate_every_epochs", 1)) == 0:
            is_full = ((epoch + 1) % full_val_every == 0) or ((epoch + 1) == epochs)
            val_samples = full_val_samples if is_full else quick_val_samples
            if force_val_samples is not None:
                val_samples = min(val_samples, force_val_samples)
            print(
                f"[pretrain][epoch {epoch+1}] Iniciando validación "
                f"mode={'full' if is_full else 'quick'} samples={val_samples} "
                f"use_precomputed_noisy_wer={bool(eval_cfg.get('use_precomputed_noisy_wer', True))}"
            )
            val = validate_wer_ecu911(
                generator,
                val_loader,
                whisper_model,
                whisper_proc,
                train_device=ctx.devices.train_device,
                whisper_device=ctx.devices.validate_device,
                sample_rate=int(cfg["audio"]["target_sr"]),
                max_samples=val_samples,
                chunk_seconds=float(eval_cfg.get("whisper_chunk_seconds", 30.0)),
                overlap_seconds=float(eval_cfg.get("whisper_overlap_seconds", 1.0)),
                condition_on_prev_tokens=bool(eval_cfg.get("condition_on_prev_tokens", False)),
                use_precomputed_noisy_wer=bool(eval_cfg.get("use_precomputed_noisy_wer", True)),
                precomputed_noisy_wer_max=eval_cfg.get("precomputed_noisy_wer_max", 5.0),
                per_sample_wer_cap=eval_cfg.get("per_sample_wer_cap", None),
                generator_chunk_seconds=float(eval_cfg.get("generator_eval_chunk_seconds", 24.0)),
                generator_overlap_seconds=float(eval_cfg.get("generator_eval_overlap_seconds", 1.0)),
                generator_min_chunk_seconds=float(eval_cfg.get("generator_eval_min_chunk_seconds", 8.0)),
                generator_direct_max_seconds=float(eval_cfg.get("generator_eval_direct_max_seconds", 12.0)),
            )
            print(
                f"[pretrain][epoch {epoch+1}] val_done "
                f"wer_noisy={val.get('wer_noisy', float('nan')):.4f} "
                f"wer_noisy_corpus={val.get('wer_noisy_corpus', float('nan')):.4f} "
                f"wer_enh={val.get('wer_enh', float('nan')):.4f} "
                f"wer_enh_corpus={val.get('wer_enh_corpus', float('nan')):.4f} "
                f"gain={val.get('wer_gain', float('nan')):.4f} "
                f"gain_corpus={val.get('wer_gain_corpus', float('nan')):.4f} "
                f"pre_used={int(val.get('noisy_precomputed_used', 0.0))} "
                f"pre_rej={int(val.get('noisy_precomputed_rejected', 0.0))} "
                f"skip_enh_oom={int(val.get('skipped_enhance_oom', 0.0))} "
                f"n={int(val.get('n', 0.0))}"
            )
            ctx.logger.log_scalars("val_full" if is_full else "val_quick", global_step, val)
            _clear_cuda_memory(device, ctx.devices.validate_device)

            state = build_state(
                epoch=epoch + 1,
                global_step=global_step,
                best_score=ctx.ckpt.best_score,
                models={"generator": generator},
                optimizers={"G": opt_g},
                amp=ctx.amp,
            )
            # Best checkpoint solo con validación full para evitar ruido.
            ctx.ckpt.save(state, score=(val.get("wer_enh") if is_full else None))

    return {"global_step": global_step, "best": ctx.ckpt.best_score}

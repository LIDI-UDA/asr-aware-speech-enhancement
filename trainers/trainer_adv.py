from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.ecu911_dataset import create_ecu911_dataloader, create_spc_dataloader
from engine.logging import MetricAccumulator
from engine.losses import multi_resolution_stft_loss, si_sdr_loss
from engine.validation import validate_wer_ecu911
from models.discriminator import create_discriminator
from models.generator import create_generator
from trainers.common import build_state, create_optimizer, load_resume_state, load_whisper_asr


def _batch_noisy_clean(batch):
    if "waveform_noisy" in batch and "waveform_clean" in batch:
        return batch["waveform_noisy"], batch["waveform_clean"]
    return batch["waveform"], batch["waveform"]


def _build_adv_loader(cfg):
    spc_dir = Path(cfg["paths"].get("spc_corpus_dir", ""))
    if spc_dir.exists():
        try:
            return create_spc_dataloader(cfg, corpus_dir=str(spc_dir), stage="train", use_degradation=True)
        except Exception as e:
            print(f"[adversarial] SPC no usable ({type(e).__name__}: {e}). Fallback a ECU911.")
    return create_ecu911_dataloader(cfg, stage="train", purpose="default")


def run_adversarial(ctx, resume: str | None = None):
    cfg = ctx.config
    device = ctx.devices.train_device

    generator = create_generator(cfg).to(device)
    discriminator = create_discriminator(cfg).to(device)

    opt_g = create_optimizer(generator, cfg["optimizer"]["generator"])
    opt_d = create_optimizer(discriminator, cfg["optimizer"]["discriminator"])

    ctx.amp.register_optimizer("G", opt_g)
    ctx.amp.register_optimizer("D", opt_d)

    train_loader = _build_adv_loader(cfg)
    val_loader = create_ecu911_dataloader(cfg, stage="val", purpose="default")

    whisper_model, whisper_proc = load_whisper_asr(cfg, ctx.devices.validate_device, for_evaluation=True)

    epoch0, global_step = load_resume_state(
        ctx.ckpt,
        resume,
        models={"generator": generator, "discriminator": discriminator},
        optimizers={"G": opt_g, "D": opt_d},
        amp=ctx.amp,
    )

    lcfg = cfg.get("losses", {})
    fft_sizes = [int(x) for x in lcfg.get("mrstft_fft_sizes", [256, 512, 1024])]
    hop_sizes = [int(x) for x in lcfg.get("mrstft_hop_sizes", [64, 128, 256])]

    epochs = int(cfg["training"]["epochs"])
    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    eval_cfg = cfg.get("evaluation", {})
    full_val_samples = int(eval_cfg.get("max_val_samples", 128))
    quick_val_samples = int(eval_cfg.get("quick_val_max_samples", min(32, full_val_samples)))
    full_val_every = max(1, int(eval_cfg.get("full_val_every_epochs", 5)))
    force_val_samples = eval_cfg.get("force_val_samples", None)
    force_val_samples = None if force_val_samples is None else max(1, int(force_val_samples))

    for epoch in range(epoch0, epochs):
        generator.train()
        discriminator.train()
        meter = MetricAccumulator()
        epoch_meter = MetricAccumulator()

        for batch in tqdm(train_loader, desc=f"[adv] epoch {epoch+1}/{epochs}"):
            noisy, clean = _batch_noisy_clean(batch)
            noisy = noisy.to(device)
            clean = clean.to(device)

            ctx.amp.zero_grad("D")
            with ctx.amp.autocast():
                fake = generator(noisy).detach()
                pred_real = discriminator(clean)
                pred_fake = discriminator(fake)
                loss_d = 0.5 * (F.relu(1.0 - pred_real).mean() + F.relu(1.0 + pred_fake).mean())

            ctx.amp.backward("D", loss_d)
            ctx.amp.clip_grad_norm_("D", discriminator.parameters(), grad_clip)
            ctx.amp.step("D")
            ctx.amp.update("D")

            ctx.amp.zero_grad("G")
            with ctx.amp.autocast():
                enh = generator(noisy)
                pred = discriminator(enh)
                loss_adv = -pred.mean()
                loss_mrstft = multi_resolution_stft_loss(enh, clean, fft_sizes=fft_sizes, hop_sizes=hop_sizes)
                loss_sisdr = si_sdr_loss(enh, clean)
                loss = loss_mrstft + 0.4 * loss_sisdr + 0.1 * loss_adv

            ctx.amp.backward("G", loss)
            ctx.amp.clip_grad_norm_("G", generator.parameters(), grad_clip)
            ctx.amp.step("G")
            ctx.amp.update("G")

            meter.add(
                loss_g=float(loss.detach()),
                loss_d=float(loss_d.detach()),
                loss_mrstft=float(loss_mrstft.detach()),
                loss_sisdr=float(loss_sisdr.detach()),
                loss_adv=float(loss_adv.detach()),
            )
            epoch_meter.add(
                loss_g=float(loss.detach()),
                loss_d=float(loss_d.detach()),
                loss_mrstft=float(loss_mrstft.detach()),
                loss_sisdr=float(loss_sisdr.detach()),
                loss_adv=float(loss_adv.detach()),
            )
            global_step += 1
            if global_step % int(cfg["training"].get("log_every", 50)) == 0:
                ctx.logger.log_scalars("train", global_step, meter.mean_dict(clear=True))

        epoch_logs = epoch_meter.mean_dict(clear=True)
        if epoch_logs:
            print(
                "[adversarial][epoch "
                f"{epoch+1}] "
                + " | ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_logs.items()))
            )
            ctx.logger.log_scalars("epoch_train", epoch + 1, epoch_logs)

        if (epoch + 1) % int(cfg["training"].get("validate_every_epochs", 1)) == 0:
            is_full = ((epoch + 1) % full_val_every == 0) or ((epoch + 1) == epochs)
            val_samples = full_val_samples if is_full else quick_val_samples
            if force_val_samples is not None:
                val_samples = min(val_samples, force_val_samples)
            print(
                f"[adv][epoch {epoch+1}] Iniciando validación "
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
            )
            print(
                f"[adv][epoch {epoch+1}] val_done "
                f"wer_noisy={val.get('wer_noisy', float('nan')):.4f} "
                f"wer_noisy_corpus={val.get('wer_noisy_corpus', float('nan')):.4f} "
                f"wer_enh={val.get('wer_enh', float('nan')):.4f} "
                f"wer_enh_corpus={val.get('wer_enh_corpus', float('nan')):.4f} "
                f"gain={val.get('wer_gain', float('nan')):.4f} "
                f"gain_corpus={val.get('wer_gain_corpus', float('nan')):.4f} "
                f"pre_used={int(val.get('noisy_precomputed_used', 0.0))} "
                f"pre_rej={int(val.get('noisy_precomputed_rejected', 0.0))} "
                f"n={int(val.get('n', 0.0))}"
            )
            ctx.logger.log_scalars("val_full" if is_full else "val_quick", global_step, val)

            state = build_state(
                epoch=epoch + 1,
                global_step=global_step,
                best_score=ctx.ckpt.best_score,
                models={"generator": generator, "discriminator": discriminator},
                optimizers={"G": opt_g, "D": opt_d},
                amp=ctx.amp,
            )
            ctx.ckpt.save(state, score=(val.get("wer_enh") if is_full else None))

    return {"global_step": global_step, "best": ctx.ckpt.best_score}

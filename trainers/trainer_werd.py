from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ecu911_dataset import ReplayWERDataset, collate_ecu911, create_ecu911_dataloader
from engine.logging import MetricAccumulator
from engine.validation import validate_discriminator_correlation
from models.wer_discriminator import create_wer_discriminator
from trainers.common import build_state, load_resume_state
from utils.wer_replay_buffer import WERReplayBuffer


def _fit_wer_calibration(dataset, wer_cap: float) -> Tuple[float, float]:
    wers = []
    for s in getattr(dataset, "samples", []):
        w = dataset._get_sample_wer(s) if hasattr(dataset, "_get_sample_wer") else s.get("wer", None)
        if w is None:
            continue
        w = float(w)
        if np.isfinite(w):
            wers.append(min(max(0.0, w), wer_cap))

    if len(wers) < 10:
        return 1.0, 0.5

    x = np.log1p(np.asarray(wers, dtype=np.float64))
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = max(1e-4, q3 - q1)
    sigma = max(0.15, iqr / 1.349)
    return med, sigma


def _fit_logwer_quantile_reference(dataset, wer_cap: float) -> np.ndarray | None:
    wers = []
    for s in getattr(dataset, "samples", []):
        w = dataset._get_sample_wer(s) if hasattr(dataset, "_get_sample_wer") else s.get("wer", None)
        if w is None:
            continue
        w = float(w)
        if np.isfinite(w):
            wers.append(min(max(0.0, w), wer_cap))

    if len(wers) < 10:
        return None

    x = np.log1p(np.asarray(wers, dtype=np.float64))
    x = np.sort(x)
    return x.astype(np.float32, copy=False)


def _fit_sample_scalar_reference(dataset, key: str) -> Tuple[float, float, int]:
    vals = []
    if not key:
        return 0.0, 1.0, 0
    for s in getattr(dataset, "samples", []):
        if key not in s:
            continue
        try:
            v = float(s[key])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)

    if len(vals) < 10:
        return 0.0, 1.0, len(vals)

    x = np.asarray(vals, dtype=np.float64)
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = max(1e-6, q3 - q1)
    sigma = max(1e-3, iqr / 1.349)
    return med, sigma, len(vals)


def _quality_target_from_wer(wer_tensor: torch.Tensor, wer_cap: float, mu: float, sigma: float) -> torch.Tensor:
    w = wer_tensor.clamp(min=0.0, max=wer_cap)
    x = torch.log1p(w)
    return torch.sigmoid(-((x - mu) / sigma))


def _z_target_from_wer(wer_tensor: torch.Tensor, wer_cap: float, mu: float, sigma: float) -> torch.Tensor:
    w = wer_tensor.clamp(min=0.0, max=wer_cap)
    x = torch.log1p(w)
    return -((x - mu) / max(1e-6, sigma))


def _uncertainty_target_from_metric(
    metric_tensor: torch.Tensor,
    mu: float,
    sigma: float,
    higher_is_worse: bool,
) -> torch.Tensor:
    z = (metric_tensor - float(mu)) / max(1e-6, float(sigma))
    return torch.sigmoid(z if higher_is_worse else -z)


def _quantile_normal_target_from_wer(
    wer_tensor: torch.Tensor,
    wer_cap: float,
    quantile_ref: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    w = wer_tensor.clamp(min=0.0, max=wer_cap)
    x = torch.log1p(w)
    if quantile_ref is None or quantile_ref.numel() < 10:
        # Mantener convención: mayor target => mejor calidad (menor WER).
        return -x

    qref = quantile_ref.to(device=x.device, dtype=x.dtype)
    idx = torch.bucketize(x, qref, right=False)
    n = float(qref.numel())
    u = (idx.to(dtype=x.dtype) + 0.5) / (n + 1.0)
    u = u.clamp(min=eps, max=1.0 - eps)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
    return -z


def _targets_from_batch(
    batch,
    target_mode: str,
    wer_cap: float,
    mu: float,
    sigma: float,
    device: torch.device,
    quantile_ref: torch.Tensor | None = None,
    quantile_eps: float = 1e-4,
):
    if "wer" not in batch:
        if "quality_scores" in batch:
            return batch["quality_scores"].to(device).clamp(0.0, 1.0)
        raise RuntimeError("No hay 'wer' ni 'quality_scores' para D_WER")

    wer_t = batch["wer"].to(device)
    if target_mode == "log_wer":
        return torch.log1p(wer_t.clamp(min=0.0, max=wer_cap))
    if target_mode == "quantile_logwer":
        return _quantile_normal_target_from_wer(
            wer_t,
            wer_cap=wer_cap,
            quantile_ref=quantile_ref,
            eps=quantile_eps,
        )

    return _quality_target_from_wer(wer_t, wer_cap=wer_cap, mu=mu, sigma=sigma)


def _rank_listwise_hard_from_quality(
    scores: torch.Tensor,
    quality: torch.Tensor,
    hard_fraction: float = 0.6,
    margin: float = 0.0,
    min_delta: float = 1e-3,
    delta_power: float = 1.0,
) -> Tuple[torch.Tensor, int]:
    """
    Ranking relativo listwise sobre todos los pares válidos.
    Usa hard-mining por dificultad y pondera por separación de target.
    """
    if scores.numel() < 2:
        return scores.new_tensor(0.0), 0

    qdiff = quality[:, None] - quality[None, :]  # (B,B)
    valid = qdiff > float(min_delta)
    npairs = int(valid.sum().item())
    if npairs < 1:
        return scores.new_tensor(0.0), 0

    pdiff = scores[:, None] - scores[None, :]  # (B,B)
    hardness = -(pdiff - margin)               # mayor => par más difícil

    if hard_fraction < 0.999:
        k = max(1, int(round(npairs * max(0.0, hard_fraction))))
        hard_vals = hardness[valid]
        if k < npairs:
            thr = torch.topk(hard_vals, k=k, largest=True).values.min()
            valid = valid & (hardness >= thr)

    sel_pdiff = pdiff[valid]
    sel_qdiff = qdiff[valid].clamp(min=0.0)
    weights = torch.pow(sel_qdiff + 1e-6, float(max(0.0, delta_power)))
    losses = F.softplus(-(sel_pdiff - margin))
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return loss, int(valid.sum().item())


def _build_replay_loader_for_epoch(cfg: Dict, base_batch_size: int) -> tuple[DataLoader | None, int, int]:
    rcfg = cfg.get("wer_discriminator", {})
    if not bool(rcfg.get("use_replay", False)):
        return None, 0, 0

    replay_dir = str(rcfg.get("replay_dir", "data/replay"))
    replay_capacity = int(rcfg.get("replay_capacity", 20000))
    replay = WERReplayBuffer(root_dir=replay_dir, capacity=max(1, replay_capacity))
    total_items = len(replay)
    if total_items < 1:
        print(f"[pretrain_discriminator] replay vacío en '{replay_dir}', se omite.")
        return None, 0, 0

    replay_samples_cfg = int(rcfg.get("replay_samples_per_epoch", 0))
    replay_samples = total_items if replay_samples_cfg <= 0 else min(replay_samples_cfg, total_items)
    items = replay.sample(replay_samples, stratify=bool(rcfg.get("replay_stratify", True)))
    if len(items) < 1:
        print(f"[pretrain_discriminator] replay sin muestras válidas en '{replay_dir}', se omite.")
        return None, total_items, 0

    replay_batch_size = int(rcfg.get("replay_batch_size", base_batch_size))
    replay_batch_size = max(1, replay_batch_size)
    wer_cap = float(rcfg.get("wer_cap", 50.0))
    dataset = ReplayWERDataset(
        items=items,
        sample_rate=int(cfg["audio"]["target_sr"]),
        normalize=bool(cfg["audio"]["normalize_rms"]),
        wer_cap=wer_cap,
    )
    num_workers = int(cfg["data"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=replay_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg["data"]["pin_memory"]),
        prefetch_factor=(cfg["data"].get("prefetch_factor", 2) if num_workers > 0 else None),
        collate_fn=collate_ecu911,
        drop_last=False,
    )
    return loader, total_items, len(items)


def run_pretrain_discriminator(ctx, resume: str | None = None):
    cfg = ctx.config
    device = ctx.devices.whisper_device

    model = create_wer_discriminator(cfg, device=device).to(device)
    print(
        "[pretrain_discriminator] "
        f"use_acoustic_branch={bool(getattr(model, 'use_acoustic_branch', False))} "
        f"acoustic_dim={int(getattr(model, 'acoustic_dim', 0))} "
        f"use_wave_multiscale_branch={bool(getattr(model, 'use_wave_multiscale_branch', False))} "
        f"wave_dim={int(getattr(model, 'wave_dim', 0))} "
        f"wave_scales={list(getattr(model, 'wave_scales', ())) if hasattr(model, 'wave_scales') else []}"
    )
    wd_cfg = cfg.get("models", {}).get("wer_discriminator", {})
    opt_cfg = cfg["optimizer"]["wer_discriminator"]
    unfreeze_last_n = int(wd_cfg.get("unfreeze_last_n_layers", 0))
    encoder_lr_scale = float(wd_cfg.get("encoder_lr_scale", 0.1))
    base_lr = float(opt_cfg.get("lr", 1e-4))
    all_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head_params = [p for n, p in all_trainable if not n.startswith("encoder.")]
    enc_params = [p for n, p in all_trainable if n.startswith("encoder.")]
    if unfreeze_last_n > 0 and len(enc_params) > 0:
        opt = AdamW(
            [
                {"params": head_params, "lr": base_lr},
                {"params": enc_params, "lr": base_lr * encoder_lr_scale},
            ],
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
        print(
            "[pretrain_discriminator] "
            f"fine_tune_encoder_layers={unfreeze_last_n} "
            f"encoder_lr_scale={encoder_lr_scale}"
        )
    else:
        opt = AdamW(
            head_params,
            lr=base_lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
    ctx.amp.register_optimizer("WERD", opt)

    wd_data_cfg = cfg.get("wer_discriminator", {})
    batch_size = int(wd_data_cfg.get("batch_size", 32))
    train_loader = create_ecu911_dataloader(cfg, stage="train", batch_size=batch_size, purpose="wer_disc")
    val_loader = create_ecu911_dataloader(cfg, stage="val", batch_size=batch_size, purpose="wer_disc")

    use_replay = bool(wd_data_cfg.get("use_replay", False))
    replay_loss_weight = float(wd_data_cfg.get("replay_loss_weight", 1.0))
    replay_start_epoch = max(1, int(wd_data_cfg.get("replay_start_epoch", 1)))
    if use_replay and replay_loss_weight <= 0.0:
        print("[pretrain_discriminator] replay desactivado: replay_loss_weight <= 0.")
        use_replay = False

    wer_cap = float(wd_data_cfg.get("wer_cap", 50.0))
    mu, sigma = _fit_wer_calibration(train_loader.dataset, wer_cap=wer_cap)
    quantile_eps = float(wd_cfg.get("quantile_eps", 1e-4))
    quantile_ref_np = _fit_logwer_quantile_reference(train_loader.dataset, wer_cap=wer_cap)
    quantile_ref = None
    if quantile_ref_np is not None:
        quantile_ref = torch.as_tensor(quantile_ref_np, dtype=torch.float32, device=device)
    decoder_aux_key = str(wd_cfg.get("decoder_aux_key", "asr_avg_entropy")).strip()
    decoder_aux_weight = float(wd_cfg.get("decoder_aux_weight", 0.0))
    decoder_aux_higher_worse = bool(wd_cfg.get("decoder_aux_higher_worse", True))
    decoder_aux_mu, decoder_aux_sigma, decoder_aux_n = _fit_sample_scalar_reference(
        train_loader.dataset,
        key=decoder_aux_key,
    )
    if decoder_aux_weight > 0.0:
        if decoder_aux_n < 10:
            print(
                "[pretrain_discriminator] "
                f"decoder_aux desactivado: key='{decoder_aux_key}' insuficiente (n={decoder_aux_n})."
            )
            decoder_aux_weight = 0.0
        else:
            print(
                "[pretrain_discriminator] "
                f"decoder_aux key={decoder_aux_key} weight={decoder_aux_weight} "
                f"higher_is_worse={decoder_aux_higher_worse} "
                f"ref_mu={decoder_aux_mu:.4f} ref_sigma={decoder_aux_sigma:.4f} n={decoder_aux_n}"
            )

    resume_weights_only = bool(wd_cfg.get("resume_weights_only", False))
    reset_best_on_resume_weights_only = bool(wd_cfg.get("reset_best_on_resume_weights_only", True))
    epoch0, global_step = load_resume_state(
        ctx.ckpt,
        resume,
        models={"wer_discriminator": model},
        optimizers={"WERD": opt},
        amp=ctx.amp,
        load_optimizers=(not resume_weights_only),
        load_amp=(not resume_weights_only),
        load_counters=(not resume_weights_only),
    )
    if resume_weights_only and resume:
        print("[pretrain_discriminator] resume_weights_only=True (optimizer/amp/epoch reset).")
        if reset_best_on_resume_weights_only:
            ctx.ckpt.best_score = None
            print("[pretrain_discriminator] best_score reset para nueva búsqueda de best.")

    target_mode = wd_cfg.get("target_mode", "hybrid")  # quality_sigmoid | log_wer | quantile_logwer | hybrid
    loss_mode = wd_cfg.get("loss_mode", "smoothl1_rank")

    sup_weight = float(wd_cfg.get("sup_weight", 1.0))
    rank_weight = float(wd_cfg.get("rank_weight", 0.7))
    sup_prob_weight = float(wd_cfg.get("sup_prob_weight", 0.7))
    sup_logit_weight = float(wd_cfg.get("sup_logit_weight", 0.3))
    rank_margin = float(wd_cfg.get("rank_margin", 0.10))
    rank_hard_fraction = float(wd_cfg.get("rank_hard_fraction", 0.5))
    selection_metric = str(wd_cfg.get("selection_metric", "spearman_quality"))
    validation_head = str(wd_cfg.get("validation_head", "abs")).strip().lower()
    if validation_head not in ("abs", "rel"):
        raise ValueError(f"validation_head inválido: {validation_head}. Usa 'abs' o 'rel'.")
    pair_min_delta = float(wd_cfg.get("relative_pair_min_delta", 1e-3))
    delta_power = float(wd_cfg.get("relative_delta_power", 1.0))
    print(
        "[pretrain_discriminator] "
        f"target_mode={target_mode} loss_mode={loss_mode} "
        f"selection_metric={selection_metric} "
        f"validation_head={validation_head}"
    )
    if target_mode == "quantile_logwer":
        if quantile_ref is None:
            print("[pretrain_discriminator] quantile_ref insuficiente -> fallback implícito a log1p(WER).")
        else:
            q = quantile_ref_np
            print(
                "[pretrain_discriminator] "
                f"quantile_ref n={int(q.shape[0])} "
                f"logwer[min/med/max]=({float(np.min(q)):.4f}/{float(np.median(q)):.4f}/{float(np.max(q)):.4f}) "
                f"quantile_eps={quantile_eps}"
            )

    distill_weight = float(wd_data_cfg.get("distill_weight", 0.0))
    distill_warmup = int(wd_data_cfg.get("distill_warmup_steps", 500))

    grad_clip = float(wd_cfg.get("grad_clip", cfg["training"].get("grad_clip", 5.0)))
    use_amp = bool(wd_cfg.get("use_amp", False))
    enable_grad_accum = bool(wd_cfg.get("enable_grad_accum", True))
    accum_cfg = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    accum_steps = accum_cfg if enable_grad_accum else 1
    skipped_nonfinite = 0
    print(
        "[pretrain_discriminator] "
        f"use_amp={use_amp} grad_clip={grad_clip} "
        f"enable_grad_accum={enable_grad_accum} accum_steps={accum_steps}"
    )
    epochs = int(cfg["training"]["epochs"])
    early_stop_patience = max(0, int(wd_data_cfg.get("early_stop_patience", 0)))
    early_stop_min_delta = float(wd_data_cfg.get("early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    if ctx.ckpt.best_score is not None and np.isfinite(float(ctx.ckpt.best_score)):
        best_seen = float(ctx.ckpt.best_score)
    else:
        best_seen = -float("inf")
    should_stop = False

    for epoch in range(epoch0, epochs):
        model.train()
        meter = MetricAccumulator()
        epoch_meter = MetricAccumulator()
        ctx.amp.zero_grad("WERD")
        epoch_loaders = [("train", train_loader, 1.0)]
        replay_batches = 0
        use_replay_epoch = use_replay and ((epoch + 1) >= replay_start_epoch)
        if use_replay_epoch:
            replay_loader, replay_total, replay_epoch = _build_replay_loader_for_epoch(cfg, base_batch_size=batch_size)
            if replay_loader is not None and replay_epoch > 0:
                epoch_loaders.append(("replay", replay_loader, replay_loss_weight))
                print(
                    "[pretrain_discriminator] "
                    f"replay activo: total={replay_total} sampled={replay_epoch} "
                    f"batch_size={replay_loader.batch_size} loss_weight={replay_loss_weight:.3f}"
                )

        for loader_tag, active_loader, source_loss_weight in epoch_loaders:
            desc = f"[pretrain_discriminator] epoch {epoch+1}/{epochs}"
            if loader_tag == "replay":
                desc += " [replay]"
            for it, batch in enumerate(tqdm(active_loader, desc=desc)):
                wave = batch["waveform"].to(device)
                dur = batch.get("durations")
                if dur is not None:
                    dur = dur.to(device)
                audio_paths = batch.get("audio_paths", None)

                target = _targets_from_batch(
                    batch,
                    target_mode,
                    wer_cap=wer_cap,
                    mu=mu,
                    sigma=sigma,
                    device=device,
                    quantile_ref=quantile_ref,
                    quantile_eps=quantile_eps,
                )
                mask = batch.get("quality_mask")
                if mask is not None:
                    mask = mask.to(device)
                    target = target[mask]

                amp_ctx = ctx.amp.autocast() if use_amp else nullcontext()
                with amp_ctx:
                    out = model(wave, durations=dur, audio_paths=audio_paths, output="both")
                    logits_abs = out["abs"]
                    logits_rel = out["rel"]
                    if mask is not None:
                        logits_abs = logits_abs[mask]
                        logits_rel = logits_rel[mask]

                    if target_mode in ("log_wer", "quantile_logwer"):
                        sup_loss = F.smooth_l1_loss(logits_abs, target)
                    elif target_mode == "hybrid":
                        if "wer" not in batch:
                            raise RuntimeError("target_mode=hybrid requiere 'wer' en batch.")
                        wer_t = batch["wer"].to(device)
                        if mask is not None:
                            wer_t = wer_t[mask]
                        target_q = _quality_target_from_wer(wer_t, wer_cap=wer_cap, mu=mu, sigma=sigma)
                        target_z = _z_target_from_wer(wer_t, wer_cap=wer_cap, mu=mu, sigma=sigma)
                        sup_prob = F.smooth_l1_loss(torch.sigmoid(logits_abs), target_q)
                        sup_logit = F.smooth_l1_loss(logits_abs, target_z)
                        sup_loss = sup_prob_weight * sup_prob + sup_logit_weight * sup_logit
                    else:
                        sup_loss = F.smooth_l1_loss(torch.sigmoid(logits_abs), target)

                    if "wer" in batch:
                        # El ranking está definido como "higher is better".
                        # Por eso usamos calidad derivada de WER (descendente con WER).
                        rank_target = _quality_target_from_wer(batch["wer"].to(device), wer_cap=wer_cap, mu=mu, sigma=sigma)
                        if mask is not None:
                            rank_target = rank_target[mask]
                    else:
                        rank_target = target

                    rank_loss, npairs = _rank_listwise_hard_from_quality(
                        logits_rel,
                        rank_target,
                        hard_fraction=rank_hard_fraction,
                        margin=rank_margin,
                        min_delta=pair_min_delta,
                        delta_power=delta_power,
                    )

                    if loss_mode == "ranknet":
                        loss = rank_loss
                    else:
                        loss = sup_weight * sup_loss + rank_weight * rank_loss

                    if distill_weight > 0.0 and "wer" in batch:
                        warm = min(1.0, global_step / max(1, distill_warmup))
                        tw = _quality_target_from_wer(batch["wer"].to(device), wer_cap=wer_cap, mu=mu, sigma=sigma)
                        if mask is not None:
                            tw = tw[mask]
                        distill = F.smooth_l1_loss(torch.sigmoid(logits_abs), tw)
                        loss = loss + warm * distill_weight * distill
                    else:
                        distill = logits_abs.new_tensor(0.0)

                    decoder_aux = logits_abs.new_tensor(0.0)
                    if decoder_aux_weight > 0.0 and (decoder_aux_key in batch):
                        dec_t = batch[decoder_aux_key].to(device)
                        dec_mask = batch.get(f"{decoder_aux_key}_mask")
                        if dec_mask is not None:
                            dec_mask = dec_mask.to(device)
                        if mask is not None:
                            dec_t = dec_t[mask]
                            if dec_mask is not None:
                                dec_mask = dec_mask[mask]

                        dec_logits = logits_abs
                        if dec_mask is not None:
                            dec_t = dec_t[dec_mask]
                            dec_logits = dec_logits[dec_mask]

                        if dec_t.numel() > 0:
                            dec_target = _uncertainty_target_from_metric(
                                dec_t,
                                mu=decoder_aux_mu,
                                sigma=decoder_aux_sigma,
                                higher_is_worse=decoder_aux_higher_worse,
                            )
                            # Convención: mayor incertidumbre esperada para peor WER.
                            pred_unc = torch.sigmoid(-dec_logits)
                            decoder_aux = F.smooth_l1_loss(pred_unc, dec_target)
                            loss = loss + decoder_aux_weight * decoder_aux

                    if source_loss_weight != 1.0:
                        loss = loss * source_loss_weight

                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    ctx.amp.zero_grad("WERD")
                    continue

                loss_step = loss / float(accum_steps)
                ctx.amp.backward("WERD", loss_step)
                do_step = ((it + 1) % accum_steps == 0) or ((it + 1) == len(active_loader))

                if do_step:
                    grad_norm = ctx.amp.clip_grad_norm_("WERD", [p for p in model.parameters() if p.requires_grad], grad_clip)
                    if not np.isfinite(float(grad_norm)):
                        skipped_nonfinite += 1
                        ctx.amp.zero_grad("WERD")
                        continue
                    ctx.amp.step("WERD")
                    ctx.amp.update("WERD")
                    ctx.amp.zero_grad("WERD")
                    global_step += 1
                else:
                    grad_norm = 0.0

                replay_flag = float(loader_tag == "replay")
                replay_batches += int(replay_flag)
                meter.add(
                    loss=float(loss.detach()),
                    sup=float(sup_loss.detach()),
                    rank=float(rank_loss.detach()),
                    distill=float(distill.detach()),
                    aux_decoder=float(decoder_aux.detach()),
                    pairs=float(npairs),
                    grad_norm=float(grad_norm),
                    replay=replay_flag,
                )
                epoch_meter.add(
                    loss=float(loss.detach()),
                    sup=float(sup_loss.detach()),
                    rank=float(rank_loss.detach()),
                    distill=float(distill.detach()),
                    aux_decoder=float(decoder_aux.detach()),
                    pairs=float(npairs),
                    grad_norm=float(grad_norm),
                    replay=replay_flag,
                )
                if do_step and global_step % int(cfg["training"].get("log_every", 50)) == 0:
                    ctx.logger.log_scalars("train", global_step, meter.mean_dict(clear=True))

        epoch_logs = epoch_meter.mean_dict(clear=True)
        if epoch_logs:
            print(
                "[pretrain_discriminator][epoch "
                f"{epoch+1}] "
                + " | ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_logs.items()))
                + f" | replay_batches={replay_batches}"
                + f" | skipped_nonfinite={skipped_nonfinite}"
            )
            ctx.logger.log_scalars("epoch_train", epoch + 1, epoch_logs)
            ctx.logger.log_scalars(
                "epoch_train_debug",
                epoch + 1,
                {
                    "skipped_nonfinite": float(skipped_nonfinite),
                    "replay_batches": float(replay_batches),
                },
            )
            skipped_nonfinite = 0

        if (epoch + 1) % int(cfg["training"].get("validate_every_epochs", 1)) == 0:
            print(
                f"[pretrain_discriminator][epoch {epoch+1}] Iniciando validación "
                f"correlación en split=val"
            )
            val = validate_discriminator_correlation(
                model,
                val_loader,
                device=device,
                output_head=validation_head,
                rank_min_delta=pair_min_delta,
                decoder_metric_key=(decoder_aux_key if decoder_aux_weight > 0.0 else None),
                decoder_metric_higher_worse=decoder_aux_higher_worse,
            )
            print(
                f"[pretrain_discriminator][epoch {epoch+1}] val_done "
                f"pearson_q={val.get('pearson_quality', float('nan')):.4f} "
                f"spearman_q={val.get('spearman_quality', float('nan')):.4f} "
                f"spearman_logwer={val.get('spearman_logwer', float('nan')):.4f} "
                f"spearman_dec={val.get('spearman_decoder_metric', float('nan')):.4f} "
                f"rank_acc={val.get('rank_accuracy', float('nan')):.4f} "
                f"rank_pairs={int(val.get('rank_pairs_logwer', 0.0))} "
                f"n_q={int(val.get('n_quality', 0.0))}"
            )

            ctx.logger.log_scalars("val", global_step, val)
            score = val.get(selection_metric, None)
            if (score is None) or (not np.isfinite(float(score))):
                score = None
            state = build_state(
                epoch=epoch + 1,
                global_step=global_step,
                best_score=ctx.ckpt.best_score,
                models={"wer_discriminator": model},
                optimizers={"WERD": opt},
                amp=ctx.amp,
            )
            ctx.ckpt.save(state, score=score)

            if score is not None:
                cur = float(score)
                if cur > (best_seen + early_stop_min_delta):
                    best_seen = cur
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                        print(
                            "[pretrain_discriminator] early stop: "
                            f"patience={early_stop_patience}, best={best_seen:.4f}, "
                            f"last={cur:.4f}, epoch={epoch+1}"
                        )
                        should_stop = True
        if should_stop:
            break

    return {
        "global_step": global_step,
        "best": ctx.ckpt.best_score,
        "target_mu": mu,
        "target_sigma": sigma,
        "target_quantile_n": int(0 if quantile_ref is None else quantile_ref.numel()),
        "decoder_aux_key": decoder_aux_key,
        "decoder_aux_n": int(decoder_aux_n),
        "use_replay": bool(use_replay),
        "replay_loss_weight": float(replay_loss_weight),
    }

from __future__ import annotations

import gc
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.ecu911_dataset import create_ecu911_dataloader
from engine.logging import MetricAccumulator
from engine.losses import multi_resolution_stft_loss, pairwise_rank_loss
from engine.validation import validate_wer_ecu911
from models.generator import create_generator
from models.wer_discriminator import create_wer_discriminator
from trainers.common import build_state, create_optimizer, load_resume_state, load_whisper_asr
from utils.mel_transform import create_mel_transform


def _load_checkpoint_if_exists(model: torch.nn.Module, ckpt_path: str | None, key: str) -> bool:
    if not ckpt_path:
        return False
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get(key, ckpt), strict=False)
    return True


def _default_stage_best_ckpt(cfg: dict, stage: str) -> Path:
    exp_name = str(cfg.get("experiment_name", "speech_enhancement"))
    return Path(cfg["paths"]["checkpoints"]) / exp_name / stage / "best.pt"


def _resolve_ckpt_or_default(explicit_path: str | None, default_path: Path) -> str | None:
    if explicit_path:
        return explicit_path
    if default_path.exists():
        return str(default_path)
    return None


def _compute_wer_adv_loss(
    logits_enh: torch.Tensor,
    logits_noisy: torch.Tensor,
    mode: str,
    pair_margin: float,
    target_quality: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = mode.strip().lower()
    if mode == "pairwise":
        loss_adv = pairwise_rank_loss(logits_enh, logits_noisy)
        adv_gap = (logits_enh - logits_noisy).mean()
        adv_quality = torch.sigmoid(logits_enh).mean()
        return loss_adv, adv_gap, adv_quality
    if mode == "pairwise_margin":
        loss_adv = F.softplus(-(logits_enh - logits_noisy - pair_margin)).mean()
        adv_gap = (logits_enh - logits_noisy).mean()
        adv_quality = torch.sigmoid(logits_enh).mean()
        return loss_adv, adv_gap, adv_quality
    if mode == "pairwise_margin_focus":
        # Enfatiza samples que D_WER ya considera claramente malos en la entrada noisy.
        # La idea es evitar empujar con la misma fuerza a ejemplos que ya están razonablemente bien.
        q_noisy = torch.sigmoid(logits_noisy).detach()
        q_enh = torch.sigmoid(logits_enh)
        focus = torch.clamp(target_quality - q_noisy, min=0.0)
        if torch.any(focus > 0):
            focus = focus / focus.sum().clamp_min(1e-6)
            loss_adv = (F.softplus(-(q_enh - q_noisy - pair_margin)) * focus).sum()
        else:
            loss_adv = q_enh.new_tensor(0.0)
        adv_gap = (q_enh - q_noisy).mean()
        adv_quality = q_enh.mean()
        return loss_adv, adv_gap, adv_quality
    if mode == "abs_mean":
        loss_adv = -logits_enh.mean()
        adv_gap = (logits_enh - logits_noisy).mean()
        adv_quality = torch.sigmoid(logits_enh).mean()
        return loss_adv, adv_gap, adv_quality
    if mode == "abs_margin":
        q_enh = torch.sigmoid(logits_enh)
        q_noisy = torch.sigmoid(logits_noisy)
        loss_adv = F.relu(target_quality - q_enh).mean()
        adv_gap = (q_enh - q_noisy).mean()
        adv_quality = q_enh.mean()
        return loss_adv, adv_gap, adv_quality
    raise ValueError(
        f"wer_adv_loss_mode inválido: {mode}. "
        "Usa 'pairwise', 'pairwise_margin', 'pairwise_margin_focus', 'abs_mean' o 'abs_margin'."
    )


def _semantic_anchor_loss(whisper_model, mel_transform, noisy: torch.Tensor, enh: torch.Tensor, device: torch.device):
    expected_frames = int(getattr(whisper_model.model.encoder.config, "max_source_positions", 1500) * 2)

    def _fit_mel_frames(mel: torch.Tensor, target_frames: int) -> torch.Tensor:
        t = int(mel.shape[-1])
        if t < target_frames:
            return torch.nn.functional.pad(mel, (0, target_frames - t))
        if t > target_frames:
            return mel[..., :target_frames]
        return mel

    mel_noisy = mel_transform(noisy.to(device))
    mel_enh = mel_transform(enh.to(device))
    mel_noisy = _fit_mel_frames(mel_noisy, expected_frames)
    mel_enh = _fit_mel_frames(mel_enh, expected_frames)

    enc = whisper_model.model.encoder
    feat_noisy = enc(mel_noisy).last_hidden_state
    feat_enh = enc(mel_enh).last_hidden_state
    return F.l1_loss(feat_enh, feat_noisy.detach())


def _train_chunk_batch(
    noisy: torch.Tensor,
    durations: torch.Tensor | None,
    sample_rate: int,
    chunk_seconds: float,
    random_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Recorta batch a una ventana fija para evitar OOM en finetune con audios largos.
    No afecta evaluación, solo entrenamiento.
    """
    if chunk_seconds <= 0:
        return noisy, durations

    target_len = int(sample_rate * chunk_seconds)
    if target_len <= 0:
        return noisy, durations

    # Fallback simple si no hay duraciones reales en el batch.
    if durations is None:
        if noisy.shape[-1] <= target_len:
            return noisy, None
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
        return noisy[..., start:end], dur_out

    # Recorte por sample usando duración real para no entrenar sobre padding.
    durations_cpu = durations.detach().float().cpu()
    chunks = []
    chunk_durations = []
    for i in range(noisy.shape[0]):
        true_len = int(round(float(durations_cpu[i].item()) * float(sample_rate)))
        true_len = max(1, min(true_len, int(noisy.shape[-1])))
        wav = noisy[i, :true_len]

        if true_len > target_len:
            if random_chunk:
                max_start = true_len - target_len
                start = int(torch.randint(0, max_start + 1, (1,), device=noisy.device).item())
            else:
                start = 0
            wav = wav[start : start + target_len]
            cur_len = target_len
        else:
            cur_len = true_len
            if wav.shape[-1] < target_len:
                wav = F.pad(wav, (0, target_len - wav.shape[-1]))

        chunks.append(wav)
        chunk_durations.append(float(cur_len) / float(sample_rate))

    noisy_out = torch.stack(chunks, dim=0)
    dur_out = torch.tensor(chunk_durations, device=noisy.device, dtype=torch.float32)
    return noisy_out, dur_out


def _resolve_chunk_seconds(
    epoch_idx: int,
    total_epochs: int,
    base_chunk_seconds: float,
    final_chunk_seconds: float,
    final_chunk_epochs: int,
) -> float:
    """
    Curriculum de contexto:
    - mayoría de epochs: chunk base (ahorra VRAM)
    - últimas N epochs: chunk más largo para reducir mismatch de contexto
    """
    if final_chunk_epochs <= 0:
        return base_chunk_seconds
    if (epoch_idx + 1) > (total_epochs - final_chunk_epochs):
        return final_chunk_seconds
    return base_chunk_seconds


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


def _clear_nontrain_cuda_memory(
    train_device: torch.device,
    *devices: torch.device,
) -> None:
    """
    Limpia caché CUDA en dispositivos auxiliares, pero evita tocar la caché de
    train_device fuera de OOM para no perder bloques reutilizables entre epochs.
    """
    filt = []
    train_key = (train_device.type, int(train_device.index if train_device.index is not None else -1))
    for dev in devices:
        key = (dev.type, int(dev.index if dev.index is not None else -1))
        if key == train_key:
            continue
        filt.append(dev)
    if len(filt) > 0:
        _clear_cuda_memory(*filt)


def run_finetune(ctx, resume: str | None = None, generator_ckpt: str | None = None, werd_ckpt: str | None = None):
    cfg = ctx.config
    train_device = ctx.devices.train_device
    whisper_device = ctx.devices.whisper_device
    fcfg = cfg.get("finetune", {})

    generator_ckpt = _resolve_ckpt_or_default(generator_ckpt, _default_stage_best_ckpt(cfg, "pretrain"))
    werd_ckpt = _resolve_ckpt_or_default(werd_ckpt, _default_stage_best_ckpt(cfg, "pretrain_discriminator"))

    generator = create_generator(cfg).to(train_device)
    loaded_gen = _load_checkpoint_if_exists(generator, generator_ckpt, key="generator")
    if loaded_gen:
        print(f"[finetune] generator checkpoint cargado: {generator_ckpt}")
    else:
        print("[finetune] ⚠️ generator sin checkpoint pretrain (entrenará desde init salvo resume).")

    # Permite usar chunking específico de D_WER durante finetune
    # sin alterar la config global de pretrain_discriminator.
    cfg_for_werd = cfg
    werd_chunk_seconds = fcfg.get("werd_chunk_seconds", None)
    werd_chunk_overlap_seconds = fcfg.get("werd_chunk_overlap_seconds", None)
    if (werd_chunk_seconds is not None) or (werd_chunk_overlap_seconds is not None):
        cfg_for_werd = deepcopy(cfg)
        mcfg_werd = cfg_for_werd.setdefault("models", {}).setdefault("wer_discriminator", {})
        if werd_chunk_seconds is not None:
            mcfg_werd["chunk_seconds"] = float(werd_chunk_seconds)
        if werd_chunk_overlap_seconds is not None:
            mcfg_werd["chunk_overlap_seconds"] = float(werd_chunk_overlap_seconds)
    mcfg_runtime = cfg_for_werd.get("models", {}).get("wer_discriminator", {})
    print(
        "[finetune] D_WER chunking: "
        f"chunk_seconds={float(mcfg_runtime.get('chunk_seconds', 20.0)):.2f} "
        f"overlap_seconds={float(mcfg_runtime.get('chunk_overlap_seconds', 2.0)):.2f}"
    )
    werd = create_wer_discriminator(cfg_for_werd, device=whisper_device).to(whisper_device)
    loaded_werd = _load_checkpoint_if_exists(werd, werd_ckpt, key="wer_discriminator")
    if loaded_werd:
        print(f"[finetune] WERD checkpoint cargado: {werd_ckpt}")
    else:
        print("[finetune] ⚠️ WERD sin checkpoint pretrain_discriminator (adv puede degradar).")

    freeze_werd = bool(fcfg.get("freeze_werd", True))
    for p in werd.parameters():
        p.requires_grad = not freeze_werd
    werd.eval()

    opt_g = create_optimizer(generator, cfg["optimizer"]["generator"])
    ctx.amp.register_optimizer("G", opt_g)

    finetune_train_wer_filter_max = fcfg.get("train_wer_filter_max", None)
    if finetune_train_wer_filter_max is not None:
        finetune_train_wer_filter_max = float(finetune_train_wer_filter_max)
        print(
            "[finetune] train WER filter activo: "
            f"thr={finetune_train_wer_filter_max:.2f} "
            "(alineando soporte de G con D_WER)."
        )

    train_loader = create_ecu911_dataloader(
        cfg,
        stage="train",
        purpose="default",
        apply_wer_filter=(finetune_train_wer_filter_max is not None),
        wer_filter_max_train=finetune_train_wer_filter_max,
        filter_eval_for_wer_filter=False,
    )
    val_loader = create_ecu911_dataloader(cfg, stage="val", purpose="default")

    semantic_whisper, _ = load_whisper_asr(cfg, whisper_device, for_evaluation=False)
    for p in semantic_whisper.parameters():
        p.requires_grad = False
    semantic_whisper.eval()

    eval_whisper, eval_proc = load_whisper_asr(cfg, ctx.devices.validate_device, for_evaluation=True)
    eval_whisper.eval()

    mel_transform = create_mel_transform(cfg, device=whisper_device)

    # Validación opcional en GPU dedicada para no fragmentar la VRAM de entrenamiento.
    validate_device = ctx.devices.validate_device
    separate_val_gen_device = bool(fcfg.get("validate_generator_on_validate_device", True))
    same_train_and_val_device = (
        (validate_device.type == train_device.type)
        and ((validate_device.index if validate_device.index is not None else -1) == (train_device.index if train_device.index is not None else -1))
    )
    use_split_val_gen = separate_val_gen_device and (not same_train_and_val_device)
    val_generator = None
    if use_split_val_gen:
        try:
            val_generator = create_generator(cfg).to(validate_device)
            val_generator.eval()
            for p in val_generator.parameters():
                p.requires_grad = False
            print(
                "[finetune] validation generator en device dedicado: "
                f"{validate_device} (train en {train_device})."
            )
        except (torch.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            is_oom = isinstance(e, torch.OutOfMemoryError) or ("out of memory" in msg)
            if not is_oom:
                raise
            val_generator = None
            print(
                "[finetune] ⚠️ no se pudo reservar generator de validación en "
                f"{validate_device}; se usa train_device={train_device}."
            )
            _clear_cuda_memory(validate_device)
    else:
        print(
            "[finetune] validation generator comparte train_device "
            f"({train_device})."
        )

    resume_werd_from_finetune_state = bool(fcfg.get("resume_werd_from_finetune_state", False))
    resume_models = {"generator": generator}
    if resume_werd_from_finetune_state:
        resume_models["wer_discriminator"] = werd
    else:
        print(
            "[finetune] resume_werd_from_finetune_state=False "
            "(D_WER se toma del ckpt de pretrain_discriminator, no de finetune/last.pt)."
        )

    epoch0, global_step = load_resume_state(
        ctx.ckpt,
        resume,
        models=resume_models,
        optimizers={"G": opt_g},
        amp=ctx.amp,
    )

    lcfg = cfg.get("losses", {})
    fft_sizes = [int(x) for x in lcfg.get("mrstft_fft_sizes", [256, 512, 1024])]
    hop_sizes = [int(x) for x in lcfg.get("mrstft_hop_sizes", [64, 128, 256])]

    use_wer_adv = bool(fcfg.get("use_wer_adv", True))
    wer_adv_weight = float(fcfg.get("wer_adv_weight", 0.25))
    wer_adv_warmup = int(fcfg.get("wer_adv_warmup_steps", 1200))
    wer_adv_every = int(fcfg.get("wer_adv_every_n_steps", 2))
    wer_adv_head = str(fcfg.get("wer_adv_head", "rel")).strip().lower()
    if wer_adv_head not in ("abs", "rel"):
        raise ValueError(f"wer_adv_head inválido: {wer_adv_head}. Usa 'abs' o 'rel'.")
    wer_adv_force_fp32 = bool(fcfg.get("wer_adv_force_fp32", True))
    wer_adv_loss_mode = str(fcfg.get("wer_adv_loss_mode", "pairwise")).strip().lower()
    if wer_adv_loss_mode not in ("pairwise", "pairwise_margin", "pairwise_margin_focus", "abs_mean", "abs_margin"):
        raise ValueError(
            f"wer_adv_loss_mode inválido: {wer_adv_loss_mode}. "
            "Usa 'pairwise', 'pairwise_margin', 'pairwise_margin_focus', 'abs_mean' o 'abs_margin'."
        )
    wer_adv_pair_margin = float(fcfg.get("wer_adv_pair_margin", 0.0))
    wer_adv_target_quality = float(fcfg.get("wer_adv_target_quality", 0.70))
    wer_adv_target_quality = float(np.clip(wer_adv_target_quality, 0.0, 1.0))
    wer_adv_auto_disable_without_ckpt = bool(fcfg.get("wer_adv_auto_disable_without_ckpt", True))

    use_sem = bool(fcfg.get("use_semantic_anchor", True))
    sem_w = float(fcfg.get("semantic_anchor_weight", 0.1))
    sem_every = int(fcfg.get("semantic_anchor_every_n_steps", 8))

    stft_w = float(fcfg.get("stft_anchor_weight", 1.0))
    id_l1_w = float(fcfg.get("identity_l1_weight", 0.05))
    wer_adv_loss_cap = float(fcfg.get("wer_adv_loss_cap", 0.0))
    skip_nonfinite_steps = bool(fcfg.get("skip_nonfinite_steps", True))

    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    accum_steps = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    epochs = int(cfg["training"]["epochs"])
    eval_cfg = cfg.get("evaluation", {})
    full_val_samples = int(eval_cfg.get("max_val_samples", 128))
    quick_val_samples = int(eval_cfg.get("quick_val_max_samples", min(32, full_val_samples)))
    full_val_every = max(1, int(eval_cfg.get("full_val_every_epochs", 5)))
    force_val_samples = eval_cfg.get("force_val_samples", None)
    force_val_samples = None if force_val_samples is None else max(1, int(force_val_samples))
    train_chunk_seconds = float(fcfg.get("train_chunk_seconds", 8.0))
    train_chunk_seconds_final = float(fcfg.get("train_chunk_seconds_final", train_chunk_seconds))
    train_chunk_final_epochs = int(fcfg.get("train_chunk_final_epochs", 0))
    train_random_chunk = bool(fcfg.get("train_random_chunk", True))
    oom_chunk_backoff = float(fcfg.get("oom_chunk_backoff", 1.0))
    oom_chunk_backoff = float(np.clip(oom_chunk_backoff, 0.1, 1.0))
    oom_min_chunk_seconds = float(fcfg.get("oom_min_chunk_seconds", train_chunk_seconds))
    oom_min_chunk_seconds = max(1.0, oom_min_chunk_seconds)
    sr = int(cfg["audio"]["target_sr"])
    has_werd_source = loaded_werd or bool(resume)
    if use_wer_adv and (not has_werd_source) and wer_adv_auto_disable_without_ckpt:
        print(
            "[finetune] ⚠️ use_wer_adv=True pero no hay checkpoint de WERD ni resume. "
            "Se desactiva loss adversarial para evitar entrenamiento inestable."
        )
        use_wer_adv = False
    if use_wer_adv and (wer_adv_loss_mode.startswith("abs") or wer_adv_loss_mode == "pairwise_margin_focus") and wer_adv_head != "abs":
        print(
            f"[finetune] ℹ️ wer_adv_loss_mode={wer_adv_loss_mode} recomienda head='abs'. "
            f"Se fuerza wer_adv_head de '{wer_adv_head}' a 'abs'."
        )
        wer_adv_head = "abs"
    if use_wer_adv:
        print(
            "[finetune] "
            f"wer_adv_mode={wer_adv_loss_mode} head={wer_adv_head} "
            f"weight={wer_adv_weight} warmup={wer_adv_warmup} every={wer_adv_every} "
            f"pair_margin={wer_adv_pair_margin:.3f} target_q={wer_adv_target_quality:.3f} "
            f"cap={wer_adv_loss_cap:.3f}"
        )
    skipped_nonfinite = 0
    skipped_nonfinite_adv = 0
    skipped_oom = 0
    adaptive_chunk_seconds = float(train_chunk_seconds)

    for epoch in range(epoch0, epochs):
        _clear_nontrain_cuda_memory(train_device, whisper_device, ctx.devices.validate_device)
        epoch_chunk_seconds = _resolve_chunk_seconds(
            epoch_idx=epoch,
            total_epochs=epochs,
            base_chunk_seconds=train_chunk_seconds,
            final_chunk_seconds=train_chunk_seconds_final,
            final_chunk_epochs=train_chunk_final_epochs,
        )
        # Si hubo OOM en epochs anteriores, no volver a un chunk mayor automáticamente.
        current_chunk_seconds = min(float(epoch_chunk_seconds), float(adaptive_chunk_seconds))
        generator.train()
        meter = MetricAccumulator()
        epoch_meter = MetricAccumulator()
        epoch_optimizer_steps = 0
        oom_log_budget = 3
        ctx.amp.zero_grad("G")
        print(
            f"[finetune][epoch {epoch+1}] train_chunk_seconds={current_chunk_seconds:.2f} "
            f"random_chunk={train_random_chunk} "
            f"oom_backoff={oom_chunk_backoff:.2f} oom_min_chunk={oom_min_chunk_seconds:.2f}"
        )

        for it, batch in enumerate(tqdm(train_loader, desc=f"[finetune] epoch {epoch+1}/{epochs}")):
            # Limpia referencias de la iteración previa antes de procesar el siguiente batch.
            durations = None
            noisy = None
            chunk_durations = None
            enh = None
            loss_stft = None
            loss_id = None
            loss = None
            loss_sem = None
            loss_wer_adv = None
            wer_adv_gap = None
            wer_adv_quality = None
            try:
                durations = batch.get("durations")
                if durations is not None:
                    durations = durations.to(train_device, non_blocking=True)
                noisy = batch["waveform"].to(train_device)
                noisy, chunk_durations = _train_chunk_batch(
                    noisy,
                    durations=durations,
                    sample_rate=sr,
                    chunk_seconds=current_chunk_seconds,
                    random_chunk=train_random_chunk,
                )
                audio_paths = batch.get("audio_paths", None)
                cur_step = int(global_step)

                with ctx.amp.autocast():
                    enh = generator(noisy)

                    loss_stft = multi_resolution_stft_loss(enh, noisy, fft_sizes=fft_sizes, hop_sizes=hop_sizes)
                    loss_id = F.l1_loss(enh, noisy)
                    loss = stft_w * loss_stft + id_l1_w * loss_id

                    do_sem = use_sem and (cur_step % max(1, sem_every) == 0)
                    if do_sem:
                        loss_sem = _semantic_anchor_loss(
                            semantic_whisper,
                            mel_transform,
                            noisy,
                            enh,
                            device=whisper_device,
                        ).to(train_device)
                        loss = loss + sem_w * loss_sem
                    else:
                        loss_sem = enh.new_tensor(0.0)
                    wer_adv_gap = enh.new_tensor(0.0)
                    wer_adv_quality = enh.new_tensor(0.0)

                    do_adv = (
                        use_wer_adv
                        and (cur_step >= wer_adv_warmup)
                        and (cur_step % max(1, wer_adv_every) == 0)
                    )
                    if do_adv:
                        d_dur = chunk_durations.to(whisper_device) if chunk_durations is not None else None
                        d_dtype = torch.float32 if wer_adv_force_fp32 else enh.dtype
                        adv_ctx = (
                            torch.autocast(device_type=whisper_device.type, enabled=False)
                            if (wer_adv_force_fp32 and whisper_device.type == "cuda")
                            else nullcontext()
                        )
                        with adv_ctx:
                            noisy_d = noisy.to(device=whisper_device, dtype=d_dtype)
                            enh_d = enh.to(device=whisper_device, dtype=d_dtype)
                            with torch.no_grad():
                                logits_noisy = werd(
                                    noisy_d,
                                    durations=d_dur,
                                    audio_paths=audio_paths,
                                    output=wer_adv_head,
                                ).float()
                            logits_enh = werd(
                                enh_d,
                                durations=d_dur,
                                audio_paths=audio_paths,
                                output=wer_adv_head,
                            ).float()

                        if torch.isfinite(logits_noisy).all() and torch.isfinite(logits_enh).all():
                            loss_wer_adv, wer_adv_gap, wer_adv_quality = _compute_wer_adv_loss(
                                logits_enh=logits_enh,
                                logits_noisy=logits_noisy,
                                mode=wer_adv_loss_mode,
                                pair_margin=wer_adv_pair_margin,
                                target_quality=wer_adv_target_quality,
                            )
                            loss_wer_adv = loss_wer_adv.to(train_device)
                            wer_adv_gap = wer_adv_gap.to(train_device)
                            wer_adv_quality = wer_adv_quality.to(train_device)
                            if wer_adv_loss_cap > 0.0:
                                loss_wer_adv = torch.clamp(loss_wer_adv, max=wer_adv_loss_cap)
                            if wer_adv_warmup > 0:
                                warm = min(1.0, (cur_step - wer_adv_warmup + 1) / float(wer_adv_warmup))
                            else:
                                warm = 1.0
                            loss = loss + wer_adv_weight * warm * loss_wer_adv
                        else:
                            skipped_nonfinite_adv += 1
                            loss_wer_adv = enh.new_tensor(0.0)
                    else:
                        loss_wer_adv = enh.new_tensor(0.0)

                if skip_nonfinite_steps and (not torch.isfinite(loss)):
                    skipped_nonfinite += 1
                    ctx.amp.zero_grad("G")
                    continue

                loss_step = loss / accum_steps
                ctx.amp.backward("G", loss_step)
                do_step = ((it + 1) % accum_steps == 0) or ((it + 1) == len(train_loader))
                if do_step:
                    grad_norm = ctx.amp.clip_grad_norm_("G", generator.parameters(), grad_clip)
                    if skip_nonfinite_steps and (not np.isfinite(float(grad_norm))):
                        skipped_nonfinite += 1
                        # Importante con AMP: si ya hubo unscale_, hay que hacer update()
                        # para evitar RuntimeError en el próximo unscale_ del mismo optimizer.
                        ctx.amp.update("G")
                        ctx.amp.zero_grad("G")
                        continue
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
                        f"[finetune][epoch {epoch+1}] OOM(it={it+1}, chunk={current_chunk_seconds:.2f}s): "
                        f"{short_msg}"
                    )
                    oom_log_budget -= 1
                skipped_oom += 1
                skipped_nonfinite += 1
                ctx.amp.zero_grad("G")
                # En OOM sí conviene intentar liberar todo, incluyendo train_device.
                _clear_cuda_memory(train_device, whisper_device, ctx.devices.validate_device)
                if oom_chunk_backoff < 0.999 and current_chunk_seconds > oom_min_chunk_seconds:
                    new_chunk = max(oom_min_chunk_seconds, current_chunk_seconds * oom_chunk_backoff)
                    if new_chunk < current_chunk_seconds:
                        print(
                            "[finetune] OOM detectado; reduciendo train_chunk_seconds "
                            f"{current_chunk_seconds:.2f} -> {new_chunk:.2f} (epoch {epoch+1}, it {it+1})"
                        )
                        current_chunk_seconds = new_chunk
                        adaptive_chunk_seconds = min(float(adaptive_chunk_seconds), float(new_chunk))
                continue

            meter.add(
                loss=float(loss.detach()),
                loss_stft=float(loss_stft.detach()),
                loss_id=float(loss_id.detach()),
                loss_sem=float(loss_sem.detach()),
                loss_wer_adv=float(loss_wer_adv.detach()),
                wer_adv_gap=float(wer_adv_gap.detach()),
                wer_adv_quality=float(wer_adv_quality.detach()),
                grad_norm=float(grad_norm),
            )
            epoch_meter.add(
                loss=float(loss.detach()),
                loss_stft=float(loss_stft.detach()),
                loss_id=float(loss_id.detach()),
                loss_sem=float(loss_sem.detach()),
                loss_wer_adv=float(loss_wer_adv.detach()),
                wer_adv_gap=float(wer_adv_gap.detach()),
                wer_adv_quality=float(wer_adv_quality.detach()),
                grad_norm=float(grad_norm),
            )
            if do_step and global_step % int(cfg["training"].get("log_every", 50)) == 0:
                ctx.logger.log_scalars("train", global_step, meter.mean_dict(clear=True))

        # Evita que queden vivos tensores grandes del último batch al pasar a validación/siguiente epoch.
        durations = None
        noisy = None
        chunk_durations = None
        enh = None
        loss_stft = None
        loss_id = None
        loss = None
        loss_sem = None
        loss_wer_adv = None
        wer_adv_gap = None
        wer_adv_quality = None
        _clear_nontrain_cuda_memory(train_device, whisper_device, ctx.devices.validate_device)

        epoch_logs = epoch_meter.mean_dict(clear=True)
        if epoch_logs:
            print(
                "[finetune][epoch "
                f"{epoch+1}] "
                + " | ".join(f"{k}={v:.4f}" for k, v in sorted(epoch_logs.items()))
                + f" | optimizer_steps={epoch_optimizer_steps}"
                + f" | skipped_nonfinite={skipped_nonfinite}"
                + f" | skipped_nonfinite_adv={skipped_nonfinite_adv}"
                + f" | skipped_oom={skipped_oom}"
            )
            ctx.logger.log_scalars("epoch_train", epoch + 1, epoch_logs)
            ctx.logger.log_scalars(
                "epoch_train_debug",
                epoch + 1,
                {
                    "optimizer_steps": float(epoch_optimizer_steps),
                    "skipped_nonfinite": float(skipped_nonfinite),
                    "skipped_nonfinite_adv": float(skipped_nonfinite_adv),
                    "skipped_oom": float(skipped_oom),
                },
            )
            skipped_nonfinite = 0
            skipped_nonfinite_adv = 0
            skipped_oom = 0
        else:
            # Caso típico: todos los batches fueron saltados (p.ej. OOM continuo).
            print(
                f"[finetune][epoch {epoch+1}] ⚠️ sin métricas de entrenamiento "
                f"(optimizer_steps={epoch_optimizer_steps}, skipped_oom={skipped_oom}, "
                f"skipped_nonfinite={skipped_nonfinite})."
            )

        if epoch_optimizer_steps == 0:
            print(
                f"[finetune][epoch {epoch+1}] ⚠️ no hubo updates de G. "
                "Probable OOM en todos los batches. "
                "Reduce batch_size o oom_min_chunk_seconds."
            )

        if (epoch + 1) % int(cfg["training"].get("validate_every_epochs", 1)) == 0:
            is_full = ((epoch + 1) % full_val_every == 0) or ((epoch + 1) == epochs)
            val_samples = full_val_samples if is_full else quick_val_samples
            if force_val_samples is not None:
                val_samples = min(val_samples, force_val_samples)
            generator_for_val = generator
            generator_val_device = train_device
            if val_generator is not None:
                # Sync de pesos hacia la copia de validación en GPU dedicada.
                with torch.no_grad():
                    val_generator.load_state_dict(generator.state_dict(), strict=True)
                val_generator.eval()
                generator_for_val = val_generator
                generator_val_device = validate_device
            print(
                f"[finetune][epoch {epoch+1}] Iniciando validación "
                f"mode={'full' if is_full else 'quick'} samples={val_samples} "
                f"gen_device={generator_val_device} "
                f"use_precomputed_noisy_wer={bool(eval_cfg.get('use_precomputed_noisy_wer', True))}"
            )
            val = validate_wer_ecu911(
                generator_for_val,
                val_loader,
                eval_whisper,
                eval_proc,
                train_device=generator_val_device,
                whisper_device=ctx.devices.validate_device,
                sample_rate=int(cfg["audio"]["target_sr"]),
                max_samples=val_samples,
                chunk_seconds=float(eval_cfg.get("whisper_chunk_seconds", 30.0)),
                overlap_seconds=float(eval_cfg.get("whisper_overlap_seconds", 1.0)),
                condition_on_prev_tokens=bool(eval_cfg.get("condition_on_prev_tokens", False)),
                use_precomputed_noisy_wer=bool(eval_cfg.get("use_precomputed_noisy_wer", True)),
                precomputed_noisy_wer_max=eval_cfg.get("precomputed_noisy_wer_max", 5.0),
                per_sample_wer_cap=eval_cfg.get("per_sample_wer_cap", None),
                generator_chunk_seconds=float(eval_cfg.get("generator_eval_chunk_seconds", current_chunk_seconds)),
                generator_overlap_seconds=float(eval_cfg.get("generator_eval_overlap_seconds", 1.0)),
                generator_min_chunk_seconds=float(eval_cfg.get("generator_eval_min_chunk_seconds", 8.0)),
                generator_direct_max_seconds=float(eval_cfg.get("generator_eval_direct_max_seconds", 12.0)),
            )
            print(
                f"[finetune][epoch {epoch+1}] val_done "
                f"wer_orig={val.get('wer_orig', float('nan')):.4f} "
                f"wer_noisy={val.get('wer_noisy', float('nan')):.4f} "
                f"wer_orig_corpus={val.get('wer_orig_corpus', float('nan')):.4f} "
                f"wer_noisy_corpus={val.get('wer_noisy_corpus', float('nan')):.4f} "
                f"wer_enh={val.get('wer_enh', float('nan')):.4f} "
                f"wer_enh_corpus={val.get('wer_enh_corpus', float('nan')):.4f} "
                f"gain={val.get('wer_gain', float('nan')):.4f} "
                f"gain_corpus={val.get('wer_gain_corpus', float('nan')):.4f} "
                f"pre_used={int(val.get('noisy_precomputed_used', 0.0))} "
                f"pre_rej={int(val.get('noisy_precomputed_rejected', 0.0))} "
                f"skip_empty_ref={int(val.get('skipped_empty_ref', 0.0))} "
                f"skip_enh_oom={int(val.get('skipped_enhance_oom', 0.0))} "
                f"n={int(val.get('n', 0.0))}"
            )
            ctx.logger.log_scalars("val_full" if is_full else "val_quick", global_step, val)
            _clear_nontrain_cuda_memory(train_device, whisper_device, ctx.devices.validate_device)

            state = build_state(
                epoch=epoch + 1,
                global_step=global_step,
                best_score=ctx.ckpt.best_score,
                models={"generator": generator, "wer_discriminator": werd},
                optimizers={"G": opt_g},
                amp=ctx.amp,
            )
            ctx.ckpt.save(state, score=val.get("wer_enh"))
            del state
            _clear_nontrain_cuda_memory(train_device, whisper_device, ctx.devices.validate_device)

    return {"global_step": global_step, "best": ctx.ckpt.best_score}

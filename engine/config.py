from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "experiment_name": "speech_enhancement",
    "paths": {
        "processed_data": "data/processed",
        "checkpoints": "checkpoints",
        "tensorboard": "runs",
        "spc_corpus_dir": "data/spc",
        "ecu911_audios": "data/ecu911/audios",
        "ecu911_metadata": "data/ecu911/metadata.csv",
        "ecu911_test_audios": "data/ecu911/test/noise_audios",
        "ecu911_test_metadata": "data/ecu911/test/metadata.csv",
    },
    "devices": {
        "train_device": None,
        "proxy_device": None,
        "whisper_device": None,
        "validate_device": None,
        "enable_data_parallel": False,
    },
    "audio": {
        "target_sr": 16000,
        "normalize_rms": True,
        "target_rms": 0.1,
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 80,
        "f_min": 0.0,
        "f_max": 8000.0,
    },
    "data": {
        "num_workers": 6,
        "pin_memory": True,
        "prefetch_factor": 2,
        "train_split": 0.9,
        "ecu911": {"batch_size": 8},
        "spc": {"batch_size": 8},
    },
    "training": {
        "epochs": 10,
        "grad_clip": 5.0,
        "log_every": 50,
        "validate_every_epochs": 1,
        "amp": True,
        "grad_accum_steps": 1,
    },
    "optimizer": {
        "generator": {"lr": 2e-4, "betas": [0.9, 0.999], "weight_decay": 1e-5},
        "discriminator": {"lr": 2e-4, "betas": [0.9, 0.999], "weight_decay": 1e-5},
        "wer_discriminator": {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 1e-5},
    },
    "models": {
        "generator": {
            "channels": 64,
            "depth": 4,
            "num_res_blocks": 2,
            "kernel_size": 7,
            "output_mode": "mask_residual",  # mask_residual | hybrid_gain_residual
            "mask_scale": 0.5,
            "gain_scale": 0.25,
            "residual_scale": 0.05,
        },
        "gan_discriminator": {
            "channels": 32,
            "num_layers": 5,
        },
        "wer_discriminator": {
            "hidden_dim": 384,
            "dropout": 0.15,
            "freeze_encoder": True,
            "unfreeze_last_n_layers": 0,
            "encoder_lr_scale": 0.1,
            "enable_grad_accum": True,
            "target_mode": "hybrid",           # quality_sigmoid | log_wer | quantile_logwer | hybrid
            "loss_mode": "smoothl1_rank",      # smoothl1_rank | relative_listwise | ranknet
            "rank_weight": 0.7,
            "sup_weight": 1.0,
            "sup_prob_weight": 0.7,
            "sup_logit_weight": 0.3,
            "rank_margin": 0.10,
            "rank_hard_fraction": 0.5,
            "relative_pair_min_delta": 1e-3,
            "relative_delta_power": 1.0,
            "quantile_eps": 1e-4,
            "selection_metric": "spearman_quality",
            "validation_head": "abs",          # abs | rel
            "use_amp": False,
            "grad_clip": 1.0,
            "resume_weights_only": False,
            "reset_best_on_resume_weights_only": True,
            "chunk_seconds": 30.0,
            "chunk_overlap_seconds": 3.0,
            "max_chunk_frames": 1200,
            "chunk_pooling_mode": "mean",      # mean | mean_topk | attn
            "chunk_topk_fraction": 0.25,
            "chunk_topk_min": 1,
            "use_domain_calibration": True,
            "channel_num_buckets": 64,
            "channel_embed_dim": 16,
            "rel_head_hidden_dim": 192,
            # Rama acústica ligera (mel-CNN) adicional a Whisper features.
            "use_acoustic_branch": False,
            "acoustic_dim": 64,
            "acoustic_chunk_pooling_mode": "mean",
            # Rama waveform multi-escala (x1/x2/x4) adicional a Whisper+mel.
            "use_wave_multiscale_branch": False,
            "wave_dim": 64,
            "wave_scales": [1, 2, 4],
            "wave_chunk_pooling_mode": "mean",
            # Supervisión auxiliar opcional usando métricas de decoder-ASR precomputadas.
            # No hay backprop a Whisper: son targets estáticos desde el dataset.
            "decoder_aux_weight": 0.0,
            "decoder_aux_key": "asr_avg_entropy",
            "decoder_aux_higher_worse": True,
        },
        "whisper": {
            "encoder_model_name": "openai/whisper-small",
        },
    },
    "losses": {
        "mrstft_fft_sizes": [256, 512, 1024],
        "mrstft_hop_sizes": [64, 128, 256],
        "mrstft_weight": 1.0,
        "si_sdr_weight": 0.4,
        "l1_weight": 0.1,
    },
    "pretrain": {
        # 0.0 desactiva chunking y usa el segmento completo.
        "train_chunk_seconds": 0.0,
        "train_random_chunk": True,
        # Backoff automático si aparece OOM en pretrain.
        "oom_chunk_backoff": 1.0,
        "oom_min_chunk_seconds": 4.0,
    },
    "wer_discriminator": {
        "batch_size": 32,
        "wer_cap": 50.0,
        "stratified_sampling": True,
        "strat_high_fraction": 0.5,
        "weighted_sampling": False,
        "weighted_mode": "log1p_wer",  # log1p_wer | wer | rank
        "weighted_alpha": 1.0,
        "weighted_power": 1.0,
        "weighted_clip_max": 4.0,
        "weighted_replacement": True,
        "weighted_num_samples": None,
        "pretrain_batch_min": 8,
        "wer_filter_max_train": 20.0,
        "wer_filter_max_eval": 50.0,
        "filter_eval_for_wer_disc": True,
        "use_longform_key": True,
        "ranking_hard_pairs": True,
        "distill_weight": 0.0,
        "distill_warmup_steps": 500,
        "use_replay": False,
        "replay_dir": "data/replay",
        "replay_capacity": 20000,
        "replay_batch_size": 0,          # 0 => usa batch_size principal
        "replay_samples_per_epoch": 0,   # 0 => usa todo el replay disponible
        "replay_stratify": True,
        "replay_loss_weight": 1.0,
        "replay_start_epoch": 1,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
    },
    "evaluation": {
        "whisper_model_name": "UDA-LIDI/openai-whisper-large-es_ecu911DM",
        "max_val_samples": 128,
        "quick_val_max_samples": 32,
        "full_val_every_epochs": 5,
        "force_val_samples": None,
        "use_precomputed_noisy_wer": True,
        # Rechaza WER precomputado fuera de rango y hace fallback a ASR online en validación.
        "precomputed_noisy_wer_max": 5.0,
        # Corta WER por-sample (noisy/enh) para evitar promedios patológicos con refs cortas.
        "per_sample_wer_cap": 5.0,
        "whisper_chunk_seconds": 30.0,
        "whisper_overlap_seconds": 1.0,
        "condition_on_prev_tokens": False,
        # Controla inferencia del GENERATOR durante validación para evitar OOM
        # en audios largos (no afecta chunking de Whisper-ASR).
        "generator_eval_chunk_seconds": 24.0,
        "generator_eval_overlap_seconds": 1.0,
        "generator_eval_min_chunk_seconds": 8.0,
        "generator_eval_direct_max_seconds": 12.0,
    },
    "preprocessing": {
        "compute_decoder_stats": False,
        # Modelo Whisper explícito para preprocessing (WER + decoder stats).
        "whisper_model_name": "UDA-LIDI/openai-whisper-large-es_ecu911DM",
    },
    "finetune": {
        "use_amp": True,
        "freeze_werd": True,
        "use_wer_adv": True,
        "wer_adv_weight": 0.25,
        "wer_adv_warmup_steps": 1200,
        "wer_adv_every_n_steps": 2,
        "wer_adv_head": "rel",  # abs | rel
        "wer_adv_loss_mode": "pairwise",  # pairwise | pairwise_margin | pairwise_margin_focus | abs_mean | abs_margin
        "wer_adv_pair_margin": 0.0,
        "wer_adv_target_quality": 0.7,
        "wer_adv_loss_cap": 0.0,  # 0 => disabled
        "wer_adv_auto_disable_without_ckpt": True,
        # Si False, al reanudar finetune solo se restaura G/optimizer/amp;
        # D_WER se mantiene desde --werd-ckpt (pretrain_discriminator).
        "resume_werd_from_finetune_state": False,
        # Si True, fuerza fp32 en el forward de D_WER durante finetune para estabilidad.
        "wer_adv_force_fp32": True,
        # Si True, ejecuta validación del generador en validate_device con una copia de G.
        "validate_generator_on_validate_device": True,
        "skip_nonfinite_steps": True,
        "train_chunk_seconds": 30.0,
        "train_chunk_seconds_final": 30.0,
        "train_chunk_final_epochs": 2,
        "train_random_chunk": True,
        # Backoff automático de chunk si hay OOM en entrenamiento.
        # 1.0 desactiva backoff; <1.0 reduce progresivamente.
        "oom_chunk_backoff": 1.0,
        "oom_min_chunk_seconds": 20.0,
        # Override opcional del chunking de D_WER SOLO durante finetune.
        # Si es None, usa models.wer_discriminator.chunk_seconds / overlap.
        "werd_chunk_seconds": None,
        "werd_chunk_overlap_seconds": None,
        "use_semantic_anchor": True,
        "semantic_anchor_weight": 0.1,
        "semantic_anchor_every_n_steps": 8,
        "stft_anchor_weight": 1.0,
        "identity_l1_weight": 0.05,
        "use_replay": False,
        "replay_batch_size": 0,
    },
}


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config no encontrado: {path}")
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        _deep_update(cfg, user_cfg)
    return cfg


def ensure_dirs(config: Dict[str, Any], stage: str) -> Dict[str, Path]:
    exp_name = config.get("experiment_name", "speech_enhancement")
    ckpt_root = Path(config["paths"]["checkpoints"]) / exp_name / stage
    tb_root = Path(config["paths"]["tensorboard"]) / exp_name / stage
    ckpt_root.mkdir(parents=True, exist_ok=True)
    tb_root.mkdir(parents=True, exist_ok=True)
    return {"ckpt_dir": ckpt_root, "tb_dir": tb_root}

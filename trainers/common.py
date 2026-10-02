from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from engine.amp import AmpManager
from engine.checkpointing import CheckpointManager
from engine.logging import TBLogger


@dataclass
class StageContext:
    stage: str
    config: Dict
    devices: any
    amp: AmpManager
    logger: TBLogger
    ckpt: CheckpointManager


def _to_cpu_state(obj):
    """
    Convierte recursivamente estructuras de state_dict a CPU para serialización,
    evitando retener referencias CUDA entre epochs.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_state(v) for v in obj)
    return obj


def create_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    return AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )   


def load_whisper_asr(config: Dict, device: torch.device, for_evaluation: bool = True):
    if for_evaluation:
        model_name = config.get("evaluation", {}).get(
            "whisper_model_name", "UDA-LIDI/openai-whisper-large-es_ecu911DM"
        )
    else:
        model_name = config.get("models", {}).get("whisper", {}).get("encoder_model_name", "openai/whisper-small")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device).eval()
    proc = WhisperProcessor.from_pretrained(model_name)
    return model, proc


def load_resume_state(
    ckpt_manager: CheckpointManager,
    resume_path: Optional[str],
    models: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    amp: AmpManager,
    load_optimizers: bool = True,
    load_amp: bool = True,
    load_counters: bool = True,
) -> Tuple[int, int]:
    state = ckpt_manager.maybe_load_resume(resume_path)
    if state is None:
        return 0, 0

    for k, model in models.items():
        if k in state and isinstance(state[k], dict):
            model.load_state_dict(state[k], strict=False)

    if load_optimizers:
        for k, opt in optimizers.items():
            key = f"opt_{k}"
            if key in state:
                try:
                    opt.load_state_dict(state[key])
                except ValueError as e:
                    # Caso común al cambiar param_groups (p.ej. head-only -> head+encoder).
                    print(
                        f"[resume] ⚠️ no se pudo cargar estado de optimizer '{k}': {e}. "
                        "Se continúa con optimizer re-inicializado."
                    )

    if load_amp and "amp" in state and isinstance(state["amp"], dict):
        amp.load_state_dict(state["amp"])

    ckpt_manager.best_score = state.get("best_score", ckpt_manager.best_score)
    if load_counters:
        return int(state.get("epoch", 0)), int(state.get("global_step", 0))
    return 0, 0


def build_state(
    epoch: int,
    global_step: int,
    best_score: Optional[float],
    models: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    amp: AmpManager,
) -> Dict:
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": best_score,
        "amp": _to_cpu_state(amp.state_dict()),
    }
    for k, m in models.items():
        state[k] = _to_cpu_state(m.state_dict())
    for k, o in optimizers.items():
        state[f"opt_{k}"] = _to_cpu_state(o.state_dict())
    return state

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch


@dataclass
class _ScalerState:
    scaler: torch.amp.GradScaler
    optimizer: torch.optim.Optimizer
    has_unscaled: bool = False


class AmpManager:
    """
    Manejo seguro de AMP con UN scaler por optimizer.
    Evita errores por unscale_ duplicado y mezcla de scalers.
    """

    def __init__(self, enabled: bool, device_type: str = "cuda"):
        self.enabled = bool(enabled and torch.cuda.is_available() and device_type == "cuda")
        self.device_type = device_type
        self._states: Dict[str, _ScalerState] = {}

    def register_optimizer(self, name: str, optimizer: torch.optim.Optimizer) -> None:
        if name in self._states:
            raise ValueError(f"Optimizer ya registrado en AMP manager: {name}")
        # API nueva de PyTorch: torch.amp.GradScaler(device_type, ...)
        scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
        self._states[name] = _ScalerState(scaler=scaler, optimizer=optimizer, has_unscaled=False)

    def zero_grad(self, name: str, set_to_none: bool = True) -> None:
        st = self._states[name]
        st.optimizer.zero_grad(set_to_none=set_to_none)
        st.has_unscaled = False

    def autocast(self, dtype: Optional[torch.dtype] = torch.float16):
        return torch.autocast(device_type=self.device_type, enabled=self.enabled, dtype=dtype)

    def backward(self, name: str, loss: torch.Tensor) -> None:
        st = self._states[name]
        st.scaler.scale(loss).backward()

    def unscale_(self, name: str) -> None:
        st = self._states[name]
        if st.has_unscaled:
            return
        st.scaler.unscale_(st.optimizer)
        st.has_unscaled = True

    def clip_grad_norm_(self, name: str, parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        st = self._states[name]
        if max_norm <= 0:
            return 0.0
        if self.enabled:
            self.unscale_(name)
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        if isinstance(total_norm, torch.Tensor):
            return float(total_norm.detach().cpu())
        return float(total_norm)

    def step(self, name: str) -> None:
        st = self._states[name]
        st.scaler.step(st.optimizer)

    def update(self, name: str) -> None:
        st = self._states[name]
        st.scaler.update()
        st.has_unscaled = False

    def state_dict(self) -> Dict[str, Dict]:
        return {k: v.scaler.state_dict() for k, v in self._states.items()}

    def load_state_dict(self, state_dict: Dict[str, Dict]) -> None:
        for name, scaler_state in state_dict.items():
            if name in self._states:
                self._states[name].scaler.load_state_dict(scaler_state)


class DummyScaler:
    """Compat mínima para casos CPU-only sin AMP."""

    def scale(self, x):
        return x

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class DeviceBundle:
    train_device: torch.device
    proxy_device: torch.device
    whisper_device: torch.device
    validate_device: torch.device
    available_cuda: List[int]


def _parse_device(spec: str | None, fallback: torch.device) -> torch.device:
    if not spec:
        return fallback
    return torch.device(spec)


def setup_devices(config: Dict) -> DeviceBundle:
    n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    available = list(range(n_cuda))

    if n_cuda == 0:
        default = torch.device("cpu")
        return DeviceBundle(default, default, default, default, available)

    # Estrategia por defecto para HPC con >=3 GPUs:
    # G/D en cuda:0, D_WER/Whisper train en cuda:1, validación en cuda:2.
    if n_cuda >= 3:
        default_train = torch.device("cuda:0")
        default_proxy = torch.device("cuda:0")
        default_whisper = torch.device("cuda:1")
        default_validate = torch.device("cuda:2")
    elif n_cuda == 2:
        default_train = torch.device("cuda:0")
        default_proxy = torch.device("cuda:0")
        default_whisper = torch.device("cuda:1")
        default_validate = torch.device("cuda:1")
    else:
        default_train = torch.device("cuda:0")
        default_proxy = torch.device("cuda:0")
        default_whisper = torch.device("cuda:0")
        default_validate = torch.device("cuda:0")

    device_cfg = config.get("devices", {})
    train_device = _parse_device(device_cfg.get("train_device"), default_train)
    proxy_device = _parse_device(device_cfg.get("proxy_device"), default_proxy)
    whisper_device = _parse_device(device_cfg.get("whisper_device"), default_whisper)
    validate_device = _parse_device(device_cfg.get("validate_device"), default_validate)

    return DeviceBundle(
        train_device=train_device,
        proxy_device=proxy_device,
        whisper_device=whisper_device,
        validate_device=validate_device,
        available_cuda=available,
    )

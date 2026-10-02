from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PatchDiscriminator1D(nn.Module):
    def __init__(self, channels: int = 32, num_layers: int = 5):
        super().__init__()
        layers = []
        in_ch = 1
        ch = channels
        for i in range(num_layers):
            stride = 2 if i < num_layers - 1 else 1
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=15, stride=stride, padding=7))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch
            ch = min(ch * 2, 512)
        layers.append(nn.Conv1d(in_ch, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        out = self.net(waveform)
        return out.mean(dim=-1).squeeze(1)


def create_discriminator(config: Dict) -> nn.Module:
    # TODO: si ya existe un discriminator específico, reemplazar este factory.
    mcfg = config.get("models", {}).get("gan_discriminator", {})
    return PatchDiscriminator1D(
        channels=int(mcfg.get("channels", 32)),
        num_layers=int(mcfg.get("num_layers", 5)),
    )

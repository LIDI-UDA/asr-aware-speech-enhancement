from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.deconv(x)))


class UNetEnhancer(nn.Module):
    """
    U-Net 1D robusto para enhancement sin pares perfectos.
    """

    def __init__(
        self,
        channels: int = 64,
        depth: int = 4,
        num_res_blocks: int = 2,
        kernel_size: int = 7,
        output_mode: str = "mask_residual",
        mask_scale: float = 0.5,
        gain_scale: float = 0.25,
        residual_scale: float = 0.05,
    ):
        super().__init__()
        self.output_mode = str(output_mode).strip().lower()
        if self.output_mode not in {"mask_residual", "hybrid_gain_residual"}:
            raise ValueError(
                f"output_mode invalido: {output_mode}. "
                "Usa 'mask_residual' o 'hybrid_gain_residual'."
            )
        self.mask_scale = float(mask_scale)
        self.gain_scale = float(gain_scale)
        self.residual_scale = float(residual_scale)
        self.in_proj = nn.Conv1d(1, channels, kernel_size=kernel_size, padding=kernel_size // 2)

        down = []
        ch = channels
        self.skip_channels = []
        for _ in range(depth):
            down.append(DownBlock(ch, ch * 2))
            ch *= 2
            self.skip_channels.append(ch)
        self.down = nn.ModuleList(down)

        bottleneck = []
        for _ in range(max(1, num_res_blocks * 2)):
            bottleneck.append(ResBlock(ch, kernel_size=kernel_size))
        self.bottleneck = nn.Sequential(*bottleneck)

        up = []
        for _ in range(depth):
            up.append(UpBlock(ch, ch // 2))
            ch = ch // 2
        self.up = nn.ModuleList(up)

        self.refine = nn.Sequential(
            *[ResBlock(channels, kernel_size=kernel_size) for _ in range(max(1, num_res_blocks))]
        )
        if self.output_mode == "mask_residual":
            self.mask_proj = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            self.gain_proj = None
            self.residual_proj = None
        else:
            self.mask_proj = None
            self.gain_proj = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            self.residual_proj = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            x = waveform.unsqueeze(1)
        else:
            x = waveform

        x = self.in_proj(x)
        skips = []
        cur = x
        for down in self.down:
            # Guardar skip ANTES de downsample para alinear canales en la subida.
            skips.append(cur)
            cur = down(cur)

        cur = self.bottleneck(cur)

        for i, up in enumerate(self.up):
            cur = up(cur)
            skip = skips[-(i + 1)]
            if cur.shape[-1] != skip.shape[-1]:
                m = min(cur.shape[-1], skip.shape[-1])
                cur = cur[..., :m]
                skip = skip[..., :m]
            cur = cur + skip

        cur = self.refine(cur)

        if waveform.ndim == 2:
            inp = waveform.unsqueeze(1)
        else:
            inp = waveform

        if self.output_mode == "mask_residual":
            if self.mask_proj is None:
                raise RuntimeError("mask_proj no inicializado para output_mode=mask_residual.")
            mask = torch.tanh(self.mask_proj(cur))
            # Modo legacy: solo reescala contenido existente.
            enh = inp + self.mask_scale * mask * inp
        else:
            if self.gain_proj is None or self.residual_proj is None:
                raise RuntimeError("Heads hibridos no inicializados para output_mode=hybrid_gain_residual.")
            gain = 1.0 + self.gain_scale * torch.tanh(self.gain_proj(cur))
            delta = self.residual_scale * torch.tanh(self.residual_proj(cur))
            # Modo hibrido:
            # - gain corrige energia y estructura ya presentes
            # - delta permite correccion aditiva donde la mascara es insuficiente
            enh = inp * gain + delta
        enh = enh.squeeze(1)
        return enh.clamp(min=-1.0, max=1.0)


def create_generator(config: Dict) -> nn.Module:
    mcfg = config.get("models", {}).get("generator", {})
    return UNetEnhancer(
        channels=int(mcfg.get("channels", 64)),
        depth=int(mcfg.get("depth", 4)),
        num_res_blocks=int(mcfg.get("num_res_blocks", 2)),
        kernel_size=int(mcfg.get("kernel_size", 7)),
        output_mode=str(mcfg.get("output_mode", "mask_residual")),
        mask_scale=float(mcfg.get("mask_scale", 0.5)),
        gain_scale=float(mcfg.get("gain_scale", 0.25)),
        residual_scale=float(mcfg.get("residual_scale", 0.05)),
    )

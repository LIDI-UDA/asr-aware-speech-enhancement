from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    tgt = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(est * tgt, dim=-1, keepdim=True)
    s_target = dot * tgt / (torch.sum(tgt**2, dim=-1, keepdim=True) + eps)
    e_noise = est - s_target

    ratio = (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
    return -10.0 * torch.log10(ratio + eps).mean()


def stft_mag_l1(x: torch.Tensor, y: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    # Ventana Hann explícita para evitar warning de spectral leakage.
    # STFT en float32 por estabilidad numérica bajo autocast.
    x_f = x.to(torch.float32)
    y_f = y.to(torch.float32)
    window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
    x_stft = torch.stft(x_f, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    y_stft = torch.stft(y_f, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    return F.l1_loss(torch.abs(x_stft), torch.abs(y_stft))


def multi_resolution_stft_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    fft_sizes: List[int],
    hop_sizes: List[int],
) -> torch.Tensor:
    assert len(fft_sizes) == len(hop_sizes), "fft_sizes y hop_sizes deben tener misma longitud"
    losses = []
    for n_fft, hop in zip(fft_sizes, hop_sizes):
        losses.append(stft_mag_l1(x, y, n_fft=n_fft, hop_length=hop))
    return torch.stack(losses).mean()


def pairwise_rank_loss(logits_hi: torch.Tensor, logits_lo: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(logits_hi - logits_lo)).mean()


def rank_hard_pairs(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if logits.numel() < 2:
        return logits.new_tensor(0.0), 0

    order = torch.argsort(targets)
    lo_idx = order[: len(order) // 2]
    hi_idx = order[len(order) // 2 :]
    n = min(len(lo_idx), len(hi_idx))
    if n == 0:
        return logits.new_tensor(0.0), 0

    lo = logits[lo_idx[:n]]
    hi = logits[hi_idx[-n:]]
    return pairwise_rank_loss(hi, lo), int(n)

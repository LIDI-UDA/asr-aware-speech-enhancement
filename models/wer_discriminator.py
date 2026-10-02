from __future__ import annotations

import math
import re
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel

from utils.mel_transform import create_mel_transform


class AttentivePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,T,D), mask: (B,T) con True en posiciones válidas.
        w = self.att(x).squeeze(-1)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.to(dtype=torch.bool)
            # Evita filas sin válidos (softmax sobre todo -inf -> NaN).
            has_valid = torch.any(mask, dim=-1, keepdim=True)
            safe_mask = torch.where(has_valid, mask, torch.ones_like(mask))
            w = w.masked_fill(~safe_mask, torch.finfo(w.dtype).min)
            w = torch.softmax(w, dim=-1)
            w = w * safe_mask.to(dtype=w.dtype)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            w = torch.softmax(w, dim=-1)
        return torch.einsum("bt,btd->bd", w, x)


class WERDiscriminator(nn.Module):
    """
    D_WER: encoder Whisper congelado + agregación por chunks para audios largos.
    """
    CH_RE = re.compile(r"CH(\d+)-", re.IGNORECASE)

    def __init__(
        self,
        whisper_model_name: str,
        mel_transform: nn.Module,
        hidden_dim: int = 384,
        dropout: float = 0.15,
        freeze_encoder: bool = True,
        unfreeze_last_n_layers: int = 0,
        chunk_seconds: float = 20.0,
        chunk_overlap_seconds: float = 2.0,
        max_chunk_frames: int = 1200,
        sample_rate: int = 16000,
        hop_length: int = 160,
        chunk_pooling_mode: str = "mean",
        chunk_topk_fraction: float = 0.25,
        chunk_topk_min: int = 1,
        use_domain_calibration: bool = True,
        channel_num_buckets: int = 64,
        channel_embed_dim: int = 16,
        rel_head_hidden_dim: int = 192,
        use_acoustic_branch: bool = False,
        acoustic_dim: int = 64,
        acoustic_chunk_pooling_mode: str = "mean",
        use_wave_multiscale_branch: bool = False,
        wave_dim: int = 64,
        wave_scales: Sequence[int] = (1, 2, 4),
        wave_chunk_pooling_mode: str = "mean",
    ):
        super().__init__()
        self.mel_transform = mel_transform

        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.encoder = whisper.encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

            # Fine-tuning controlado de las últimas capas del encoder.
            n = max(0, int(unfreeze_last_n_layers))
            if n > 0 and hasattr(self.encoder, "layers"):
                layers = list(self.encoder.layers)
                n = min(n, len(layers))
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                # Normalización final también ayuda a calibrar.
                if hasattr(self.encoder, "layer_norm"):
                    for p in self.encoder.layer_norm.parameters():
                        p.requires_grad = True

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.chunk_seconds = float(chunk_seconds)
        self.chunk_overlap_seconds = float(chunk_overlap_seconds)
        self.max_chunk_frames = int(max_chunk_frames)
        # Whisper encoder en HF espera longitud fija de mel frames.
        # En checkpoints Whisper estándar: max_source_positions=1500 => 3000 frames.
        self.expected_mel_frames = int(getattr(self.encoder.config, "max_source_positions", 1500) * 2)

        dim = int(self.encoder.config.d_model)
        self.temporal_pool = AttentivePool(dim)
        self.chunk_pooling_mode = str(chunk_pooling_mode).lower().strip()
        if self.chunk_pooling_mode not in {"mean", "mean_topk", "attn"}:
            raise ValueError(
                f"chunk_pooling_mode inválido: {chunk_pooling_mode}. "
                "Usa: mean | mean_topk | attn."
            )
        self.chunk_topk_fraction = float(min(1.0, max(0.0, chunk_topk_fraction)))
        self.chunk_topk_min = max(1, int(chunk_topk_min))
        self.chunk_score = nn.Linear(dim, 1) if self.chunk_pooling_mode in {"mean_topk", "attn"} else None
        self.use_acoustic_branch = bool(use_acoustic_branch)
        self.acoustic_dim = max(0, int(acoustic_dim)) if self.use_acoustic_branch else 0
        self.acoustic_chunk_pooling_mode = str(acoustic_chunk_pooling_mode).lower().strip()
        if self.acoustic_chunk_pooling_mode not in {"mean", "mean_topk", "attn"}:
            raise ValueError(
                f"acoustic_chunk_pooling_mode inválido: {acoustic_chunk_pooling_mode}. "
                "Usa: mean | mean_topk | attn."
            )
        if self.use_acoustic_branch:
            # Rama acústica ligera (mel-CNN) para complementar señal semántica.
            self.acoustic_backbone = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 2), padding=(2, 2)),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.acoustic_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, self.acoustic_dim),
                nn.LayerNorm(self.acoustic_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.acoustic_score = (
                nn.Linear(self.acoustic_dim, 1)
                if self.acoustic_chunk_pooling_mode in {"mean_topk", "attn"}
                else None
            )
        else:
            self.acoustic_backbone = None
            self.acoustic_proj = None
            self.acoustic_score = None

        self.use_wave_multiscale_branch = bool(use_wave_multiscale_branch)
        self.wave_dim = max(1, int(wave_dim)) if self.use_wave_multiscale_branch else 0
        if wave_scales is None:
            scales = [1, 2, 4]
        elif isinstance(wave_scales, (int, float)):
            scales = [max(1, int(wave_scales))]
        else:
            scales = [max(1, int(s)) for s in list(wave_scales)]
        self.wave_scales = tuple(sorted(set(scales))) if len(scales) > 0 else (1,)
        self.wave_chunk_pooling_mode = str(wave_chunk_pooling_mode).lower().strip()
        if self.wave_chunk_pooling_mode not in {"mean", "mean_topk", "attn"}:
            raise ValueError(
                f"wave_chunk_pooling_mode inválido: {wave_chunk_pooling_mode}. "
                "Usa: mean | mean_topk | attn."
            )
        if self.use_wave_multiscale_branch:
            # Rama waveform multi-escala tipo HiFiGAN-lite (x1/x2/x4 por defecto).
            self.wave_backbone = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.wave_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * len(self.wave_scales), self.wave_dim),
                nn.LayerNorm(self.wave_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.wave_score = (
                nn.Linear(self.wave_dim, 1)
                if self.wave_chunk_pooling_mode in {"mean_topk", "attn"}
                else None
            )
        else:
            self.wave_backbone = None
            self.wave_proj = None
            self.wave_score = None

        self.feat_dim = dim + self.acoustic_dim + self.wave_dim
        self.use_domain_calibration = bool(use_domain_calibration)
        self.channel_num_buckets = max(2, int(channel_num_buckets))
        self.channel_embed_dim = max(1, int(channel_embed_dim))
        if self.use_domain_calibration:
            self.channel_embedding = nn.Embedding(self.channel_num_buckets, self.channel_embed_dim)
            meta_in = 3 + self.channel_embed_dim  # log_dur, rms_db_norm, crest_norm + channel emb
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_in, self.feat_dim),
                nn.LayerNorm(self.feat_dim),
                nn.GELU(),
                nn.Linear(self.feat_dim, self.feat_dim),
            )
        else:
            self.channel_embedding = None
            self.meta_proj = None

        self.head_abs = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        rel_h = max(64, int(rel_head_hidden_dim))
        self.head_rel_delta = nn.Sequential(
            nn.Linear(self.feat_dim, rel_h),
            nn.LayerNorm(rel_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rel_h, 1),
        )
        # Compatibilidad con código existente que referencia model.head
        self.head = self.head_abs

    def _fit_mel_frames(self, mel: torch.Tensor) -> torch.Tensor:
        t = int(mel.shape[-1])
        if t < self.expected_mel_frames:
            mel = F.pad(mel, (0, self.expected_mel_frames - t))
        if t > self.expected_mel_frames:
            mel = mel[..., : self.expected_mel_frames]
        return mel

    def _forward_chunk_from_mel(self, mel: torch.Tensor, valid_mel_frames: Optional[int] = None) -> torch.Tensor:
        # Construye attention_mask sobre mel frames válidos (antes de padding).
        attn_mask = None
        if valid_mel_frames is not None:
            valid_mel_frames = max(1, min(int(valid_mel_frames), int(mel.shape[-1])))
            attn_mask = torch.zeros(
                (int(mel.shape[0]), int(mel.shape[-1])),
                device=mel.device,
                dtype=torch.long,
            )
            attn_mask[:, :valid_mel_frames] = 1

        # Compatibilidad defensiva: algunos wrappers/versions pueden no aceptar
        # attention_mask o pueden esperar otra forma.
        try:
            h = self.encoder(mel, attention_mask=attn_mask).last_hidden_state if attn_mask is not None else self.encoder(mel).last_hidden_state
        except (TypeError, ValueError, RuntimeError):
            h = self.encoder(mel).last_hidden_state
            attn_mask = None

        # Conv frontend de Whisper reduce resolución temporal (~x2).
        token_mask = None
        if attn_mask is not None:
            t_out = int(h.shape[1])
            t_in = int(attn_mask.shape[1])
            valid_tokens = int(math.ceil(float(valid_mel_frames) * float(t_out) / float(max(1, t_in))))
            valid_tokens = max(1, min(t_out, valid_tokens))
            token_mask = torch.zeros((int(h.shape[0]), t_out), device=h.device, dtype=torch.bool)
            token_mask[:, :valid_tokens] = True

        pooled = self.temporal_pool(h, mask=token_mask)  # (1, d)
        return pooled

    def _forward_chunk(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, T)
        mel = self.mel_transform(wav)  # (1, 80, frames)
        valid_mel_frames = int(mel.shape[-1])
        mel = self._fit_mel_frames(mel)
        return self._forward_chunk_from_mel(mel, valid_mel_frames=valid_mel_frames)

    def _forward_acoustic_from_mel(self, mel: torch.Tensor) -> torch.Tensor:
        if (not self.use_acoustic_branch) or (self.acoustic_backbone is None) or (self.acoustic_proj is None):
            raise RuntimeError("Rama acústica no está habilitada.")
        x = mel.unsqueeze(1)  # (1,1,80,T)
        f = self.acoustic_backbone(x)
        f = self.acoustic_proj(f)
        return f

    def _downsample_wave(self, wav: torch.Tensor, factor: int) -> torch.Tensor:
        if factor <= 1:
            return wav
        if int(wav.shape[-1]) < factor:
            return wav
        return F.avg_pool1d(wav, kernel_size=factor, stride=factor)

    def _forward_wave_multiscale_from_wav(self, wav: torch.Tensor) -> torch.Tensor:
        if (not self.use_wave_multiscale_branch) or (self.wave_backbone is None) or (self.wave_proj is None):
            raise RuntimeError("Rama waveform multi-escala no está habilitada.")
        # wav: (1,T) o (1,1,T)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.ndim != 3:
            raise ValueError(f"wav inválido para rama waveform: shape={tuple(wav.shape)}")

        scale_feats = []
        for s in self.wave_scales:
            cur = self._downsample_wave(wav, int(s))
            if int(cur.shape[-1]) < 8:
                cur = F.pad(cur, (0, 8 - int(cur.shape[-1])))
            feat_s = self.wave_backbone(cur).squeeze(-1)  # (1,32)
            scale_feats.append(feat_s)
        f = torch.cat(scale_feats, dim=-1)  # (1, 32 * n_scales)
        return self.wave_proj(f)

    def _chunk_waveform(self, wav_1d: torch.Tensor):
        total = int(wav_1d.shape[-1])
        chunk_samples = max(1, int(self.chunk_seconds * self.sample_rate))
        overlap_samples = max(0, int(self.chunk_overlap_seconds * self.sample_rate))
        hop = max(1, chunk_samples - overlap_samples)

        chunks = []
        for start in range(0, total, hop):
            end = min(total, start + chunk_samples)
            chunk = wav_1d[start:end]
            if chunk.numel() < int(0.5 * chunk_samples):
                if start > 0:
                    break
            chunks.append(chunk)
            if end >= total:
                break
        return chunks

    def _aggregate_chunk_feats(
        self,
        chunk_feats: torch.Tensor,
        mode: str,
        score_layer: Optional[nn.Linear],
    ) -> torch.Tensor:
        # chunk_feats: (num_chunks, dim)
        if chunk_feats.ndim != 2:
            raise ValueError(f"chunk_feats debe tener shape (N,D), recibido: {tuple(chunk_feats.shape)}")

        n = int(chunk_feats.shape[0])
        if n == 1 or mode == "mean":
            return chunk_feats.mean(dim=0, keepdim=True)

        assert score_layer is not None
        scores = score_layer(chunk_feats).squeeze(-1)  # (N,)

        if mode == "attn":
            w = torch.softmax(scores, dim=0)
            return torch.sum(w.unsqueeze(-1) * chunk_feats, dim=0, keepdim=True)

        # mean_topk: mezcla mean global + mean de los chunks con mayor score (más "difíciles")
        k = int(round(float(n) * self.chunk_topk_fraction))
        k = max(self.chunk_topk_min, k)
        k = max(1, min(n, k))
        top_idx = torch.topk(scores, k=k, largest=True).indices
        top_mean = chunk_feats[top_idx].mean(dim=0)
        global_mean = chunk_feats.mean(dim=0)
        return (0.5 * (top_mean + global_mean)).unsqueeze(0)

    def _aggregate_chunks(self, chunk_feats: torch.Tensor) -> torch.Tensor:
        return self._aggregate_chunk_feats(
            chunk_feats=chunk_feats,
            mode=self.chunk_pooling_mode,
            score_layer=self.chunk_score,
        )

    def _aggregate_acoustic_chunks(self, chunk_feats: torch.Tensor) -> torch.Tensor:
        return self._aggregate_chunk_feats(
            chunk_feats=chunk_feats,
            mode=self.acoustic_chunk_pooling_mode,
            score_layer=self.acoustic_score,
        )

    def _aggregate_wave_chunks(self, chunk_feats: torch.Tensor) -> torch.Tensor:
        return self._aggregate_chunk_feats(
            chunk_feats=chunk_feats,
            mode=self.wave_chunk_pooling_mode,
            score_layer=self.wave_score,
        )

    def _channel_bucket(self, audio_path: Optional[str]) -> int:
        if (audio_path is None) or (not isinstance(audio_path, str)):
            return 0
        m = self.CH_RE.search(audio_path)
        if m is None:
            return 0
        try:
            ch = int(m.group(1))
        except Exception:
            return 0
        return max(0, min(self.channel_num_buckets - 1, ch + 1))

    def _domain_features(self, wav: torch.Tensor, true_len: int, audio_path: Optional[str], dtype: torch.dtype) -> torch.Tensor:
        w = wav[: max(1, int(true_len))]
        dur_s = float(max(1, int(true_len))) / float(self.sample_rate)
        log_dur = math.log1p(dur_s) / math.log1p(360.0)
        rms = torch.sqrt(torch.mean(w * w) + 1e-8)
        rms_db = 20.0 * torch.log10(rms + 1e-8)
        rms_db_norm = ((rms_db + 80.0) / 80.0).clamp(0.0, 1.0)
        peak = torch.max(torch.abs(w))
        crest = peak / (rms + 1e-6)
        crest_norm = (torch.log1p(crest) / math.log1p(50.0)).clamp(0.0, 1.0)

        num = torch.tensor(
            [log_dur, float(rms_db_norm.detach().item()), float(crest_norm.detach().item())],
            device=w.device,
            dtype=dtype,
        )
        if (not self.use_domain_calibration) or (self.channel_embedding is None):
            return num

        ch = self._channel_bucket(audio_path)
        ch_id = torch.tensor(ch, device=w.device, dtype=torch.long)
        ch_emb = self.channel_embedding(ch_id).to(dtype=dtype)
        return torch.cat([num, ch_emb], dim=0)

    def forward(
        self,
        waveform: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        audio_paths: Optional[Sequence[str]] = None,
        output: str = "abs",
    ):
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
        output = str(output).strip().lower()
        if output not in ("abs", "rel", "both"):
            raise ValueError(f"output inválido: {output}. Usa 'abs', 'rel' o 'both'.")

        outs = []
        meta_feats = []
        for i in range(waveform.shape[0]):
            wav = waveform[i]
            true_len = int(wav.shape[-1])
            if durations is not None:
                true_len = max(1, int(float(durations[i]) * self.sample_rate))
                wav = wav[: min(true_len, int(wav.shape[-1]))]
            else:
                wav = wav[:true_len]

            chunks = self._chunk_waveform(wav)
            chunk_feats = []
            acoustic_feats = []
            wave_feats = []
            for c in chunks:
                mel_raw = self.mel_transform(c.unsqueeze(0))
                valid_mel_frames = int(mel_raw.shape[-1])
                mel = self._fit_mel_frames(mel_raw)
                chunk_feats.append(self._forward_chunk_from_mel(mel, valid_mel_frames=valid_mel_frames))
                if self.use_acoustic_branch:
                    # Rama acústica usa mel sin padding para evitar sesgo por ceros.
                    acoustic_feats.append(self._forward_acoustic_from_mel(mel_raw))
                if self.use_wave_multiscale_branch:
                    wave_feats.append(self._forward_wave_multiscale_from_wav(c.unsqueeze(0)))
            chunk_feats_t = torch.cat(chunk_feats, dim=0)
            f = self._aggregate_chunks(chunk_feats_t)
            if self.use_acoustic_branch and len(acoustic_feats) > 0:
                ac_t = torch.cat(acoustic_feats, dim=0)
                f_ac = self._aggregate_acoustic_chunks(ac_t)
                f = torch.cat([f, f_ac], dim=-1)
            if self.use_wave_multiscale_branch and len(wave_feats) > 0:
                w_t = torch.cat(wave_feats, dim=0)
                f_w = self._aggregate_wave_chunks(w_t)
                f = torch.cat([f, f_w], dim=-1)
            outs.append(f)
            if self.use_domain_calibration:
                ap = None
                if audio_paths is not None and i < len(audio_paths):
                    ap = audio_paths[i]
                meta_feats.append(self._domain_features(wav, true_len=true_len, audio_path=ap, dtype=f.dtype))

        feat = torch.cat(outs, dim=0)
        if self.use_domain_calibration and self.meta_proj is not None and len(meta_feats) == feat.shape[0]:
            meta = torch.stack(meta_feats, dim=0)
            feat = feat + self.meta_proj(meta)

        abs_logits = self.head_abs(feat).squeeze(-1)
        rel_logits = abs_logits + self.head_rel_delta(feat).squeeze(-1)

        if output == "abs":
            return abs_logits
        if output == "rel":
            return rel_logits
        return {"abs": abs_logits, "rel": rel_logits}


def create_wer_discriminator(config: Dict, device: torch.device) -> nn.Module:
    mcfg = config.get("models", {}).get("wer_discriminator", {})
    whisper_name = config.get("models", {}).get("whisper", {}).get("encoder_model_name", "openai/whisper-small")
    mel_transform = create_mel_transform(config, device=device)

    model = WERDiscriminator(
        whisper_model_name=whisper_name,
        mel_transform=mel_transform,
        hidden_dim=int(mcfg.get("hidden_dim", 384)),
        dropout=float(mcfg.get("dropout", 0.15)),
        freeze_encoder=bool(mcfg.get("freeze_encoder", True)),
        unfreeze_last_n_layers=int(mcfg.get("unfreeze_last_n_layers", 0)),
        chunk_seconds=float(mcfg.get("chunk_seconds", 20.0)),
        chunk_overlap_seconds=float(mcfg.get("chunk_overlap_seconds", 2.0)),
        max_chunk_frames=int(mcfg.get("max_chunk_frames", 1200)),
        sample_rate=int(config["audio"]["target_sr"]),
        hop_length=int(config["audio"]["hop_length"]),
        chunk_pooling_mode=str(mcfg.get("chunk_pooling_mode", "mean")),
        chunk_topk_fraction=float(mcfg.get("chunk_topk_fraction", 0.25)),
        chunk_topk_min=int(mcfg.get("chunk_topk_min", 1)),
        use_domain_calibration=bool(mcfg.get("use_domain_calibration", True)),
        channel_num_buckets=int(mcfg.get("channel_num_buckets", 64)),
        channel_embed_dim=int(mcfg.get("channel_embed_dim", 16)),
        rel_head_hidden_dim=int(mcfg.get("rel_head_hidden_dim", 192)),
        use_acoustic_branch=bool(mcfg.get("use_acoustic_branch", False)),
        acoustic_dim=int(mcfg.get("acoustic_dim", 64)),
        acoustic_chunk_pooling_mode=str(mcfg.get("acoustic_chunk_pooling_mode", "mean")),
        use_wave_multiscale_branch=bool(mcfg.get("use_wave_multiscale_branch", False)),
        wave_dim=int(mcfg.get("wave_dim", 64)),
        wave_scales=mcfg.get("wave_scales", [1, 2, 4]),
        wave_chunk_pooling_mode=str(mcfg.get("wave_chunk_pooling_mode", "mean")),
    )
    return model

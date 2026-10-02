"""
Metrics - WER/CER Computation
==============================
WER/CER usando Whisper + normalización consistente del proyecto.

CLAVES:
- normalize_text_for_wer() viene de utils/text.py (única fuente de verdad).
- Whisper transcribe en chunks para audios largos (sin timestamps).
- dtype/device siempre alineado con el modelo (evita float vs half).
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from jiwer import wer, cer

from utils.text import normalize_text_for_wer


# ============================================================================
# WER/CER usando Whisper
# ============================================================================

class WhisperWERMetric:
    """
    Calcula WER/CER usando el modelo Whisper para transcripción.
    Soporta long-form por chunks sin timestamps.
    """

    def __init__(
        self,
        whisper_model,
        whisper_processor,
        device: torch.device,
        sample_rate: int = 16000,
        chunk_seconds: float = 30.0,
    ):
        self.whisper_model = whisper_model
        self.whisper_processor = whisper_processor
        self.device = device
        self.sample_rate = int(sample_rate)
        self.chunk_seconds = float(chunk_seconds)

        self.whisper_model.eval()
        self.whisper_model.to(device)

    @torch.no_grad()
    def _transcribe_once(self, waveform_1d: torch.Tensor) -> str:
        """
        Transcribe un chunk (<= chunk_seconds) usando Whisper.
        waveform_1d: (samples,) float
        """
        if waveform_1d.ndim == 2:
            waveform_1d = waveform_1d.squeeze(0)

        # processor requiere numpy float32
        wav_np = waveform_1d.detach().to(torch.float32).cpu().numpy()

        inputs = self.whisper_processor(
            wav_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        input_features = inputs.input_features  # (1, 80, T)

        # ✅ alinear dtype con modelo
        model_dtype = next(self.whisper_model.parameters()).dtype
        input_features = input_features.to(self.device, dtype=model_dtype, non_blocking=True)

        predicted_ids = self.whisper_model.generate(input_features)

        transcription = self.whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return (transcription or "").strip()

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> str:
        """
        Transcribe waveform completo.
        Si es largo, lo procesa en chunks consecutivos de chunk_seconds y concatena.
        """
        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        waveform = waveform.detach()
        if waveform.numel() == 0:
            return ""

        chunk_len = int(self.chunk_seconds * self.sample_rate)
        if chunk_len <= 0 or waveform.shape[0] <= chunk_len:
            return self._transcribe_once(waveform)

        texts: List[str] = []
        start = 0
        while start < waveform.shape[0]:
            end = min(start + chunk_len, waveform.shape[0])
            chunk = waveform[start:end]
            texts.append(self._transcribe_once(chunk))
            start = end

        # concatenación simple (suficiente para WER estable sin timestamps)
        return " ".join([t for t in texts if t])

    @torch.no_grad()
    def transcribe_batch(self, waveforms: torch.Tensor) -> List[str]:
        """
        waveforms: (B, T) o list/tuple de tensores
        """
        if isinstance(waveforms, (list, tuple)):
            return [self.transcribe(w) for w in waveforms]

        if waveforms.ndim == 3:
            waveforms = waveforms.squeeze(1)

        return [self.transcribe(w) for w in waveforms]

    def compute_wer(self, references: List[str], hypotheses: List[str]) -> float:
        refs = [normalize_text_for_wer(r) for r in references]
        hyps = [normalize_text_for_wer(h) for h in hypotheses]

        # evita edge cases de strings vacías
        refs = [r if r else "<empty>" for r in refs]
        hyps = [h if h else "<empty>" for h in hyps]

        try:
            return float(wer(refs, hyps))
        except Exception:
            return 1.0

    def compute_cer(self, references: List[str], hypotheses: List[str]) -> float:
        refs = [normalize_text_for_wer(r) for r in references]
        hyps = [normalize_text_for_wer(h) for h in hypotheses]
        refs = [r if r else "<empty>" for r in refs]
        hyps = [h if h else "<empty>" for h in hyps]

        try:
            return float(cer(refs, hyps))
        except Exception:
            return 1.0

    @torch.no_grad()
    def evaluate_dataset(self, dataloader, max_samples: Optional[int] = None) -> Dict[str, float]:
        all_refs, all_hyps = [], []
        n = 0

        for batch in dataloader:
            waveforms = batch["waveform"]
            refs = batch["transcripts"]

            hyps = self.transcribe_batch(waveforms)

            all_refs.extend(refs)
            all_hyps.extend(hyps)
            n += len(refs)

            if max_samples is not None and n >= max_samples:
                break

        return {
            "wer": self.compute_wer(all_refs, all_hyps),
            "cer": self.compute_cer(all_refs, all_hyps),
            "num_samples": n,
        }


# ============================================================================
# Comparación Before/After Enhancement
# ============================================================================

class EnhancementEvaluator:
    """
    Evalúa impacto de enhancement comparando WER/CER noisy vs enhanced.
    """

    def __init__(self, generator_model, whisper_metric: WhisperWERMetric, device: torch.device):
        self.generator = generator_model
        self.whisper_metric = whisper_metric
        self.device = device

        self.generator.eval()
        self.generator.to(device)

    @torch.no_grad()
    def evaluate_enhancement(self, dataloader, max_samples: Optional[int] = None) -> Dict[str, float]:
        refs = []
        hyps_noisy = []
        hyps_enh = []
        n = 0

        for batch in dataloader:
            if max_samples is not None and n >= max_samples:
                break

            wave_noisy = batch["waveform"].to(self.device)
            texts = batch["transcripts"]

            wave_enh = self.generator(wave_noisy)

            # Whisper metric trabaja en CPU/GPU internamente pero transcribe() hace cpu->numpy
            hyp_n = self.whisper_metric.transcribe_batch(wave_noisy.cpu())
            hyp_e = self.whisper_metric.transcribe_batch(wave_enh.cpu())

            refs.extend(texts)
            hyps_noisy.extend(hyp_n)
            hyps_enh.extend(hyp_e)

            n += len(texts)

        wer_noisy = self.whisper_metric.compute_wer(refs, hyps_noisy)
        wer_enh = self.whisper_metric.compute_wer(refs, hyps_enh)

        cer_noisy = self.whisper_metric.compute_cer(refs, hyps_noisy)
        cer_enh = self.whisper_metric.compute_cer(refs, hyps_enh)

        return {
            "wer_noisy": wer_noisy,
            "wer_enhanced": wer_enh,
            "wer_improvement": wer_noisy - wer_enh,
            "cer_noisy": cer_noisy,
            "cer_enhanced": cer_enh,
            "cer_improvement": cer_noisy - cer_enh,
            "num_samples": n,
        }


# ============================================================================
# ASR Good Pool
# ============================================================================

class ASRGoodPool:
    """
    Guarda mejores outputs del generator basándose en WER (normalización consistente).
    """

    def __init__(self, whisper_metric: WhisperWERMetric, pool_size: int = 500, wer_threshold: float = 0.3):
        self.whisper_metric = whisper_metric
        self.pool_size = pool_size
        self.wer_threshold = wer_threshold
        self.pool = []  # (waveform, reference, sample_wer)

    def add_samples(self, waveforms: torch.Tensor, references: List[str]):
        hyps = self.whisper_metric.transcribe_batch(waveforms)

        for waveform, ref, hyp in zip(waveforms, references, hyps):
            r = normalize_text_for_wer(ref)
            h = normalize_text_for_wer(hyp)
            if not r:
                continue
            try:
                sample_wer = float(wer([r], [h]))
            except Exception:
                sample_wer = 1.0

            if sample_wer < self.wer_threshold:
                self.pool.append((waveform.cpu(), ref, sample_wer))

        if len(self.pool) > self.pool_size:
            self.pool.sort(key=lambda x: x[2])
            self.pool = self.pool[:self.pool_size]

    def get_pool_samples(self) -> List[torch.Tensor]:
        return [item[0] for item in self.pool]

    def get_pool_size(self) -> int:
        return len(self.pool)

    def get_average_wer(self) -> float:
        if not self.pool:
            return 1.0
        return float(np.mean([x[2] for x in self.pool]))

"""
Audio Utilities
================
Funciones básicas de audio: resample, normalización, chunking.
Cada función hace UNA cosa bien.

Tipos de datos:
- Waveforms: torch.FloatTensor, shape (samples,) o (batch, samples)
- Sample rate: int (8000 o 16000 Hz)
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import torchaudio.functional as AF

import torch.nn.functional as F



# ============================================================================
# RESAMPLE - 8kHz → 16kHz
# ============================================================================

def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio de orig_sr a target_sr.

    Args:
        waveform: (samples,) o (batch, samples)
        orig_sr: Frecuencia original (8000)
        target_sr: Frecuencia objetivo (16000)

    Returns:
        waveform resampled: misma shape pero diferentes samples
    """
    if orig_sr == target_sr:
        return waveform

    # Asegurar shape 2D para torchaudio
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Resample
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    )
    waveform_resampled = resampler(waveform)

    if squeeze_output:
        waveform_resampled = waveform_resampled.squeeze(0)

    return waveform_resampled


# ============================================================================
# NORMALIZACIÓN RMS
# ============================================================================

def normalize_rms(
    waveform: torch.Tensor,
    target_rms: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normaliza audio a un RMS objetivo.

    Args:
        waveform: (samples,) o (batch, samples)
        target_rms: RMS objetivo (0.1 por defecto)
        eps: Epsilon para estabilidad

    Returns:
        waveform normalizado: misma shape
    """
    if waveform.ndim == 1:
        # Single audio
        current_rms = torch.sqrt(torch.mean(waveform ** 2) + eps)
        scale = target_rms / (current_rms + eps)
        return waveform * scale
    else:
        # Batch: normalizar cada utterance independientemente
        current_rms = torch.sqrt(torch.mean(waveform ** 2, dim=1, keepdim=True) + eps)
        scale = target_rms / (current_rms + eps)
        return waveform * scale


# ============================================================================
# CHUNKING ALEATORIO - Para entrenamiento
# ============================================================================

def random_chunk(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_duration_min: float = 2.0,
    chunk_duration_max: float = 6.0,
) -> torch.Tensor:
    """
    Extrae un chunk aleatorio del waveform.
    Usado en TRAINING para alimentar al modelo en piezas manejables.

    Args:
        waveform: (samples,)
        sample_rate: 16000
        chunk_duration_min: Duración mínima en segundos
        chunk_duration_max: Duración máxima en segundos

    Returns:
        chunk: (chunk_samples,)
    """
    num_samples = waveform.shape[0]

    # Duración aleatoria entre min y max
    chunk_duration = np.random.uniform(chunk_duration_min, chunk_duration_max)
    chunk_samples = int(chunk_duration * sample_rate)

    # Si el audio es más corto que el chunk, pad
    if num_samples < chunk_samples:
        pad_length = chunk_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_length), value=0.0)
        return waveform

    # Inicio aleatorio
    start_idx = np.random.randint(0, num_samples - chunk_samples + 1)
    chunk = waveform[start_idx : start_idx + chunk_samples]

    return chunk


def batch_random_chunks(
    waveforms: list,
    sample_rate: int,
    chunk_duration_min: float = 2.0,
    chunk_duration_max: float = 6.0,
) -> torch.Tensor:
    """
    Extrae chunks aleatorios de una lista de waveforms y los apila.

    Args:
        waveforms: Lista de tensors (samples_i,) con diferentes longitudes
        sample_rate: 16000
        chunk_duration_min: segundos
        chunk_duration_max: segundos

    Returns:
        batch_chunks: (batch, max_chunk_samples) con padding si necesario
    """
    chunks = []
    for waveform in waveforms:
        chunk = random_chunk(waveform, sample_rate, chunk_duration_min, chunk_duration_max)
        chunks.append(chunk)

    # Pad al mismo tamaño (max length en batch)
    max_len = max(chunk.shape[0] for chunk in chunks)
    padded_chunks = []
    for chunk in chunks:
        if chunk.shape[0] < max_len:
            pad_length = max_len - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad_length), value=0.0)
        padded_chunks.append(chunk)

    batch_chunks = torch.stack(padded_chunks, dim=0)  # (batch, max_len)
    return batch_chunks


# ============================================================================
# SLIDING WINDOW - Para inferencia en audios largos
# ============================================================================

def sliding_window_inference(
    waveform: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: int,
    window_duration: float = 10.0,
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Procesa audio largo usando sliding window con overlap.
    Usado en INFERENCIA para audios > max_duration.

    Args:
        waveform: (samples,) audio largo
        model: Generator model
        sample_rate: 16000
        window_duration: Duración de cada ventana en segundos
        overlap: Fracción de overlap (0.5 = 50%)
        device: Device para model

    Returns:
        enhanced: (samples,) audio mejorado completo
    """
    num_samples = waveform.shape[0]
    window_samples = int(window_duration * sample_rate)
    hop_samples = int(window_samples * (1 - overlap))

    # Si el audio es más corto que la ventana, procesar todo
    if num_samples <= window_samples:
        waveform_batch = waveform.unsqueeze(0).to(device)  # (1, samples)
        with torch.no_grad():
            enhanced = model(waveform_batch)
        return enhanced.squeeze(0).cpu()

    # Procesar por ventanas
    enhanced_chunks = []
    weights = []

    for start_idx in range(0, num_samples, hop_samples):
        end_idx = min(start_idx + window_samples, num_samples)
        chunk = waveform[start_idx:end_idx]

        # Pad si es necesario
        if chunk.shape[0] < window_samples:
            pad_length = window_samples - chunk.shape[0]
            chunk = torch.nn.functional.pad(chunk, (0, pad_length), value=0.0)

        # Forward
        chunk_batch = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced_chunk = model(chunk_batch).squeeze(0).cpu()

        # Remover padding
        if end_idx - start_idx < window_samples:
            enhanced_chunk = enhanced_chunk[:end_idx - start_idx]

        enhanced_chunks.append(enhanced_chunk)

        # Weight para overlap-add (hanning window)
        weight = torch.hann_window(enhanced_chunk.shape[0])
        weights.append(weight)

        if end_idx >= num_samples:
            break

    # Overlap-add
    output = torch.zeros(num_samples)
    weight_sum = torch.zeros(num_samples)

    idx = 0
    for chunk, weight in zip(enhanced_chunks, weights):
        length = chunk.shape[0]
        output[idx:idx + length] += chunk * weight
        weight_sum[idx:idx + length] += weight
        idx += hop_samples

    # Normalizar
    output = output / (weight_sum + 1e-8)

    return output


# ============================================================================
# LOAD & SAVE
# ============================================================================

def load_audio(
    filepath: str,
    target_sr: int = 16000,
    normalize: bool = True,
    target_rms: float = 0.1,
) -> Tuple[torch.Tensor, int]:
    """
    Carga audio desde archivo.

    Args:
        filepath: Path al archivo .wav
        target_sr: Resample a esta frecuencia (None para no resample)
        normalize: Si normalizar RMS
        target_rms: RMS objetivo

    Returns:
        waveform: (samples,)
        sample_rate: int
    """
    waveform, orig_sr = torchaudio.load(filepath)

    # Convertir a mono si es stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # Resample si es necesario
    if target_sr is not None and orig_sr != target_sr:
        waveform = resample_audio(waveform, orig_sr, target_sr)
        sample_rate = target_sr
    else:
        sample_rate = orig_sr

    # Normalizar
    if normalize:
        waveform = normalize_rms(waveform, target_rms)

    return waveform, sample_rate


def save_audio(
    filepath: str,
    waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """
    Guarda audio a archivo .wav.

    Args:
        filepath: Path donde guardar
        waveform: (samples,) o (1, samples)
        sample_rate: Frecuencia de muestreo
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(filepath, waveform.cpu(), sample_rate)


# ============================================================================
# Simulate Telefone Noise
# ============================================================================


class TelephoneDegradation:
    """
    Simula canal telefónico 8kHz real (PSTN / radio / ECU911).
    Entrada y salida: 16 kHz.

    Interfaz sin cambios:
        TelephoneDegradation(sample_rate=16000, snr_range=(20, 30))
    """

    def __init__(self, sample_rate=16000, snr_range=(20, 30)):
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        # parámetros internos optimizados
        self._tele_band_low = 300
        self._tele_band_high = 3400
        self._tele_rate = 8000
        self._mu_channels = 256

        # AGC ligero
        self._agc_target = 0.12
        self._agc_min_gain = 0.6
        self._agc_max_gain = 3.0
        self._agc_window_ms = 20

        # Variación de volumen (segundo hablante)
        self._second_speaker_boost_db = (2.0, 6.0)  # 2-6 dB como en tu diseño
        self._min_segment_s = 1.0
        self._max_segment_s = 3.0

        # dBFS objetivo y rango (ECU911)
        self._dbfs_mean = -20.43
        self._dbfs_std = 3.0
        self._dbfs_min = -28.16
        self._dbfs_max = -10.32

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Asegurar 1D y obtener device/dtype
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        device = waveform.device
        dtype = waveform.dtype

        # Trabajar sobre copia para no mutar input
        wav = waveform.clone()

        # 1. Band-pass telefónico
        wav = AF.highpass_biquad(wav, self.sample_rate, self._tele_band_low)
        wav = AF.lowpass_biquad(wav, self.sample_rate, self._tele_band_high)

        # 2. Downsample a 8 kHz
        if self.sample_rate != self._tele_rate:
            wav = AF.resample(wav, self.sample_rate, self._tele_rate)

        # 3. μ-law encode/decode (económico)
        wav = AF.mu_law_decoding(
            AF.mu_law_encoding(wav, quantization_channels=self._mu_channels),
            quantization_channels=self._mu_channels
        )

        # 4. AGC ligero (ventana RMS eficiente)
        win_samples = max(1, int(self._agc_window_ms / 1000.0 * self._tele_rate))
        wav_2d = wav.unsqueeze(0).unsqueeze(0)  # (1,1,L)
        sq = wav_2d * wav_2d
        pad = win_samples // 2
        sq_padded = F.pad(sq, (pad, pad), mode='replicate')
        env = torch.sqrt(F.avg_pool1d(sq_padded, kernel_size=win_samples, stride=1).squeeze())
        env = env[: wav.shape[0]]

        gain = self._agc_target / (env + 1e-8)
        gain = torch.clamp(gain, min=self._agc_min_gain, max=self._agc_max_gain)
        kernel_size = max(3, win_samples // 8)
        g_3d = gain.unsqueeze(0).unsqueeze(0)
        g_pad = F.pad(g_3d, (kernel_size // 2, kernel_size // 2), mode='replicate')
        kernel = torch.ones(1, 1, kernel_size, device=device, dtype=dtype) / float(kernel_size)
        gain_smooth = F.conv1d(g_pad, kernel).squeeze()[: wav.shape[0]]
        wav = wav * gain_smooth

        # 5. Ruido blanco a SNR controlado
        snr_db = float(np.random.uniform(*self.snr_range))
        sig_power = torch.mean(wav ** 2)
        noise = torch.randn_like(wav, device=device, dtype=dtype)
        noise_power = torch.mean(noise ** 2)
        desired_noise_power = sig_power / (10 ** (snr_db / 10.0) + 1e-12)
        noise_scale = torch.sqrt(desired_noise_power / (noise_power + 1e-12))
        wav = wav + noise * noise_scale

        # 6. Distorsión armónica leve (soft clipping)
        dist_amount = 0.03
        if dist_amount > 0:
            distorted = torch.tanh(wav * (1.0 + dist_amount * 4.0))
            wav = (1.0 - dist_amount) * wav + dist_amount * distorted

        # 7. VARIACIÓN DE VOLUMEN: simula turnos entre hablantes (2-6 dB boost en segmentos alternos)
        wav = self._apply_volume_variation(wav)

        # 8. Upsample a sample_rate (si era diferente)
        if self.sample_rate != self._tele_rate:
            wav = AF.resample(wav, self._tele_rate, self.sample_rate)

        # 9. Ajuste final de energía a dBFS dentro del rango pedido
        wav = self._adjust_energy_level(wav)

        return wav

    def _apply_volume_variation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Aplica boost en segmentos alternos para simular segundo hablante."""
        L = len(waveform)
        # segment duration aleatoria entre 1-3s
        seg_dur = float(np.random.uniform(self._min_segment_s, self._max_segment_s))
        seg_samples = int(seg_dur * self._tele_rate)

        wav = waveform.clone()

        if L < seg_samples * 2:
            # audio corto: aplicar boost en segunda mitad
            boost_db = float(np.random.uniform(*self._second_speaker_boost_db))
            boost = 10 ** (boost_db / 20.0)
            mid = L // 2
            wav[mid:] = wav[mid:] * boost
            return wav

        # para audios más largos, alternar segmentos
        num_segments = max(1, L // seg_samples)
        for i in range(num_segments):
            start = i * seg_samples
            end = min((i + 1) * seg_samples, L)
            # impares -> boost
            if i % 2 == 1:
                boost_db = float(np.random.uniform(*self._second_speaker_boost_db))
                boost = 10 ** (boost_db / 20.0)
                wav[start:end] = wav[start:end] * boost

        return wav

    def _adjust_energy_level(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ajusta RMS final para aproximarse a un dBFS muestreado desde
        normal(mean=-20.43, std=3.0) y clip al rango [-28.16, -10.32].
        """
        wav = waveform.clone()
        # calcular RMS actual
        rms = torch.sqrt(torch.mean(wav ** 2) + 1e-12).item()

        # muestrear target dBFS y recortar al rango
        target_dbfs = float(np.random.normal(self._dbfs_mean, self._dbfs_std))
        target_dbfs = float(np.clip(target_dbfs, self._dbfs_min, self._dbfs_max))

        # convertir dBFS -> rms amplitude (asumiendo full scale = 1.0)
        target_rms = 10 ** (target_dbfs / 20.0)

        # escalar
        if rms > 0:
            scale = target_rms / (rms + 1e-12)
            wav = wav * scale
        else:
            # si RMS es 0 (señal nula), devolver sin cambios
            pass

        return wav

# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing audio utilities...")

    # Test resample
    waveform_8k = torch.randn(8000 * 3)  # 3 segundos @ 8kHz
    waveform_16k = resample_audio(waveform_8k, 8000, 16000)
    print(f"Resample 8kHz → 16kHz: {waveform_8k.shape} → {waveform_16k.shape}")
    assert waveform_16k.shape[0] == 16000 * 3

    # Test normalize
    waveform_noisy = torch.randn(16000) * 10
    waveform_norm = normalize_rms(waveform_noisy, target_rms=0.1)
    rms = torch.sqrt(torch.mean(waveform_norm ** 2))
    print(f"Normalize RMS: {rms:.4f} (target: 0.1)")
    assert abs(rms - 0.1) < 0.01

    # Test random chunk
    waveform_long = torch.randn(16000 * 10)  # 10 segundos
    chunk = random_chunk(waveform_long, 16000, chunk_duration_min=2.0, chunk_duration_max=6.0)
    print(f"Random chunk: {chunk.shape[0] / 16000:.2f} segundos")
    assert 2.0 <= chunk.shape[0] / 16000 <= 6.0

    print("\n✓ Audio utilities funcionan correctamente")

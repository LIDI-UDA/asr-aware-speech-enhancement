"""
Mel Spectrogram Differentiable Transform
=========================================
Transformación de audio a mel spectrogram usando torchaudio (GPU).
Mantiene gradientes para que L_sem pueda backpropagear al waveform.

Input: waveform (batch, samples) @ 16kHz
Output: mel spectrogram (batch, n_mels, time_frames)
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class DifferentiableMelTransform(nn.Module):
    """
    Mel spectrogram transform compatible con Whisper.
    Usa torchaudio.transforms en GPU para mantener gradientes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        """
        Args:
            sample_rate: Frecuencia de muestreo (16kHz para Whisper)
            n_fft: Tamaño de FFT
            hop_length: Hop en samples (10ms para 16kHz)
            n_mels: Número de mel bins (80 para Whisper)
            f_min: Frecuencia mínima
            f_max: Frecuencia máxima
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel spectrogram transform (differentiable)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,  # Power spectrogram
            norm="slaney",
            mel_scale="slaney",
        )

        # Amplitude to DB (differentiable)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convierte waveform a mel spectrogram.

        Args:
            waveform: (batch, samples) @ 16kHz

        Returns:
            mel_spec: (batch, n_mels, time_frames) en dB scale
        """
        # waveform: (batch, samples)

        # Mel spectrogram: (batch, n_mels, time)
        mel_spec = self.mel_transform(waveform)

        # Convertir a dB scale (como Whisper)
        mel_spec = self.amplitude_to_db(mel_spec)

        # Normalizar a rango [-1, 1] aproximadamente (similar a Whisper)
        # Whisper usa normalización por utterance, aquí simplificamos
        mel_spec = (mel_spec + 40) / 40  # Heurística: -40dB a 0dB → [-1, 0]

        return mel_spec

    def to(self, device):
        """Override para mover mel_transform al device correcto."""
        super().to(device)
        self.mel_transform = self.mel_transform.to(device)
        self.amplitude_to_db = self.amplitude_to_db.to(device)
        return self


def create_mel_transform(config: dict, device: torch.device) -> DifferentiableMelTransform:
    """
    Factory function para crear mel transform desde config.

    Args:
        config: Diccionario con audio config
        device: Device donde colocar el transform

    Returns:
        DifferentiableMelTransform en el device especificado
    """
    mel_transform = DifferentiableMelTransform(
        sample_rate=config["audio"]["target_sr"],
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        n_mels=config["audio"]["n_mels"],
        f_min=config["audio"]["f_min"],
        f_max=config["audio"]["f_max"],
    )

    mel_transform = mel_transform.to(device)
    mel_transform.eval()  # No tiene parámetros entrenables

    return mel_transform


# ============================================================================
# Tests rápidos
# ============================================================================

if __name__ == "__main__":
    # Test básico
    print("Testing DifferentiableMelTransform...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear transform
    mel_transform = DifferentiableMelTransform(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
    ).to(device)

    # Audio de prueba: 3 segundos @ 16kHz
    batch_size = 2
    duration = 3.0
    num_samples = int(16000 * duration)

    waveform = torch.randn(batch_size, num_samples, device=device, requires_grad=True)

    # Forward
    mel_spec = mel_transform(waveform)

    print(f"Waveform shape: {waveform.shape}")
    print(f"Mel spec shape: {mel_spec.shape}")
    print(f"Mel spec range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")

    # Test gradientes
    loss = mel_spec.mean()
    loss.backward()

    print(f"Gradientes en waveform: {waveform.grad is not None}")
    print(f"Grad norm: {waveform.grad.norm():.4f}")

    print("\n✓ DifferentiableMelTransform funciona correctamente con gradientes")

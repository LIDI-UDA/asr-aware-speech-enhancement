"""
Data Utilities - Dataset y DataLoader
======================================

ECU911Dataset:
  - Carga desde pickle preparado (ecu911_prepared.pkl)
  - Soporta splits: train | val | test
  - Propaga quality_score al batch cuando está disponible

SpanishConversationalDataset:
  - Corpus limpio con timestamps
  - Split train / val

collate_ecu911:
  - quality_scores como (batch,) float32 — shape correcta para BCEWithLogitsLoss
  - quality_mask para ignorar samples sin target

create_ecu911_dataloader:
  - Factory con soporte completo train / val / test
  - purpose="wer_disc" para targets robustos del D_WER sin rehacer preprocessing
"""

import torch
import numpy as np
import pandas as pd
import pickle
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import math
from torch.utils.data import Sampler

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.audio import load_audio, normalize_rms, TelephoneDegradation
import torch.nn.functional as F


EXTRA_FLOAT_SAMPLE_KEYS = (
    "asr_avg_entropy",
    "asr_avg_logprob",
    "asr_no_speech_prob",
    "asr_compression_ratio",
)


def _resolve_expected_preprocessing_whisper_model(config: dict) -> Optional[str]:
    pcfg = config.get("preprocessing", {})
    name = pcfg.get("whisper_model_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    wcfg = config.get("models", {}).get("whisper", {})
    for key in ("model_name", "encoder_model_name"):
        v = wcfg.get(key, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ev = config.get("evaluation", {}).get("whisper_model_name", None)
    if isinstance(ev, str) and ev.strip():
        return ev.strip()
    return None


# ============================================================================
# ECU911Dataset
# ============================================================================
class ECU911Dataset(Dataset):
    """
    Dataset para ECU911 (train / val / test).

    Cada sample del pickle contiene al menos:
      - audio_path, transcript, duration
      - wer o wer_longform (si existe)
      - (opcional) quality_score

    purpose:
      - "default":
          usa quality_score guardado si existe; si no, deriva desde WER.
      - "wer_disc":
          SIEMPRE deriva targets robustos desde WER y puede filtrar outliers
          (recomendado para pretrain del D_WER).

    Target robusto (para WER arbitrariamente grande):
      q = sigmoid(-(log1p(min(WER, wer_cap)) - mu) / sigma)

    Donde mu/sigma se auto-calibran (si auto_calibrate=True) usando el split actual
    (después de filtrar), para asegurar varianza suficiente en targets.
    """

    def __init__(
        self,
        prepared_data_path: str,
        stage: str,
        sample_rate: int = 16000,
        normalize: bool = True,
        purpose: str = "default",              # "default" | "wer_disc"

        # Para construir targets (NO cambia el WER guardado)
        wer_cap: float = 50.0,

        # Filtrado opcional de outliers por WER:
        # - D_WER lo habilita por defecto
        # - finetune de G puede activarlo explícitamente para alinear soporte con D_WER
        wer_filter_max_train: Optional[float] = 20.0,
        wer_filter_max_eval: Optional[float] = 50.0,
        apply_wer_filter: bool = False,
        filter_eval_for_wer_filter: bool = True,

        use_longform_key: bool = True,
        expected_preproc_whisper_model: Optional[str] = None,
        expected_preproc_chunk_seconds: Optional[float] = None,
    ):
        assert stage in ("train", "val", "test"), \
            f"Stage inválido: '{stage}'. Debe ser 'train', 'val' o 'test'."
        assert purpose in ("default", "wer_disc"), \
            f"purpose inválido: '{purpose}'. Debe ser 'default' o 'wer_disc'."

        self.stage = stage
        self.sample_rate = int(sample_rate)
        self.normalize = bool(normalize)

        self.purpose = purpose
        self.wer_cap = float(wer_cap)

        self.wer_filter_max_train = float(wer_filter_max_train) if wer_filter_max_train is not None else None
        self.wer_filter_max_eval = float(wer_filter_max_eval) if wer_filter_max_eval is not None else None
        self.apply_wer_filter = bool(apply_wer_filter)
        self.filter_eval_for_wer_filter = bool(filter_eval_for_wer_filter)

        self.use_longform_key = bool(use_longform_key)

        prepared_path = Path(prepared_data_path)
        if not prepared_path.exists():
            raise FileNotFoundError(
                f"Pickle no encontrado: {prepared_path}\n"
                f"Ejecuta primero: python preprocessing.py"
            )

        with open(prepared_path, "rb") as f:
            prepared_data = pickle.load(f)

        if stage not in prepared_data:
            available = [k for k in prepared_data if isinstance(prepared_data[k], list)]
            raise KeyError(
                f"Stage '{stage}' no encontrado en el pickle.\n"
                f"Disponibles: {available}"
            )

        samples = prepared_data[stage]
        metadata = prepared_data.get("metadata", {}) if isinstance(prepared_data, dict) else {}
        self._warn_preprocessing_compatibility(
            metadata=metadata if isinstance(metadata, dict) else {},
            expected_model=expected_preproc_whisper_model,
            expected_chunk_seconds=expected_preproc_chunk_seconds,
        )

        # Validar que audios existen
        missing = [s for s in samples if not Path(s["audio_path"]).exists()]
        if missing:
            print(
                f"  [WARN][ECU911][{stage}] {len(missing)} audios no encontrados. "
                f"Ejemplo: {missing[0]['audio_path']}"
            )
            samples = [s for s in samples if Path(s["audio_path"]).exists()]

        # Filtrado opcional de outliers por WER.
        if self.apply_wer_filter:
            if self.stage == "train" and self.wer_filter_max_train is not None:
                samples = self._filter_by_wer(samples, self.wer_filter_max_train, tag="train")
            elif self.stage in ("val", "test") and self.filter_eval_for_wer_filter and self.wer_filter_max_eval is not None:
                samples = self._filter_by_wer(samples, self.wer_filter_max_eval, tag=self.stage)

        self.samples = samples
        self.has_quality_scores = all(("quality_score" in s) for s in self.samples)

        print(f"  [ECU911][{stage}] {len(self.samples)} audios cargados | purpose={self.purpose}")
        if self.has_quality_scores:
            print(f"  [ECU911][{stage}] quality_score en pickle: disponible")
        if self.purpose == "wer_disc":
            print(f"  [ECU911][{stage}] target(q)=1/(1+min(WER, wer_cap)) con wer_cap={self.wer_cap}")

    def _warn_preprocessing_compatibility(
        self,
        metadata: Dict[str, Any],
        expected_model: Optional[str],
        expected_chunk_seconds: Optional[float],
    ) -> None:
        if not metadata:
            print(
                f"  [WARN][ECU911][{self.stage}] metadata no disponible en pickle; "
                "no se puede verificar compatibilidad de preprocessing."
            )
            return

        saved_model = metadata.get("preprocessing_whisper_model_name", None)
        saved_chunk = metadata.get("whisper_chunk_seconds", None)

        # Compatibilidad de modelo Whisper usado para construir labels/stats.
        if expected_model:
            if isinstance(saved_model, str) and saved_model.strip():
                if saved_model.strip() != expected_model.strip():
                    print(
                        f"  [WARN][ECU911][{self.stage}] incompatibilidad whisper_model_name: "
                        f"pickle='{saved_model}' vs config='{expected_model}'."
                    )
            else:
                print(
                    f"  [WARN][ECU911][{self.stage}] pickle sin 'preprocessing_whisper_model_name'; "
                    f"config espera '{expected_model}'."
                )

        # Compatibilidad de chunk_seconds usado en transcripción long-form.
        if expected_chunk_seconds is not None:
            try:
                exp = float(expected_chunk_seconds)
            except Exception:
                exp = None
            try:
                got = float(saved_chunk) if saved_chunk is not None else None
            except Exception:
                got = None

            if exp is not None:
                if got is None:
                    print(
                        f"  [WARN][ECU911][{self.stage}] pickle sin 'whisper_chunk_seconds'; "
                        f"config espera {exp:.2f}s."
                    )
                elif abs(got - exp) > 1e-6:
                    print(
                        f"  [WARN][ECU911][{self.stage}] incompatibilidad whisper_chunk_seconds: "
                        f"pickle={got:.2f}s vs config={exp:.2f}s."
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample_wer(self, sample: Dict[str, Any]) -> Optional[float]:
        if self.use_longform_key and ("wer_longform" in sample):
            try:
                return float(sample["wer_longform"])
            except Exception:
                return None
        if "wer" in sample:
            try:
                return float(sample["wer"])
            except Exception:
                return None
        return None

    def _filter_by_wer(self, samples: List[Dict[str, Any]], thr: float, tag: str):
        before = len(samples)
        kept = []
        dropped = 0
        for s in samples:
            w = self._get_sample_wer(s)
            if w is None:
                dropped += 1
                continue
            if not np.isfinite(float(w)):
                dropped += 1
                continue
            if float(w) > float(thr):
                dropped += 1
                continue
            kept.append(s)
        print(
            f"  [ECU911][{self.stage}][{self.purpose}] "
            f"filtro outliers({tag}): {before} -> {len(kept)} "
            f"(dropped={dropped}, thr={thr})"
        )
        return kept

    def _quality_from_wer(self, wer_value: float) -> float:
        w = float(wer_value)
        if not np.isfinite(w) or w < 0.0:
            w = 0.0
        w = min(w, self.wer_cap)
        return float(1.0 / (1.0 + w))

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        waveform, _sr = load_audio(
            sample["audio_path"],
            target_sr=self.sample_rate,
            normalize=self.normalize,
        )

        if waveform.ndim == 2:
            waveform = waveform.squeeze(0)

        duration = float(sample.get("duration", waveform.shape[0] / self.sample_rate))

        result = {
            "waveform": waveform,
            "transcript": sample.get("transcript", ""),
            "duration": duration,
            "audio_path": sample["audio_path"],
            "wer": float(sample["wer"]),
            "stage": self.stage,
        }

        w = self._get_sample_wer(sample)
        if w is not None:
            result["wer"] = float(w)

        # Targets / masks
        quality = None

        if self.purpose == "wer_disc":
            if w is not None:
                quality = self._quality_from_wer(w)
        else:
            if "quality_score" in sample:
                try:
                    quality = float(sample["quality_score"])
                except Exception:
                    quality = None
            elif w is not None:
                quality = self._quality_from_wer(w)

        if quality is not None:
            quality = float(max(0.0, min(1.0, quality)))
            result["quality_score"] = quality
            result["quality_mask"] = True
        else:
            result["quality_mask"] = False

        # Señales auxiliares opcionales de decoder-ASR precomputadas en el pickle.
        for key in EXTRA_FLOAT_SAMPLE_KEYS:
            if key not in sample:
                continue
            try:
                v = float(sample[key])
            except Exception:
                continue
            if np.isfinite(v):
                result[key] = v

        return result

class ReplayWERDataset(Dataset):
    def __init__(self, items, sample_rate: int = 16000, normalize: bool = True, wer_cap: float = 50.0):
        self.items = items
        self.sr = int(sample_rate)
        self.normalize = bool(normalize)
        self.wer_cap = wer_cap

    def _q(self, wer):
        w = 0.0 if wer is None else float(wer)
        w = max(0.0, min(w, self.wer_cap))
        return float(1.0 / (1.0 + w))


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        w, _ = load_audio(it.audio_path, target_sr=self.sr, normalize=self.normalize)
        if w.ndim == 2:
            w = w.squeeze(0)
        return {
            "waveform": w,
            "transcript": it.transcript,
            "duration": float(it.duration),
            "quality_score": float(it.quality),
            "quality_mask": True,

            "wer": float(it.wer) if it.wer is not None else None,
            "audio_path": it.audio_path,
        }

class StratifiedWERBatchSampler(Sampler[List[int]]):
    """
    BatchSampler estratificado por WER para evitar batches con targets casi iguales.

    Idea:
      - Ordena índices por WER (preferir wer_longform si existe)
      - Divide en dos grupos: LOW (mitad inferior) y HIGH (mitad superior)
      - Cada batch toma batch_size/2 de LOW y batch_size/2 de HIGH (shuffle dentro)
    """

    def __init__(
        self,
        dataset: "ECU911Dataset",
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
        high_fraction: float = 0.5,  # por defecto mitad-high mitad-low
        prefer_log: bool = True,     # ordenar por log1p(WER) (más estable)
    ):
        assert batch_size >= 2, "batch_size debe ser >=2 para estratificar."
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.high_fraction = float(high_fraction)
        self.prefer_log = bool(prefer_log)

        # Precompute WER por índice
        wers = []
        for i in range(len(dataset)):
            s = dataset.samples[i]
            w = dataset._get_sample_wer(s)
            if w is None or (not np.isfinite(float(w))):
                w = 0.0
            w = float(max(0.0, w))
            if self.prefer_log:
                w = float(np.log1p(w))
            wers.append((w, i))

        # Orden ascendente
        wers.sort(key=lambda x: x[0])
        self.sorted_indices = [i for _, i in wers]

        # Split LOW/HIGH
        n = len(self.sorted_indices)
        cut = max(1, int(round(n * (1.0 - self.high_fraction))))
        cut = min(n - 1, cut)  # asegurar ambos grupos no vacíos
        self.low = self.sorted_indices[:cut]
        self.high = self.sorted_indices[cut:]

        self.rng = np.random.default_rng(self.seed)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        # Shuffles independientes por epoch/iter
        low = self.low.copy()
        high = self.high.copy()
        self.rng.shuffle(low)
        self.rng.shuffle(high)

        half_high = int(round(self.batch_size * self.high_fraction))
        half_high = max(1, min(self.batch_size - 1, half_high))
        half_low = self.batch_size - half_high

        # Iteradores cíclicos (para no quedarnos sin uno de los grupos)
        i_low = 0
        i_high = 0

        nbatches = len(self)
        for _ in range(nbatches):
            batch = []

            for _k in range(half_low):
                if i_low >= len(low):
                    self.rng.shuffle(low)
                    i_low = 0
                batch.append(low[i_low])
                i_low += 1

            for _k in range(half_high):
                if i_high >= len(high):
                    self.rng.shuffle(high)
                    i_high = 0
                batch.append(high[i_high])
                i_high += 1

            self.rng.shuffle(batch)

            if self.drop_last and len(batch) < self.batch_size:
                continue

            yield batch

# ============================================================================
# SpanishConversationalDataset (SPC)
# ============================================================================

class SpanishConversationalDataset(Dataset):
    """
    Dataset para Spanish Conversational Speech Corpus.
    Split train/val con semilla fija.
    """

    TIMESTAMP_RE = re.compile(
        r"\[(?P<s>\d+(?:\.\d+)?),(?P<e>\d+(?:\.\d+)?)\]\s*"
        r"(?P<spk>\S+)?\s*(?P<gender>\w+)?\s*(?P<text>.+)"
    )

    FILTER_PATTERNS = [
        r"voice\s+(?:data\s+)?collection\s+for\s+beijing\s+magic\s+data",
        r"voice\s+collection",
        r"beijing\s+magic",
    ]

    def __init__(
        self,
        corpus_dir: str,
        sample_rate: int = 16000,
        normalize: bool = True,
        degradation: Optional[TelephoneDegradation] = None,
        min_segment_duration: float = 0.2,
        min_words: int = 1,
        filter_metadata_text: bool = True,
        seed: int = 42,
        stage: str = "train",
        train_split: float = 0.9,
    ):
        assert stage in ("train", "val")

        self.corpus_dir = Path(corpus_dir)
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.degradation = degradation
        self.min_segment_duration = float(min_segment_duration)
        self.min_words = int(min_words)
        self.filter_metadata_text = bool(filter_metadata_text)

        wav_dir = self.corpus_dir / "WAV"
        txt_dir = self.corpus_dir / "TXT"
        if not wav_dir.exists():
            raise FileNotFoundError(f"SPC WAV/ no encontrado: {wav_dir}")
        if not txt_dir.exists():
            raise FileNotFoundError(f"SPC TXT/ no encontrado: {txt_dir}")

        wav_files = list(wav_dir.glob("*.wav"))

        segments: List[Dict[str, Any]] = []
        parse_stats = {
            "total_lines": 0,
            "bad_format": 0,
            "bad_timestamp": 0,
            "filtered_meta": 0,
            "short_words": 0,
            "short_duration": 0,
            "out_of_bounds": 0,
        }

        def _parse_line(raw: str):
            """
            Soporta variantes comunes:
            - [s,e] \\t speaker \\t gender \\t text
            - [s,e] \\t text
            - [s,e] speaker gender text
            """
            parts = [p.strip() for p in raw.split("\t")]

            # Formato tabular largo
            if len(parts) >= 4 and parts[0].startswith("[") and parts[0].endswith("]"):
                ts = parts[0]
                text = parts[3]
            # Formato tabular corto
            elif len(parts) >= 2 and parts[0].startswith("[") and parts[0].endswith("]"):
                ts = parts[0]
                text = parts[-1]
            else:
                # Fallback regex sobre línea completa
                m = self.TIMESTAMP_RE.match(raw)
                if not m:
                    return None, None, "bad_format"
                ts = f"[{m.group('s')},{m.group('e')}]"
                text = (m.group("text") or "").strip()

            try:
                start_sec, end_sec = [float(x.strip()) for x in ts[1:-1].split(",")]
            except Exception:
                return None, None, "bad_timestamp"

            return (start_sec, end_sec), text, None
        for wav_path in wav_files:
            txt_path = txt_dir / (wav_path.stem + ".txt")
            if not txt_path.exists():
                continue

            try:
                waveform, sr = load_audio(
                    str(wav_path),
                    target_sr=self.sample_rate,
                    normalize=self.normalize,
                )
            except Exception:
                continue

            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parse_stats["total_lines"] += 1

                parsed_ts, text, err = _parse_line(line)
                if err is not None:
                    parse_stats[err] += 1
                    continue
                start_sec, end_sec = parsed_ts
                text = text.strip()

                if self.filter_metadata_text:
                    t = text.lower().strip()
                    if any(re.search(p, t) for p in self.FILTER_PATTERNS):
                        parse_stats["filtered_meta"] += 1
                        continue

                if len(text.split()) < self.min_words:
                    parse_stats["short_words"] += 1
                    continue

                duration = end_sec - start_sec
                if duration < self.min_segment_duration:
                    parse_stats["short_duration"] += 1
                    continue

                if int(end_sec * sr) > waveform.shape[-1]:
                    parse_stats["out_of_bounds"] += 1
                    continue

                segments.append(
                    {
                        "audio_path": str(wav_path),
                        "start_sec": float(start_sec),
                        "end_sec": float(end_sec),
                        "duration": float(duration),
                        "transcript": text,
                    }
                )

        random.seed(seed)
        random.shuffle(segments)
        n_train = int(len(segments) * train_split)
        if stage == "train":
            self.segments = segments[:n_train]
        else:
            self.segments = segments[n_train:]

        print(
            f"[SPC][{stage}] segmentos={len(self.segments)} "
            f"(total_raw={len(segments)}, lines={parse_stats['total_lines']}, "
            f"bad_format={parse_stats['bad_format']}, bad_timestamp={parse_stats['bad_timestamp']}, "
            f"short_words={parse_stats['short_words']}, short_duration={parse_stats['short_duration']}, "
            f"filtered_meta={parse_stats['filtered_meta']}, out_of_bounds={parse_stats['out_of_bounds']})"
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seg = self.segments[idx]

        waveform, sr = load_audio(
            seg["audio_path"],
            target_sr=self.sample_rate,
            normalize=self.normalize,
        )

        start = int(seg["start_sec"] * sr)
        end = int(seg["end_sec"] * sr)
        chunk = waveform[..., start:end]
        if chunk.ndim == 2:
            chunk = chunk.squeeze(0)

        clean = chunk
        noisy = chunk
        if self.degradation is not None:
            noisy = self.degradation(clean)

        return {
            "waveform_clean": clean,
            "waveform_noisy": noisy,
            "transcript": seg["transcript"],
            "duration": float(seg["duration"]),
            "audio_path": seg["audio_path"],
        }


# ============================================================================
# Collates
# ============================================================================


def collate_ecu911(batch: List[Dict]) -> Dict:
    """
    - waveforms padded (B,T)
    - transcripts list[str]
    - durations (B,)
    - quality_scores (B,) + quality_mask (B,) si aplica
    - wer (B,) si aplica
    """
    max_len = max(item["waveform"].shape[0] for item in batch)
    waveforms = []

    for item in batch:
        w = item["waveform"]
        if w.shape[0] < max_len:
            w = F.pad(w, (0, max_len - w.shape[0]))
        waveforms.append(w)

    out = {
        "waveform": torch.stack(waveforms, dim=0),
        "transcripts": [item.get("transcript", "") for item in batch],
        "durations": torch.tensor([float(item["duration"]) for item in batch], dtype=torch.float32),
        "audio_paths": [item["audio_path"] for item in batch],
    }

    # ✅ WER: usa clave "wer" (no "wers") para que el step sea simple
    # (y de paso mantén "wers" si quieres por compatibilidad)
    if "wer" in batch[0]:
        wer_t = torch.tensor([float(item.get("wer", 0.0)) for item in batch], dtype=torch.float32)
        out["wer"] = wer_t
        out["wers"] = wer_t  # opcional: compat

    # ✅ quality score + mask
    # FIX: si no existe quality_mask en el item, asumimos True (válido)
    if ("quality_score" in batch[0]) or ("quality_mask" in batch[0]):
        qs = []
        qm = []
        for item in batch:
            m = bool(item.get("quality_mask", True))  # 👈 FIX CLAVE
            qm.append(m)
            qs.append(float(item.get("quality_score", 0.0)) if m else 0.0)

        out["quality_scores"] = torch.tensor(qs, dtype=torch.float32)  # (B,)
        out["quality_mask"] = torch.tensor(qm, dtype=torch.bool)       # (B,)

    # Señales auxiliares opcionales (si existen en el sample original).
    for key in EXTRA_FLOAT_SAMPLE_KEYS:
        if not any((key in item) for item in batch):
            continue
        vals = []
        masks = []
        for item in batch:
            raw = item.get(key, None)
            ok = False
            v = 0.0
            if raw is not None:
                try:
                    v = float(raw)
                    ok = np.isfinite(v)
                except Exception:
                    ok = False
            vals.append(float(v) if ok else 0.0)
            masks.append(bool(ok))
        out[key] = torch.tensor(vals, dtype=torch.float32)
        out[f"{key}_mask"] = torch.tensor(masks, dtype=torch.bool)

    return out


def collate_fn_semantic_chunks(batch: List[Dict]) -> Dict:
    """Collate para SpanishConversationalDataset."""
    if "waveform_clean" not in batch[0]:
        raise ValueError("Expected 'waveform_clean' in batch")

    max_len = max(item["waveform_clean"].shape[-1] for item in batch)

    clean = []
    noisy = []
    for item in batch:
        c = item["waveform_clean"]
        n = item["waveform_noisy"]
        if c.shape[-1] < max_len:
            c = torch.nn.functional.pad(c, (0, max_len - c.shape[-1]))
            n = torch.nn.functional.pad(n, (0, max_len - n.shape[-1]))
        clean.append(c)
        noisy.append(n)

    return {
        "waveform_clean": torch.stack(clean, dim=0),
        "waveform_noisy": torch.stack(noisy, dim=0),
        "transcripts": [item["transcript"] for item in batch],
        "durations": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
        "audio_paths": [item["audio_path"] for item in batch],
    }


# ============================================================================
# DataLoader factories
# ============================================================================

def create_ecu911_dataloader(
    config: dict,
    stage: str = "train",
    batch_size: Optional[int] = None,
    purpose: str = "default",  # "default" | "wer_disc"
    apply_wer_filter: Optional[bool] = None,
    wer_filter_max_train: Optional[float] = None,
    wer_filter_max_eval: Optional[float] = None,
    filter_eval_for_wer_filter: Optional[bool] = None,
) -> DataLoader:
    assert stage in ("train", "val", "test"), f"Stage inválido: {stage}"

    prepared_data_path = Path(config["paths"]["processed_data"]) / "ecu911_prepared.pkl"
    if not prepared_data_path.exists():
        raise FileNotFoundError(
            f"Pickle no encontrado: {prepared_data_path}\n"
            f"Ejecuta primero: python preprocessing.py"
        )

    wd_cfg = config.get("wer_discriminator", {})
    wer_cap = float(wd_cfg.get("wer_cap", 50.0))
    if wer_filter_max_train is None:
        wer_filter_train = wd_cfg.get("wer_filter_max_train", 20.0)
    else:
        wer_filter_train = wer_filter_max_train
    if wer_filter_max_eval is None:
        wer_filter_eval = wd_cfg.get("wer_filter_max_eval", 50.0)
    else:
        wer_filter_eval = wer_filter_max_eval
    if apply_wer_filter is None:
        apply_wer_filter = (purpose == "wer_disc")
    if filter_eval_for_wer_filter is None:
        filter_eval_for_wer_filter = bool(wd_cfg.get("filter_eval_for_wer_disc", True))
    expected_preproc_whisper_model = _resolve_expected_preprocessing_whisper_model(config)
    expected_preproc_chunk_seconds = float(config.get("evaluation", {}).get("whisper_chunk_seconds", 30.0))

    dataset = ECU911Dataset(
        prepared_data_path=str(prepared_data_path),
        stage=stage,
        sample_rate=int(config["audio"]["target_sr"]),
        normalize=bool(config["audio"]["normalize_rms"]),
        purpose=purpose,
        wer_cap=wer_cap,
        wer_filter_max_train=wer_filter_train,
        wer_filter_max_eval=wer_filter_eval,
        apply_wer_filter=bool(apply_wer_filter),
        filter_eval_for_wer_filter=bool(filter_eval_for_wer_filter),
        use_longform_key=bool(wd_cfg.get("use_longform_key", True)),
        expected_preproc_whisper_model=expected_preproc_whisper_model,
        expected_preproc_chunk_seconds=expected_preproc_chunk_seconds,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"[ECU911][{stage}] Dataset vacío.")

    if batch_size is None:
        batch_size = int(config["data"]["ecu911"]["batch_size"])

    # ✅ Recomendación fuerte: para pretrain D, NO uses batch_size=2
    if purpose == "wer_disc":
        bs_min = int(wd_cfg.get("pretrain_batch_min", 8))
        if batch_size < bs_min:
            #print(f"  [ECU911][{stage}] ⚠️ batch_size={batch_size} muy bajo para D. Forzando a {bs_min}.")
            print(f"  [WARN][ECU911][{stage}] batch_size={batch_size} muy bajo para D")
            #batch_size = bs_min

    use_strat = bool(wd_cfg.get("stratified_sampling", True))
    use_weighted = bool(wd_cfg.get("weighted_sampling", False))
    seed = int(config.get("seed", 42))

    def _build_weighted_sampler(ds: "ECU911Dataset") -> WeightedRandomSampler:
        mode = str(wd_cfg.get("weighted_mode", "log1p_wer"))  # log1p_wer | wer | rank
        alpha = float(wd_cfg.get("weighted_alpha", 1.0))
        power = float(wd_cfg.get("weighted_power", 1.0))
        clip_max = float(wd_cfg.get("weighted_clip_max", 4.0))
        replacement = bool(wd_cfg.get("weighted_replacement", True))
        n_samples_cfg = wd_cfg.get("weighted_num_samples", None)

        vals = []
        for s in ds.samples:
            w = ds._get_sample_wer(s)
            if w is None or (not np.isfinite(float(w))):
                w = 0.0
            w = float(max(0.0, min(float(w), wer_cap)))
            if mode == "wer":
                v = w
            else:
                v = float(np.log1p(w))
            vals.append(v)
        arr = np.asarray(vals, dtype=np.float64)

        if mode == "rank":
            order = np.argsort(arr)
            rank = np.empty_like(order, dtype=np.float64)
            rank[order] = np.linspace(0.0, 1.0, num=len(order), endpoint=False, dtype=np.float64)
            tail = rank
        else:
            med = float(np.median(arr))
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            iqr = max(1e-6, q3 - q1)
            tail = np.clip((arr - med) / iqr, a_min=0.0, a_max=None)

        weights = 1.0 + alpha * np.power(tail, power)
        weights = np.clip(weights, a_min=1e-4, a_max=clip_max)

        if n_samples_cfg is None:
            num_samples = len(ds)
        else:
            num_samples = max(1, int(n_samples_cfg))

        g = torch.Generator()
        g.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
            generator=g,
        )
        print(
            "  [ECU911][train][wer_disc] usando WeightedRandomSampler "
            f"(mode={mode}, alpha={alpha}, power={power}, clip_max={clip_max}, "
            f"replacement={replacement}, num_samples={num_samples}, "
            f"w_min={weights.min():.3f}, w_mean={weights.mean():.3f}, w_max={weights.max():.3f})"
        )
        return sampler

    # Weighted sampler (train + wer_disc) tiene prioridad sobre estratificado.
    if (purpose == "wer_disc") and (stage == "train") and use_weighted:
        sampler = _build_weighted_sampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"]["pin_memory"]),
            prefetch_factor=(config["data"].get("prefetch_factor", 2) if int(config["data"]["num_workers"]) > 0 else None),
            collate_fn=collate_ecu911,
            drop_last=False,
        )
        return dataloader

    # ✅ SOLO estratificar en train + wer_disc
    if (purpose == "wer_disc") and (stage == "train") and use_strat:
        high_frac = float(wd_cfg.get("strat_high_fraction", 0.5))
        batch_sampler = StratifiedWERBatchSampler(
            dataset=dataset,
            batch_size=int(batch_size),
            drop_last=False,
            seed=seed,
            high_fraction=high_frac,
            prefer_log=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=bool(config["data"]["pin_memory"]),
            prefetch_factor=(config["data"].get("prefetch_factor", 2) if int(config["data"]["num_workers"]) > 0 else None),
            collate_fn=collate_ecu911,
        )
        print(f"  [ECU911][train][wer_disc] usando StratifiedWERBatchSampler (high_frac={high_frac})")
        return dataloader

    # Default loader
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(stage == "train"),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"]),
        prefetch_factor=(config["data"].get("prefetch_factor", 2) if int(config["data"]["num_workers"]) > 0 else None),
        collate_fn=collate_ecu911,
        drop_last=False,
    )
    return dataloader


def create_spc_dataloader(
    config: dict,
    corpus_dir: str,
    use_degradation: bool = True,
    stage: str = "train",
    **kwargs,
) -> DataLoader:
    """
    Crea DataLoader para SPC.

    Args:
        config          : configuración completa
        corpus_dir      : ruta al corpus SPC
        use_degradation : aplicar TelephoneDegradation
        stage           : "train" | "val"
    """
    degradation = None
    if use_degradation:
        degradation = TelephoneDegradation(
            sample_rate=config["audio"]["target_sr"],
            **config.get("telephone_degradation", {}),
        )

    dataset = SpanishConversationalDataset(
        corpus_dir=corpus_dir,
        sample_rate=config["audio"]["target_sr"],
        normalize=config["audio"]["normalize_rms"],
        degradation=degradation,
        seed=config.get("seed", 42),
        stage=stage,
        train_split=config["data"]["train_split"],
        **kwargs,
    )

    dl = DataLoader(
        dataset,
        batch_size=config["data"]["spc"]["batch_size"],
        shuffle=(stage == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        prefetch_factor=(
            config["data"].get("prefetch_factor", 2)
            if config["data"]["num_workers"] > 0 else None
        ),
        collate_fn=collate_fn_semantic_chunks,
        drop_last=False,
    )
    return dl

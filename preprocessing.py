"""
Preprocessing - FASE 0: Preparación de Datos
=============================================

Genera:
  - data/processed/ecu911_prepared.pkl  →  {"train": [...], "val": [...], "test": [...], "metadata": {...}}
  - data/processed/clean_prepared.pkl   →  {"train": [...], "val": [...], "metadata": {...}}

CAMBIO CLAVE:
- WER de ECU911 TRAIN se calcula en modo long-form por chunks (sin timestamps).
- Normalización de texto ÚNICA (utils.text.normalize_text_for_wer).
- quality_score más estable: 1/(1+WER) para soportar WER>1.
"""

import argparse
import sys
import zlib
import torch
import pandas as pd
import pickle
import random
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer as compute_wer_fn

# Permite ejecutar este script como "python preprocessing.py" desde cualquier cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import load_config
from utils.text import normalize_text_for_wer
from utils.audio import load_audio, resample_audio, normalize_rms, save_audio


# ============================================================================
# TEXT / SCORE UTILITIES
# ============================================================================

def quality_score_from_wer(wer: float) -> float:
    """
    Target estable para D:
      quality = 1/(1+WER)
    - soporta WER>1 sin saturar todo a 0
    - sigue siendo [0,1]
    """
    try:
        wer = float(wer)
    except Exception:
        wer = 999.0
    if wer < 0:
        wer = 0.0
    return float(1.0 / (1.0 + wer))


def parse_decimal(val) -> float:
    """Convierte '0,390070922' o 0.39 a float. Devuelve 1.0 si falla."""
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return 1.0


def is_valid_transcript(text) -> bool:
    """True si el texto tiene al menos un carácter alfanumérico."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    return len(text) > 0 and any(c.isalnum() for c in text)


# ============================================================================
# WHISPER UTILITIES
# ============================================================================

def load_whisper(
    model_name: str,
    device: torch.device,
) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Carga Whisper model + processor."""
    print(f"\nCargando Whisper '{model_name}' en {device}...")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    )
    model.to(device).eval()
    processor = WhisperProcessor.from_pretrained(model_name)
    print("✓ Whisper listo")
    return model, processor


def _resolve_preproc_whisper_model_name(config: dict) -> str:
    pcfg = config.get("preprocessing", {})
    name = pcfg.get("whisper_model_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    wcfg = config.get("models", {}).get("whisper", {})
    for key in ("model_name", "encoder_model_name"):
        val = wcfg.get(key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    eval_name = config.get("evaluation", {}).get("whisper_model_name", None)
    if isinstance(eval_name, str) and eval_name.strip():
        return eval_name.strip()
    return "openai/whisper-small"


def compute_wer_with_whisper_longform(
    waveform: torch.Tensor,
    reference_text: str,
    whisper_model: WhisperForConditionalGeneration,
    whisper_processor: WhisperProcessor,
    device: torch.device,
    sample_rate: int = 16000,
    chunk_seconds: float = 30.0,
    collect_decoder_stats: bool = False,
) -> Tuple[float, float, Dict[str, float] | None]:
    """
    Calcula dos WERs:
      - wer_longform: transcribe por chunks consecutivos de chunk_seconds y concatena
      - wer_first30 : solo el primer chunk (diagnóstico)

    NO requiere timestamps.

    Nota: WER puede ser >1 (inserciones). Eso es normal.
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    waveform = waveform.detach().to(torch.float32).cpu()
    empty_stats = {
        "asr_avg_entropy": float("nan"),
        "asr_avg_logprob": float("nan"),
        "asr_no_speech_prob": float("nan"),
        "asr_compression_ratio": float("nan"),
        "asr_token_count": 0.0,
    }
    if waveform.numel() == 0:
        return 1.0, 1.0, (empty_stats if collect_decoder_stats else None)

    ref = normalize_text_for_wer(reference_text)
    if not ref:
        return 1.0, 1.0, (empty_stats if collect_decoder_stats else None)

    chunk_len = int(chunk_seconds * sample_rate)
    if chunk_len <= 0:
        chunk_len = waveform.shape[0]

    model_dtype = next(whisper_model.parameters()).dtype
    tokenizer = whisper_processor.tokenizer
    nospeech_token_id = None
    try:
        tid = tokenizer.convert_tokens_to_ids("<|nospeech|>")
        if isinstance(tid, int) and tid >= 0:
            nospeech_token_id = int(tid)
    except Exception:
        nospeech_token_id = None

    def _transcribe_chunk(w: torch.Tensor):
        w_np = w.numpy()
        feats = whisper_processor(
            w_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feats.input_features.to(device=device, dtype=model_dtype)
        attention_mask = getattr(feats, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        gen_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            # Evita warning de logits processors duplicados en algunos checkpoints.
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
        }
        if collect_decoder_stats:
            gen_kwargs["return_dict_in_generate"] = True
        if attention_mask is not None:
            out = whisper_model.generate(input_features, attention_mask=attention_mask, **gen_kwargs)
        else:
            out = whisper_model.generate(input_features, **gen_kwargs)
        if collect_decoder_stats:
            ids = out.sequences
            # Deriva logits paso-a-paso con teacher forcing para evitar depender de
            # output_scores en generate (no siempre soportado por todas las versiones).
            if ids.shape[1] > 1:
                decoder_input_ids = ids[:, :-1]
                with torch.no_grad():
                    fwd = whisper_model(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                # logits alineados para predecir ids[:, 1:]
                scores = [fwd.logits[:, i, :] for i in range(fwd.logits.shape[1])]
            else:
                scores = []
        else:
            ids = out
            scores = []
        txt = whisper_processor.batch_decode(ids, skip_special_tokens=True)[0]
        return (txt or "").strip(), ids, scores

    hyps = []
    hyp_first = ""
    entropy_sum = 0.0
    logprob_sum = 0.0
    token_count = 0
    nospeech_prob_sum = 0.0
    nospeech_count = 0
    with torch.no_grad():
        # Una sola pasada por chunks: evita transcribir dos veces el primer chunk.
        start = 0
        chunk_idx = 0
        while start < waveform.shape[0]:
            end = min(start + chunk_len, waveform.shape[0])
            chunk = waveform[start:end]
            t, ids, scores = _transcribe_chunk(chunk)
            if t:
                hyps.append(t)
            if chunk_idx == 0:
                hyp_first = normalize_text_for_wer(t)

            if collect_decoder_stats and (len(scores) > 0):
                if nospeech_token_id is not None:
                    step0 = scores[0].float()[0]
                    p0 = torch.softmax(step0, dim=-1)
                    if nospeech_token_id < int(p0.numel()):
                        nospeech_prob_sum += float(p0[nospeech_token_id].item())
                        nospeech_count += 1
                seq = ids[0]
                gen_tokens = seq[-len(scores):]
                for step, step_logits in enumerate(scores):
                    logits = step_logits.float()[0]
                    logp = torch.log_softmax(logits, dim=-1)
                    p = torch.softmax(logits, dim=-1)
                    ent = float(-(p * logp).sum().item())
                    tok = int(gen_tokens[step].item())
                    tok_logp = float(logp[tok].item())
                    entropy_sum += ent
                    logprob_sum += tok_logp
                    token_count += 1

            start = end
            chunk_idx += 1

    hyp_long = normalize_text_for_wer(" ".join(hyps))
    try:
        wer_first = float(compute_wer_fn([ref], [hyp_first]))
    except Exception:
        wer_first = 1.0
    try:
        wer_long = float(compute_wer_fn([ref], [hyp_long]))
    except Exception:
        wer_long = 1.0

    stats = None
    if collect_decoder_stats:
        hyp_text_raw = " ".join(hyps).strip()
        if hyp_text_raw:
            raw_b = hyp_text_raw.encode("utf-8", errors="ignore")
            comp_b = zlib.compress(raw_b)
            comp_ratio = float(len(raw_b) / max(1, len(comp_b)))
        else:
            comp_ratio = float("nan")
        if token_count > 0:
            stats = {
                "asr_avg_entropy": float(entropy_sum / float(token_count)),
                "asr_avg_logprob": float(logprob_sum / float(token_count)),
                "asr_no_speech_prob": (
                    float(nospeech_prob_sum / float(nospeech_count))
                    if nospeech_count > 0 else float("nan")
                ),
                "asr_compression_ratio": comp_ratio,
                "asr_token_count": float(token_count),
            }
        else:
            stats = empty_stats

    return wer_long, wer_first, stats


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def process_and_save_audio(
    audio_path: Path,
    output_dir: Path,
    target_sr: int,
    normalize: bool,
    target_rms: float,
) -> Tuple[torch.Tensor, Path]:
    """
    Carga un audio, resamplea a target_sr, normaliza RMS y lo guarda.
    """
    waveform, orig_sr = load_audio(str(audio_path), target_sr=None, normalize=False)
    waveform_16k = resample_audio(waveform, orig_sr=orig_sr, target_sr=target_sr)

    if normalize:
        waveform_16k = normalize_rms(waveform_16k, target_rms=target_rms)

    waveform_16k = waveform_16k.to(torch.float32)
    if waveform_16k.ndim == 2:
        waveform_16k = waveform_16k.squeeze(0)

    out_path = output_dir / f"{audio_path.stem}_16k.wav"
    save_audio(str(out_path), waveform_16k, sample_rate=target_sr)
    return waveform_16k, out_path


# ============================================================================
# ECU911 TRAIN
# ============================================================================

def process_ecu911_train(config: dict) -> Dict:
    print("\n" + "=" * 80)
    print("PROCESANDO ECU911 TRAIN  (WER long-form con Whisper)")
    print("=" * 80)

    audio_dir     = Path(config["paths"]["ecu911_audios"])
    metadata_path = Path(config["paths"]["ecu911_metadata"])
    processed_dir = Path(config["paths"]["processed_data"])
    audio_out_dir = processed_dir / "ecu911_16khz" / "train"

    _check_path_exists(audio_dir,     "Directorio de audios train")
    _check_path_exists(metadata_path, "Metadata train")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCargando metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    print(f"Filas: {len(metadata)}")

    _check_columns(metadata, {"TEXTO_V2", "ARCHIVO_NOMBRE"}, "metadata train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper_model_name = _resolve_preproc_whisper_model_name(config)
    whisper_model, whisper_processor = load_whisper(whisper_model_name, device)

    chunk_seconds = float(config.get("evaluation", {}).get("whisper_chunk_seconds", 30.0))
    compute_decoder_stats = bool(config.get("preprocessing", {}).get("compute_decoder_stats", False))
    print(f"[preprocessing] compute_decoder_stats={compute_decoder_stats}")
    sr = int(config["audio"]["target_sr"])

    valid_samples    = []
    skipped_missing  = 0
    skipped_invalid  = 0
    skipped_wer_fail = 0

    print(f"\nProcesando {len(metadata)} audios...")
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):

        audio_filename = str(row["ARCHIVO_NOMBRE"]).strip()
        if not audio_filename.lower().endswith(".wav"):
            audio_filename += ".wav"

        audio_path = audio_dir / audio_filename
        if not audio_path.exists():
            skipped_missing += 1
            continue

        transcript = str(row.get("TEXTO_V2", "")).strip()
        if not is_valid_transcript(transcript):
            skipped_invalid += 1
            continue

        try:
            waveform_16k, out_path = process_and_save_audio(
                audio_path  = audio_path,
                output_dir  = audio_out_dir,
                target_sr   = sr,
                normalize   = config["audio"]["normalize_rms"],
                target_rms  = config["audio"]["target_rms"],
            )
        except Exception as e:
            print(f"\n  ⚠️  Audio '{audio_filename}': {e}")
            continue

        try:
            wer_long, wer_first, dec_stats = compute_wer_with_whisper_longform(
                waveform=waveform_16k,
                reference_text=transcript,
                whisper_model=whisper_model,
                whisper_processor=whisper_processor,
                device=device,
                sample_rate=sr,
                chunk_seconds=chunk_seconds,
                collect_decoder_stats=compute_decoder_stats,
            )
        except Exception as e:
            print(f"\n  ⚠️  WER '{audio_filename}': {e}")
            skipped_wer_fail += 1
            continue

        sample_out = {
            "audio_path"    : str(out_path),
            "transcript"    : transcript,
            "duration"      : waveform_16k.shape[0] / sr,
            "tra_id"        : str(row.get("TRA_ID", "")),
            "incident_grade": str(row.get("INCIDENTGRADENAME", "")),
            "incident_type" : str(row.get("INCIDENTYPENAME", "")),

            # ✅ targets “completos”
            "wer"           : float(wer_long),         # por compatibilidad, ahora “wer” = longform
            "wer_longform"  : float(wer_long),
            "wer_first30"   : float(wer_first),
            "quality_score" : quality_score_from_wer(wer_long),
        }
        if dec_stats is not None:
            sample_out.update(dec_stats)
        valid_samples.append(sample_out)

    print(f"\nProcesados OK     : {len(valid_samples)}")
    print(f"Skipped – faltante: {skipped_missing}")
    print(f"Skipped – transcript: {skipped_invalid}")
    print(f"Skipped – WER fail: {skipped_wer_fail}")

    if not valid_samples:
        raise RuntimeError("No se procesó ningún audio de ECU911 train.")

    _print_wer_stats(valid_samples, "TRAIN (longform)")
    if compute_decoder_stats:
        vals_e = [float(s["asr_avg_entropy"]) for s in valid_samples if "asr_avg_entropy" in s]
        vals_lp = [float(s["asr_avg_logprob"]) for s in valid_samples if "asr_avg_logprob" in s]
        vals_ns = [float(s["asr_no_speech_prob"]) for s in valid_samples if "asr_no_speech_prob" in s]
        vals_cr = [float(s["asr_compression_ratio"]) for s in valid_samples if "asr_compression_ratio" in s]
        print(
            "[preprocessing] decoder_stats "
            f"n_entropy={len(vals_e)} n_logprob={len(vals_lp)} n_nospeech={len(vals_ns)} n_compr={len(vals_cr)} "
            f"mean_entropy={(sum(vals_e)/len(vals_e)):.4f} "
            f"mean_logprob={(sum(vals_lp)/len(vals_lp)):.4f} "
            f"mean_nospeech={(sum(vals_ns)/len(vals_ns)):.4f} "
            f"mean_compr={(sum(vals_cr)/len(vals_cr)):.4f}"
            if (vals_e and vals_lp and vals_ns and vals_cr)
            else f"[preprocessing] decoder_stats disponibles parciales: "
                 f"entropy={len(vals_e)} logprob={len(vals_lp)} "
                 f"nospeech={len(vals_ns)} compr={len(vals_cr)}"
        )

    random.seed(config["seed"])
    random.shuffle(valid_samples)
    n = int(len(valid_samples) * config["data"]["train_split"])

    print(f"\nSplit → Train: {n} | Val: {len(valid_samples) - n}")
    return {"train": valid_samples[:n], "val": valid_samples[n:]}


# ============================================================================
# ECU911 TEST
# ============================================================================

def process_ecu911_test(config: dict) -> List[Dict]:
    print("\n" + "=" * 80)
    print("PROCESANDO ECU911 TEST  (WER del CSV)")
    print("=" * 80)

    audio_dir     = Path(config["paths"]["ecu911_test_audios"])
    metadata_path = Path(config["paths"]["ecu911_test_metadata"])
    processed_dir = Path(config["paths"]["processed_data"])
    audio_out_dir = processed_dir / "ecu911_16khz" / "test"

    _check_path_exists(audio_dir,     "Directorio de audios test")
    _check_path_exists(metadata_path, "Metadata test")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path, dtype=str)
    _check_columns(metadata, {"text", "archivo_nombre", "wer"}, "metadata test")

    sr = int(config["audio"]["target_sr"])

    valid_samples   = []
    skipped_missing = 0
    skipped_invalid = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_filename = str(row["archivo_nombre"]).strip()
        if not audio_filename.lower().endswith(".wav"):
            audio_filename += ".wav"

        audio_path = audio_dir / audio_filename
        if not audio_path.exists():
            skipped_missing += 1
            continue

        transcript = str(row.get("text", "")).strip()
        if not is_valid_transcript(transcript):
            skipped_invalid += 1
            continue

        wer_csv = parse_decimal(row.get("wer", "1"))

        try:
            waveform_16k, out_path = process_and_save_audio(
                audio_path  = audio_path,
                output_dir  = audio_out_dir,
                target_sr   = sr,
                normalize   = config["audio"]["normalize_rms"],
                target_rms  = config["audio"]["target_rms"],
            )
        except Exception as e:
            print(f"\n  ⚠️  Audio '{audio_filename}': {e}")
            continue

        valid_samples.append({
            "audio_path"   : str(out_path),
            "transcript"   : transcript,
            "duration"     : waveform_16k.shape[0] / sr,
            "tra_id"       : str(row.get("tra_id", "")),
            "wer"          : float(wer_csv),
            "quality_score": quality_score_from_wer(wer_csv),
        })

    print(f"\nProcesados OK     : {len(valid_samples)}")
    print(f"Skipped – faltante: {skipped_missing}")
    print(f"Skipped – transcript: {skipped_invalid}")

    if not valid_samples:
        raise RuntimeError("No se procesó ningún audio de ECU911 test.")

    _print_wer_stats(valid_samples, "TEST")
    return valid_samples


# ============================================================================
# SPC CLEAN (sin cambios sustanciales)
# ============================================================================

def process_clean_dataset(config: dict) -> Dict:
    print("\n" + "=" * 80)
    print("PROCESANDO SPC CLEAN")
    print("=" * 80)

    base_dir      = Path(config["paths"]["spc_corpus_dir"])
    wav_dir       = base_dir / "WAV"
    txt_dir       = base_dir / "TXT"
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    _check_path_exists(wav_dir, "SPC WAV/")
    _check_path_exists(txt_dir, "SPC TXT/")

    _filter_patterns = [
        r"voice\s+(?:data\s+)?collection\s+for\s+beijing\s+magic\s+data",
        r"voice\s+collection",
        r"beijing\s+magic",
        r"ok\s+ok",
    ]

    def should_filter(text: str) -> bool:
        t = text.lower().strip()
        return any(re.search(p, t) for p in _filter_patterns)

    wav_files = list(wav_dir.glob("*.wav"))
    print(f"\nWAVs encontrados: {len(wav_files)}")

    valid_samples = []
    sr = int(config["audio"]["target_sr"])

    for wav_path in tqdm(wav_files, desc="SPC"):
        txt_path = txt_dir / (wav_path.stem + ".txt")
        if not txt_path.exists():
            continue

        try:
            waveform, _ = load_audio(
                str(wav_path),
                target_sr  = sr,
                normalize  = config["audio"]["normalize_rms"],
            )
        except Exception as e:
            print(f"  Error {wav_path.name}: {e}")
            continue

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            ts = parts[0].strip()
            if not (ts.startswith("[") and ts.endswith("]")):
                continue

            try:
                start_sec, end_sec = [float(x) for x in ts[1:-1].split(",")]
            except (ValueError, IndexError):
                continue

            text = parts[3].strip()
            if should_filter(text) or len(text) < 10:
                continue

            duration = end_sec - start_sec
            if not (1.0 <= duration <= 10.0) or start_sec < 0 or end_sec <= start_sec:
                continue

            if int(end_sec * sr) > waveform.shape[-1]:
                continue

            valid_samples.append({
                "audio_path": str(wav_path),
                "start_sec" : start_sec,
                "end_sec"   : end_sec,
                "duration"  : duration,
                "transcript": text,
            })

    if not valid_samples:
        raise RuntimeError("Sin chunks SPC. Verifica formato TXT.")

    random.seed(config["seed"])
    random.shuffle(valid_samples)
    n = int(len(valid_samples) * config["data"]["train_split"])

    result = {
        "train"   : valid_samples[:n],
        "val"     : valid_samples[n:],
        "metadata": {"sample_rate": sr, "source": "spc"},
    }

    out_path = Path(config["paths"]["processed_data"]) / "clean_prepared.pkl"
    _save_pickle(result, out_path, "SPC")
    return result


# ============================================================================
# VERIFICACIÓN (ajustada para WER>1)
# ============================================================================

def verify_ecu911_pickle(pkl_path: Path) -> bool:
    print(f"\nVerificando {pkl_path} ...")

    if not pkl_path.exists():
        print(f"  ❌ Archivo no existe: {pkl_path}")
        return False

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    missing_splits = {"train", "val", "test"} - set(data.keys())
    if missing_splits:
        print(f"  ❌ Splits faltantes: {missing_splits}")
        return False

    all_ok = True
    required_fields = {"audio_path", "transcript", "wer", "quality_score"}

    for split in ["train", "val", "test"]:
        samples = data[split]
        if not samples:
            print(f"  ⚠️  '{split}' está vacío")
            all_ok = False
            continue

        missing_files = invalid_values = missing_fields_n = 0

        for s in samples:
            if not required_fields.issubset(s.keys()):
                missing_fields_n += 1
                continue
            if not Path(s["audio_path"]).exists():
                missing_files += 1

            qs  = float(s.get("quality_score", -1.0))
            w   = float(s.get("wer", -1.0))
            # ✅ WER puede ser >1. Validamos solo que sea >=0
            if not (0.0 <= qs <= 1.0) or w < 0.0:
                invalid_values += 1

        ok_split = (missing_fields_n == 0 and missing_files == 0 and invalid_values == 0)
        print(f"\n  {'✓' if ok_split else '❌'} {split}: {len(samples)} samples")

        if missing_fields_n:
            print(f"    ❌ {missing_fields_n} samples con campos faltantes")
            all_ok = False
        if missing_files:
            print(f"    ❌ {missing_files} archivos no encontrados en disco")
            all_ok = False
        if invalid_values:
            print(f"    ❌ {invalid_values} values inválidos")
            all_ok = False

        s0 = samples[0]
        print(f"    Ejemplo → wer={float(s0.get('wer',0)):.3f} quality={float(s0.get('quality_score',0)):.3f} dur={float(s0.get('duration',0)):.1f}s")

    print("\n" + ("✅ OK" if all_ok else "❌ PROBLEMAS"))
    return all_ok


# ============================================================================
# HELPERS
# ============================================================================

def _check_path_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} no encontrado: {path}")


def _check_columns(df: pd.DataFrame, required: set, label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {label}: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


def _print_wer_stats(samples: List[Dict], label: str = "") -> None:
    wers = [float(s["wer"]) for s in samples if "wer" in s]
    if not wers:
        return
    tag = f" [{label}]" if label else ""
    print(f"\nEstadísticas WER{tag} ({len(wers)} audios):")
    print(f"  Media   : {sum(wers)/len(wers):.4f}")
    print(f"  Min     : {min(wers):.4f}")
    print(f"  Max     : {max(wers):.4f}")
    print(f"  Mediana : {sorted(wers)[len(wers)//2]:.4f}")
    qs = [float(s["quality_score"]) for s in samples if "quality_score" in s]
    if qs:
        print(f"  Quality score media: {sum(qs)/len(qs):.4f}")


def _save_pickle(data: dict, path: Path, label: str = "") -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)
    info = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    tag  = f" [{label}]" if label else ""
    print(f"  ✓ Guardado{tag}: {path}  {info}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("PREPROCESSING — PREPARACIÓN DE DATOS")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Preprocessing ECU911/SPC")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta YAML de config. Si no se pasa, intenta config.hpc.yaml o usa defaults.",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        fallback = Path("config.hpc.yaml")
        cfg_path = str(fallback) if fallback.exists() else None
    config = load_config(cfg_path)
    print(f"✓ Config cargada ({cfg_path if cfg_path else 'defaults'})")

    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    ecu911_pkl = processed_dir / "ecu911_prepared.pkl"
    ecu911     = {}

    # PASO 1: TEST
    ecu911["test"] = process_ecu911_test(config)
    _save_pickle(ecu911, ecu911_pkl, "checkpoint test")

    # PASO 2: TRAIN
    tv = process_ecu911_train(config)
    ecu911["train"] = tv["train"]
    ecu911["val"]   = tv["val"]
    ecu911["metadata"] = {
        "sample_rate"    : int(config["audio"]["target_sr"]),
        "original_sr"    : int(config.get("audio", {}).get("original_sr", config["audio"]["target_sr"])),
        "has_wer_labels" : True,
        "wer_mode"       : "longform_chunks",
        "preprocessing_whisper_model_name": _resolve_preproc_whisper_model_name(config),
        "whisper_chunk_seconds": float(config.get("evaluation", {}).get("whisper_chunk_seconds", 30.0)),
        "quality_formula": "1/(1+wer)",
        "has_decoder_stats": bool(config.get("preprocessing", {}).get("compute_decoder_stats", False)),
        "decoder_stats_keys": [
            "asr_avg_entropy",
            "asr_avg_logprob",
            "asr_no_speech_prob",
            "asr_compression_ratio",
            "asr_token_count",
        ],
    }
    _save_pickle(ecu911, ecu911_pkl, "ECU911 completo")

    # PASO 3: SPC
    clean = process_clean_dataset(config)

    # VERIFICACIÓN
    ok = verify_ecu911_pickle(ecu911_pkl)

    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"ECU911 train: {len(ecu911.get('train', []))}")
    print(f"ECU911 val  : {len(ecu911.get('val', []))}")
    print(f"ECU911 test : {len(ecu911.get('test', []))}")
    print(f"SPC train   : {len(clean.get('train', []))}")
    print(f"SPC val     : {len(clean.get('val', []))}")
    print(f"\n{'✅' if ok else '❌'} Verificación: {'OK' if ok else 'PROBLEMAS'}")


if __name__ == "__main__":
    main()

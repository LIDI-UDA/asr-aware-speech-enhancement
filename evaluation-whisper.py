#!/usr/bin/env python3
"""
evaluation-whisper.py

Evalúa Generator + Whisper sobre ECU911 en subset {train,val,test,all}.

Correcciones importantes:
- Normalización de texto consistente: lowercase + sin tildes + sin puntuación.
- Evalúa 2 modos:
  (A) single_shot: Whisper sobre audio completo (si cabe).
  (B) longform_chunked: Whisper por chunks (default 30s) + concat.
- Pasa attention_mask a Whisper.generate para evitar comportamiento inesperado.
- Guarda CSV con raw/norm + WER/CER + mejora.
"""

import argparse
import os
import time
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jiwer
import soundfile as sf
from tqdm import tqdm

from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging as hf_logging
hf_logging.set_verbosity_error()

# Proyecto
from models.generator import create_generator
from utils.data import create_ecu911_dataloader
from utils.audio import sliding_window_inference
from engine.config import load_config as load_train_config

# ------------------------
# Texto / Normalización
# ------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def normalize_text_for_wer(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = strip_accents(s)
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------------------
# CER simple
# ------------------------
def char_error_rate(ref: str, hyp: str) -> float:
    r = "" if ref is None else str(ref)
    h = "" if hyp is None else str(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if r[i - 1] == h[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n

# ------------------------
# Audio IO
# ------------------------
def write_wav(path: str, waveform: torch.Tensor, sr: int):
    arr = waveform.detach().cpu().float().numpy()
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    sf.write(path, arr, sr)

def ensure_mono_1d(w: torch.Tensor) -> torch.Tensor:
    # acepta (T), (1,T), (C,T)
    if w.ndim == 1:
        return w
    if w.ndim == 2:
        if w.shape[0] == 1:
            return w[0]
        return w.mean(dim=0)
    raise ValueError(f"Waveform shape no soportado: {tuple(w.shape)}")

# ------------------------
# Whisper loader
# ------------------------
def try_load_whisper(model_name: str, device: torch.device, hf_token: Optional[str] = None,
                     retries: int = 3, sleep_sec: int = 5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[Whisper] Cargando modelo '{model_name}' (attempt {attempt}/{retries})...")
            kwargs = {}
            if hf_token:
                kwargs["use_auth_token"] = hf_token

            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                **kwargs,
            ).to(device)
            model.eval()

            proc = WhisperProcessor.from_pretrained(model_name, **kwargs)
            print("[Whisper] Cargado correctamente.")
            return model, proc
        except Exception as e:
            last_exc = e
            print(f"[Whisper] Error cargando (attempt {attempt}): {type(e).__name__}: {e}")
            if attempt < retries:
                print(f"[Whisper] Reintentando en {sleep_sec}s...")
                time.sleep(sleep_sec)

    raise RuntimeError(
        f"No se pudo cargar Whisper después de {retries} intentos. Último error: {last_exc}\n"
        f"Sugerencia: pasa token HF con --hf-token <TOKEN>."
    )

# ------------------------
# Whisper transcribe helpers
# ------------------------
@torch.no_grad()
def whisper_transcribe(
    whisper_model,
    whisper_processor,
    audio_1d: torch.Tensor,
    sr: int,
    device: torch.device,
    language: str = "es",
    task: str = "transcribe",
    condition_on_prev_tokens: bool = False,
) -> str:
    """
    audio_1d: torch (T,) CPU o GPU.
    Retorna texto raw (sin normalizar).
    """
    if audio_1d.device != torch.device("cpu"):
        audio_np = audio_1d.detach().cpu().float().numpy()
    else:
        audio_np = audio_1d.detach().float().numpy()

    feats = whisper_processor(
        audio_np,
        sampling_rate=sr,
        return_tensors="pt"
    )
    input_features = feats.input_features.to(device, dtype=torch.float16)

    # IMPORTANTE: attention_mask si existe
    attention_mask = None
    if hasattr(feats, "attention_mask") and feats.attention_mask is not None:
        attention_mask = feats.attention_mask.to(device)

    gen_kwargs = dict(language=language, task=task)
    # Para chunking, conviene desactivar “arrastre” entre chunks
    gen_kwargs["condition_on_prev_tokens"] = bool(condition_on_prev_tokens)

    if attention_mask is not None:
        generated_ids = whisper_model.generate(input_features, attention_mask=attention_mask, **gen_kwargs)
    else:
        generated_ids = whisper_model.generate(input_features, **gen_kwargs)

    text = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

@torch.no_grad()
def whisper_transcribe_longform_chunked(
    whisper_model,
    whisper_processor,
    audio_1d: torch.Tensor,
    sr: int,
    device: torch.device,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
    language: str = "es",
    task: str = "transcribe",
) -> Tuple[str, int]:
    """
    Chunking simple + concat. Retorna (texto_raw, num_chunks).
    overlap_seconds ayuda a no cortar palabras; después concatenamos.
    """
    audio_1d = ensure_mono_1d(audio_1d)
    T = int(audio_1d.shape[-1])
    chunk = int(round(chunk_seconds * sr))
    overlap = int(round(overlap_seconds * sr))
    chunk = max(1, chunk)
    overlap = max(0, min(overlap, chunk - 1))

    if T <= chunk:
        txt = whisper_transcribe(
            whisper_model, whisper_processor, audio_1d, sr, device,
            language=language, task=task, condition_on_prev_tokens=False
        )
        return txt, 1

    hop = chunk - overlap
    texts: List[str] = []
    n_chunks = 0
    for start in range(0, T, hop):
        end = min(T, start + chunk)
        seg = audio_1d[start:end].contiguous()
        txt = whisper_transcribe(
            whisper_model, whisper_processor, seg, sr, device,
            language=language, task=task, condition_on_prev_tokens=False
        )
        texts.append(txt.strip())
        n_chunks += 1
        if end >= T:
            break

    # concat naïve (suficiente para WER con normalización)
    joined = " ".join([t for t in texts if t])
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined, n_chunks

# ------------------------
# Evaluación por subset
# ------------------------
def evaluate_subset(
    checkpoint_path: str,
    config: dict,
    subset: str,
    max_samples: int,
    outdir: Path,
    generator_device: torch.device,
    whisper_device: torch.device,
    whisper_model,
    whisper_processor,
    chunk_seconds: float,
    overlap_seconds: float,
    enh_window_seconds: float,
    enh_overlap: float,
    enh_direct_max_seconds: float,
    mode: str,  # "single_shot" | "longform" | "both"
):
    print(f"\n=== Evaluando subset: {subset} ===")

    subset_out = outdir / subset
    plots_dir = subset_out / "plots"
    samples_dir = subset_out / "samples"
    stats_dir = subset_out / "stats"
    for d in (subset_out, plots_dir, samples_dir, stats_dir):
        d.mkdir(parents=True, exist_ok=True)

    sr = int(config["audio"]["target_sr"])

    # ---- Generator
    print(f"[{subset}] Cargando generator desde {checkpoint_path} ...")
    generator = create_generator(config).to(generator_device)
    ck = torch.load(checkpoint_path, map_location=generator_device, weights_only=False)
    if "generator" in ck:
        generator.load_state_dict(ck["generator"])
    else:
        generator.load_state_dict(ck)
    generator.eval()

    # ---- Dataset / loader (usar el mismo dataset, pero evaluamos determinístico)
    print(f"[{subset}] Creando dataloader stage='{subset}' ...")
    base_loader = create_ecu911_dataloader(config, stage=subset)
    ds = base_loader.dataset

    max_samples = min(int(max_samples), len(ds))
    print(f"[{subset}] samples disponibles: {len(ds)}, evaluando: {max_samples}")

    from torch.utils.data import Subset, DataLoader
    loader = DataLoader(
        Subset(ds, list(range(max_samples))),
        batch_size=1,
        shuffle=False,
        num_workers=getattr(base_loader, "num_workers", 0),
        collate_fn=getattr(base_loader, "collate_fn", None),
        pin_memory=True,
    )

    rows: List[Dict] = []

    def compute_wer(ref_norm: str, hyp_norm: str) -> float:
        # jiwer espera strings
        return float(jiwer.wer(ref_norm, hyp_norm))

    with torch.no_grad():
        pbar = tqdm(loader, total=max_samples, desc=f"[{subset}] Evaluating", unit="audio")
        for i, batch in enumerate(pbar):
            waveform = batch["waveform"]  # (1,T) o (B,T)
            transcripts = batch.get("transcripts", batch.get("transcript", [""]))
            audio_paths = batch.get("audio_paths", [None])
            durations = batch.get("durations", None)

            w = waveform[0].to(generator_device, non_blocking=True)
            w_mono = ensure_mono_1d(w)

            ref_raw = transcripts[0] if isinstance(transcripts, (list, tuple)) else str(transcripts)
            ref_raw = "" if ref_raw is None else str(ref_raw)

            # recorte por duration si está (evita padding silencioso si tu collate mete pad)
            if durations is not None:
                dur_s = float(durations[0])
                true_len = int(round(dur_s * sr))
                true_len = max(1, min(true_len, int(w_mono.shape[-1])))
                w_mono = w_mono[:true_len].contiguous()

            # ---- Enhance (ventaneado para audios largos y menor consumo VRAM)
            dur_s = float(w_mono.shape[-1]) / float(sr)
            use_direct = dur_s <= float(enh_direct_max_seconds)
            gen_t0 = time.perf_counter()
            if use_direct:
                with torch.autocast(
                    device_type=generator_device.type,
                    enabled=(generator_device.type == "cuda"),
                ):
                    enh = generator(w_mono.unsqueeze(0))
                    if isinstance(enh, tuple):
                        enh = enh[0]
                    enh = enh.squeeze(0)
                    enh = ensure_mono_1d(enh).detach()
            else:
                enh = sliding_window_inference(
                    waveform=w_mono.detach().cpu(),
                    model=generator,
                    sample_rate=sr,
                    window_duration=float(enh_window_seconds),
                    overlap=float(enh_overlap),
                    device=generator_device,
                ).to(generator_device)
                enh = ensure_mono_1d(enh).detach()
            gen_infer_seconds = float(time.perf_counter() - gen_t0)
            gen_audio_seconds = float(dur_s)
            gen_seconds_per_audio_second = (
                gen_infer_seconds / gen_audio_seconds if gen_audio_seconds > 0.0 else float("nan")
            )
            gen_rtf = gen_seconds_per_audio_second

            # ---- Transcribe
            # Original
            pred_orig_single = None
            pred_orig_long = None
            n_chunks_orig = 0

            # Enhanced
            pred_enh_single = None
            pred_enh_long = None
            n_chunks_enh = 0

            if mode in ("single_shot", "both"):
                pred_orig_single = whisper_transcribe(
                    whisper_model, whisper_processor, w_mono, sr, whisper_device,
                    language="es", task="transcribe", condition_on_prev_tokens=False
                )
                pred_enh_single = whisper_transcribe(
                    whisper_model, whisper_processor, enh, sr, whisper_device,
                    language="es", task="transcribe", condition_on_prev_tokens=False
                )

            if mode in ("longform", "both"):
                pred_orig_long, n_chunks_orig = whisper_transcribe_longform_chunked(
                    whisper_model, whisper_processor, w_mono, sr, whisper_device,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    language="es", task="transcribe",
                )
                pred_enh_long, n_chunks_enh = whisper_transcribe_longform_chunked(
                    whisper_model, whisper_processor, enh, sr, whisper_device,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    language="es", task="transcribe",
                )

            # ---- Normalizar
            ref_norm = normalize_text_for_wer(ref_raw)

            def pack_metrics(hyp_raw: Optional[str]) -> Tuple[str, str, float, float, int, int]:
                if hyp_raw is None:
                    return "", "", float("nan"), float("nan"), 0, 0
                hyp_raw = str(hyp_raw)
                hyp_norm = normalize_text_for_wer(hyp_raw)
                wer = compute_wer(ref_norm, hyp_norm)
                cer = char_error_rate(ref_norm, hyp_norm)
                return hyp_raw, hyp_norm, float(wer), float(cer), len(ref_norm.split()), len(hyp_norm.split())

            org_raw_s, org_norm_s, wer_org_s, cer_org_s, wref, wors = pack_metrics(pred_orig_single)
            enh_raw_s, enh_norm_s, wer_enh_s, cer_enh_s, _, wens = pack_metrics(pred_enh_single)

            org_raw_l, org_norm_l, wer_org_l, cer_org_l, _, _ = pack_metrics(pred_orig_long)
            enh_raw_l, enh_norm_l, wer_enh_l, cer_enh_l, _, _ = pack_metrics(pred_enh_long)

            # ---- Mejora
            imp_single = (wer_org_s - wer_enh_s) if np.isfinite(wer_org_s) and np.isfinite(wer_enh_s) else float("nan")
            imp_long = (wer_org_l - wer_enh_l) if np.isfinite(wer_org_l) and np.isfinite(wer_enh_l) else float("nan")

            audio_id = Path(audio_paths[0]).stem if (audio_paths and audio_paths[0]) else f"{subset}_sample_{i}"

            rows.append({
                "audio_id": audio_id,
                "audio_path": audio_paths[0] if audio_paths else None,

                "reference_raw": ref_raw,
                "reference_norm": ref_norm,

                "transcript_orig_raw_single": org_raw_s,
                "transcript_orig_norm_single": org_norm_s,
                "transcript_enh_raw_single": enh_raw_s,
                "transcript_enh_norm_single": enh_norm_s,
                "wer_orig_single": wer_org_s,
                "wer_enh_single": wer_enh_s,
                "cer_orig_single": cer_org_s,
                "cer_enh_single": cer_enh_s,
                "improvement_single": imp_single,

                "transcript_orig_raw_long": org_raw_l,
                "transcript_orig_norm_long": org_norm_l,
                "transcript_enh_raw_long": enh_raw_l,
                "transcript_enh_norm_long": enh_norm_l,
                "wer_orig_long": wer_org_l,
                "wer_enh_long": wer_enh_l,
                "cer_orig_long": cer_org_l,
                "cer_enh_long": cer_enh_l,
                "improvement_long": imp_long,

                "num_chunks_orig": int(n_chunks_orig),
                "num_chunks_enh": int(n_chunks_enh),
                "chunk_seconds": float(chunk_seconds),
                "overlap_seconds": float(overlap_seconds),
                "audio_duration_seconds": gen_audio_seconds,
                "generator_inference_seconds": gen_infer_seconds,
                "generator_seconds_per_audio_second": gen_seconds_per_audio_second,
                "generator_rtf": gen_rtf,
                "generator_inference_mode": ("direct" if use_direct else "sliding_window"),
                "generator_window_seconds": float(enh_window_seconds),
                "generator_window_overlap": float(enh_overlap),
                "generator_direct_max_seconds": float(enh_direct_max_seconds),

                "sample_index": int(i),
            })

            # postfix
            if len(rows) >= 1:
                df_tmp = pd.DataFrame(rows)
                show_mode = "long" if mode == "longform" else ("both" if mode == "both" else "single")
                if show_mode in ("single", "both") and df_tmp["wer_orig_single"].notna().any():
                    m1 = float(df_tmp["wer_orig_single"].dropna().mean())
                    m2 = float(df_tmp["wer_enh_single"].dropna().mean())
                    pbar.set_postfix({"WER_orig_single": f"{m1:.2f}", "WER_enh_single": f"{m2:.2f}"})
                if show_mode in ("long", "both") and df_tmp["wer_orig_long"].notna().any():
                    m1 = float(df_tmp["wer_orig_long"].dropna().mean())
                    m2 = float(df_tmp["wer_enh_long"].dropna().mean())
                    pbar.set_postfix({"WER_orig_long": f"{m1:.2f}", "WER_enh_long": f"{m2:.2f}"})

            # JSON per-audio
            (samples_dir / f"{audio_id}.json").write_text(
                json.dumps(rows[-1], ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            del w, w_mono, enh
            if generator_device.type == "cuda" and ((i + 1) % 5 == 0):
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = subset_out / "results_per_audio.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{subset}] Saved per-audio CSV -> {csv_path}")

    # ---- Summary
    def summarize(series: pd.Series) -> Dict:
        s = series.dropna()
        if len(s) == 0:
            return {"count": 0}
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "25%": float(s.quantile(0.25)),
            "75%": float(s.quantile(0.75)),
        }

    summary = {
        "wer_orig_single": summarize(df["wer_orig_single"]) if "wer_orig_single" in df else {"count": 0},
        "wer_enh_single": summarize(df["wer_enh_single"]) if "wer_enh_single" in df else {"count": 0},
        "wer_orig_long": summarize(df["wer_orig_long"]) if "wer_orig_long" in df else {"count": 0},
        "wer_enh_long": summarize(df["wer_enh_long"]) if "wer_enh_long" in df else {"count": 0},
        "improvement_single": summarize(df["improvement_single"]) if "improvement_single" in df else {"count": 0},
        "improvement_long": summarize(df["improvement_long"]) if "improvement_long" in df else {"count": 0},
    }

    # counts improved/degraded por modo
    def counts_for(col: str) -> Dict:
        if col not in df:
            return {"total": 0}
        s = df[col].dropna()
        total = int(len(s))
        if total == 0:
            return {"total": 0}
        improved = int((s > 0).sum())
        equal = int((s == 0).sum())
        degraded = int((s < 0).sum())
        return {
            "total": total,
            "improved": improved,
            "equal": equal,
            "degraded": degraded,
            "pct_improved": float(improved / total),
        }

    summary["counts_single"] = counts_for("improvement_single")
    summary["counts_long"] = counts_for("improvement_long")

    (stats_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{subset}] Saved summary -> {stats_dir/'summary.json'}")

    # ---- Plots (solo para el modo disponible)
    sns.set(style="whitegrid")
    try:
        if "improvement_long" in df and df["improvement_long"].notna().any():
            series = df["improvement_long"].dropna()
            plt.figure(figsize=(8, 5))
            sns.histplot(series, bins=40, kde=True)
            plt.title(f"Histogram: WER improvement LONG ({subset})")
            plt.xlabel("WER improvement (orig - enh)")
            plt.tight_layout()
            plt.savefig(plots_dir / "hist_improvement_long.png", dpi=200)
            plt.close()

        if "improvement_single" in df and df["improvement_single"].notna().any():
            series = df["improvement_single"].dropna()
            plt.figure(figsize=(8, 5))
            sns.histplot(series, bins=40, kde=True)
            plt.title(f"Histogram: WER improvement SINGLE ({subset})")
            plt.xlabel("WER improvement (orig - enh)")
            plt.tight_layout()
            plt.savefig(plots_dir / "hist_improvement_single.png", dpi=200)
            plt.close()

        # Scatter: WER original vs WER enhanced (referencia: diagonal y=x)
        if (
            "wer_orig_long" in df and "wer_enh_long" in df
            and df["wer_orig_long"].notna().any() and df["wer_enh_long"].notna().any()
        ):
            dfl = df[["wer_orig_long", "wer_enh_long"]].dropna()
            if len(dfl) > 0:
                plt.figure(figsize=(6, 6))
                sns.scatterplot(data=dfl, x="wer_orig_long", y="wer_enh_long", s=35, alpha=0.8)
                vmax = float(max(dfl["wer_orig_long"].max(), dfl["wer_enh_long"].max()))
                plt.plot([0.0, vmax], [0.0, vmax], linestyle="--", linewidth=1.2, color="black", label="y=x")
                plt.title(f"Scatter WER LONG ({subset})")
                plt.xlabel("WER original")
                plt.ylabel("WER enhanced")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "scatter_wer_long_orig_vs_enh.png", dpi=200)
                plt.close()

        if (
            "wer_orig_single" in df and "wer_enh_single" in df
            and df["wer_orig_single"].notna().any() and df["wer_enh_single"].notna().any()
        ):
            dfs = df[["wer_orig_single", "wer_enh_single"]].dropna()
            if len(dfs) > 0:
                plt.figure(figsize=(6, 6))
                sns.scatterplot(data=dfs, x="wer_orig_single", y="wer_enh_single", s=35, alpha=0.8)
                vmax = float(max(dfs["wer_orig_single"].max(), dfs["wer_enh_single"].max()))
                plt.plot([0.0, vmax], [0.0, vmax], linestyle="--", linewidth=1.2, color="black", label="y=x")
                plt.title(f"Scatter WER SINGLE ({subset})")
                plt.xlabel("WER original")
                plt.ylabel("WER enhanced")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / "scatter_wer_single_orig_vs_enh.png", dpi=200)
                plt.close()
    except Exception as e:
        print(f"[{subset}] Error generando plots: {e}")

    # ---- Save a few audio samples
    # elegimos 3 mejores + 2 peores del modo long si existe; sino single
    key = "improvement_long" if ("improvement_long" in df and df["improvement_long"].notna().any()) else "improvement_single"
    if key in df and df[key].notna().any():
        df_sorted = df.sort_values(key, ascending=False)
        pick = []
        pick += df_sorted.head(3)["sample_index"].tolist()
        pick += df_sorted.tail(2)["sample_index"].tolist()
        pick = list(dict.fromkeys(pick))[:5]
    else:
        pick = list(range(min(5, len(df))))

    print(f"[{subset}] Guardando {len(pick)} samples (wav+txt) en {samples_dir} ...")
    for si in pick:
        item = ds[int(si)]
        wav_path = Path(item["audio_path"])
        audio_id = wav_path.stem

        # cargar audio como en dataset
        try:
            from utils.audio import load_audio
            w0, sr0 = load_audio(str(wav_path), target_sr=sr, normalize=config["audio"].get("normalize_rms", False))
            if w0.ndim == 1:
                w0 = w0.unsqueeze(0)
        except Exception:
            w0, sr0 = torchaudio.load(str(wav_path))
            if sr0 != sr:
                w0 = torchaudio.functional.resample(w0, sr0, sr)

        w0m = ensure_mono_1d(w0)
        w0m = w0m.detach()

        with torch.no_grad():
            if (w0m.numel() / float(sr)) <= float(enh_direct_max_seconds):
                with torch.autocast(
                    device_type=generator_device.type,
                    enabled=(generator_device.type == "cuda"),
                ):
                    enh = generator(w0m.to(generator_device).unsqueeze(0))
                    if isinstance(enh, tuple):
                        enh = enh[0]
                    enh = ensure_mono_1d(enh.squeeze(0)).detach().cpu()
            else:
                enh = sliding_window_inference(
                    waveform=w0m.detach().cpu(),
                    model=generator,
                    sample_rate=sr,
                    window_duration=float(enh_window_seconds),
                    overlap=float(enh_overlap),
                    device=generator_device,
                ).detach().cpu()

        write_wav(str(samples_dir / f"{audio_id}_orig.wav"), w0m, sr)
        write_wav(str(samples_dir / f"{audio_id}_enh.wav"), enh, sr)

        row = df[df["audio_id"] == audio_id]
        if not row.empty:
            r = row.iloc[0]
            txt = (
                f"AUDIO_ID: {audio_id}\nPATH: {r.get('audio_path','')}\n\n"
                f"REFERENCE_RAW:\n{r.get('reference_raw','')}\n\n"
                f"REFERENCE_NORM:\n{r.get('reference_norm','')}\n\n"
                f"--- SINGLE_SHOT ---\n"
                f"ORIG_RAW:\n{r.get('transcript_orig_raw_single','')}\n\n"
                f"ORIG_NORM:\n{r.get('transcript_orig_norm_single','')}\n\n"
                f"ENH_RAW:\n{r.get('transcript_enh_raw_single','')}\n\n"
                f"ENH_NORM:\n{r.get('transcript_enh_norm_single','')}\n\n"
                f"WER_ORIG_SINGLE={r.get('wer_orig_single', None)} | WER_ENH_SINGLE={r.get('wer_enh_single', None)} | IMP_SINGLE={r.get('improvement_single', None)}\n\n"
                f"--- LONGFORM_CHUNKED ---\n"
                f"chunks(orig)={r.get('num_chunks_orig',0)} chunks(enh)={r.get('num_chunks_enh',0)} chunk_s={r.get('chunk_seconds',0)} overlap_s={r.get('overlap_seconds',0)}\n\n"
                f"ORIG_RAW:\n{r.get('transcript_orig_raw_long','')}\n\n"
                f"ORIG_NORM:\n{r.get('transcript_orig_norm_long','')}\n\n"
                f"ENH_RAW:\n{r.get('transcript_enh_raw_long','')}\n\n"
                f"ENH_NORM:\n{r.get('transcript_enh_norm_long','')}\n\n"
                f"WER_ORIG_LONG={r.get('wer_orig_long', None)} | WER_ENH_LONG={r.get('wer_enh_long', None)} | IMP_LONG={r.get('improvement_long', None)}\n"
            )
            (samples_dir / f"{audio_id}_transcripts.txt").write_text(txt, encoding="utf-8")

    # reporte
    if key in df and df[key].notna().any():
        counts = summary["counts_long"] if key == "improvement_long" else summary["counts_single"]
        report_text = (
            f"[{subset}] Evaluation completed.\n"
            f"- Samples evaluated: {len(df)}\n"
            f"- Mode used for headline: {key}\n"
            f"- Improved: {counts.get('improved',0)} ({counts.get('pct_improved',0)*100:.2f}%)\n"
            f"- Equal: {counts.get('equal',0)}\n"
            f"- Degraded: {counts.get('degraded',0)}\n"
            f"Results CSV: {csv_path}\n"
            f"Summary: {stats_dir / 'summary.json'}\n"
            f"Samples: {samples_dir}\n"
        )
    else:
        report_text = f"[{subset}] Evaluation completed. (No finite improvements computed)\n"

    (subset_out / "report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)

    return df, summary

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--generator-device", type=str, default=None,
                        help="Device para generator (default: config.devices.train_device o --device)")
    parser.add_argument("--whisper-device", type=str, default=None,
                        help="Device para Whisper (default: config.devices.validate_device o --device)")
    parser.add_argument("--subset", type=str, choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["single_shot", "longform", "both"], default="both",
                        help="Eval mode: single_shot, longform(chunked), or both")
    parser.add_argument("--chunk-seconds", type=float, default=30.0)
    parser.add_argument("--overlap-seconds", type=float, default=1.0)
    parser.add_argument("--enh-window-seconds", type=float, default=12.0,
                        help="Ventana (s) para enhancement por sliding window en audios largos")
    parser.add_argument("--enh-overlap", type=float, default=0.5,
                        help="Overlap fraccional [0,1) para enhancement por ventanas")
    parser.add_argument("--enh-direct-max-seconds", type=float, default=20.0,
                        help="Si audio <= este umbral, usa forward directo del generator")
    args = parser.parse_args()

    # Usa el loader del pipeline para aplicar defaults (audio/data/models/paths).
    config = load_train_config(args.config)
    if args.device is not None:
        generator_device = torch.device(args.device)
        whisper_device = torch.device(args.device)
    else:
        cfg_dev = config.get("devices", {})
        generator_device = torch.device(args.generator_device or cfg_dev.get("train_device", "cuda:0" if torch.cuda.is_available() else "cpu"))
        whisper_device = torch.device(args.whisper_device or cfg_dev.get("validate_device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    print(
        {
            "generator_device": str(generator_device),
            "whisper_device": str(whisper_device),
        }
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    whisper_name = (
        config.get("evaluation", {}).get("whisper_model_name")
        or config.get("models", {}).get("whisper", {}).get("model_name")
        or "UDA-LIDI/openai-whisper-large-es_ecu911DM"
    )
    whisper_model, whisper_processor = try_load_whisper(
        whisper_name, whisper_device, hf_token=args.hf_token, retries=3, sleep_sec=5
    )

    subsets = ["train", "val", "test"] if args.subset == "all" else [args.subset]

    summary_all = {}
    for s in subsets:
        _, summary = evaluate_subset(
            checkpoint_path=args.checkpoint,
            config=config,
            subset=s,
            max_samples=args.max_samples,
            outdir=outdir,
            generator_device=generator_device,
            whisper_device=whisper_device,
            whisper_model=whisper_model,
            whisper_processor=whisper_processor,
            chunk_seconds=float(args.chunk_seconds),
            overlap_seconds=float(args.overlap_seconds),
            enh_window_seconds=float(args.enh_window_seconds),
            enh_overlap=float(args.enh_overlap),
            enh_direct_max_seconds=float(args.enh_direct_max_seconds),
            mode=args.mode,
        )
        summary_all[s] = summary

    (outdir / "summary_by_subset.json").write_text(
        json.dumps(summary_all, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved aggregated summary -> {outdir/'summary_by_subset.json'}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# Permite ejecutar el script directamente: python scripts/analyze_werd.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ecu911_dataset import create_ecu911_dataloader
from engine.config import load_config
from engine.stats import safe_corr
from models.wer_discriminator import create_wer_discriminator


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _summarize(arr: np.ndarray) -> Dict:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
    }


@torch.no_grad()
def run_analysis(
    config: Dict,
    checkpoint_path: str,
    subset: str,
    device: torch.device,
    out_dir: Path,
    max_samples: int | None,
    batch_size_override: int | None,
    output_head: str,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    model = create_wer_discriminator(config, device=device).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("wer_discriminator", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    if batch_size_override is not None:
        loader = create_ecu911_dataloader(
            config,
            stage=subset,
            batch_size=int(batch_size_override),
            purpose="wer_disc",
        )
    else:
        loader = create_ecu911_dataloader(config, stage=subset, purpose="wer_disc")

    rows: List[Dict] = []
    seen = 0
    pbar = tqdm(loader, desc=f"[analyze_werd] {subset}", leave=False)
    for batch in pbar:
        wave = batch["waveform"].to(device)
        dur = batch.get("durations")
        if dur is not None:
            dur = dur.to(device)

        logits = model(
            wave,
            durations=dur,
            audio_paths=batch.get("audio_paths", None),
            output=output_head,
        ).detach().float().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        neg_logits = -logits

        wer_v = None
        if "wer" in batch:
            wer_v = batch["wer"].detach().float().cpu().numpy()
        q_v = None
        if "quality_scores" in batch:
            q_v = batch["quality_scores"].detach().float().cpu().numpy()
        mask = batch.get("quality_mask")
        if mask is not None:
            mask = mask.detach().cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(logits, dtype=bool)

        durations = batch.get("durations")
        if durations is not None:
            durations = durations.detach().float().cpu().numpy()
        else:
            durations = np.full_like(logits, np.nan, dtype=np.float32)

        audio_paths = batch.get("audio_paths", [None] * len(logits))

        for i in range(len(logits)):
            if max_samples is not None and seen >= max_samples:
                break
            if not bool(mask[i]):
                continue

            w = _to_float(wer_v[i]) if wer_v is not None else float("nan")
            q = _to_float(q_v[i]) if q_v is not None else float("nan")
            rows.append(
                {
                    "audio_path": audio_paths[i] if i < len(audio_paths) else None,
                    "duration_s": _to_float(durations[i]),
                    "wer": w,
                    "log1p_wer": np.log1p(max(0.0, w)) if np.isfinite(w) else float("nan"),
                    "quality": q,
                    "logit": _to_float(logits[i]),
                    "prob": _to_float(probs[i]),
                    "neg_logit": _to_float(neg_logits[i]),
                }
            )
            seen += 1

        if max_samples is not None and seen >= max_samples:
            break

    if not rows:
        raise RuntimeError("No se recolectaron filas para análisis D_WER.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rows.csv", index=False)

    corr = {
        "prob_vs_quality": safe_corr(df["prob"].values, df["quality"].values),
        "neglogit_vs_log1pwer": safe_corr(df["neg_logit"].values, df["log1p_wer"].values),
        "prob_vs_log1pwer": safe_corr(df["prob"].values, df["log1p_wer"].values),
        "logit_vs_quality": safe_corr(df["logit"].values, df["quality"].values),
    }

    bins = [0, 5, 10, 20, 30, 60, 120, 1e9]
    labels = ["0-5", "5-10", "10-20", "20-30", "30-60", "60-120", "120+"]
    dfd = df.copy()
    dfd["dur_bin"] = pd.cut(dfd["duration_s"], bins=bins, labels=labels, include_lowest=True)
    by_dur = []
    for b in labels:
        sub = dfd[dfd["dur_bin"] == b]
        if len(sub) < 3:
            continue
        c1 = safe_corr(sub["prob"].values, sub["quality"].values)
        c2 = safe_corr(sub["neg_logit"].values, sub["log1p_wer"].values)
        by_dur.append(
            {
                "dur_bin": b,
                "n": int(len(sub)),
                "spearman_prob_quality": float(c1["spearman"]),
                "spearman_neglogit_logwer": float(c2["spearman"]),
            }
        )

    dfd["wer_decile"] = pd.qcut(dfd["wer"].rank(method="first"), 10, labels=False, duplicates="drop")
    by_decile = (
        dfd.groupby("wer_decile", dropna=True)[["wer", "prob", "neg_logit", "quality"]]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "subset": subset,
        "output_head": output_head,
        "n": int(len(df)),
        "stats": {
            "wer": _summarize(df["wer"].values),
            "quality": _summarize(df["quality"].values),
            "logit": _summarize(df["logit"].values),
            "prob": _summarize(df["prob"].values),
        },
        "correlations": corr,
        "by_duration_bin": by_dur,
        "by_wer_decile_means": by_decile,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="quality", y="prob", alpha=0.7, s=20)
    plt.title("D_WER: prob vs quality")
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_prob_vs_quality.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="log1p_wer", y="neg_logit", alpha=0.7, s=20)
    plt.title("D_WER: -logit vs log1p(WER)")
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_neglogit_vs_logwer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(df["logit"].dropna(), bins=50, kde=True)
    plt.title("Logit distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_logit.png", dpi=180)
    plt.close()

    if by_dur:
        bdf = pd.DataFrame(by_dur)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=bdf, x="dur_bin", y="spearman_prob_quality", color="#2f6db2")
        plt.title("Spearman(prob, quality) by duration bin")
        plt.tight_layout()
        plt.savefig(plot_dir / "bar_spearman_by_duration.png", dpi=180)
        plt.close()

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subset", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="analysis/werd")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output-head", type=str, default="abs", choices=["abs", "rel"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg.get("devices", {}).get("whisper_device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    outdir = Path(args.outdir) / args.subset
    summary = run_analysis(
        config=cfg,
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        device=device,
        out_dir=outdir,
        max_samples=args.max_samples,
        batch_size_override=args.batch_size,
        output_head=str(args.output_head),
    )
    print(json.dumps(summary["correlations"], indent=2, ensure_ascii=False))
    print(f"[analyze_werd] saved -> {outdir}")


if __name__ == "__main__":
    main()

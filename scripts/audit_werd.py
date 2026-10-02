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

# Permite ejecutar: python scripts/audit_werd.py
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


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Dict:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    n = int(xv.size)
    if n < 5 or n_boot <= 0:
        return {"n": n, "pearson_ci95": [float("nan"), float("nan")], "spearman_ci95": [float("nan"), float("nan")]}

    rng = np.random.default_rng(seed)
    pvals = np.zeros(n_boot, dtype=np.float64)
    svals = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = safe_corr(xv[idx], yv[idx])
        pvals[i] = float(c["pearson"])
        svals[i] = float(c["spearman"])

    pvals = pvals[np.isfinite(pvals)]
    svals = svals[np.isfinite(svals)]
    return {
        "n": n,
        "pearson_ci95": [float(np.percentile(pvals, 2.5)), float(np.percentile(pvals, 97.5))] if pvals.size else [float("nan"), float("nan")],
        "spearman_ci95": [float(np.percentile(svals, 2.5)), float(np.percentile(svals, 97.5))] if svals.size else [float("nan"), float("nan")],
    }


@torch.no_grad()
def _collect_rows(
    config: Dict,
    subset: str,
    purpose: str,
    output_head: str,
    device: torch.device,
    checkpoint_path: str | None,
    batch_size_override: int | None,
) -> pd.DataFrame:
    model = None
    if checkpoint_path:
        model = create_wer_discriminator(config, device=device).to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("wer_discriminator", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

    if batch_size_override is None:
        loader = create_ecu911_dataloader(config, stage=subset, purpose=purpose)
    else:
        loader = create_ecu911_dataloader(
            config,
            stage=subset,
            batch_size=int(batch_size_override),
            purpose=purpose,
        )

    rows: List[Dict] = []
    for batch in tqdm(loader, desc=f"[audit_werd] collect {subset}", leave=False):
        wave = batch["waveform"].to(device)
        dur = batch.get("durations")
        if dur is not None:
            dur = dur.to(device)

        logits = None
        probs = None
        neg_logits = None
        if model is not None:
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
            mask = np.ones(wave.shape[0], dtype=bool)

        durations = batch.get("durations")
        if durations is not None:
            durations = durations.detach().float().cpu().numpy()
        else:
            durations = np.full((wave.shape[0],), np.nan, dtype=np.float32)
        audio_paths = batch.get("audio_paths", [None] * wave.shape[0])

        for i in range(wave.shape[0]):
            if not bool(mask[i]):
                continue
            w = _to_float(wer_v[i]) if wer_v is not None else float("nan")
            q = _to_float(q_v[i]) if q_v is not None else float("nan")
            row = {
                "audio_path": audio_paths[i] if i < len(audio_paths) else None,
                "duration_s": _to_float(durations[i]),
                "wer": w,
                "log1p_wer": np.log1p(max(0.0, w)) if np.isfinite(w) else float("nan"),
                "quality": q,
            }
            if logits is not None:
                row["logit"] = _to_float(logits[i])
                row["prob"] = _to_float(probs[i])
                row["neg_logit"] = _to_float(neg_logits[i])
            rows.append(row)

    if not rows:
        raise RuntimeError("No se recolectaron filas para auditoría D_WER.")
    return pd.DataFrame(rows)


def _rank(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).argsort().argsort().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--subset", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--purpose", type=str, default="default", choices=["default", "wer_disc"])
    ap.add_argument("--output-head", type=str, default="abs", choices=["abs", "rel"])
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="analysis/werd_audit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        dflt = cfg.get("devices", {}).get("whisper_device", "cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(dflt)

    out_dir = Path(args.outdir) / args.purpose / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_rows(
        config=cfg,
        subset=args.subset,
        purpose=args.purpose,
        output_head=str(args.output_head),
        device=device,
        checkpoint_path=args.checkpoint,
        batch_size_override=args.batch_size,
    )
    df.to_csv(out_dir / "rows.csv", index=False)

    c_q_lw = safe_corr(df["quality"].values, df["log1p_wer"].values)
    c_q_nlw = safe_corr(df["quality"].values, -df["log1p_wer"].values)
    b_q_lw = _bootstrap_corr(df["quality"].values, df["log1p_wer"].values, n_boot=args.bootstrap, seed=args.bootstrap_seed)
    b_q_nlw = _bootstrap_corr(df["quality"].values, -df["log1p_wer"].values, n_boot=args.bootstrap, seed=args.bootstrap_seed)

    summary = {
        "subset": args.subset,
        "purpose": args.purpose,
        "n": int(len(df)),
        "label_consistency": {
            "quality_vs_log1pwer": c_q_lw,
            "quality_vs_neglog1pwer": c_q_nlw,
            "bootstrap_quality_vs_log1pwer": b_q_lw,
            "bootstrap_quality_vs_neglog1pwer": b_q_nlw,
        },
    }

    mask = np.isfinite(df["quality"].values) & np.isfinite(df["log1p_wer"].values)
    lab = df.loc[mask].copy()
    if len(lab) > 0:
        rq = _rank(lab["quality"].values)
        rlw = _rank((-lab["log1p_wer"]).values)
        lab["rank_gap_quality_vs_wer"] = np.abs(rq - rlw)
        lab = lab.sort_values("rank_gap_quality_vs_wer", ascending=False)
        lab.head(max(1, int(args.top_k))).to_csv(out_dir / "label_disagreements_topk.csv", index=False)

    if args.checkpoint:
        c_prob_q = safe_corr(df["prob"].values, df["quality"].values)
        c_nlog_lw = safe_corr(df["neg_logit"].values, df["log1p_wer"].values)
        b_prob_q = _bootstrap_corr(df["prob"].values, df["quality"].values, n_boot=args.bootstrap, seed=args.bootstrap_seed + 1)
        b_nlog_lw = _bootstrap_corr(df["neg_logit"].values, df["log1p_wer"].values, n_boot=args.bootstrap, seed=args.bootstrap_seed + 2)
        summary["model"] = {
            "checkpoint": args.checkpoint,
            "output_head": str(args.output_head),
            "prob_vs_quality": c_prob_q,
            "neglogit_vs_log1pwer": c_nlog_lw,
            "bootstrap_prob_vs_quality": b_prob_q,
            "bootstrap_neglogit_vs_log1pwer": b_nlog_lw,
            "gap_to_label_spearman_quality": float(c_q_nlw["spearman"] - c_prob_q["spearman"]),
        }

        m = np.isfinite(df["quality"].values) & np.isfinite(df["prob"].values)
        cmp_df = df.loc[m].copy()
        if len(cmp_df) > 0:
            rq = _rank(cmp_df["quality"].values)
            rp = _rank(cmp_df["prob"].values)
            cmp_df["rank_gap_model_vs_quality"] = np.abs(rq - rp)
            cmp_df = cmp_df.sort_values("rank_gap_model_vs_quality", ascending=False)
            cmp_df.head(max(1, int(args.top_k))).to_csv(out_dir / "model_disagreements_topk.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="log1p_wer", y="quality", alpha=0.7, s=20)
    plt.title("Label consistency: quality vs log1p(WER)")
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_quality_vs_logwer.png", dpi=180)
    plt.close()

    if args.checkpoint:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x="quality", y="prob", alpha=0.7, s=20)
        plt.title("Model: prob vs quality")
        plt.tight_layout()
        plt.savefig(plot_dir / "scatter_model_prob_vs_quality.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x="log1p_wer", y="neg_logit", alpha=0.7, s=20)
        plt.title("Model: -logit vs log1p(WER)")
        plt.tight_layout()
        plt.savefig(plot_dir / "scatter_model_neglogit_vs_logwer.png", dpi=180)
        plt.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[audit_werd] saved -> {out_dir}")


if __name__ == "__main__":
    main()

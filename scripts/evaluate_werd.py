#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import seaborn as sns
except Exception:
    sns = None

# Permite ejecutar: python scripts/evaluate_werd.py
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


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while (j + 1) < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def _pair_rank_accuracy(pred: np.ndarray, target: np.ndarray, min_delta: float = 1e-3) -> Dict[str, float]:
    pv = np.asarray(pred, dtype=np.float64)
    tv = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pv) & np.isfinite(tv)
    pv = pv[mask]
    tv = tv[mask]
    if pv.size < 2:
        return {"acc": float("nan"), "npairs": 0.0}

    tdiff = tv[:, None] - tv[None, :]
    valid = tdiff > float(min_delta)
    npairs = int(np.sum(valid))
    if npairs < 1:
        return {"acc": float("nan"), "npairs": 0.0}

    pdiff = pv[:, None] - pv[None, :]
    sel = pdiff[valid]
    acc = float(np.mean((sel > 0.0).astype(np.float64) + 0.5 * (sel == 0.0).astype(np.float64)))
    return {"acc": acc, "npairs": float(npairs)}


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Dict:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    n = int(xv.size)
    if n < 5 or n_boot <= 0:
        return {
            "n": n,
            "pearson_ci95": [float("nan"), float("nan")],
            "spearman_ci95": [float("nan"), float("nan")],
        }

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
        "p95": float(np.percentile(x, 95)),
    }


def _compute_prediction_errors(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mask = np.isfinite(df["prob"].values) & np.isfinite(df["quality"].values)
    if np.any(mask):
        err = df["prob"].values[mask] - df["quality"].values[mask]
        out["mae_prob_quality"] = float(np.mean(np.abs(err)))
        out["rmse_prob_quality"] = float(np.sqrt(np.mean(np.square(err))))
        out["bias_prob_quality"] = float(np.mean(err))
    else:
        out["mae_prob_quality"] = float("nan")
        out["rmse_prob_quality"] = float("nan")
        out["bias_prob_quality"] = float("nan")
    return out


def _kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return float("nan")
    try:
        return float(df["x"].corr(df["y"], method="kendall"))
    except Exception:
        return float("nan")


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size < 2:
        return float("nan")
    denom = float(np.sum(np.square(yt - np.mean(yt))))
    if denom <= 0.0:
        return float("nan")
    num = float(np.sum(np.square(yt - yp)))
    return float(1.0 - (num / denom))


def _precision_recall_at_k(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    higher_is_better: bool,
) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(t)
    s = s[mask]
    t = t[mask]
    n = int(s.size)
    if n < 2:
        return {"k": 0.0, "precision": float("nan"), "recall": float("nan"), "overlap": 0.0}
    k = max(1, min(int(k), n))
    order_s = np.argsort(-s if higher_is_better else s, kind="mergesort")
    order_t = np.argsort(-t if higher_is_better else t, kind="mergesort")
    pred_top = set(order_s[:k].tolist())
    true_top = set(order_t[:k].tolist())
    inter = len(pred_top & true_top)
    return {
        "k": float(k),
        "precision": float(inter / float(k)),
        "recall": float(inter / float(k)),
        "overlap": float(inter),
    }


def _ndcg_at_k(
    scores: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    higher_is_better: bool,
) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(t)
    s = s[mask]
    t = t[mask]
    n = int(s.size)
    if n < 2:
        return {"k": 0.0, "ndcg": float("nan")}
    k = max(1, min(int(k), n))
    if not higher_is_better:
        s = -s
        t = -t
    t = t - np.min(t)
    gains = np.maximum(0.0, t)
    order_pred = np.argsort(-s, kind="mergesort")[:k]
    order_true = np.argsort(-gains, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gains[order_pred] * discounts))
    idcg = float(np.sum(gains[order_true] * discounts))
    return {"k": float(k), "ndcg": float(dcg / idcg) if idcg > 0.0 else float("nan")}


def _calibration_bins(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(t)
    p = p[mask]
    t = t[mask]
    if p.size == 0:
        return pd.DataFrame(), {"ece": float("nan"), "mce": float("nan"), "n": 0}
    p = np.clip(p, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: List[Dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    n = int(p.size)
    for i in range(int(n_bins)):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == int(n_bins) - 1:
            sel = (p >= lo) & (p <= hi)
        else:
            sel = (p >= lo) & (p < hi)
        count = int(np.sum(sel))
        if count == 0:
            continue
        conf = float(np.mean(p[sel]))
        acc = float(np.mean(t[sel]))
        gap = abs(conf - acc)
        ece += (count / float(n)) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin_idx": i,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_pred": conf,
                "mean_target": acc,
                "abs_gap": gap,
            }
        )
    return pd.DataFrame(rows), {"ece": float(ece), "mce": float(mce), "n": n}


def _compute_summary(
    df: pd.DataFrame,
    *,
    subset: str,
    purpose: str,
    output_head: str,
    checkpoint: str,
    bootstrap: int,
    bootstrap_seed: int,
) -> Dict:
    corr_prob_quality = safe_corr(df["prob"].values, df["quality"].values)
    corr_neglogit_logwer = safe_corr(df["neg_logit"].values, df["log1p_wer"].values)
    corr_prob_neglogwer = safe_corr(df["prob"].values, -df["log1p_wer"].values)
    kendall_prob_quality = _kendall_corr(df["prob"].values, df["quality"].values)
    kendall_neglogit_logwer = _kendall_corr(df["neg_logit"].values, df["log1p_wer"].values)
    boot_prob_quality = _bootstrap_corr(
        df["prob"].values, df["quality"].values, n_boot=int(bootstrap), seed=int(bootstrap_seed)
    )
    boot_neglogit_logwer = _bootstrap_corr(
        df["neg_logit"].values, df["log1p_wer"].values, n_boot=int(bootstrap), seed=int(bootstrap_seed + 1)
    )
    rank_prob_quality = _pair_rank_accuracy(df["prob"].values, df["quality"].values)
    rank_neglogit_logwer = _pair_rank_accuracy(df["neg_logit"].values, df["log1p_wer"].values)
    cal_bins, cal_summary = _calibration_bins(df["prob"].values, df["quality"].values, n_bins=10)
    n = max(1, int(len(df)))
    top10 = max(1, int(round(0.10 * n)))
    top20 = max(1, int(round(0.20 * n)))

    return {
        "subset": subset,
        "purpose": purpose,
        "output_head": str(output_head),
        "checkpoint": checkpoint,
        "n": int(len(df)),
        "stats": {
            "wer": _summarize(df["wer"].values),
            "quality": _summarize(df["quality"].values),
            "logit": _summarize(df["logit"].values),
            "prob": _summarize(df["prob"].values),
            "duration_s": _summarize(df["duration_s"].values),
            "per_audio_inference_ms": _summarize(df["per_audio_inference_ms"].values),
            "per_audio_seconds_per_second": _summarize(df["per_audio_seconds_per_second"].values),
        },
        "prediction": _compute_prediction_errors(df),
        "prediction_extra": {
            "r2_prob_quality": _r2_score(df["quality"].values, df["prob"].values),
            "r2_neglogit_log1pwer": _r2_score(df["log1p_wer"].values, df["neg_logit"].values),
        },
        "calibration": {
            **cal_summary,
            "bins": cal_bins.to_dict(orient="records"),
        },
        "correlations": {
            "prob_vs_quality": corr_prob_quality,
            "neglogit_vs_log1pwer": corr_neglogit_logwer,
            "prob_vs_neglog1pwer": corr_prob_neglogwer,
            "kendall_prob_vs_quality": kendall_prob_quality,
            "kendall_neglogit_vs_log1pwer": kendall_neglogit_logwer,
            "bootstrap_prob_vs_quality": boot_prob_quality,
            "bootstrap_neglogit_vs_log1pwer": boot_neglogit_logwer,
            "rank_prob_vs_quality": rank_prob_quality,
            "rank_neglogit_vs_log1pwer": rank_neglogit_logwer,
        },
        "retrieval": {
            "quality_top10": _precision_recall_at_k(df["prob"].values, df["quality"].values, k=top10, higher_is_better=True),
            "quality_top20": _precision_recall_at_k(df["prob"].values, df["quality"].values, k=top20, higher_is_better=True),
            "wer_top10_hardest": _precision_recall_at_k(
                df["neg_logit"].values, df["log1p_wer"].values, k=top10, higher_is_better=True
            ),
            "wer_top20_hardest": _precision_recall_at_k(
                df["neg_logit"].values, df["log1p_wer"].values, k=top20, higher_is_better=True
            ),
            "ndcg_quality_top10": _ndcg_at_k(df["prob"].values, df["quality"].values, k=top10, higher_is_better=True),
            "ndcg_quality_top20": _ndcg_at_k(df["prob"].values, df["quality"].values, k=top20, higher_is_better=True),
            "ndcg_wer_top10_hardest": _ndcg_at_k(
                df["neg_logit"].values, df["log1p_wer"].values, k=top10, higher_is_better=True
            ),
            "ndcg_wer_top20_hardest": _ndcg_at_k(
                df["neg_logit"].values, df["log1p_wer"].values, k=top20, higher_is_better=True
            ),
        },
    }


def _extract_metric_table(summary: Dict, label: str) -> List[Dict[str, object]]:
    return [
        {"metric": "prob_quality_pearson", label: _to_float(summary["correlations"]["prob_vs_quality"].get("pearson"))},
        {"metric": "prob_quality_spearman", label: _to_float(summary["correlations"]["prob_vs_quality"].get("spearman"))},
        {"metric": "prob_quality_kendall", label: _to_float(summary["correlations"].get("kendall_prob_vs_quality"))},
        {"metric": "neglogit_logwer_pearson", label: _to_float(summary["correlations"]["neglogit_vs_log1pwer"].get("pearson"))},
        {"metric": "neglogit_logwer_spearman", label: _to_float(summary["correlations"]["neglogit_vs_log1pwer"].get("spearman"))},
        {"metric": "neglogit_logwer_kendall", label: _to_float(summary["correlations"].get("kendall_neglogit_vs_log1pwer"))},
        {"metric": "rank_prob_quality_acc", label: _to_float(summary["correlations"]["rank_prob_vs_quality"].get("acc"))},
        {"metric": "rank_neglogit_logwer_acc", label: _to_float(summary["correlations"]["rank_neglogit_vs_log1pwer"].get("acc"))},
        {"metric": "mae_prob_quality", label: _to_float(summary["prediction"].get("mae_prob_quality"))},
        {"metric": "rmse_prob_quality", label: _to_float(summary["prediction"].get("rmse_prob_quality"))},
        {"metric": "bias_prob_quality", label: _to_float(summary["prediction"].get("bias_prob_quality"))},
        {"metric": "r2_prob_quality", label: _to_float(summary["prediction_extra"].get("r2_prob_quality"))},
        {"metric": "calibration_ece", label: _to_float(summary["calibration"].get("ece"))},
        {"metric": "calibration_mce", label: _to_float(summary["calibration"].get("mce"))},
        {"metric": "quality_top10_precision", label: _to_float(summary["retrieval"]["quality_top10"].get("precision"))},
        {"metric": "quality_top20_precision", label: _to_float(summary["retrieval"]["quality_top20"].get("precision"))},
        {"metric": "wer_top10_precision", label: _to_float(summary["retrieval"]["wer_top10_hardest"].get("precision"))},
        {"metric": "wer_top20_precision", label: _to_float(summary["retrieval"]["wer_top20_hardest"].get("precision"))},
        {"metric": "ndcg_quality_top10", label: _to_float(summary["retrieval"]["ndcg_quality_top10"].get("ndcg"))},
        {"metric": "ndcg_quality_top20", label: _to_float(summary["retrieval"]["ndcg_quality_top20"].get("ndcg"))},
        {"metric": "ndcg_wer_top10", label: _to_float(summary["retrieval"]["ndcg_wer_top10_hardest"].get("ndcg"))},
        {"metric": "ndcg_wer_top20", label: _to_float(summary["retrieval"]["ndcg_wer_top20_hardest"].get("ndcg"))},
        {"metric": "mean_inference_ms", label: _to_float(summary["stats"]["per_audio_inference_ms"].get("mean"))},
        {"metric": "mean_seconds_per_second", label: _to_float(summary["stats"]["per_audio_seconds_per_second"].get("mean"))},
        {"metric": "n", label: int(summary.get("n", 0))},
    ]

@torch.no_grad()
def _collect_rows(
    config: Dict,
    checkpoint_path: str,
    subset: str,
    purpose: str,
    output_head: str,
    device: torch.device,
    batch_size_override: int | None,
    max_samples: int | None,
) -> pd.DataFrame:
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
    seen = 0
    pbar = tqdm(loader, desc=f"[evaluate_werd] {subset}", leave=False)
    for batch in pbar:
        wave = batch["waveform"].to(device)
        dur = batch.get("durations")
        if dur is not None:
            dur = dur.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = model(
            wave,
            durations=dur,
            audio_paths=batch.get("audio_paths", None),
            output=output_head,
        ).detach().float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_seconds = float(time.perf_counter() - t0)

        probs = 1.0 / (1.0 + np.exp(-logits))
        neg_logits = -logits

        wer_v = batch.get("wer")
        if wer_v is not None:
            wer_v = wer_v.detach().float().cpu().numpy()
        q_v = batch.get("quality_scores")
        if q_v is not None:
            q_v = q_v.detach().float().cpu().numpy()
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
        batch_infer_ms = 1000.0 * infer_seconds
        per_audio_ms = batch_infer_ms / max(1, len(logits))

        for i in range(len(logits)):
            if max_samples is not None and seen >= max_samples:
                break
            if not bool(mask[i]):
                continue
            w = _to_float(wer_v[i]) if wer_v is not None else float("nan")
            q = _to_float(q_v[i]) if q_v is not None else float("nan")
            dur_s = _to_float(durations[i])
            rows.append(
                {
                    "audio_path": audio_paths[i] if i < len(audio_paths) else None,
                    "duration_s": dur_s,
                    "wer": w,
                    "log1p_wer": np.log1p(max(0.0, w)) if np.isfinite(w) else float("nan"),
                    "quality": q,
                    "logit": _to_float(logits[i]),
                    "prob": _to_float(probs[i]),
                    "neg_logit": _to_float(neg_logits[i]),
                    "batch_size_eval": int(len(logits)),
                    "batch_inference_ms": float(batch_infer_ms),
                    "per_audio_inference_ms": float(per_audio_ms),
                    "per_audio_seconds_per_second": (
                        float(per_audio_ms) / 1000.0 / dur_s if np.isfinite(dur_s) and dur_s > 0.0 else float("nan")
                    ),
                }
            )
            seen += 1

        if max_samples is not None and seen >= max_samples:
            break

    if not rows:
        raise RuntimeError("No se recolectaron filas para evaluación D_WER.")
    return pd.DataFrame(rows)


def _save_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    if sns is not None:
        sns.set(style="whitegrid")
    else:
        plt.style.use("ggplot")

    plt.figure(figsize=(6, 6))
    if sns is not None:
        sns.scatterplot(data=df, x="quality", y="prob", alpha=0.7, s=20)
    else:
        plt.scatter(df["quality"], df["prob"], alpha=0.7, s=20)
    plt.title("D_WER: prob vs quality")
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_prob_vs_quality.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    if sns is not None:
        sns.scatterplot(data=df, x="log1p_wer", y="neg_logit", alpha=0.7, s=20)
    else:
        plt.scatter(df["log1p_wer"], df["neg_logit"], alpha=0.7, s=20)
    plt.title("D_WER: -logit vs log1p(WER)")
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_neglogit_vs_logwer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.histplot(df["logit"].dropna(), bins=50, kde=True)
    else:
        plt.hist(df["logit"].dropna(), bins=50)
    plt.title("D_WER logit distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_logit.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.histplot(df["prob"].dropna(), bins=50, kde=True)
    else:
        plt.hist(df["prob"].dropna(), bins=50)
    plt.title("D_WER prob distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "hist_prob.png", dpi=180)
    plt.close()

    if "per_audio_inference_ms" in df and df["per_audio_inference_ms"].notna().any():
        plt.figure(figsize=(6, 4))
        if sns is not None:
            sns.histplot(df["per_audio_inference_ms"].dropna(), bins=40, kde=True)
        else:
            plt.hist(df["per_audio_inference_ms"].dropna(), bins=40)
        plt.title("D_WER inference per audio (ms)")
        plt.tight_layout()
        plt.savefig(plot_dir / "hist_inference_ms.png", dpi=180)
        plt.close()

    cal_df, _ = _calibration_bins(df["prob"].values, df["quality"].values, n_bins=10)
    if not cal_df.empty:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.2)
        if sns is not None:
            sns.lineplot(data=cal_df, x="mean_pred", y="mean_target", marker="o")
        else:
            plt.plot(cal_df["mean_pred"], cal_df["mean_target"], marker="o")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Mean predicted quality")
        plt.ylabel("Mean target quality")
        plt.title("D_WER calibration: prob vs quality")
        plt.tight_layout()
        plt.savefig(plot_dir / "calibration_prob_vs_quality.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 4))
        if sns is not None:
            sns.barplot(data=cal_df, x="bin_idx", y="abs_gap", color="#4C78A8")
        else:
            plt.bar(cal_df["bin_idx"].astype(str), cal_df["abs_gap"], color="#4C78A8")
        plt.xlabel("Calibration bin")
        plt.ylabel("Absolute gap")
        plt.title("D_WER calibration gap by bin")
        plt.tight_layout()
        plt.savefig(plot_dir / "calibration_gap_by_bin.png", dpi=180)
        plt.close()


def _save_comparison_outputs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    summary_a: Dict,
    summary_b: Dict,
    out_dir: Path,
    label_a: str,
    label_b: str,
) -> None:
    key = "audio_path"
    if key not in df_a.columns or key not in df_b.columns:
        return

    cols = [
        "audio_path",
        "duration_s",
        "wer",
        "quality",
        "prob",
        "neg_logit",
        "per_audio_inference_ms",
        "per_audio_seconds_per_second",
    ]
    left = df_a[[c for c in cols if c in df_a.columns]].copy()
    right = df_b[[c for c in cols if c in df_b.columns]].copy()
    left = left.rename(columns={c: f"{c}_{label_a}" for c in left.columns if c != key})
    right = right.rename(columns={c: f"{c}_{label_b}" for c in right.columns if c != key})
    merged = left.merge(right, on=key, how="inner")

    if f"prob_{label_a}" in merged.columns and f"quality_{label_a}" in merged.columns:
        merged[f"abs_err_prob_quality_{label_a}"] = np.abs(merged[f"prob_{label_a}"] - merged[f"quality_{label_a}"])
    if f"prob_{label_b}" in merged.columns and f"quality_{label_b}" in merged.columns:
        merged[f"abs_err_prob_quality_{label_b}"] = np.abs(merged[f"prob_{label_b}"] - merged[f"quality_{label_b}"])
    if (
        f"abs_err_prob_quality_{label_a}" in merged.columns
        and f"abs_err_prob_quality_{label_b}" in merged.columns
    ):
        merged["delta_abs_err_prob_quality_b_minus_a"] = (
            merged[f"abs_err_prob_quality_{label_b}"] - merged[f"abs_err_prob_quality_{label_a}"]
        )
    merged.to_csv(out_dir / "comparison_rows.csv", index=False)

    tbl = pd.DataFrame(_extract_metric_table(summary_a, label_a)).merge(
        pd.DataFrame(_extract_metric_table(summary_b, label_b)),
        on="metric",
        how="outer",
    )
    tbl["delta_b_minus_a"] = tbl[label_b] - tbl[label_a]

    higher_better = {
        "prob_quality_pearson",
        "prob_quality_spearman",
        "neglogit_logwer_pearson",
        "neglogit_logwer_spearman",
        "rank_prob_quality_acc",
        "rank_neglogit_logwer_acc",
    }
    lower_better = {
        "mae_prob_quality",
        "rmse_prob_quality",
        "mean_inference_ms",
        "mean_seconds_per_second",
    }
    better = []
    for _, r in tbl.iterrows():
        metric = str(r["metric"])
        va = _to_float(r.get(label_a))
        vb = _to_float(r.get(label_b))
        if not (np.isfinite(va) and np.isfinite(vb)):
            better.append("nan")
        elif metric in higher_better:
            better.append(label_b if vb > va else (label_a if va > vb else "tie"))
        elif metric in lower_better or metric == "bias_prob_quality":
            if metric == "bias_prob_quality":
                va = abs(va)
                vb = abs(vb)
            better.append(label_b if vb < va else (label_a if va < vb else "tie"))
        else:
            better.append("n/a")
    tbl["better"] = better
    tbl.to_csv(out_dir / "comparison_metrics.csv", index=False)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if (
        f"prob_{label_a}" in merged.columns
        and f"prob_{label_b}" in merged.columns
        and f"quality_{label_a}" in merged.columns
    ):
        plot_df = pd.concat(
            [
                pd.DataFrame({"quality": merged[f"quality_{label_a}"], "prob": merged[f"prob_{label_a}"], "model": label_a}),
                pd.DataFrame({"quality": merged[f"quality_{label_a}"], "prob": merged[f"prob_{label_b}"], "model": label_b}),
            ],
            ignore_index=True,
        )
        plt.figure(figsize=(6.5, 6.0))
        if sns is not None:
            sns.scatterplot(data=plot_df, x="quality", y="prob", hue="model", alpha=0.6, s=18)
        else:
            for model_name, grp in plot_df.groupby("model"):
                plt.scatter(grp["quality"], grp["prob"], alpha=0.6, s=18, label=model_name)
            plt.legend()
        plt.title("D_WER comparison: prob vs quality")
        plt.tight_layout()
        plt.savefig(plot_dir / "compare_prob_vs_quality.png", dpi=180)
        plt.close()

    if (
        f"neg_logit_{label_a}" in merged.columns
        and f"neg_logit_{label_b}" in merged.columns
        and f"wer_{label_a}" in merged.columns
    ):
        plot_df = pd.concat(
            [
                pd.DataFrame({
                    "log1p_wer": np.log1p(np.maximum(0.0, merged[f"wer_{label_a}"])),
                    "neg_logit": merged[f"neg_logit_{label_a}"],
                    "model": label_a,
                }),
                pd.DataFrame({
                    "log1p_wer": np.log1p(np.maximum(0.0, merged[f"wer_{label_a}"])),
                    "neg_logit": merged[f"neg_logit_{label_b}"],
                    "model": label_b,
                }),
            ],
            ignore_index=True,
        )
        plt.figure(figsize=(6.5, 6.0))
        if sns is not None:
            sns.scatterplot(data=plot_df, x="log1p_wer", y="neg_logit", hue="model", alpha=0.6, s=18)
        else:
            for model_name, grp in plot_df.groupby("model"):
                plt.scatter(grp["log1p_wer"], grp["neg_logit"], alpha=0.6, s=18, label=model_name)
            plt.legend()
        plt.title("D_WER comparison: -logit vs log1p(WER)")
        plt.tight_layout()
        plt.savefig(plot_dir / "compare_neglogit_vs_logwer.png", dpi=180)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluación paper-ready del discriminador D_WER.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--checkpoint-b", type=str, default=None)
    ap.add_argument("--label-a", type=str, default="model_a")
    ap.add_argument("--label-b", type=str, default="model_b")
    ap.add_argument("--subset", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--purpose", type=str, default="wer_disc", choices=["default", "wer_disc"])
    ap.add_argument("--output-head", type=str, default="abs", choices=["abs", "rel"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--outdir", type=str, default="analysis/werd_eval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        device = torch.device(args.device)
    else:
        dflt = cfg.get("devices", {}).get("whisper_device", "cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(dflt)

    out_dir = Path(args.outdir) / args.purpose / args.subset
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_rows(
        config=cfg,
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        purpose=args.purpose,
        output_head=str(args.output_head),
        device=device,
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
    )
    df.to_csv(out_dir / "rows.csv", index=False)

    m = np.isfinite(df["quality"].values) & np.isfinite(df["prob"].values)
    cmp_df = df.loc[m].copy()
    if len(cmp_df) > 0:
        rq = _rankdata(cmp_df["quality"].values)
        rp = _rankdata(cmp_df["prob"].values)
        cmp_df["rank_gap_model_vs_quality"] = np.abs(rq - rp)
        cmp_df = cmp_df.sort_values("rank_gap_model_vs_quality", ascending=False)
        cmp_df.head(max(1, int(args.top_k))).to_csv(out_dir / "model_disagreements_topk.csv", index=False)

    summary = _compute_summary(
        df,
        subset=args.subset,
        purpose=args.purpose,
        output_head=str(args.output_head),
        checkpoint=args.checkpoint,
        bootstrap=int(args.bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    cal_df, _ = _calibration_bins(df["prob"].values, df["quality"].values, n_bins=10)
    if not cal_df.empty:
        cal_df.to_csv(out_dir / "calibration_bins.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_plots(df, plot_dir)

    if args.checkpoint_b:
        out_dir_b = Path(args.outdir) / args.purpose / f"{args.subset}__{args.label_b}"
        plot_dir_b = out_dir_b / "plots"
        out_dir_b.mkdir(parents=True, exist_ok=True)
        plot_dir_b.mkdir(parents=True, exist_ok=True)
        df_b = _collect_rows(
            config=cfg,
            checkpoint_path=args.checkpoint_b,
            subset=args.subset,
            purpose=args.purpose,
            output_head=str(args.output_head),
            device=device,
            batch_size_override=args.batch_size,
            max_samples=args.max_samples,
        )
        df_b.to_csv(out_dir_b / "rows.csv", index=False)
        summary_b = _compute_summary(
            df_b,
            subset=args.subset,
            purpose=args.purpose,
            output_head=str(args.output_head),
            checkpoint=args.checkpoint_b,
            bootstrap=int(args.bootstrap),
            bootstrap_seed=int(args.bootstrap_seed) + 10,
        )
        cal_df_b, _ = _calibration_bins(df_b["prob"].values, df_b["quality"].values, n_bins=10)
        if not cal_df_b.empty:
            cal_df_b.to_csv(out_dir_b / "calibration_bins.csv", index=False)
        (out_dir_b / "summary.json").write_text(json.dumps(summary_b, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_plots(df_b, plot_dir_b)

        cmp_out = Path(args.outdir) / args.purpose / f"{args.subset}__compare_{args.label_a}_vs_{args.label_b}"
        cmp_out.mkdir(parents=True, exist_ok=True)
        _save_comparison_outputs(
            df_a=df,
            df_b=df_b,
            summary_a=summary,
            summary_b=summary_b,
            out_dir=cmp_out,
            label_a=str(args.label_a),
            label_b=str(args.label_b),
        )
        print(f"[evaluate_werd] comparison saved -> {cmp_out}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[evaluate_werd] saved -> {out_dir}")


if __name__ == "__main__":
    main()

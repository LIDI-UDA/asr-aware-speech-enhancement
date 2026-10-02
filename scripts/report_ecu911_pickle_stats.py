#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Permite ejecutar como "python3 scripts/..." sin depender de PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import load_config

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DECODER_KEYS = (
    "asr_avg_entropy",
    "asr_avg_logprob",
    "asr_no_speech_prob",
    "asr_compression_ratio",
    "asr_token_count",
)

CATEGORICAL_KEYS = (
    "incident_type",
    "incident_grade",
    "tra_id",
)

WER_BINS_DEFAULT = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf"))
DUR_BINS_DEFAULT = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, float("inf"))

CH_RE = re.compile(r"CH(\d+)-", re.IGNORECASE)


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return f


def _summary(vals: List[float]) -> Optional[Dict[str, float]]:
    if len(vals) < 1:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _fmt_stats(name: str, st: Optional[Dict[str, float]]) -> str:
    if st is None:
        return f"{name}: n=0"
    return (
        f"{name}: n={int(st['n'])} "
        f"mean={st['mean']:.4f} median={st['median']:.4f} std={st['std']:.4f} "
        f"min={st['min']:.4f} p95={st['p95']:.4f} max={st['max']:.4f}"
    )


def _pct(a: int, b: int) -> float:
    if b <= 0:
        return float("nan")
    return 100.0 * float(a) / float(b)


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    if len(x) != len(y):
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    if len(x) < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": float(len(x))}
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    m = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[m]
    ya = ya[m]
    n = int(xa.size)
    if n < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": float(n)}

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa <= 0.0 or sb <= 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

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

    p = _pearson(xa, ya)
    rs = _rankdata(xa)
    rt = _rankdata(ya)
    s = _pearson(rs, rt)
    return {"pearson": p, "spearman": s, "n": float(n)}


def _hist_counts(vals: Sequence[float], bins: Sequence[float]) -> Dict[str, Any]:
    if len(vals) < 1:
        return {"bins": [], "total": 0}
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return {"bins": [], "total": 0}

    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bins debe tener al menos 2 edges")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bins debe ser estrictamente creciente")

    out = []
    total = int(arr.size)
    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i < (len(edges) - 2):
            mask = (arr >= left) & (arr < right)
            label = f"[{left:.2f}, {right:.2f})"
        else:
            mask = (arr >= left) & (arr <= right)
            label = f"[{left:.2f}, +inf)"
        c = int(np.sum(mask))
        out.append({"label": label, "count": c, "pct": _pct(c, total)})
    return {"bins": out, "total": total}


def _top_counts(values: Sequence[Any], topk: int) -> List[Dict[str, Any]]:
    cnt = Counter(str(v).strip() for v in values if str(v).strip() != "")
    out = []
    total = int(sum(cnt.values()))
    for k, v in cnt.most_common(max(0, int(topk))):
        out.append({"value": k, "count": int(v), "pct": _pct(int(v), total)})
    return out


def _extract_channel(audio_path: Any) -> Optional[str]:
    if audio_path is None:
        return None
    s = str(audio_path)
    m = CH_RE.search(s)
    if m is None:
        return None
    return f"CH{m.group(1)}"


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if plt is not None:
        plt.close(fig)


def _collect_split_plot_data(samples: List[Dict[str, Any]], prefer_longform: bool) -> Dict[str, Any]:
    wer_eff: List[float] = []
    durations: List[float] = []
    words: List[float] = []
    chars: List[float] = []
    quality: List[float] = []
    quality_expected: List[float] = []
    quality_abs_err: List[float] = []
    decoder_vals: Dict[str, List[float]] = {k: [] for k in DECODER_KEYS}
    channel_vals: List[str] = []
    cat_vals: Dict[str, List[str]] = {k: [] for k in CATEGORICAL_KEYS}

    for s in samples:
        w_eff, _ = _effective_wer(s, prefer_longform=prefer_longform)
        if w_eff is not None:
            wer_eff.append(float(w_eff))

        d = _as_float(s.get("duration", None))
        if d is not None and d >= 0.0:
            durations.append(d)

        tr = str(s.get("transcript", "") or "").strip()
        if tr:
            words.append(float(len(tr.split())))
            chars.append(float(len(tr)))

        q = _as_float(s.get("quality_score", None))
        if q is not None and w_eff is not None:
            quality.append(float(q))
            q_exp = 1.0 / (1.0 + max(0.0, float(w_eff)))
            quality_expected.append(float(q_exp))
            quality_abs_err.append(abs(float(q) - float(q_exp)))

        for k in DECODER_KEYS:
            vk = _as_float(s.get(k, None))
            if vk is not None:
                decoder_vals[k].append(vk)

        ap = s.get("audio_path", None)
        ch = _extract_channel(ap)
        if ch is not None:
            channel_vals.append(ch)

        for k in CATEGORICAL_KEYS:
            v = s.get(k, None)
            if v is not None and str(v).strip() != "":
                cat_vals[k].append(str(v).strip())

    return {
        "wer_eff": wer_eff,
        "durations": durations,
        "words": words,
        "chars": chars,
        "quality": quality,
        "quality_expected": quality_expected,
        "quality_abs_err": quality_abs_err,
        "decoder_vals": decoder_vals,
        "channels": channel_vals,
        "categorical": cat_vals,
    }


def _plot_hist_bar(
    title: str,
    hist_data: Dict[str, Any],
    x_label: str,
    out_path: Path,
    color: str = "#4C72B0",
) -> bool:
    bins = hist_data.get("bins", [])
    if plt is None or len(bins) < 1:
        return False
    labels = [b["label"] for b in bins]
    counts = [int(b["count"]) for b in bins]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(labels)), counts, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel(x_label)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, out_path)
    return True


def _plot_scatter_quality_vs_wer(
    split: str,
    wer_vals: Sequence[float],
    q_vals: Sequence[float],
    out_path: Path,
) -> bool:
    if plt is None:
        return False
    if len(wer_vals) < 2 or len(q_vals) < 2:
        return False
    n = min(len(wer_vals), len(q_vals))
    x = np.asarray(wer_vals[:n], dtype=np.float64)
    y = np.asarray(q_vals[:n], dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return False

    x_max = max(1.0, float(np.percentile(x, 99)))
    curve_x = np.linspace(0.0, x_max, 300)
    curve_y = 1.0 / (1.0 + curve_x)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(x, y, s=14, alpha=0.35, color="#4C72B0", label="samples")
    ax.plot(curve_x, curve_y, color="#DD8452", linewidth=2.0, label="q=1/(1+wer)")
    ax.set_title(f"{split}: quality_score vs WER")
    ax.set_xlabel("WER (effective)")
    ax.set_ylabel("quality_score")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save_fig(fig, out_path)
    return True


def _plot_decoder_hists(
    split: str,
    decoder_vals: Dict[str, List[float]],
    out_path: Path,
) -> bool:
    if plt is None:
        return False
    keys = [k for k in DECODER_KEYS if len(decoder_vals.get(k, [])) > 0]
    if len(keys) < 1:
        return False

    n = len(keys)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 3.8 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = axes.reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx >= n:
                ax.axis("off")
                continue
            key = keys[idx]
            arr = np.asarray(decoder_vals[key], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                ax.hist(arr, bins=40, color="#55A868", alpha=0.85)
                ax.set_title(f"{key} (n={arr.size})")
                ax.grid(alpha=0.2)
            else:
                ax.set_title(f"{key} (n=0)")
            idx += 1
    fig.suptitle(f"{split}: decoder metric distributions", fontsize=12)
    _save_fig(fig, out_path)
    return True


def _plot_top_categories(
    split: str,
    group_name: str,
    top_vals: List[Dict[str, Any]],
    out_path: Path,
) -> bool:
    if plt is None:
        return False
    if len(top_vals) < 1:
        return False
    labels = [str(x["value"]) for x in top_vals]
    counts = [int(x["count"]) for x in top_vals]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(range(len(labels)), counts, color="#C44E52", alpha=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"{split}: top {group_name}")
    ax.grid(axis="x", alpha=0.2)
    _save_fig(fig, out_path)
    return True


def _plot_cross_split_boxplots(
    split_to_vals: Dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    if plt is None:
        return False
    labels = []
    series = []
    for k in sorted(split_to_vals.keys()):
        arr = np.asarray(split_to_vals[k], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size < 1:
            continue
        labels.append(k)
        series.append(arr)
    if len(series) < 1:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.boxplot(series, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    _save_fig(fig, out_path)
    return True


def _generate_plots(
    data: Dict[str, Any],
    splits: List[str],
    prefer_longform: bool,
    plots_dir: Path,
    topk_categories: int,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"base_dir": str(plots_dir), "files": []}
    if plt is None:
        manifest["warning"] = "matplotlib no disponible; no se generaron plots"
        return manifest

    plots_dir.mkdir(parents=True, exist_ok=True)

    split_wer = {}
    split_dur = {}

    for split in splits:
        samples = data.get(split, [])
        if not isinstance(samples, list):
            continue
        pdata = _collect_split_plot_data(samples=samples, prefer_longform=prefer_longform)
        split_wer[split] = list(pdata["wer_eff"])
        split_dur[split] = list(pdata["durations"])

        h_wer = _hist_counts(pdata["wer_eff"], WER_BINS_DEFAULT)
        h_dur = _hist_counts(pdata["durations"], DUR_BINS_DEFAULT)

        out = plots_dir / f"{split}__wer_hist.png"
        if _plot_hist_bar(f"{split}: WER effective distribution", h_wer, "WER bins", out, color="#4C72B0"):
            manifest["files"].append(str(out))

        out = plots_dir / f"{split}__duration_hist.png"
        if _plot_hist_bar(f"{split}: duration distribution", h_dur, "Duration bins (s)", out, color="#8172B3"):
            manifest["files"].append(str(out))

        out = plots_dir / f"{split}__quality_vs_wer_scatter.png"
        if _plot_scatter_quality_vs_wer(split, pdata["wer_eff"], pdata["quality"], out):
            manifest["files"].append(str(out))

        out = plots_dir / f"{split}__quality_formula_error_hist.png"
        h_qerr = _hist_counts(pdata["quality_abs_err"], (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, float("inf")))
        if _plot_hist_bar(f"{split}: |quality - 1/(1+wer)|", h_qerr, "Absolute error bins", out, color="#CCB974"):
            manifest["files"].append(str(out))

        out = plots_dir / f"{split}__decoder_metrics_hist.png"
        if _plot_decoder_hists(split, pdata["decoder_vals"], out):
            manifest["files"].append(str(out))

        top_channel = _top_counts(pdata["channels"], topk=topk_categories)
        out = plots_dir / f"{split}__channel_top.png"
        if _plot_top_categories(split, "channels", top_channel, out):
            manifest["files"].append(str(out))

        for k in CATEGORICAL_KEYS:
            top_k = _top_counts(pdata["categorical"].get(k, []), topk=topk_categories)
            out = plots_dir / f"{split}__{k}_top.png"
            if _plot_top_categories(split, k, top_k, out):
                manifest["files"].append(str(out))

    out = plots_dir / "splits__wer_boxplot.png"
    if _plot_cross_split_boxplots(split_wer, "WER effective by split", "WER effective", out):
        manifest["files"].append(str(out))

    out = plots_dir / "splits__duration_boxplot.png"
    if _plot_cross_split_boxplots(split_dur, "Duration by split", "Duration (s)", out):
        manifest["files"].append(str(out))

    return manifest


def _effective_wer(sample: Dict[str, Any], prefer_longform: bool) -> tuple[Optional[float], str]:
    if prefer_longform:
        wlf = _as_float(sample.get("wer_longform", None))
        if wlf is not None:
            return wlf, "wer_longform"
    w = _as_float(sample.get("wer", None))
    if w is not None:
        return w, "wer"
    return None, "none"


def _iter_splits(data: Dict[str, Any], requested: Optional[Iterable[str]]) -> List[str]:
    if requested is None:
        requested = [k for k, v in data.items() if isinstance(v, list)]
    out = []
    for s in requested:
        if s in data and isinstance(data[s], list):
            out.append(s)
    return out


def _analyze_split(
    samples: List[Dict[str, Any]],
    split: str,
    prefer_longform: bool,
    precomputed_noisy_wer_max: Optional[float],
    topk_outliers: int,
    topk_categories: int,
    wer_bins: Sequence[float],
    dur_bins: Sequence[float],
) -> Dict[str, Any]:
    total = len(samples)
    missing_audio = 0
    empty_transcript = 0

    duration_vals: List[float] = []
    wer_vals: List[float] = []
    wer_vals_raw: List[float] = []
    wer_vals_long: List[float] = []
    quality_vals: List[float] = []
    quality_abs_err_vs_1_over_1_plus_wer: List[float] = []
    decoder_vals: Dict[str, List[float]] = {k: [] for k in DECODER_KEYS}
    transcript_word_counts: List[float] = []
    transcript_char_counts: List[float] = []

    source_counts = {"wer_longform": 0, "wer": 0, "none": 0}
    pre_ok = 0
    pre_rej = 0
    pre_na = 0
    outliers = []
    field_presence = Counter()
    audio_paths = []
    channels = []
    categorical_values: Dict[str, List[str]] = {k: [] for k in CATEGORICAL_KEYS}

    for s in samples:
        field_presence.update(s.keys())

        ap = s.get("audio_path", None)
        audio_paths.append(str(ap) if ap is not None else "")
        if (ap is None) or (not Path(str(ap)).exists()):
            missing_audio += 1

        tr = str(s.get("transcript", "") or "")
        if len(tr.strip()) == 0:
            empty_transcript += 1
        else:
            transcript_word_counts.append(float(len(tr.strip().split())))
            transcript_char_counts.append(float(len(tr.strip())))

        d = _as_float(s.get("duration", None))
        if d is not None and d >= 0.0:
            duration_vals.append(d)

        w_raw = _as_float(s.get("wer", None))
        if w_raw is not None:
            wer_vals_raw.append(w_raw)

        w_lf = _as_float(s.get("wer_longform", None))
        if w_lf is not None:
            wer_vals_long.append(w_lf)

        w_eff, src = _effective_wer(s, prefer_longform=prefer_longform)
        source_counts[src] = source_counts.get(src, 0) + 1
        if w_eff is not None:
            wer_vals.append(w_eff)
            outliers.append(
                {
                    "audio_path": str(ap),
                    "wer": float(w_eff),
                    "source": src,
                    "duration": float(d) if d is not None else None,
                }
            )
            if precomputed_noisy_wer_max is None:
                pre_ok += 1
            elif 0.0 <= float(w_eff) <= float(precomputed_noisy_wer_max):
                pre_ok += 1
            else:
                pre_rej += 1
        else:
            pre_na += 1

        q = _as_float(s.get("quality_score", None))
        if q is not None:
            quality_vals.append(q)
            if w_eff is not None:
                q_exp = 1.0 / (1.0 + max(0.0, float(w_eff)))
                quality_abs_err_vs_1_over_1_plus_wer.append(abs(float(q) - float(q_exp)))

        for k in DECODER_KEYS:
            vk = _as_float(s.get(k, None))
            if vk is not None:
                decoder_vals[k].append(vk)

        ch = _extract_channel(ap)
        if ch is not None:
            channels.append(ch)
        for k in CATEGORICAL_KEYS:
            v = s.get(k, None)
            if v is not None and str(v).strip() != "":
                categorical_values[k].append(str(v).strip())

    outliers.sort(key=lambda x: x["wer"], reverse=True)
    outliers = outliers[: max(0, int(topk_outliers))]

    wer_arr = np.asarray(wer_vals, dtype=np.float64) if len(wer_vals) > 0 else np.asarray([], dtype=np.float64)
    counts_over = {
        "gt_1": int(np.sum(wer_arr > 1.0)) if wer_arr.size > 0 else 0,
        "gt_2": int(np.sum(wer_arr > 2.0)) if wer_arr.size > 0 else 0,
        "gt_5": int(np.sum(wer_arr > 5.0)) if wer_arr.size > 0 else 0,
        "gt_10": int(np.sum(wer_arr > 10.0)) if wer_arr.size > 0 else 0,
        "gt_20": int(np.sum(wer_arr > 20.0)) if wer_arr.size > 0 else 0,
    }

    expected_quality = [1.0 / (1.0 + max(0.0, float(w))) for w in wer_vals]
    quality_consistency = _safe_corr(quality_vals[: len(expected_quality)], expected_quality[: len(quality_vals)])
    path_counts = Counter([p for p in audio_paths if p != ""])
    duplicate_paths = int(sum(1 for _, c in path_counts.items() if c > 1))

    field_coverage = {}
    for k, c in sorted(field_presence.items()):
        field_coverage[k] = {
            "count": int(c),
            "pct": _pct(int(c), int(total)),
        }

    result = {
        "split": split,
        "n_samples": total,
        "missing_audio_paths": missing_audio,
        "missing_audio_paths_pct": _pct(missing_audio, total),
        "empty_transcripts": empty_transcript,
        "empty_transcripts_pct": _pct(empty_transcript, total),
        "unique_audio_paths": int(len(path_counts)),
        "duplicate_audio_paths": duplicate_paths,
        "effective_wer_source_counts": source_counts,
        "precomputed_noisy_wer_valid": {
            "threshold": (None if precomputed_noisy_wer_max is None else float(precomputed_noisy_wer_max)),
            "ok": int(pre_ok),
            "rejected": int(pre_rej),
            "na": int(pre_na),
        },
        "field_coverage": field_coverage,
        "duration_stats": _summary(duration_vals),
        "transcript_words_stats": _summary(transcript_word_counts),
        "transcript_chars_stats": _summary(transcript_char_counts),
        "wer_effective_stats": _summary(wer_vals),
        "wer_raw_stats": _summary(wer_vals_raw),
        "wer_longform_stats": _summary(wer_vals_long),
        "wer_effective_counts_over": counts_over,
        "wer_effective_hist": _hist_counts(wer_vals, wer_bins),
        "duration_hist": _hist_counts(duration_vals, dur_bins),
        "quality_score_stats": _summary(quality_vals),
        "quality_abs_err_vs_1_over_1_plus_wer_stats": _summary(quality_abs_err_vs_1_over_1_plus_wer),
        "quality_vs_expected_from_wer_corr": quality_consistency,
        "decoder_stats": {k: _summary(v) for k, v in decoder_vals.items()},
        "channel_top": _top_counts(channels, topk=topk_categories),
        "categorical_top": {k: _top_counts(v, topk=topk_categories) for k, v in categorical_values.items()},
        "top_wer_outliers": outliers,
    }
    return result


def _split_overlap(data: Dict[str, Any], splits: List[str]) -> Dict[str, Any]:
    split_paths = {}
    split_tra = {}
    for s in splits:
        samples = data[s]
        paths = set()
        tra = set()
        for it in samples:
            ap = str(it.get("audio_path", "")).strip()
            if ap:
                paths.add(ap)
            tid = str(it.get("tra_id", "")).strip()
            if tid:
                tra.add(tid)
        split_paths[s] = paths
        split_tra[s] = tra

    pairs = []
    for a, b in combinations(splits, 2):
        pa = split_paths.get(a, set())
        pb = split_paths.get(b, set())
        ta = split_tra.get(a, set())
        tb = split_tra.get(b, set())
        inter_p = pa & pb
        inter_t = ta & tb
        pairs.append(
            {
                "pair": f"{a}__{b}",
                "path_overlap": int(len(inter_p)),
                "path_overlap_pct_of_min_split": _pct(len(inter_p), max(1, min(len(pa), len(pb)))),
                "tra_id_overlap": int(len(inter_t)),
                "tra_id_overlap_pct_of_min_split": _pct(len(inter_t), max(1, min(len(ta), len(tb)))),
            }
        )

    return {
        "paths_unique_per_split": {k: int(len(v)) for k, v in split_paths.items()},
        "tra_unique_per_split": {k: int(len(v)) for k, v in split_tra.items()},
        "pairwise": pairs,
    }


def _print_hist(title: str, h: Dict[str, Any], indent: str = "    ") -> None:
    print(f"{indent}{title}:")
    bins = h.get("bins", [])
    if len(bins) < 1:
        print(f"{indent}  (n=0)")
        return
    for b in bins:
        print(f"{indent}  {b['label']}: n={b['count']} ({b['pct']:.2f}%)")


def _print_top(title: str, vals: List[Dict[str, Any]], indent: str = "    ") -> None:
    print(f"{indent}{title}:")
    if len(vals) < 1:
        print(f"{indent}  (sin datos)")
        return
    for i, it in enumerate(vals, start=1):
        print(f"{indent}  {i}. {it['value']} -> n={it['count']} ({it['pct']:.2f}%)")


def _print_report(metadata: Dict[str, Any], split_reports: List[Dict[str, Any]], overlap: Dict[str, Any]) -> None:
    print("=" * 100)
    print("ECU911 PICKLE REPORT")
    print("=" * 100)
    if metadata:
        print("[metadata]")
        for k in sorted(metadata.keys()):
            v = metadata[k]
            if isinstance(v, (list, tuple)):
                print(f"  - {k}: {list(v)}")
            else:
                print(f"  - {k}: {v}")
    else:
        print("[metadata] (no disponible)")

    print("-" * 100)
    print("[split-overlap]")
    print(f"  paths_unique_per_split={overlap.get('paths_unique_per_split', {})}")
    print(f"  tra_unique_per_split={overlap.get('tra_unique_per_split', {})}")
    for p in overlap.get("pairwise", []):
        print(
            f"  {p['pair']}: "
            f"path_overlap={p['path_overlap']} ({p['path_overlap_pct_of_min_split']:.2f}% of min split), "
            f"tra_overlap={p['tra_id_overlap']} ({p['tra_id_overlap_pct_of_min_split']:.2f}% of min split)"
        )

    for rep in split_reports:
        print("-" * 100)
        print(f"[split={rep['split']}] n={rep['n_samples']}")
        print(
            f"  missing_audio_paths={rep['missing_audio_paths']} "
            f"({rep['missing_audio_paths_pct']:.2f}%) "
            f"empty_transcripts={rep['empty_transcripts']} "
            f"({rep['empty_transcripts_pct']:.2f}%)"
        )
        print(
            f"  unique_audio_paths={rep['unique_audio_paths']} "
            f"duplicate_audio_paths={rep['duplicate_audio_paths']}"
        )
        print(f"  effective_wer_source_counts={rep['effective_wer_source_counts']}")
        print(f"  precomputed_noisy_wer_valid={rep['precomputed_noisy_wer_valid']}")
        print("  " + _fmt_stats("duration", rep["duration_stats"]))
        print("  " + _fmt_stats("transcript_words", rep["transcript_words_stats"]))
        print("  " + _fmt_stats("transcript_chars", rep["transcript_chars_stats"]))
        print("  " + _fmt_stats("wer_effective", rep["wer_effective_stats"]))
        print("  " + _fmt_stats("wer_raw", rep["wer_raw_stats"]))
        print("  " + _fmt_stats("wer_longform", rep["wer_longform_stats"]))
        print(f"  wer_effective_counts_over={rep['wer_effective_counts_over']}")
        print("  " + _fmt_stats("quality_score", rep["quality_score_stats"]))
        print(
            "  "
            + _fmt_stats(
                "quality_abs_err_vs_1_over_1_plus_wer",
                rep["quality_abs_err_vs_1_over_1_plus_wer_stats"],
            )
        )
        qc = rep.get("quality_vs_expected_from_wer_corr", {})
        print(
            f"  quality_vs_expected_from_wer_corr: "
            f"pearson={qc.get('pearson', float('nan')):.4f} "
            f"spearman={qc.get('spearman', float('nan')):.4f} "
            f"n={int(qc.get('n', 0.0))}"
        )
        _print_hist("wer_effective_hist", rep.get("wer_effective_hist", {}), indent="  ")
        _print_hist("duration_hist", rep.get("duration_hist", {}), indent="  ")
        print("  decoder_stats:")
        for k in DECODER_KEYS:
            print("    " + _fmt_stats(k, rep["decoder_stats"].get(k)))
        _print_top("channel_top", rep.get("channel_top", []), indent="  ")
        cat = rep.get("categorical_top", {})
        for k in CATEGORICAL_KEYS:
            _print_top(f"{k}_top", cat.get(k, []), indent="  ")
        if len(rep["top_wer_outliers"]) > 0:
            print("  top_wer_outliers:")
            for i, o in enumerate(rep["top_wer_outliers"], start=1):
                print(
                    f"    {i}. wer={o['wer']:.4f} src={o['source']} "
                    f"dur={o.get('duration')} path={o['audio_path']}"
                )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _build_markdown(
    pkl_path: Path,
    metadata: Dict[str, Any],
    prefer_longform: bool,
    pre_max: Optional[float],
    split_reports: List[Dict[str, Any]],
    overlap: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# ECU911 Pickle Stats Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- pickle: `{pkl_path}`")
    lines.append(f"- prefer_longform: `{bool(prefer_longform)}`")
    lines.append(f"- precomputed_noisy_wer_max: `{pre_max}`")
    lines.append("")

    lines.append("## Metadata")
    if metadata:
        for k in sorted(metadata.keys()):
            lines.append(f"- `{k}`: `{metadata[k]}`")
    else:
        lines.append("- (no metadata)")
    lines.append("")

    lines.append("## Split Overlap")
    rows = []
    for p in overlap.get("pairwise", []):
        rows.append(
            [
                p["pair"],
                str(p["path_overlap"]),
                f"{p['path_overlap_pct_of_min_split']:.2f}%",
                str(p["tra_id_overlap"]),
                f"{p['tra_id_overlap_pct_of_min_split']:.2f}%",
            ]
        )
    if len(rows) > 0:
        lines.append(
            _md_table(
                headers=[
                    "pair",
                    "path_overlap",
                    "path_overlap_pct_of_min_split",
                    "tra_id_overlap",
                    "tra_id_overlap_pct_of_min_split",
                ],
                rows=rows,
            )
        )
    else:
        lines.append("- (no pairwise overlap)")
    lines.append("")

    for rep in split_reports:
        split = rep["split"]
        lines.append(f"## Split `{split}`")
        lines.append("")
        rows_main = [
            ["n_samples", str(rep["n_samples"])],
            ["missing_audio_paths", f"{rep['missing_audio_paths']} ({rep['missing_audio_paths_pct']:.2f}%)"],
            ["empty_transcripts", f"{rep['empty_transcripts']} ({rep['empty_transcripts_pct']:.2f}%)"],
            ["unique_audio_paths", str(rep["unique_audio_paths"])],
            ["duplicate_audio_paths", str(rep["duplicate_audio_paths"])],
            ["effective_wer_source_counts", json.dumps(rep["effective_wer_source_counts"], ensure_ascii=False)],
            ["precomputed_noisy_wer_valid", json.dumps(rep["precomputed_noisy_wer_valid"], ensure_ascii=False)],
        ]
        lines.append(_md_table(headers=["metric", "value"], rows=rows_main))
        lines.append("")

        def _stats_rows(name: str, st: Optional[Dict[str, float]]) -> List[List[str]]:
            if st is None:
                return [[name, "n=0", "", "", "", "", "", ""]]
            return [
                [
                    name,
                    f"n={int(st['n'])}",
                    f"{st['mean']:.4f}",
                    f"{st['median']:.4f}",
                    f"{st['std']:.4f}",
                    f"{st['min']:.4f}",
                    f"{st['p95']:.4f}",
                    f"{st['max']:.4f}",
                ]
            ]

        rows_stats: List[List[str]] = []
        rows_stats += _stats_rows("duration", rep.get("duration_stats"))
        rows_stats += _stats_rows("transcript_words", rep.get("transcript_words_stats"))
        rows_stats += _stats_rows("transcript_chars", rep.get("transcript_chars_stats"))
        rows_stats += _stats_rows("wer_effective", rep.get("wer_effective_stats"))
        rows_stats += _stats_rows("wer_raw", rep.get("wer_raw_stats"))
        rows_stats += _stats_rows("wer_longform", rep.get("wer_longform_stats"))
        rows_stats += _stats_rows("quality_score", rep.get("quality_score_stats"))
        rows_stats += _stats_rows("quality_abs_err_vs_1_over_1_plus_wer", rep.get("quality_abs_err_vs_1_over_1_plus_wer_stats"))

        lines.append("### Numeric Stats")
        lines.append(
            _md_table(
                headers=["name", "n", "mean", "median", "std", "min", "p95", "max"],
                rows=rows_stats,
            )
        )
        lines.append("")

        qc = rep.get("quality_vs_expected_from_wer_corr", {})
        lines.append("### Quality Consistency")
        lines.append(
            _md_table(
                headers=["pearson", "spearman", "n"],
                rows=[[
                    f"{qc.get('pearson', float('nan')):.4f}",
                    f"{qc.get('spearman', float('nan')):.4f}",
                    str(int(qc.get("n", 0.0))),
                ]],
            )
        )
        lines.append("")

        lines.append("### WER Threshold Counts")
        wc = rep.get("wer_effective_counts_over", {})
        rows_wc = [[k, str(v)] for k, v in wc.items()]
        lines.append(_md_table(headers=["threshold", "count"], rows=rows_wc))
        lines.append("")

        for hname in ("wer_effective_hist", "duration_hist"):
            h = rep.get(hname, {})
            rows_h = []
            for b in h.get("bins", []):
                rows_h.append([b["label"], str(b["count"]), f"{b['pct']:.2f}%"])
            lines.append(f"### {hname}")
            if rows_h:
                lines.append(_md_table(headers=["bin", "count", "pct"], rows=rows_h))
            else:
                lines.append("- (n=0)")
            lines.append("")

        lines.append("### Decoder Stats")
        rows_dec: List[List[str]] = []
        for k in DECODER_KEYS:
            st = rep.get("decoder_stats", {}).get(k)
            if st is None:
                rows_dec.append([k, "n=0", "", "", "", "", "", ""])
            else:
                rows_dec.append([
                    k,
                    f"n={int(st['n'])}",
                    f"{st['mean']:.4f}",
                    f"{st['median']:.4f}",
                    f"{st['std']:.4f}",
                    f"{st['min']:.4f}",
                    f"{st['p95']:.4f}",
                    f"{st['max']:.4f}",
                ])
        lines.append(_md_table(headers=["key", "n", "mean", "median", "std", "min", "p95", "max"], rows=rows_dec))
        lines.append("")

        lines.append("### Categorical Top")
        rows_cat: List[List[str]] = []
        for it in rep.get("channel_top", []):
            rows_cat.append(["channel", it["value"], str(it["count"]), f"{it['pct']:.2f}%"])
        for key in CATEGORICAL_KEYS:
            for it in rep.get("categorical_top", {}).get(key, []):
                rows_cat.append([key, it["value"], str(it["count"]), f"{it['pct']:.2f}%"])
        if len(rows_cat) > 0:
            lines.append(_md_table(headers=["group", "value", "count", "pct"], rows=rows_cat))
        else:
            lines.append("- (sin datos)")
        lines.append("")

        out = rep.get("top_wer_outliers", [])
        lines.append("### Top WER Outliers")
        if len(out) > 0:
            rows_o = []
            for i, o in enumerate(out, start=1):
                rows_o.append([str(i), f"{o['wer']:.4f}", str(o["source"]), str(o.get("duration")), str(o["audio_path"])])
            lines.append(_md_table(headers=["rank", "wer", "source", "duration", "audio_path"], rows=rows_o))
        else:
            lines.append("- (sin outliers)")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description="Reporte de estadísticas de ecu911_prepared.pkl")
    ap.add_argument("--config", type=str, default=None, help="Config YAML (opcional)")
    ap.add_argument("--pkl", type=str, default=None, help="Ruta explícita a ecu911_prepared.pkl")
    ap.add_argument("--splits", type=str, nargs="*", default=None, help="Splits a analizar (default: todos)")
    ap.add_argument("--prefer-longform", action="store_true", help="Priorizar wer_longform como WER efectivo")
    ap.add_argument(
        "--precomputed-noisy-wer-max",
        type=float,
        default=None,
        help="Threshold para simular aceptación/rechazo de WER precomputado",
    )
    ap.add_argument("--topk-outliers", type=int, default=5, help="Top K outliers por WER efectivo")
    ap.add_argument("--topk-categories", type=int, default=10, help="Top K valores por variable categórica")
    ap.add_argument(
        "--plots-dir",
        type=str,
        default="analysis/pickle_stats/plots",
        help="Directorio para guardar plots (PNG)",
    )
    ap.add_argument("--no-plots", action="store_true", help="Desactivar generación de plots")
    ap.add_argument("--json-out", type=str, default=None, help="Guardar reporte en JSON")
    ap.add_argument("--md-out", type=str, default=None, help="Guardar reporte en Markdown (paper-friendly)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    default_pkl = None
    default_pre_max = None
    default_prefer_longform = True
    if cfg:
        default_pkl = str(Path(cfg["paths"]["processed_data"]) / "ecu911_prepared.pkl")
        default_pre_max = cfg.get("evaluation", {}).get("precomputed_noisy_wer_max", 5.0)
        default_prefer_longform = bool(cfg.get("wer_discriminator", {}).get("use_longform_key", True))

    pkl_path = Path(args.pkl or default_pkl or "data/processed/ecu911_prepared.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe pickle: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Formato inesperado en pickle: {type(data).__name__}")

    splits = _iter_splits(data, args.splits)
    if len(splits) < 1:
        raise RuntimeError(f"No hay splits válidos. Disponibles: {[k for k, v in data.items() if isinstance(v, list)]}")

    prefer_longform = bool(args.prefer_longform or default_prefer_longform)
    pre_max = args.precomputed_noisy_wer_max
    if pre_max is None:
        pre_max = default_pre_max

    split_reports = []
    for split in splits:
        rep = _analyze_split(
            samples=data[split],
            split=split,
            prefer_longform=prefer_longform,
            precomputed_noisy_wer_max=(None if pre_max is None else float(pre_max)),
            topk_outliers=int(args.topk_outliers),
            topk_categories=int(args.topk_categories),
            wer_bins=WER_BINS_DEFAULT,
            dur_bins=DUR_BINS_DEFAULT,
        )
        split_reports.append(rep)

    metadata = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}
    overlap = _split_overlap(data=data, splits=splits)
    _print_report(metadata=metadata, split_reports=split_reports, overlap=overlap)

    plot_manifest: Dict[str, Any] = {}
    if not bool(args.no_plots):
        plot_manifest = _generate_plots(
            data=data,
            splits=splits,
            prefer_longform=prefer_longform,
            plots_dir=Path(args.plots_dir),
            topk_categories=int(args.topk_categories),
        )
        if "warning" in plot_manifest:
            print(f"\n[plots] {plot_manifest['warning']}")
        else:
            print(
                f"\n[plots] guardados en: {plot_manifest.get('base_dir')} "
                f"(n={len(plot_manifest.get('files', []))})"
            )

    out = {
        "pkl_path": str(pkl_path),
        "prefer_longform": bool(prefer_longform),
        "precomputed_noisy_wer_max": (None if pre_max is None else float(pre_max)),
        "metadata": metadata,
        "split_overlap": overlap,
        "splits": split_reports,
        "plots": plot_manifest,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[json] guardado en: {out_path}")

    if args.md_out:
        md_text = _build_markdown(
            pkl_path=pkl_path,
            metadata=metadata,
            prefer_longform=prefer_longform,
            pre_max=(None if pre_max is None else float(pre_max)),
            split_reports=split_reports,
            overlap=overlap,
        )
        md_path = Path(args.md_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8")
        print(f"[md] guardado en: {md_path}")


if __name__ == "__main__":
    main()

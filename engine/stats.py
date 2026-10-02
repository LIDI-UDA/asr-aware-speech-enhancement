from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def safe_corr(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    xv = np.asarray(list(x), dtype=np.float64)
    yv = np.asarray(list(y), dtype=np.float64)

    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]

    out = {"pearson": float("nan"), "spearman": float("nan"), "n": int(xv.size)}
    if xv.size < 3:
        return out

    # Pearson
    xstd = xv.std()
    ystd = yv.std()
    if xstd > 0 and ystd > 0:
        out["pearson"] = float(np.corrcoef(xv, yv)[0, 1])

    # Spearman sin scipy
    xr = xv.argsort().argsort().astype(np.float64)
    yr = yv.argsort().argsort().astype(np.float64)
    xr_std = xr.std()
    yr_std = yr.std()
    if xr_std > 0 and yr_std > 0:
        out["spearman"] = float(np.corrcoef(xr, yr)[0, 1])

    return out

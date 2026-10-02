from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter


@dataclass
class MetricAccumulator:
    values: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, **metrics: float) -> None:
        for k, v in metrics.items():
            if v is None:
                continue
            v = float(v)
            if not np.isfinite(v):
                continue
            self.values.setdefault(k, []).append(v)

    def mean_dict(self, clear: bool = True) -> Dict[str, float]:
        out = {}
        for k, vals in self.values.items():
            finite = [x for x in vals if np.isfinite(x)]
            if not finite:
                continue
            out[k] = float(np.mean(finite))
        if clear:
            self.values.clear()
        return out


class TBLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, prefix: str, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if np.isfinite(v):
                self.writer.add_scalar(f"{prefix}/{k}", float(v), step)

    def close(self) -> None:
        self.writer.close()

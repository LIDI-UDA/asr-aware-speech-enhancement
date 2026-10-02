from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def __init__(self, ckpt_dir: Path, stage: str, mode: str):
        self.ckpt_dir = Path(ckpt_dir)
        self.stage = stage
        self.mode = mode  # "min" o "max"
        self.last_path = self.ckpt_dir / "last.pt"
        self.best_path = self.ckpt_dir / "best.pt"
        self.best_score: Optional[float] = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def save(self, state: Dict[str, Any], score: Optional[float] = None) -> Dict[str, bool]:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, self.last_path)

        saved_best = False
        if score is not None and self._is_better(float(score)):
            self.best_score = float(score)
            torch.save(state, self.best_path)
            saved_best = True

        return {"saved_last": True, "saved_best": saved_best}

    def maybe_load_resume(self, resume_path: Optional[str]) -> Optional[Dict[str, Any]]:
        path = Path(resume_path) if resume_path else self.last_path
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

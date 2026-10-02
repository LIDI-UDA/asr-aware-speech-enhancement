from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch


@dataclass
class ReplayItem:
    audio_path: str
    transcript: str
    duration: float
    quality: float
    wer: Optional[float] = None
    step: int = 0


class WERReplayBuffer:
    """
    Replay buffer en disco con index jsonl.

    API pública:
    - add_item(item)
    - add_item_from_tensor(item, waveform, sample_rate)
    - sample(n, stratify=True)
    """

    def __init__(self, root_dir: str, capacity: int = 600, seed: int = 42):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.wav_dir = self.root / "wavs"
        self.wav_dir.mkdir(parents=True, exist_ok=True)

        self.capacity = int(capacity)
        self.rng = random.Random(seed)

        self.index_path = self.root / "index.jsonl"
        self.items: List[ReplayItem] = []
        self._load()

    def _load(self) -> None:
        self.items = []
        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                item = ReplayItem(
                    audio_path=str(d.get("audio_path", "")),
                    transcript=str(d.get("transcript", "")),
                    duration=float(d.get("duration", 0.0)),
                    quality=float(d.get("quality", d.get("quality_score", 0.0))),
                    wer=(None if d.get("wer", None) is None else float(d.get("wer"))),
                    step=int(d.get("step", 0)),
                )
                if item.audio_path and Path(item.audio_path).exists():
                    self.items.append(item)
        self._truncate_save()

    def _truncate_save(self) -> None:
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

        with self.index_path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.items)

    def _new_wav_path(self, step: int) -> Path:
        ts = int(time.time() * 1000)
        rnd = self.rng.randint(0, 10_000_000)
        return self.wav_dir / f"enh_step{step}_{ts}_{rnd}.wav"

    def add_item(self, item: ReplayItem) -> ReplayItem:
        q = float(max(0.0, min(1.0, float(item.quality))))
        d = float(max(0.0, float(item.duration)))
        fixed = ReplayItem(
            audio_path=item.audio_path,
            transcript=str(item.transcript or ""),
            duration=d,
            quality=q,
            wer=None if item.wer is None else float(item.wer),
            step=int(item.step),
        )
        self.items.append(fixed)
        self._truncate_save()
        return fixed

    def add_item_from_tensor(self, item: ReplayItem, waveform: torch.Tensor, sample_rate: int) -> ReplayItem:
        wav = waveform.detach().cpu().float()
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav.squeeze(0)
            else:
                wav = wav.mean(dim=0)
        out_path = self._new_wav_path(step=int(item.step))
        sf.write(str(out_path), wav.numpy(), int(sample_rate), subtype="FLOAT")

        stored = ReplayItem(
            audio_path=str(out_path),
            transcript=item.transcript,
            duration=float(item.duration),
            quality=float(item.quality),
            wer=item.wer,
            step=int(item.step),
        )
        return self.add_item(stored)

    def sample(self, n: int, stratify: bool = True) -> List[ReplayItem]:
        if len(self.items) == 0:
            return []
        n = min(int(n), len(self.items))

        if not stratify or len(self.items) < 30:
            return self.rng.sample(self.items, n)

        lo = [it for it in self.items if it.quality <= 0.33]
        mid = [it for it in self.items if 0.33 < it.quality <= 0.66]
        hi = [it for it in self.items if it.quality > 0.66]

        out: List[ReplayItem] = []
        for bucket in (lo, mid, hi):
            if len(out) >= n:
                break
            k = min(max(1, n // 3), len(bucket))
            if k > 0:
                out.extend(self.rng.sample(bucket, k))

        if len(out) < n:
            remain = [it for it in self.items if it not in out]
            need = n - len(out)
            if remain:
                out.extend(self.rng.sample(remain, min(need, len(remain))))

        self.rng.shuffle(out)
        return out[:n]

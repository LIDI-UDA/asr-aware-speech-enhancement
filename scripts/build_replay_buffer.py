#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Permite ejecutar como "python3 scripts/..." sin depender de PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ecu911_dataset import create_ecu911_dataloader
from engine.config import load_config
from engine.devices import setup_devices
from engine.validation import whisper_transcribe_longform_chunked
from models.generator import create_generator
from trainers.common import load_whisper_asr
from utils.text import normalize_text_for_wer
from utils.wer_replay_buffer import ReplayItem, WERReplayBuffer
from jiwer import wer


def parse_args():
    p = argparse.ArgumentParser(description="Build offline replay buffer with coherent Whisper WER")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--generator-ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default="data/replay")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1, help="Batch para loader base (recomendado=1 por VRAM)")
    p.add_argument("--enhance-chunk-seconds", type=float, default=8.0, help="Chunk para inferencia del generador")
    p.add_argument("--enhance-overlap-seconds", type=float, default=1.0, help="Overlap entre chunks del generador")
    p.add_argument("--enhance-min-chunk-seconds", type=float, default=2.0, help="Chunk mínimo si hay fallback por OOM")
    return p.parse_args()


def _triangular_window(length: int, overlap: int, is_first: bool, is_last: bool, device: torch.device) -> torch.Tensor:
    w = torch.ones(length, dtype=torch.float32, device=device)
    ov = int(max(0, min(overlap, length // 2)))
    if ov <= 0:
        return w
    if not is_first:
        w[:ov] = torch.linspace(0.0, 1.0, steps=ov, device=device, dtype=torch.float32)
    if not is_last:
        w[-ov:] = torch.minimum(w[-ov:], torch.linspace(1.0, 0.0, steps=ov, device=device, dtype=torch.float32))
    return w


def _enhance_waveform_chunked(
    generator: torch.nn.Module,
    waveform_1d: torch.Tensor,
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
) -> torch.Tensor:
    if waveform_1d.ndim == 2:
        waveform_1d = waveform_1d.squeeze(0)
    wav = waveform_1d.detach().to(torch.float32).cpu()
    total = int(wav.shape[-1])
    if total < 1:
        return wav

    chunk = int(round(float(chunk_seconds) * float(sample_rate)))
    if chunk <= 0 or chunk >= total:
        with torch.inference_mode():
            inp = wav.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = generator(inp)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(0).detach().float().cpu()
        return out[:total]

    overlap = int(round(float(overlap_seconds) * float(sample_rate)))
    overlap = max(0, min(overlap, chunk - 1))
    hop = max(1, chunk - overlap)

    acc = torch.zeros(total, dtype=torch.float32)
    wsum = torch.zeros(total, dtype=torch.float32)

    with torch.inference_mode():
        for start in range(0, total, hop):
            end = min(total, start + chunk)
            seg = wav[start:end]
            seg_len = int(end - start)
            if seg_len < chunk:
                seg = F.pad(seg, (0, chunk - seg_len))

            seg_dev = seg.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = generator(seg_dev)
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(0).detach().float().cpu()[:seg_len]

            win = _triangular_window(
                length=seg_len,
                overlap=overlap,
                is_first=(start == 0),
                is_last=(end >= total),
                device=out.device,
            )
            acc[start:end] += out * win
            wsum[start:end] += win

    enh = acc / wsum.clamp_min(1e-6)
    return enh.clamp(min=-1.0, max=1.0)


def _enhance_with_fallback(
    generator: torch.nn.Module,
    waveform_1d: torch.Tensor,
    device: torch.device,
    sample_rate: int,
    chunk_seconds: float,
    overlap_seconds: float,
    min_chunk_seconds: float,
) -> torch.Tensor:
    cur = float(max(chunk_seconds, min_chunk_seconds))
    min_chunk = float(max(0.5, min_chunk_seconds))
    while True:
        try:
            return _enhance_waveform_chunked(
                generator=generator,
                waveform_1d=waveform_1d,
                device=device,
                sample_rate=sample_rate,
                chunk_seconds=cur,
                overlap_seconds=overlap_seconds,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" not in msg) or (cur <= min_chunk + 1e-6):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cur = max(min_chunk, cur / 2.0)
            print(f"[build_replay] OOM en generador, fallback chunk_seconds={cur:.2f}s")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    devices = setup_devices(cfg)

    loader = create_ecu911_dataloader(
        cfg,
        stage=args.split,
        batch_size=max(1, int(args.batch_size)),
        purpose="default",
    )

    gen = create_generator(cfg).to(devices.train_device)
    ckpt = torch.load(args.generator_ckpt, map_location="cpu", weights_only=False)
    gen.load_state_dict(ckpt.get("generator", ckpt), strict=False)
    gen.eval()

    whisper_model, whisper_proc = load_whisper_asr(cfg, devices.whisper_device)

    replay = WERReplayBuffer(root_dir=args.out, capacity=max(args.max_samples, 1))

    sr = int(cfg["audio"]["target_sr"])
    chunk = float(cfg["evaluation"].get("whisper_chunk_seconds", 30.0))
    overlap = float(cfg["evaluation"].get("whisper_overlap_seconds", 1.0))

    n = 0
    for batch in tqdm(loader, desc="build replay"):
        wave = batch["waveform"]
        durs = batch["durations"]
        texts = batch["transcripts"]

        for i in range(wave.shape[0]):
            if n >= args.max_samples:
                break

            true_len = max(1, int(float(durs[i]) * sr))
            noisy_i = wave[i][:true_len]
            e = _enhance_with_fallback(
                generator=gen,
                waveform_1d=noisy_i,
                device=devices.train_device,
                sample_rate=sr,
                chunk_seconds=float(args.enhance_chunk_seconds),
                overlap_seconds=float(args.enhance_overlap_seconds),
                min_chunk_seconds=float(args.enhance_min_chunk_seconds),
            )
            ref = normalize_text_for_wer(texts[i])

            hyp = whisper_transcribe_longform_chunked(
                whisper_model,
                whisper_proc,
                e.to(devices.whisper_device),
                sr=sr,
                device=devices.whisper_device,
                chunk_seconds=chunk,
                overlap_seconds=overlap,
                condition_on_prev_tokens=False,
            )
            hyp = normalize_text_for_wer(hyp)

            try:
                w = float(wer([ref if ref else "<empty>"], [hyp if hyp else "<empty>"]))
            except Exception:
                w = 1.0
            q = float(1.0 / (1.0 + max(0.0, w)))

            item = ReplayItem(
                audio_path="",
                transcript=texts[i],
                duration=float(durs[i]),
                quality=q,
                wer=w,
                step=0,
            )
            replay.add_item_from_tensor(item=item, waveform=e, sample_rate=sr)
            n += 1

        if n >= args.max_samples:
            break

    print({"saved": len(replay), "out": str(Path(args.out))})


if __name__ == "__main__":
    main()

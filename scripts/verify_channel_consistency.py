#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Permite ejecutar como "python3 scripts/..." sin depender de PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config import load_config


CH_RE = re.compile(r"CH(\d+)-", re.IGNORECASE)


def _parse_ch_token(path_like: Any) -> tuple[Optional[str], Optional[int]]:
    if path_like is None:
        return None, None
    m = CH_RE.search(str(path_like))
    if m is None:
        return None, None
    idx = int(m.group(1))
    return f"CH{idx}", idx


def _load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Formato inesperado en pickle: {type(data).__name__}")
    return data


def _iter_splits(data: Dict[str, Any], requested: Optional[Iterable[str]]) -> List[str]:
    if requested is None:
        requested = [k for k, v in data.items() if isinstance(v, list)]
    out: List[str] = []
    for s in requested:
        if s in data and isinstance(data[s], list):
            out.append(s)
    return out


def _resolve_default_pkl(cfg_path: Optional[str]) -> Path:
    if cfg_path:
        cfg = load_config(cfg_path)
        return Path(cfg["paths"]["processed_data"]) / "ecu911_prepared.pkl"
    return Path("data/processed/ecu911_prepared.pkl")


def _probe_ffprobe(path: str) -> Optional[Dict[str, Any]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,sample_rate,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(out.stdout)
    except Exception:
        return None

    streams = data.get("streams", [])
    if not streams:
        return None
    s0 = streams[0]
    ch = s0.get("channels", None)
    sr = s0.get("sample_rate", None)
    dur = data.get("format", {}).get("duration", None)
    codec = s0.get("codec_name", None)
    try:
        ch = int(ch) if ch is not None else None
    except Exception:
        ch = None
    try:
        sr = int(sr) if sr is not None else None
    except Exception:
        sr = None
    try:
        dur = float(dur) if dur is not None else None
    except Exception:
        dur = None
    return {"channels": ch, "sample_rate": sr, "duration": dur, "codec": codec, "backend": "ffprobe"}


def _probe_librosa(path: str) -> Optional[Dict[str, Any]]:
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    try:
        y, sr = librosa.load(path, sr=None, mono=False)
    except Exception:
        return None

    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            ch = 1
            n = y.shape[0]
        else:
            ch = int(y.shape[0])
            n = int(y.shape[-1])
    else:
        return None
    dur = float(n) / float(sr) if sr and sr > 0 else None
    return {"channels": int(ch), "sample_rate": int(sr), "duration": dur, "codec": None, "backend": "librosa"}


def _probe_audio(path: str, backend: str) -> Optional[Dict[str, Any]]:
    backend = backend.lower().strip()
    if backend == "ffprobe":
        return _probe_ffprobe(path)
    if backend == "librosa":
        return _probe_librosa(path)
    # auto
    info = _probe_ffprobe(path)
    if info is not None:
        return info
    return _probe_librosa(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Verifica si CHxx del nombre corresponde al número real de canales "
            "del archivo de audio (usando ffprobe o librosa)."
        )
    )
    ap.add_argument("--config", type=str, default=None, help="Config YAML (opcional).")
    ap.add_argument("--pkl", type=str, default=None, help="Ruta explícita a ecu911_prepared.pkl.")
    ap.add_argument("--splits", type=str, nargs="*", default=["train", "val"], help="Splits a revisar.")
    ap.add_argument("--n-samples", type=int, default=5, help="Número de audios a mostrar en detalle.")
    ap.add_argument(
        "--scan-size",
        type=int,
        default=200,
        help="Cantidad de audios para estadística global (usa -1 para todos).",
    )
    ap.add_argument("--seed", type=int, default=42, help="Semilla.")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "ffprobe", "librosa"],
        help="Backend para leer metadatos del audio.",
    )
    ap.add_argument("--json-out", type=str, default=None, help="Guardar reporte en JSON.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    pkl_path = Path(args.pkl) if args.pkl else _resolve_default_pkl(args.config)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe pickle: {pkl_path}")

    data = _load_pickle(pkl_path)
    splits = _iter_splits(data, args.splits)
    if len(splits) < 1:
        raise RuntimeError(f"No hay splits válidos para: {args.splits}")

    pool = []
    for s in splits:
        for item in data.get(s, []):
            p = str(item.get("audio_path", "")).strip()
            if not p:
                continue
            if not Path(p).exists():
                continue
            ch_lbl, ch_num = _parse_ch_token(p)
            pool.append({"split": s, "audio_path": p, "ch_label": ch_lbl, "ch_num": ch_num})

    if len(pool) < 1:
        raise RuntimeError("No se encontraron audios válidos en los splits solicitados.")

    rng.shuffle(pool)

    if int(args.scan_size) < 0:
        scan_pool = pool
    else:
        scan_pool = pool[: max(1, int(args.scan_size))]

    checked = []
    channel_count_dist = Counter()
    token_num_equals_real = 0
    token_num_total = 0
    backend_used = Counter()

    for it in scan_pool:
        info = _probe_audio(it["audio_path"], backend=args.backend)
        if info is None:
            continue
        real_ch = info.get("channels", None)
        if real_ch is not None:
            channel_count_dist[int(real_ch)] += 1
        ch_num = it.get("ch_num", None)
        same = None
        if ch_num is not None and real_ch is not None:
            same = bool(int(ch_num) == int(real_ch))
            token_num_total += 1
            token_num_equals_real += int(same)

        checked.append(
            {
                "split": it["split"],
                "audio_path": it["audio_path"],
                "ch_label": it["ch_label"],
                "ch_number_from_name": ch_num,
                "real_channels": real_ch,
                "sample_rate": info.get("sample_rate", None),
                "duration": info.get("duration", None),
                "codec": info.get("codec", None),
                "name_number_equals_real_channels": same,
                "backend": info.get("backend", None),
            }
        )
        backend_used[info.get("backend", "unknown")] += 1

    if len(checked) < 1:
        raise RuntimeError(
            "No se pudo leer metadata de audio con el backend seleccionado. "
            "Prueba --backend ffprobe o --backend librosa."
        )

    samples = checked[: max(1, int(args.n_samples))]

    print("=" * 100)
    print("CHANNEL COUNT CHECK (ffprobe/librosa)")
    print("=" * 100)
    print(f"pickle: {pkl_path}")
    print(f"splits usados: {splits}")
    print(f"backend solicitado: {args.backend}")
    print(f"backend usado: {dict(backend_used)}")
    print(f"audios revisados: {len(checked)}")
    print(f"distribución canales reales: {dict(channel_count_dist)}")
    if token_num_total > 0:
        pct = 100.0 * float(token_num_equals_real) / float(token_num_total)
        print(
            "coincidencia numérica (CHxx == canales_reales): "
            f"{token_num_equals_real}/{token_num_total} ({pct:.2f}%)"
        )

    print("\n[samples]")
    for i, it in enumerate(samples, start=1):
        print(
            f"{i}. split={it['split']} "
            f"name={it['ch_label']}({it['ch_number_from_name']}) "
            f"real_channels={it['real_channels']} "
            f"sr={it['sample_rate']} dur={it['duration']} "
            f"eq={it['name_number_equals_real_channels']} "
            f"path={it['audio_path']}"
        )

    report = {
        "pickle": str(pkl_path),
        "splits": list(splits),
        "backend_requested": args.backend,
        "backend_used": dict(backend_used),
        "checked": int(len(checked)),
        "real_channel_count_distribution": {str(k): int(v) for k, v in channel_count_dist.items()},
        "name_number_equals_real_channels": {
            "matches": int(token_num_equals_real),
            "total": int(token_num_total),
            "pct": (100.0 * float(token_num_equals_real) / float(max(1, token_num_total))),
        },
        "sample_rows": samples,
        "note": (
            "CHxx en nombre suele representar identificador de canal/sistema, "
            "no número físico de canales del stream (mono/stereo)."
        ),
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[json] guardado en: {out_path}")


if __name__ == "__main__":
    main()


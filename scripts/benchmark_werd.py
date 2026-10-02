#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import queue
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from tqdm import tqdm

# Permite ejecutar: python scripts/benchmark_werd.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.ecu911_dataset import create_ecu911_dataloader
from engine.config import load_config
from engine.validation import validate_discriminator_correlation
from models.wer_discriminator import create_wer_discriminator


def _parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("No se recibieron seeds válidas.")
    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return -1


def _parse_gpu_ids(raw: str) -> List[int]:
    out: List[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _run_train(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env_overrides: Dict[str, str] | None = None,
    stream_stdout: bool = True,
) -> Tuple[int, Dict | None]:
    last_payload = None
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        env.update(env_overrides)

    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert p.stdout is not None
        for line in p.stdout:
            if stream_stdout:
                print(line, end="")
            f.write(line)
            s = line.strip()
            if s.startswith("{") and s.endswith("}") and ("'stage'" in s or '"stage"' in s):
                try:
                    payload = ast.literal_eval(s)
                    if isinstance(payload, dict):
                        last_payload = payload
                except Exception:
                    continue
        ret = int(p.wait())
    return ret, last_payload


@torch.no_grad()
def _evaluate_checkpoint(
    config: Dict,
    checkpoint_path: Path,
    subset: str,
    device: torch.device,
    batch_size_override: int | None,
) -> Dict[str, float]:
    model = create_wer_discriminator(config, device=device).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt.get("wer_discriminator", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    if batch_size_override is None:
        loader = create_ecu911_dataloader(config, stage=subset, purpose="wer_disc")
    else:
        loader = create_ecu911_dataloader(
            config,
            stage=subset,
            batch_size=int(batch_size_override),
            purpose="wer_disc",
        )
    return validate_discriminator_correlation(model, loader, device=device)


def _series_stats(arr: np.ndarray) -> Dict:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
    }


def _prepare_seed_config(
    base_cfg: Dict,
    seed: int,
    epochs_override: int | None,
    exp_base: str,
    out_dir: Path,
    visible_gpu: int | None,
) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["experiment_name"] = f"{exp_base}_werdbmk_s{seed}"
    if epochs_override is not None:
        cfg.setdefault("training", {})["epochs"] = int(epochs_override)

    # Si entrenamos en paralelo por GPU, aislamos cada proceso con CUDA_VISIBLE_DEVICES.
    # Dentro del proceso sólo se ve 1 GPU, por eso usamos cuda:0 en config.
    if visible_gpu is not None:
        cfg.setdefault("devices", {})
        cfg["devices"]["train_device"] = "cuda:0"
        cfg["devices"]["whisper_device"] = "cuda:0"
        cfg["devices"]["validate_device"] = "cuda:0"

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = seed_dir / "config.seed.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    return {
        "seed": int(seed),
        "cfg": cfg,
        "cfg_path": cfg_path,
        "seed_dir": seed_dir,
        "visible_gpu": visible_gpu,
    }


def _train_one_seed(
    seed_pack: Dict,
    python_bin: str,
    extra_train_args: str,
    stream_stdout: bool,
) -> Dict:
    seed = int(seed_pack["seed"])
    cfg = seed_pack["cfg"]
    cfg_path = Path(seed_pack["cfg_path"])
    seed_dir = Path(seed_pack["seed_dir"])
    visible_gpu = seed_pack.get("visible_gpu", None)

    train_cmd = [
        python_bin,
        "train.py",
        "--stage",
        "pretrain_discriminator",
        "--config",
        str(cfg_path),
    ]
    if extra_train_args.strip():
        train_cmd.extend(shlex.split(extra_train_args))

    env_overrides = None
    if visible_gpu is not None:
        env_overrides = {"CUDA_VISIBLE_DEVICES": str(visible_gpu)}

    print(
        "[benchmark_werd] "
        f"seed={seed} "
        + (f"gpu={visible_gpu} " if visible_gpu is not None else "")
        + f"-> {' '.join(train_cmd)}"
    )
    ret_code, payload = _run_train(
        cmd=train_cmd,
        cwd=PROJECT_ROOT,
        log_path=seed_dir / "train.log",
        env_overrides=env_overrides,
        stream_stdout=stream_stdout,
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / cfg["experiment_name"] / "pretrain_discriminator"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else (last_ckpt if last_ckpt.exists() else None)

    return {
        "seed": seed,
        "cfg": cfg,
        "cfg_path": str(cfg_path),
        "seed_dir": str(seed_dir),
        "visible_gpu": visible_gpu,
        "exit_code": int(ret_code),
        "payload": payload or {},
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
    }


def _train_seed_group(
    gpu_id: int,
    group: List[Dict],
    python_bin: str,
    extra_train_args: str,
    result_queue: queue.Queue | None = None,
) -> List[Dict]:
    out: List[Dict] = []
    for item in group:
        res = _train_one_seed(
            seed_pack=item,
            python_bin=python_bin,
            extra_train_args=extra_train_args,
            stream_stdout=False,  # evitar logs intercalados al paralelizar
        )
        out.append(res)
        if result_queue is not None:
            result_queue.put(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size-eval", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="analysis/werd_benchmark")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--python-bin", type=str, default=sys.executable)
    ap.add_argument("--extra-train-args", type=str, default="")
    ap.add_argument(
        "--parallel-gpus",
        type=str,
        default="",
        help="Lista de GPUs para entrenamiento paralelo, ej: '0,1,2'.",
    )
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    parallel_gpu_ids = _parse_gpu_ids(args.parallel_gpus) if args.parallel_gpus.strip() else []
    parallel_mode = len(parallel_gpu_ids) > 0
    base_cfg = load_config(args.config)
    exp_base = str(base_cfg.get("experiment_name", "speech_enhancement"))
    run_tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outdir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_packs: List[Dict] = []
    if parallel_mode:
        for i, seed in enumerate(seeds):
            gpu = int(parallel_gpu_ids[i % len(parallel_gpu_ids)])
            seed_packs.append(
                _prepare_seed_config(
                    base_cfg=base_cfg,
                    seed=int(seed),
                    epochs_override=args.epochs,
                    exp_base=exp_base,
                    out_dir=out_dir,
                    visible_gpu=gpu,
                )
            )
    else:
        for seed in seeds:
            seed_packs.append(
                _prepare_seed_config(
                    base_cfg=base_cfg,
                    seed=int(seed),
                    epochs_override=args.epochs,
                    exp_base=exp_base,
                    out_dir=out_dir,
                    visible_gpu=None,
                )
            )

    train_results: List[Dict] = []
    if parallel_mode:
        by_gpu: Dict[int, List[Dict]] = {int(g): [] for g in parallel_gpu_ids}
        for p in seed_packs:
            by_gpu[int(p["visible_gpu"])].append(p)

        print(
            "[benchmark_werd] parallel mode "
            f"gpus={parallel_gpu_ids} "
            f"groups={{"
            + ", ".join(f"{g}:{len(v)}" for g, v in sorted(by_gpu.items()))
            + "}}"
        )
        result_q: queue.Queue = queue.Queue()
        expected = int(len(seed_packs))
        with cf.ThreadPoolExecutor(max_workers=len(by_gpu)) as ex:
            futs = [
                ex.submit(
                    _train_seed_group,
                    gpu_id=int(gpu),
                    group=group,
                    python_bin=args.python_bin,
                    extra_train_args=args.extra_train_args,
                    result_queue=result_q,
                )
                for gpu, group in sorted(by_gpu.items())
                if len(group) > 0
            ]
            pbar_train = tqdm(total=expected, desc="[benchmark_werd] train seeds", leave=True)
            while len(train_results) < expected:
                try:
                    res = result_q.get(timeout=1.0)
                    train_results.append(res)
                    bvals = [_safe_float((r.get("payload") or {}).get("best")) for r in train_results]
                    bvals = [v for v in bvals if np.isfinite(v)]
                    mean_best = float(np.mean(bvals)) if len(bvals) > 0 else float("nan")
                    pbar_train.update(1)
                    pbar_train.set_postfix(
                        done=f"{len(train_results)}/{expected}",
                        mean_best=f"{mean_best:.4f}" if np.isfinite(mean_best) else "nan",
                    )
                except queue.Empty:
                    if all(f.done() for f in futs):
                        break
            pbar_train.close()
            # Propaga excepciones de workers si alguna falló.
            for fut in futs:
                fut.result()
            # Drena cola remanente por seguridad.
            while len(train_results) < expected and (not result_q.empty()):
                train_results.append(result_q.get())
    else:
        pbar_train = tqdm(total=len(seed_packs), desc="[benchmark_werd] train seeds", leave=True)
        for p in seed_packs:
            res = _train_one_seed(
                seed_pack=p,
                python_bin=args.python_bin,
                extra_train_args=args.extra_train_args,
                stream_stdout=True,
            )
            train_results.append(res)
            bvals = [_safe_float((r.get("payload") or {}).get("best")) for r in train_results]
            bvals = [v for v in bvals if np.isfinite(v)]
            mean_best = float(np.mean(bvals)) if len(bvals) > 0 else float("nan")
            pbar_train.update(1)
            pbar_train.set_postfix(
                done=f"{len(train_results)}/{len(seed_packs)}",
                mean_best=f"{mean_best:.4f}" if np.isfinite(mean_best) else "nan",
            )
        pbar_train.close()

    if args.device:
        eval_device = torch.device(args.device)
    else:
        eval_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_rows: List[Dict] = []
    sorted_train_results = sorted(train_results, key=lambda x: int(x["seed"]))
    pbar_eval = tqdm(total=len(sorted_train_results), desc="[benchmark_werd] eval seeds", leave=True)
    for tr in sorted_train_results:
        seed = int(tr["seed"])
        cfg = tr["cfg"]
        seed_dir = Path(tr["seed_dir"])
        ckpt_path = Path(tr["checkpoint_path"]) if tr.get("checkpoint_path") else None

        eval_metrics = {}
        if ckpt_path is not None and ckpt_path.exists():
            try:
                eval_metrics = _evaluate_checkpoint(
                    config=cfg,
                    checkpoint_path=ckpt_path,
                    subset=args.subset,
                    device=eval_device,
                    batch_size_override=args.batch_size_eval,
                )
            except Exception as e:
                eval_metrics = {"eval_error": str(e)}
        else:
            eval_metrics = {"eval_error": "checkpoint_not_found"}

        payload = tr.get("payload", {}) or {}
        row = {
            "seed": seed,
            "visible_gpu": tr.get("visible_gpu", None),
            "exit_code": int(tr["exit_code"]),
            "experiment_name": str(cfg["experiment_name"]),
            "checkpoint_path": str(ckpt_path) if ckpt_path is not None else None,
            "train_best": _safe_float(payload.get("best")),
            "train_global_step": _safe_int(payload.get("global_step")),
            "train_target_mu": _safe_float(payload.get("target_mu")),
            "train_target_sigma": _safe_float(payload.get("target_sigma")),
            "eval_spearman_quality": _safe_float(eval_metrics.get("spearman_quality")),
            "eval_spearman_logwer": _safe_float(eval_metrics.get("spearman_logwer")),
            "eval_pearson_quality": _safe_float(eval_metrics.get("pearson_quality")),
            "eval_pearson_logwer": _safe_float(eval_metrics.get("pearson_logwer")),
            "eval_n_quality": _safe_int(eval_metrics.get("n_quality")),
            "eval_n_logwer": _safe_int(eval_metrics.get("n_logwer")),
            "eval_error": eval_metrics.get("eval_error", None),
        }
        run_rows.append(row)
        (seed_dir / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        svals = [r["eval_spearman_logwer"] for r in run_rows if np.isfinite(r.get("eval_spearman_logwer", float("nan")))]
        mean_s = float(np.mean(svals)) if len(svals) > 0 else float("nan")
        pbar_eval.update(1)
        pbar_eval.set_postfix(
            done=f"{len(run_rows)}/{len(sorted_train_results)}",
            mean_slog=f"{mean_s:.4f}" if np.isfinite(mean_s) else "nan",
        )
    pbar_eval.close()

    df = pd.DataFrame(run_rows)
    df.to_csv(out_dir / "runs.csv", index=False)

    summary = {
        "config": args.config,
        "seeds": seeds,
        "subset": args.subset,
        "parallel_gpus": parallel_gpu_ids if parallel_mode else [],
        "eval_device": str(eval_device),
        "n_runs": int(len(df)),
        "metrics": {
            "eval_spearman_quality": _series_stats(df["eval_spearman_quality"].values),
            "eval_spearman_logwer": _series_stats(df["eval_spearman_logwer"].values),
            "eval_pearson_quality": _series_stats(df["eval_pearson_quality"].values),
            "eval_pearson_logwer": _series_stats(df["eval_pearson_logwer"].values),
            "train_best": _series_stats(df["train_best"].values),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sns.set(style="whitegrid")
    ok = df[np.isfinite(df["eval_spearman_logwer"].values)].copy()
    if len(ok) > 0:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=ok, x="seed", y="eval_spearman_logwer", color="#2f6db2")
        plt.title("D_WER Benchmark: Spearman(logWER) by seed")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_seed_spearman_logwer.png", dpi=180)
        plt.close()

    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print(f"[benchmark_werd] saved -> {out_dir}")


if __name__ == "__main__":
    main()

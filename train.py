#!/usr/bin/env python3
from __future__ import annotations

import argparse

from engine.amp import AmpManager
from engine.checkpointing import CheckpointManager
from engine.config import ensure_dirs, load_config
from engine.devices import setup_devices
from engine.logging import TBLogger
from engine.seed import set_seed
from trainers.common import StageContext
from trainers.trainer_adv import run_adversarial
from trainers.trainer_finetune import run_finetune
from trainers.trainer_pretrain import run_pretrain
from trainers.trainer_werd import run_pretrain_discriminator


def parse_args():
    p = argparse.ArgumentParser(description="Speech Enhancement Training")
    p.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pretrain", "adversarial", "finetune", "pretrain_discriminator"],
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--generator-ckpt", type=str, default=None)
    p.add_argument("--werd-ckpt", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    dirs = ensure_dirs(config, args.stage)
    devices = setup_devices(config)
    print(
        {
            "train_device": str(devices.train_device),
            "whisper_device": str(devices.whisper_device),
            "validate_device": str(devices.validate_device),
            "cuda_available": devices.available_cuda,
        }
    )

    if args.stage == "pretrain_discriminator":
        best_mode = "max"
    else:
        best_mode = "min"

    amp_enabled = bool(config["training"].get("amp", True))
    if args.stage == "finetune":
        amp_enabled = bool(config.get("finetune", {}).get("use_amp", amp_enabled))
    amp = AmpManager(enabled=amp_enabled, device_type=devices.train_device.type)
    logger = TBLogger(log_dir=str(dirs["tb_dir"]))
    ckpt = CheckpointManager(ckpt_dir=dirs["ckpt_dir"], stage=args.stage, mode=best_mode)

    ctx = StageContext(
        stage=args.stage,
        config=config,
        devices=devices,
        amp=amp,
        logger=logger,
        ckpt=ckpt,
    )

    try:
        if args.stage == "pretrain":
            out = run_pretrain(ctx, resume=args.resume)
        elif args.stage == "adversarial":
            out = run_adversarial(ctx, resume=args.resume)
        elif args.stage == "finetune":
            out = run_finetune(
                ctx,
                resume=args.resume,
                generator_ckpt=args.generator_ckpt,
                werd_ckpt=args.werd_ckpt,
            )
        elif args.stage == "pretrain_discriminator":
            out = run_pretrain_discriminator(ctx, resume=args.resume)
        else:
            raise ValueError(f"Stage no soportado: {args.stage}")
    finally:
        logger.close()

    print({"stage": args.stage, **out})


if __name__ == "__main__":
    main()

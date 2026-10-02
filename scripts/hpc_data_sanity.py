#!/usr/bin/env python3
from __future__ import annotations

import argparse

from data.ecu911_dataset import create_ecu911_dataloader, create_spc_dataloader
from engine.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Quick data sanity check for HPC")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--check-spc", action="store_true")
    return p.parse_args()


def _peek_batch(loader, name: str):
    batch = next(iter(loader))
    print(
        {
            "name": name,
            "num_samples": len(loader.dataset),
            "batch_wave_shape": tuple(batch["waveform"].shape) if "waveform" in batch else None,
            "keys": sorted(list(batch.keys())),
        }
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    train = create_ecu911_dataloader(cfg, stage="train", purpose="default")
    val = create_ecu911_dataloader(cfg, stage="val", purpose="default")
    dwer = create_ecu911_dataloader(
        cfg,
        stage="train",
        batch_size=int(cfg.get("wer_discriminator", {}).get("batch_size", 32)),
        purpose="wer_disc",
    )

    _peek_batch(train, "ecu911_train")
    _peek_batch(val, "ecu911_val")
    _peek_batch(dwer, "ecu911_train_wer_disc")

    if args.check_spc:
        spc_dir = cfg["paths"].get("spc_corpus_dir")
        if spc_dir:
            spc = create_spc_dataloader(cfg, corpus_dir=spc_dir, stage="train", use_degradation=True)
            _peek_batch(spc, "spc_train")


if __name__ == "__main__":
    main()

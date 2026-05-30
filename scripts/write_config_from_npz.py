#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from coras.data import describe_npz


def choose_target_domains(npz_path: Path, n_target: int, min_count: int, explicit: list[str] | None = None) -> list[str]:
    data = np.load(npz_path, allow_pickle=True)
    domains = data["domains"].astype(str) if "domains" in data else np.array(["default"] * len(data["labels"]))
    if explicit:
        return explicit
    uniq, counts = np.unique(domains, return_counts=True)
    eligible = [(u, int(c)) for u, c in zip(uniq, counts) if int(c) >= min_count]
    if not eligible:
        eligible = [(u, int(c)) for u, c in zip(uniq, counts)]
    # Use lexicographically last eligible domains to avoid always holding out domain_0/task_0.
    eligible = sorted(eligible, key=lambda x: (x[1], x[0]), reverse=True)
    return [u for u, _ in eligible[:max(1, min(n_target, len(eligible)))]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concrete YAML config for a CoRAS NPZ dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-domain", action="append", default=None)
    parser.add_argument("--n-target-domains", type=int, default=2)
    parser.add_argument("--min-target-count", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder", choices=["small_cnn", "resnet18"], default="small_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--adapter-epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--calib-frac", type=float, default=0.25)
    parser.add_argument("--tune-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-remaining-target-in-train", action="store_true")
    parser.add_argument("--exclude-source-with-target-eval-episodes", action="store_true")
    args = parser.parse_args()
    target_domains = choose_target_domains(args.data, args.n_target_domains, args.min_target_count, args.target_domain)
    cfg = {
        "seed": args.seed,
        "data_path": str(args.data),
        "output_dir": str(args.output_dir),
        "target_domains": target_domains,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "encoder": args.encoder,
        "pretrained": bool(args.pretrained),
        "adapter_rank": 16 if args.encoder == "small_cnn" else 32,
        "freeze_encoder_from_start": False if args.encoder == "small_cnn" else True,
        "target_prompt_tuning": True,
        "epochs": args.epochs,
        "adapter_epochs": args.adapter_epochs,
        "lr": 7e-4 if args.encoder == "small_cnn" else 3e-4,
        "adapter_lr": 1e-3,
        "weight_decay": 1e-4,
        "alpha": args.alpha,
        "tune_frac": args.tune_frac,
        "calib_frac": args.calib_frac,
        "test_frac": args.test_frac,
        "include_remaining_target_in_train": bool(args.include_remaining_target_in_train),
        "exclude_source_with_target_eval_episodes": bool(args.exclude_source_with_target_eval_episodes),
        "torch_threads": 4,
        "methods": ["top1_singleton", "topk_calibrated", "vanilla_conformal", "aps_conformal", "temperature_conformal", "prompt_only_top1", "coras", "coras_aps", "mondrian_domain"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {args.out}")
    print("Selected target domains:", target_domains)
    print("Dataset summary:", describe_npz(args.data))


if __name__ == "__main__":
    main()

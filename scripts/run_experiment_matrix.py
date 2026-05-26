#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from coras.utils import run_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoRAS training/evaluation matrix over seeds, alphas, and calibration fractions.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--calib-fracs", nargs="+", type=float, default=[0.10, 0.25, 0.40])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(args.base_config.read_text())
    base_out = Path(base["output_dir"])
    configs_dir = base_out / "matrix_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for cf in args.calib_fracs:
            cfg = dict(base)
            cfg["seed"] = int(seed)
            cfg["calib_frac"] = float(cf)
            cfg["output_dir"] = str(base_out / f"seed{seed}_calib{cf:.2f}")
            cfg_path = configs_dir / f"config_seed{seed}_calib{cf:.2f}.yaml"
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            if not args.skip_train:
                run_cmd([sys.executable, "scripts/train_action_model.py", "--config", str(cfg_path), "--device", args.device], cwd=root)
            for alpha in args.alphas:
                run_cmd([sys.executable, "scripts/evaluate_methods.py", "--config", str(cfg_path), "--device", args.device, "--alpha", str(alpha)], cwd=root)
    run_cmd([sys.executable, "scripts/aggregate_results.py", "--root", str(base_out)], cwd=root)


if __name__ == "__main__":
    main()

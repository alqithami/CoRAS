#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Check split integrity: no episode/base-index leakage across train/tune/calib/test.")
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    splits = np.load(out_dir / "split_indices.npz")
    data = np.load(cfg["data_path"], allow_pickle=True)
    episodes = data["episodes"].astype(int)
    base_index = data["base_index"].astype(int) if "base_index" in data else np.arange(len(episodes))
    domains = data["domains"].astype(str) if "domains" in data else np.array(["default"] * len(episodes))
    names = ["train", "tune", "calib", "test"]
    print("Split sizes:")
    for n in names:
        idx = splits[n]
        print(f"  {n}: samples={len(idx)} episodes={len(set(episodes[idx]))} domains={dict(zip(*np.unique(domains[idx], return_counts=True)))}")
    ok = True
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ea, eb = set(map(int, episodes[splits[a]])), set(map(int, episodes[splits[b]]))
            ba, bb = set(map(int, base_index[splits[a]])), set(map(int, base_index[splits[b]]))
            ep_inter = ea & eb
            base_inter = ba & bb
            print(f"{a} vs {b}: shared_episodes={len(ep_inter)} shared_base_indices={len(base_inter)}")
            if ep_inter or base_inter:
                ok = False
    if not ok:
        raise SystemExit("Leakage check failed. Do not use this run in the paper.")
    print("Leakage check passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description="Audit CoRAS train/tune/calib/test split integrity.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["output_dir"])
    split_path = out_dir / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}; run train_action_model.py first.")
    with np.load(cfg["data_path"], allow_pickle=True) as data:
        n = len(data["labels"])
        domains = data["domains"].astype(str) if "domains" in data else np.array(["default"] * n)
        episodes = data["episodes"].astype(np.int64) if "episodes" in data else np.arange(n, dtype=np.int64)
        tasks = data["tasks"].astype(str) if "tasks" in data else np.array(["task"] * n)
        original_index = data["source_original_index"].astype(np.int64) if "source_original_index" in data else np.arange(n, dtype=np.int64)
    split = np.load(split_path)
    rows = []
    episode_sets = {}
    original_sets = {}
    for name in ["train", "tune", "calib", "test"]:
        idx = split[name].astype(np.int64)
        episode_sets[name] = set(map(int, episodes[idx]))
        original_sets[name] = set(map(int, original_index[idx]))
        for d in sorted(set(domains[idx])):
            m = idx[domains[idx] == d]
            rows.append({
                "split": name,
                "domain": d,
                "n": int(len(m)),
                "episodes": int(len(set(map(int, episodes[m])))),
                "tasks": int(len(set(tasks[m]))),
            })
    df = pd.DataFrame(rows)
    out_base = args.out or (out_dir / "split_audit")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_base) + "_domain_counts.csv", index=False)

    overlaps = []
    names = ["train", "tune", "calib", "test"]
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlaps.append({
                "a": a,
                "b": b,
                "episode_overlap": len(episode_sets[a] & episode_sets[b]),
                "original_index_overlap": len(original_sets[a] & original_sets[b]),
            })
    target_domains = set(map(str, cfg.get("target_domains") or []))
    target_train_n = int(np.isin(domains[split["train"]], list(target_domains)).sum()) if target_domains else 0
    summary = {
        "config": str(args.config),
        "data_path": cfg["data_path"],
        "target_domains": sorted(target_domains),
        "split_sizes": {name: int(len(split[name])) for name in names},
        "overlaps": overlaps,
        "target_domain_samples_in_train": target_train_n,
        "episode_overlap_ok": all(x["episode_overlap"] == 0 for x in overlaps if x["a"] != "train" or target_train_n == 0),
        "original_index_overlap_note": "May be nonzero only for controlled-shift benchmarks where multiple shifted views of the same original frame are intentionally kept in the same episode block.",
    }
    Path(str(out_base) + "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import KMeans


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize continuous robot actions into action-token labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-fit-samples", type=int, default=200000)
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=True)
    if "actions" not in data:
        raise ValueError(f"{args.input} does not contain an actions array")
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[:, None]
    rng = np.random.default_rng(args.seed)
    fit_idx = np.arange(len(actions))
    if len(fit_idx) > args.max_fit_samples:
        fit_idx = rng.choice(fit_idx, size=args.max_fit_samples, replace=False)
    k = min(int(args.num_codes), len(fit_idx))
    km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=10, verbose=0)
    km.fit(actions[fit_idx])
    labels = km.predict(actions).astype(np.int64)
    centers = km.cluster_centers_.astype(np.float32)
    mse = float(np.mean((actions - centers[labels]) ** 2))
    arrays = {name: data[name] for name in data.files if name != "labels"}
    arrays["labels"] = labels
    arrays["codebook_centers"] = centers
    arrays["codebook_quantization_mse"] = np.array([mse], dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"Wrote {args.out}; k={k}; quantization_mse={mse:.6g}")


if __name__ == "__main__":
    main()

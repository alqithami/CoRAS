#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from PIL import Image


def as_hwc_uint8(img: np.ndarray, resize: int | None = None) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if resize and (arr.shape[0] != resize or arr.shape[1] != resize):
        arr = np.asarray(Image.fromarray(arr).resize((resize, resize), Image.BILINEAR), dtype=np.uint8)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Export robomimic HDF5 demonstrations to CoRAS NPZ.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera-key", type=str, default="agentview_image")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--domain-strategy", choices=["constant", "demo_mod"], default="constant")
    parser.add_argument("--demo-mod", type=int, default=5)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    images, actions, episodes, domains, tasks = [], [], [], [], []
    base_domain = args.domain or args.input.stem
    with h5py.File(args.input, "r") as f:
        if "data" not in f:
            raise KeyError("Expected robomimic file with top-level group 'data'.")
        demos = sorted(list(f["data"].keys()))
        for epi, demo in enumerate(demos):
            grp = f["data"][demo]
            obs = grp["obs"]
            if args.camera_key not in obs:
                raise KeyError(f"Camera key {args.camera_key!r} not found. Available obs keys: {list(obs.keys())}")
            imgs = obs[args.camera_key][()]
            acts = grp["actions"][()]
            if imgs.ndim == 4 and imgs.shape[1] in {1, 3} and imgs.shape[-1] not in {1, 3}:
                imgs = np.transpose(imgs, (0, 2, 3, 1))
            for j in range(min(len(imgs), len(acts))):
                images.append(as_hwc_uint8(imgs[j], resize=args.resize))
                actions.append(np.asarray(acts[j], dtype=np.float32).reshape(-1))
                episodes.append(epi)
                domains.append(f"demo_mod_{epi % max(1, args.demo_mod)}" if args.domain_strategy == "demo_mod" else base_domain)
                tasks.append(args.input.stem)
                if args.max_samples and len(images) >= args.max_samples:
                    break
            if args.max_samples and len(images) >= args.max_samples:
                break
    labels = np.zeros(len(images), dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        images=np.stack(images),
        actions=np.stack(actions).astype(np.float32),
        labels=labels,
        domains=np.array(domains),
        episodes=np.array(episodes, dtype=np.int64),
        tasks=np.array(tasks),
    )
    print(f"Wrote {args.out} with {len(images)} samples")


if __name__ == "__main__":
    main()

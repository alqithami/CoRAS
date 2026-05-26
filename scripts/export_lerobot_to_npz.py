#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def as_uint8_image(x: Any, resize: int | None = None) -> np.ndarray:
    arr = _to_numpy(x)
    # Some datasets expose PIL images or nested arrays; np.asarray handles PIL.
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Cannot convert image with shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if resize and (arr.shape[0] != resize or arr.shape[1] != resize):
        arr = np.asarray(Image.fromarray(arr).resize((resize, resize), Image.BILINEAR), dtype=np.uint8)
    return arr


def scalar_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    arr = _to_numpy(v)
    if arr.size == 0:
        return default
    return int(arr.reshape(-1)[0])


def choose_key(row: dict, requested: str | None, kind: str) -> str:
    if requested and requested in row:
        return requested
    if requested and requested not in row:
        raise KeyError(f"Requested {kind} key {requested!r} not found. Available keys: {list(row.keys())}")
    candidates = []
    for k in row.keys():
        kl = k.lower()
        if kind == "image" and ("image" in kl or "camera" in kl):
            candidates.append(k)
        if kind == "action" and "action" in kl:
            candidates.append(k)
    if not candidates:
        raise KeyError(f"No {kind} key found. Available keys: {list(row.keys())}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LeRobot dataset to CoRAS NPZ.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--action-key", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--domain-strategy", choices=["constant", "task", "episode", "episode_mod"], default="constant")
    parser.add_argument("--episode-mod", type=int, default=10)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError("Install LeRobot first, e.g. pip install 'lerobot[video]' or see HF LeRobot installation docs.") from exc
    ds = LeRobotDataset(args.repo_id)
    first = dict(ds[0])
    camera_key = choose_key(first, args.camera, "image")
    action_key = choose_key(first, args.action_key, "action")
    print(f"Using camera_key={camera_key}; action_key={action_key}; len={len(ds)}")
    base_domain = args.domain or args.repo_id.replace("/", "_").replace(".", "_")
    images, actions, episodes, domains, tasks = [], [], [], [], []
    n_total = len(ds)
    max_samples = n_total if args.max_samples <= 0 else min(n_total, args.max_samples)
    i = 0
    while i < n_total and len(images) < max_samples:
        row = dict(ds[i])
        image = as_uint8_image(row[camera_key], resize=args.resize)
        action = _to_numpy(row[action_key]).astype(np.float32).reshape(-1)
        epi = scalar_int(row.get("episode_index"), default=i // 100)
        task = scalar_int(row.get("task_index"), default=0)
        if args.domain_strategy == "task":
            domain = f"task_{task}"
        elif args.domain_strategy == "episode":
            domain = f"episode_{epi}"
        elif args.domain_strategy == "episode_mod":
            domain = f"episode_mod_{epi % max(1, args.episode_mod)}"
        else:
            domain = base_domain
        images.append(image); actions.append(action); episodes.append(epi); domains.append(domain); tasks.append(f"task_{task}")
        i += max(1, args.stride)
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
    unique, counts = np.unique(np.array(domains), return_counts=True)
    print(f"Wrote {args.out} with {len(images)} samples")
    print(dict(zip(unique.tolist()[:30], counts.astype(int).tolist()[:30])))


if __name__ == "__main__":
    main()

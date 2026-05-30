#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def as_uint8_images(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 4:
        raise ValueError(f"Expected images with shape (N,H,W,C) or (N,C,H,W), got {arr.shape}")
    if arr.shape[1] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(img, dtype=np.uint8))


def apply_shift(img: np.ndarray, shift: str, rng: np.random.Generator) -> np.ndarray:
    im = pil(img)
    if shift == "clean":
        return np.asarray(im, dtype=np.uint8)
    if shift == "brightness_low":
        return np.asarray(ImageEnhance.Brightness(im).enhance(0.55), dtype=np.uint8)
    if shift == "brightness_high":
        return np.asarray(ImageEnhance.Brightness(im).enhance(1.45), dtype=np.uint8)
    if shift == "contrast_low":
        return np.asarray(ImageEnhance.Contrast(im).enhance(0.55), dtype=np.uint8)
    if shift == "contrast_high":
        return np.asarray(ImageEnhance.Contrast(im).enhance(1.65), dtype=np.uint8)
    if shift == "blur":
        return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=1.25)), dtype=np.uint8)
    if shift == "gaussian_noise":
        arr = np.asarray(im, dtype=np.float32)
        noise = rng.normal(0.0, 18.0, size=arr.shape).astype(np.float32)
        return np.clip(arr + noise, 0, 255).astype(np.uint8)
    if shift == "occlusion":
        arr = np.asarray(im, dtype=np.uint8).copy()
        h, w = arr.shape[:2]
        occ_h = max(8, int(0.22 * h))
        occ_w = max(8, int(0.22 * w))
        y0 = int(rng.integers(0, max(1, h - occ_h + 1)))
        x0 = int(rng.integers(0, max(1, w - occ_w + 1)))
        fill = int(np.median(arr))
        arr[y0:y0 + occ_h, x0:x0 + occ_w, :] = fill
        return arr
    if shift == "crop_shift":
        arr = np.asarray(im, dtype=np.uint8)
        h, w = arr.shape[:2]
        pad = max(4, int(0.08 * min(h, w)))
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        dy = int(rng.integers(-pad, pad + 1))
        dx = int(rng.integers(-pad, pad + 1))
        y0 = pad + dy
        x0 = pad + dx
        return padded[y0:y0 + h, x0:x0 + w, :].astype(np.uint8)
    raise ValueError(f"Unknown shift: {shift}")


def subset_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_samples and max_samples > 0 and max_samples < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
    return idx.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PushT visual-domain-shift suite from an exported CoRAS NPZ.")
    parser.add_argument("--input", type=Path, required=True, help="Usually data/pusht_tokens_k64.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-domain", type=str, default="pusht_clean")
    parser.add_argument("--shift", action="append", default=None, help="Repeatable. Defaults to a strong six-shift suite.")
    parser.add_argument("--max-base-samples", type=int, default=0, help="Optional cap before duplicating across shifts.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shifts = args.shift or ["brightness_low", "brightness_high", "contrast_low", "occlusion", "crop_shift", "gaussian_noise"]
    with np.load(args.input, allow_pickle=True) as data:
        images0 = as_uint8_images(data["images"])
        labels0 = np.asarray(data["labels"], dtype=np.int64)
        actions0 = np.asarray(data["actions"], dtype=np.float32) if "actions" in data else None
        episodes0 = np.asarray(data["episodes"], dtype=np.int64) if "episodes" in data else np.arange(len(labels0), dtype=np.int64)
        tasks0 = np.asarray(data["tasks"]).astype(str) if "tasks" in data else np.array(["task"] * len(labels0))
        unsafe0 = np.asarray(data["unsafe"]).astype(bool) if "unsafe" in data else None
        passthrough: Dict[str, np.ndarray] = {k: np.asarray(data[k]) for k in data.files if k not in {"images", "labels", "actions", "episodes", "tasks", "domains", "unsafe"}}

    base_idx = subset_indices(len(labels0), args.max_base_samples, args.seed)
    images0 = images0[base_idx]
    labels0 = labels0[base_idx]
    episodes0 = episodes0[base_idx]
    tasks0 = tasks0[base_idx]
    if actions0 is not None:
        actions0 = actions0[base_idx]
    if unsafe0 is not None:
        unsafe0 = unsafe0[base_idx]

    all_images: List[np.ndarray] = [images0]
    all_labels: List[np.ndarray] = [labels0]
    all_episodes: List[np.ndarray] = [episodes0]
    all_tasks: List[np.ndarray] = [tasks0]
    all_domains: List[np.ndarray] = [np.array([args.source_domain] * len(labels0))]
    all_actions: List[np.ndarray] = []
    all_unsafe: List[np.ndarray] = []
    if actions0 is not None:
        all_actions.append(actions0)
    if unsafe0 is not None:
        all_unsafe.append(unsafe0)

    for shift_i, shift in enumerate(shifts):
        rng = np.random.default_rng(args.seed + 1009 * (shift_i + 1))
        shifted = np.empty_like(images0)
        for i, img in enumerate(images0):
            shifted[i] = apply_shift(img, shift, rng)
        all_images.append(shifted)
        all_labels.append(labels0)
        # Preserve episode IDs so episode-block splitting keeps all shifted views of an episode together.
        all_episodes.append(episodes0)
        all_tasks.append(tasks0)
        all_domains.append(np.array([f"pusht_{shift}"] * len(labels0)))
        if actions0 is not None:
            all_actions.append(actions0)
        if unsafe0 is not None:
            all_unsafe.append(unsafe0)

    arrays: Dict[str, np.ndarray] = {
        "images": np.concatenate(all_images, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "episodes": np.concatenate(all_episodes, axis=0).astype(np.int64),
        "tasks": np.concatenate(all_tasks, axis=0).astype(str),
        "domains": np.concatenate(all_domains, axis=0).astype(str),
    }
    if actions0 is not None:
        arrays["actions"] = np.concatenate(all_actions, axis=0).astype(np.float32)
    if unsafe0 is not None:
        arrays["unsafe"] = np.concatenate(all_unsafe, axis=0).astype(bool)
    arrays.update(passthrough)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    uniq, counts = np.unique(arrays["domains"], return_counts=True)
    print(f"Wrote {args.out} with {len(arrays['labels'])} samples from {len(base_idx)} base frames")
    print(dict(zip(uniq.tolist(), counts.astype(int).tolist())))
    print("Target domains:", " ".join([f"pusht_{s}" for s in shifts]))


if __name__ == "__main__":
    main()

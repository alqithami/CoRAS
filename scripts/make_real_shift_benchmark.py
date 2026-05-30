#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


def as_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def transform_image(img: np.ndarray, shift: str, rng: np.random.Generator) -> np.ndarray:
    arr = as_uint8_hwc(img)
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr)

    if shift == "lighting":
        # Strong but plausible camera/light change: brightness + contrast + mild color drift.
        brightness = float(rng.uniform(0.55, 1.55))
        contrast = float(rng.uniform(0.65, 1.45))
        pil = ImageEnhance.Brightness(pil).enhance(brightness)
        pil = ImageEnhance.Contrast(pil).enhance(contrast)
        out = np.asarray(pil).astype(np.float32)
        channel_scale = rng.uniform(0.90, 1.10, size=(1, 1, 3)).astype(np.float32)
        out = np.clip(out * channel_scale, 0, 255).astype(np.uint8)
        return out

    if shift == "camera_crop":
        # Simulates camera re-centering/crop/zoom. Crop 72-90% then resize back.
        scale = float(rng.uniform(0.72, 0.90))
        ch, cw = max(2, int(h * scale)), max(2, int(w * scale))
        top = int(rng.integers(0, max(1, h - ch + 1)))
        left = int(rng.integers(0, max(1, w - cw + 1)))
        crop = pil.crop((left, top, left + cw, top + ch))
        return np.asarray(crop.resize((w, h), Image.BILINEAR), dtype=np.uint8)

    if shift == "occlusion":
        # Adds one or two opaque rectangles, common in cluttered real robot views.
        out = pil.copy()
        draw = ImageDraw.Draw(out)
        n_rect = int(rng.integers(1, 3))
        for _ in range(n_rect):
            rh = int(rng.integers(max(4, h // 10), max(5, h // 3)))
            rw = int(rng.integers(max(4, w // 10), max(5, w // 3)))
            y0 = int(rng.integers(0, max(1, h - rh + 1)))
            x0 = int(rng.integers(0, max(1, w - rw + 1)))
            color = tuple(int(c) for c in rng.integers(0, 256, size=3))
            draw.rectangle((x0, y0, x0 + rw, y0 + rh), fill=color)
        return np.asarray(out, dtype=np.uint8)

    if shift == "blur":
        radius = float(rng.uniform(0.8, 2.2))
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)

    if shift == "noise":
        sigma = float(rng.uniform(8.0, 22.0))
        noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if shift == "compression":
        # Downsample-upsample artifacts without requiring JPEG dependencies.
        scale = float(rng.uniform(0.35, 0.65))
        small = pil.resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.BILINEAR)
        return np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)

    raise ValueError(f"Unknown shift {shift!r}. Supported: lighting, camera_crop, occlusion, blur, noise, compression")


def choose_target_episodes(episodes: np.ndarray, target_frac: float, seed: int, min_target_episodes: int) -> set[int]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(episodes.astype(np.int64))
    rng.shuffle(uniq)
    n_target = max(min_target_episodes, int(round(float(target_frac) * len(uniq))))
    n_target = min(len(uniq) - 1 if len(uniq) > 1 else 1, n_target)
    return set(int(x) for x in uniq[:n_target])


def main() -> None:
    p = argparse.ArgumentParser(description="Build a real-frame controlled-shift benchmark from an existing CoRAS NPZ.")
    p.add_argument("--input", type=Path, required=True, help="Tokenized CoRAS NPZ with images, labels, episodes, domains.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-domain", type=str, default="source_clean")
    p.add_argument("--target-domain-prefix", type=str, default="target")
    p.add_argument("--shifts", nargs="+", default=["lighting", "camera_crop", "occlusion", "blur"])
    p.add_argument("--target-frac", type=float, default=0.35, help="Fraction of episodes reserved for shifted target domains.")
    p.add_argument("--min-target-episodes", type=int, default=8)
    p.add_argument("--max-source-samples", type=int, default=0, help="Optional cap on clean source samples.")
    p.add_argument("--max-target-samples-per-shift", type=int, default=0, help="Optional cap per shifted target domain.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with np.load(args.input, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    if "images" not in arrays or "labels" not in arrays:
        raise ValueError("Input must include images and labels arrays.")
    images = np.asarray(arrays["images"])
    labels = np.asarray(arrays["labels"]).astype(np.int64)
    n = len(labels)
    episodes = np.asarray(arrays.get("episodes", np.arange(n, dtype=np.int64))).astype(np.int64)
    tasks = np.asarray(arrays.get("tasks", np.array(["task"] * n))).astype(str)
    source_domains_orig = np.asarray(arrays.get("domains", np.array(["original"] * n))).astype(str)
    actions = np.asarray(arrays["actions"]).astype(np.float32) if "actions" in arrays else None
    unsafe = np.asarray(arrays["unsafe"]).astype(bool) if "unsafe" in arrays else np.zeros(n, dtype=bool)

    target_eps = choose_target_episodes(episodes, args.target_frac, args.seed, args.min_target_episodes)
    source_idx = np.array([i for i, e in enumerate(episodes) if int(e) not in target_eps], dtype=np.int64)
    target_idx = np.array([i for i, e in enumerate(episodes) if int(e) in target_eps], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    if args.max_source_samples and len(source_idx) > args.max_source_samples:
        source_idx = np.sort(rng.choice(source_idx, size=args.max_source_samples, replace=False))
    if len(source_idx) == 0 or len(target_idx) == 0:
        raise ValueError("Need nonempty source and target episode sets. Lower --target-frac or min-target-episodes.")

    out_images: list[np.ndarray] = []
    out_labels: list[int] = []
    out_domains: list[str] = []
    out_episodes: list[int] = []
    out_tasks: list[str] = []
    out_actions: list[np.ndarray] = []
    out_unsafe: list[bool] = []
    out_original_indices: list[int] = []
    out_shift_names: list[str] = []

    def append_row(i: int, img: np.ndarray, domain: str, shift: str) -> None:
        out_images.append(as_uint8_hwc(img))
        out_labels.append(int(labels[i]))
        out_domains.append(str(domain))
        out_episodes.append(int(episodes[i]))
        out_tasks.append(str(tasks[i]))
        if actions is not None:
            out_actions.append(np.asarray(actions[i], dtype=np.float32))
        out_unsafe.append(bool(unsafe[i]))
        out_original_indices.append(int(i))
        out_shift_names.append(str(shift))

    for i in source_idx:
        append_row(int(i), images[int(i)], args.source_domain, "clean")

    for shift in args.shifts:
        idx = target_idx.copy()
        if args.max_target_samples_per_shift and len(idx) > args.max_target_samples_per_shift:
            idx = np.sort(rng.choice(idx, size=args.max_target_samples_per_shift, replace=False))
        for i in idx:
            local_seed = (int(args.seed) * 1000003 + int(i) * 9176 + sum((j + 1) * ord(ch) for j, ch in enumerate(shift)) % 100000) % (2**32 - 1)
            local_rng = np.random.default_rng(local_seed)
            img = transform_image(images[int(i)], shift, local_rng)
            append_row(int(i), img, f"{args.target_domain_prefix}_{shift}", shift)

    out = {
        "images": np.stack(out_images).astype(np.uint8),
        "labels": np.asarray(out_labels, dtype=np.int64),
        "domains": np.asarray(out_domains),
        "episodes": np.asarray(out_episodes, dtype=np.int64),
        "tasks": np.asarray(out_tasks),
        "unsafe": np.asarray(out_unsafe, dtype=bool),
        "source_original_index": np.asarray(out_original_indices, dtype=np.int64),
        "shift_name": np.asarray(out_shift_names),
    }
    if actions is not None:
        out["actions"] = np.stack(out_actions).astype(np.float32)
    for k in ["codebook_centers", "codebook_quantization_mse"]:
        if k in arrays:
            out[k] = arrays[k]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    uniq, counts = np.unique(out["domains"].astype(str), return_counts=True)
    meta = {
        "input": str(args.input),
        "out": str(args.out),
        "seed": args.seed,
        "source_domain": args.source_domain,
        "target_episodes": sorted(target_eps),
        "source_samples": int(len(source_idx)),
        "target_original_samples": int(len(target_idx)),
        "shifts": args.shifts,
        "domain_counts": {str(u): int(c) for u, c in zip(uniq, counts)},
    }
    (args.out.with_suffix(".metadata.json")).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out} with {len(out['labels'])} samples")
    print(meta["domain_counts"])


if __name__ == "__main__":
    main()

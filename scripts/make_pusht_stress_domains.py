#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw


def _pil(img: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def transform_image(img: np.ndarray, domain: str, seed: int, out_size: int | None = None) -> np.ndarray:
    im = _pil(img).convert("RGB")
    w, h = im.size
    rng = np.random.default_rng(seed)
    if domain == "source_clean" or domain == "target_clean":
        pass
    elif domain == "target_lighting":
        factor = float(rng.uniform(0.45, 0.75))
        im = ImageEnhance.Brightness(im).enhance(factor)
        im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(1.15, 1.65)))
    elif domain == "target_crop":
        # Translation/camera crop with resize back to original size.
        margin = int(rng.integers(max(2, w // 20), max(3, w // 8)))
        left = int(rng.integers(0, margin + 1))
        top = int(rng.integers(0, margin + 1))
        right = w - int(rng.integers(0, margin + 1))
        bottom = h - int(rng.integers(0, margin + 1))
        if right <= left + 4 or bottom <= top + 4:
            left, top, right, bottom = 0, 0, w, h
        im = im.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
    elif domain == "target_blur":
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.8, 2.0))))
        im = ImageEnhance.Sharpness(im).enhance(0.65)
    elif domain == "target_noise":
        arr = np.asarray(im).astype(np.float32)
        sigma = float(rng.uniform(8.0, 18.0))
        arr = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    elif domain == "target_occlusion":
        draw = ImageDraw.Draw(im)
        occ_w = int(rng.integers(max(4, w // 10), max(6, w // 4)))
        occ_h = int(rng.integers(max(4, h // 10), max(6, h // 4)))
        x0 = int(rng.integers(0, max(1, w - occ_w)))
        y0 = int(rng.integers(0, max(1, h - occ_h)))
        color = tuple(int(x) for x in rng.integers(0, 40, size=3))
        draw.rectangle((x0, y0, x0 + occ_w, y0 + occ_h), fill=color)
    elif domain == "target_hard":
        # Compose several deployment-style perturbations.
        im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.55, 0.80)))
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.2))))
        margin = int(rng.integers(max(2, w // 24), max(3, w // 10)))
        im = im.crop((margin, margin, w - margin, h - margin)).resize((w, h), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32)
        arr = np.clip(arr + rng.normal(0.0, float(rng.uniform(4.0, 12.0)), size=arr.shape), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
    else:
        raise ValueError(f"Unknown stress domain {domain!r}")
    if out_size and im.size != (out_size, out_size):
        im = im.resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def parse_domains(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a real-frame PushT stress/domain-shift NPZ for stronger CoRAS experiments.")
    parser.add_argument("--input", type=Path, required=True, help="Tokenized PushT NPZ, e.g. data/pusht_tokens_k64.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=96)
    parser.add_argument("--source-episode-frac", type=float, default=0.60)
    parser.add_argument("--max-base-samples", type=int, default=0, help="Optional subsample before stress generation; 0 = all samples")
    parser.add_argument("--target-domains", type=str, default="target_lighting,target_crop,target_blur,target_noise,target_occlusion,target_hard")
    parser.add_argument("--mode", choices=["single", "duplicate_target"], default="single", help="single keeps dataset size near original; duplicate_target gives every target frame every target shift")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with np.load(args.input, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    required = {"images", "labels", "episodes"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"{args.input} missing required arrays: {missing}")
    images = np.asarray(arrays["images"])
    labels = np.asarray(arrays["labels"]).astype(np.int64)
    episodes = np.asarray(arrays["episodes"]).astype(np.int64)
    domains_in = np.asarray(arrays.get("domains", np.array(["raw"] * len(labels)))).astype(str)
    tasks_in = np.asarray(arrays.get("tasks", np.array(["task"] * len(labels)))).astype(str)
    actions = np.asarray(arrays["actions"]) if "actions" in arrays else None

    base_idx = np.arange(len(labels))
    if args.max_base_samples and args.max_base_samples > 0 and args.max_base_samples < len(base_idx):
        # Episode-aware subsample: choose episodes until approximately max_base_samples are included.
        eps = np.unique(episodes)
        rng.shuffle(eps)
        chosen = []
        total = 0
        for ep in eps:
            idx = np.where(episodes == ep)[0]
            chosen.append(ep)
            total += len(idx)
            if total >= args.max_base_samples:
                break
        chosen_set = set(int(e) for e in chosen)
        base_idx = np.array([i for i in base_idx if int(episodes[i]) in chosen_set], dtype=np.int64)

    eps = np.unique(episodes[base_idx])
    rng.shuffle(eps)
    n_source = max(1, int(round(float(args.source_episode_frac) * len(eps))))
    source_eps = set(int(e) for e in eps[:n_source])
    target_domains = parse_domains(args.target_domains)
    out_images, out_labels, out_eps, out_domains, out_tasks, out_actions, out_base_idx = [], [], [], [], [], [], []
    out_shift = []

    for pos, i in enumerate(base_idx):
        ep = int(episodes[i])
        if ep in source_eps:
            doms = ["source_clean"]
        else:
            if args.mode == "duplicate_target":
                doms = ["target_clean"] + target_domains
            else:
                # Assign one deterministic target stress domain per frame. This avoids
                # same-frame leakage while keeping the public-data run lightweight.
                doms = [target_domains[int((ep * 1315423911 + i + args.seed) % len(target_domains))]]
        for d in doms:
            img = transform_image(images[i], d, seed=int(i + 10007 * (len(out_images) + 1) + args.seed), out_size=args.resize)
            out_images.append(img)
            out_labels.append(int(labels[i]))
            out_eps.append(ep)
            out_domains.append(d)
            out_tasks.append(str(tasks_in[i]))
            out_base_idx.append(int(i))
            out_shift.append(d.replace("target_", "").replace("source_", ""))
            if actions is not None:
                out_actions.append(actions[i])

    out = {
        "images": np.stack(out_images).astype(np.uint8),
        "labels": np.asarray(out_labels, dtype=np.int64),
        "domains": np.asarray(out_domains),
        "episodes": np.asarray(out_eps, dtype=np.int64),
        "tasks": np.asarray(out_tasks),
        "base_index": np.asarray(out_base_idx, dtype=np.int64),
        "stress_type": np.asarray(out_shift),
    }
    if out_actions:
        out["actions"] = np.stack(out_actions).astype(np.float32)
    for k in ["codebook_centers", "codebook_quantization_mse"]:
        if k in arrays:
            out[k] = arrays[k]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    uniq, counts = np.unique(out["domains"].astype(str), return_counts=True)
    print(f"Wrote {args.out} with {len(out['labels'])} samples from {len(base_idx)} base samples and {len(eps)} episodes")
    print("Domain counts:", dict(zip(uniq.tolist(), counts.astype(int).tolist())))
    print("Mode:", args.mode, "source_episode_frac:", args.source_episode_frac)


if __name__ == "__main__":
    main()

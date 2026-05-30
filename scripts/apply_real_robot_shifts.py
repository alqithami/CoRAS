#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def as_uint8_hwc(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def apply_shift(img: np.ndarray, mode: str, rng: np.random.Generator, severity: float) -> np.ndarray:
    arr = as_uint8_hwc(img)
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr)
    sev = float(severity)

    if mode == 'clean':
        return arr
    if mode == 'lighting':
        factor = float(rng.uniform(max(0.25, 1.0 - 0.9 * sev), 1.0 + 0.6 * sev))
        pil = ImageEnhance.Brightness(pil).enhance(factor)
        factor_c = float(rng.uniform(max(0.35, 1.0 - 0.6 * sev), 1.0 + 0.5 * sev))
        pil = ImageEnhance.Contrast(pil).enhance(factor_c)
        return np.asarray(pil, dtype=np.uint8)
    if mode == 'camera_crop':
        # Simulate slight camera re-centering and zoom. Resize back to original size.
        max_margin = max(1, int(round(min(h, w) * 0.18 * sev)))
        left = int(rng.integers(0, max_margin + 1))
        top = int(rng.integers(0, max_margin + 1))
        right = w - int(rng.integers(0, max_margin + 1))
        bottom = h - int(rng.integers(0, max_margin + 1))
        right = max(right, left + 4); bottom = max(bottom, top + 4)
        return np.asarray(pil.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    if mode == 'occlusion':
        out = arr.copy()
        side = int(round(min(h, w) * rng.uniform(0.10, 0.28) * sev))
        side = max(4, min(side, min(h, w) // 2))
        x0 = int(rng.integers(0, max(1, w - side)))
        y0 = int(rng.integers(0, max(1, h - side)))
        fill = np.array([int(x) for x in rng.integers(0, 256, size=3)], dtype=np.uint8)
        out[y0:y0 + side, x0:x0 + side] = fill
        return out
    if mode == 'blur_noise':
        radius = float(rng.uniform(0.25, 1.25) * sev)
        arr2 = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        sigma = float(rng.uniform(3.0, 16.0) * sev)
        arr2 += rng.normal(0.0, sigma, size=arr2.shape)
        return np.clip(arr2, 0, 255).astype(np.uint8)
    if mode == 'color_jitter':
        # Works for RGB images; for mostly grayscale frames this still changes sensor statistics.
        for enhancer_cls, lo, hi in [
            (ImageEnhance.Color, 1.0 - 0.7 * sev, 1.0 + 0.7 * sev),
            (ImageEnhance.Sharpness, 1.0 - 0.8 * sev, 1.0 + 1.0 * sev),
            (ImageEnhance.Contrast, 1.0 - 0.5 * sev, 1.0 + 0.7 * sev),
        ]:
            pil = enhancer_cls(pil).enhance(float(rng.uniform(max(0.1, lo), hi)))
        return np.asarray(pil, dtype=np.uint8)
    raise ValueError(f'Unknown shift mode: {mode}')


def choose_target_episodes(episodes: np.ndarray, frac: float, seed: int, min_eps: int = 6) -> set[int]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(episodes.astype(int))
    rng.shuffle(uniq)
    n = max(min_eps, int(round(len(uniq) * frac)))
    n = min(len(uniq), max(1, n))
    return set(int(x) for x in uniq[:n])


def main() -> None:
    ap = argparse.ArgumentParser(description='Create real-robot visual-shift domains from an existing CoRAS NPZ dataset.')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', nargs='+', default=['lighting', 'camera_crop', 'occlusion', 'blur_noise'])
    ap.add_argument('--target-frac', type=float, default=0.40, help='Fraction of episodes converted into shifted target domains.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--severity', type=float, default=0.85)
    ap.add_argument('--copy-target-for-each-mode', action='store_true', help='If set, duplicate target episodes once per mode. Otherwise assign one mode per target episode.')
    ap.add_argument('--source-domain-name', type=str, default='real_clean_source')
    ap.add_argument('--target-prefix', type=str, default='real_shift')
    args = ap.parse_args()

    with np.load(args.input, allow_pickle=True) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    images = np.asarray(arrays['images'])
    labels = np.asarray(arrays['labels'])
    n = len(labels)
    episodes = np.asarray(arrays.get('episodes', np.arange(n) // 100)).astype(int)
    tasks = np.asarray(arrays.get('tasks', np.array(['task'] * n))).astype(str)
    target_eps = choose_target_episodes(episodes, args.target_frac, args.seed)
    rng = np.random.default_rng(args.seed)

    out_arrays: dict[str, list] = {k: [] for k in arrays.keys() if k != 'metadata'}
    # Ensure required optional arrays exist in output.
    for k in ['domains', 'episodes', 'tasks']:
        if k not in out_arrays:
            out_arrays[k] = []

    mode_list = [m for m in args.modes if m != 'clean']
    if not mode_list:
        raise ValueError('At least one non-clean mode is required')
    episode_to_mode = {ep: mode_list[i % len(mode_list)] for i, ep in enumerate(sorted(target_eps))}

    def append_row(i: int, img: np.ndarray, domain: str, episode: int, task: str) -> None:
        for k, v in arrays.items():
            if k == 'metadata':
                continue
            if k == 'images':
                out_arrays[k].append(img)
            elif k == 'domains':
                out_arrays[k].append(domain)
            elif k == 'episodes':
                out_arrays[k].append(int(episode))
            elif k == 'tasks':
                out_arrays[k].append(str(task))
            else:
                out_arrays[k].append(v[i])
        if 'domains' not in arrays:
            out_arrays['domains'].append(domain)
        if 'episodes' not in arrays:
            out_arrays['episodes'].append(int(episode))
        if 'tasks' not in arrays:
            out_arrays['tasks'].append(str(task))

    for i in range(n):
        ep = int(episodes[i])
        task = str(tasks[i]) if len(tasks) == n else 'task'
        if ep not in target_eps:
            append_row(i, as_uint8_hwc(images[i]), args.source_domain_name, ep, task)
        else:
            if args.copy_target_for_each_mode:
                for j, mode in enumerate(mode_list):
                    shifted = apply_shift(images[i], mode, rng, args.severity)
                    # Offset episode ids per mode so episode-block splits remain independent.
                    ep2 = ep + (j + 1) * 10_000_000
                    append_row(i, shifted, f'{args.target_prefix}_{mode}', ep2, task)
            else:
                mode = episode_to_mode[ep]
                shifted = apply_shift(images[i], mode, rng, args.severity)
                append_row(i, shifted, f'{args.target_prefix}_{mode}', ep, task)

    final = {}
    for k, vals in out_arrays.items():
        if k in {'domains', 'tasks'}:
            final[k] = np.asarray(vals).astype(str)
        elif k == 'episodes':
            final[k] = np.asarray(vals, dtype=np.int64)
        elif k == 'images':
            final[k] = np.stack(vals).astype(np.uint8)
        else:
            final[k] = np.asarray(vals)
    final['shift_metadata'] = np.array([f'modes={mode_list};target_frac={args.target_frac};severity={args.severity};seed={args.seed}'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **final)
    uniq, counts = np.unique(final['domains'].astype(str), return_counts=True)
    print(f'Wrote {args.out} with {len(final["labels"])} samples from {n} input samples')
    print(dict(zip(uniq.tolist(), counts.astype(int).tolist())))

if __name__ == '__main__':
    main()

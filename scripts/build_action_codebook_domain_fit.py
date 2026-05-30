#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from sklearn.cluster import KMeans


def _filter_row_arrays(arrays: dict, keep: np.ndarray, original_n: int) -> dict:
    out = {}
    for key, val in arrays.items():
        try:
            if hasattr(val, "shape") and len(val.shape) > 0 and val.shape[0] == original_n:
                out[key] = val[keep]
            else:
                out[key] = val
        except TypeError:
            out[key] = val
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build an action-token codebook while fitting KMeans only on selected source domains. "
            "KMeans is fit in standardized action coordinates and centers are stored back in original action units."
        )
    )
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--num-codes', type=int, default=64)
    ap.add_argument('--fit-domain-regex', type=str, default='.*clean.*|.*source.*',
                    help='Regex over domains used to fit codebook. Labels are assigned for all samples.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-fit-samples', type=int, default=200000)
    args = ap.parse_args()

    with np.load(args.input, allow_pickle=True) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files if k != 'labels'}
        if 'actions' not in data:
            raise KeyError(f"{args.input} does not contain an 'actions' array")
        actions = np.asarray(data['actions'], dtype=np.float32)
        domains = np.asarray(data['domains']).astype(str) if 'domains' in data else np.array(['default'] * len(actions))

    if actions.ndim == 1:
        actions = actions[:, None]
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be 1D or 2D, got shape {actions.shape}")

    original_n = len(actions)
    finite = np.isfinite(actions).all(axis=1)
    if not finite.all():
        n_bad = int((~finite).sum())
        print(f'[warn] Dropping {n_bad} rows with non-finite actions before codebook fitting/labeling.')
        keep = np.where(finite)[0]
        actions = actions[keep]
        domains = domains[keep]
        arrays = _filter_row_arrays(arrays, keep, original_n)

    if len(actions) == 0:
        raise ValueError('No finite action rows remain after filtering.')

    mask = np.array([re.match(args.fit_domain_regex, d) is not None for d in domains], dtype=bool)
    fit_idx = np.where(mask)[0]
    if len(fit_idx) < max(2, args.num_codes):
        print(f'[warn] Only {len(fit_idx)} samples matched {args.fit_domain_regex!r}; falling back to all actions for codebook fit.')
        fit_idx = np.arange(len(actions))

    rng = np.random.default_rng(args.seed)
    if len(fit_idx) > args.max_fit_samples:
        fit_idx = rng.choice(fit_idx, size=args.max_fit_samples, replace=False)

    fit_actions = actions[fit_idx].astype(np.float32)
    mean = fit_actions.mean(axis=0, keepdims=True).astype(np.float32)
    scale = fit_actions.std(axis=0, keepdims=True).astype(np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale < 1e-6] = 1.0

    z_actions = ((actions - mean) / scale).astype(np.float32)
    z_actions = np.nan_to_num(z_actions, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(int(args.num_codes), len(fit_idx))
    if k < 2:
        raise ValueError(f'Need at least 2 samples for KMeans, got {len(fit_idx)}')

    km = KMeans(n_clusters=k, random_state=int(args.seed), n_init=10, verbose=0)
    km.fit(z_actions[fit_idx])
    labels = km.predict(z_actions).astype(np.int64)
    centers_z = km.cluster_centers_.astype(np.float32)
    centers = (centers_z * scale + mean).astype(np.float32)

    # Diagnostics in original robot action coordinates.
    diff_all = actions - centers[labels]
    mse_all = float(np.mean(diff_all ** 2))
    fit_labels = km.predict(z_actions[fit_idx]).astype(np.int64)
    diff_fit = actions[fit_idx] - centers[fit_labels]
    mse_fit = float(np.mean(diff_fit ** 2))

    arrays['actions'] = actions.astype(np.float32)
    arrays['labels'] = labels
    arrays['codebook_centers'] = centers
    arrays['codebook_centers_standardized'] = centers_z
    arrays['codebook_action_mean'] = mean.squeeze(0).astype(np.float32)
    arrays['codebook_action_scale'] = scale.squeeze(0).astype(np.float32)
    arrays['codebook_quantization_mse'] = np.array([mse_all], dtype=np.float32)
    arrays['codebook_fit_quantization_mse'] = np.array([mse_fit], dtype=np.float32)
    arrays['codebook_fit_domain_regex'] = np.array([args.fit_domain_regex])
    arrays['codebook_standardized_fit'] = np.array([1], dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'Wrote {args.out}; k={k}; fit_n={len(fit_idx)}; mse_all={mse_all:.6g}; mse_fit={mse_fit:.6g}; standardized=1')


if __name__ == '__main__':
    main()

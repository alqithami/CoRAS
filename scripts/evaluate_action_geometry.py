#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


def parse_alpha_from_name(path: Path) -> str:
    m = re.search(r'alpha([0-9]+\.[0-9]+)', path.name)
    return m.group(1) if m else 'unknown'


def action_geometry_for_file(pred_path: Path, data_path: Path) -> dict:
    pred = pd.read_csv(pred_path)
    with np.load(data_path, allow_pickle=True) as data:
        actions = np.asarray(data['actions'], dtype=np.float32) if 'actions' in data else None
        centers = np.asarray(data['codebook_centers'], dtype=np.float32) if 'codebook_centers' in data else None
    if actions is None or centers is None:
        raise ValueError(f'{data_path} must contain actions and codebook_centers for action-geometry diagnostics.')
    in_cols = sorted([c for c in pred.columns if c.startswith('in_set_')], key=lambda c: int(c.split('_')[-1]))
    if not in_cols:
        raise ValueError(f'{pred_path} has no in_set_* columns. Re-run evaluate_methods.py after v3 patch.')
    idx = pred['index'].to_numpy(dtype=int)
    y = pred['label'].to_numpy(dtype=int)
    top1 = pred['top1'].to_numpy(dtype=int)
    set_mask = pred[in_cols].to_numpy(dtype=bool)
    true_actions = actions[idx]
    label_centers = centers[y]
    top1_centers = centers[top1]

    # Per-sample nearest action center inside each prediction set.
    best_l2 = np.empty(len(pred), dtype=np.float64)
    set_diam = np.empty(len(pred), dtype=np.float64)
    for i, mask in enumerate(set_mask):
        c = centers[mask]
        if len(c) == 0:
            best_l2[i] = np.inf
            set_diam[i] = 0.0
        else:
            d = np.linalg.norm(c - true_actions[i][None, :], axis=1)
            best_l2[i] = float(np.min(d))
            if len(c) <= 1:
                set_diam[i] = 0.0
            else:
                # O(k^2) on at most 128 centers per row; acceptable for diagnostics.
                dif = c[:, None, :] - c[None, :, :]
                set_diam[i] = float(np.max(np.linalg.norm(dif, axis=-1)))

    top1_l2 = np.linalg.norm(top1_centers - true_actions, axis=1)
    quant_l2 = np.linalg.norm(label_centers - true_actions, axis=1)
    finite_best = np.isfinite(best_l2)
    covered = pred['covered'].astype(bool).to_numpy() if 'covered' in pred else set_mask[np.arange(len(y)), y]
    out = {
        'method': str(pred['method'].iloc[0]) if 'method' in pred else pred_path.stem,
        'alpha': float(pred['alpha'].iloc[0]) if 'alpha' in pred else np.nan,
        'n': int(len(pred)),
        'token_coverage': float(np.mean(covered)),
        'mean_set_size': float(np.mean(set_mask.sum(axis=1))),
        'mean_codebook_quant_l2': float(np.mean(quant_l2)),
        'p90_codebook_quant_l2': float(np.quantile(quant_l2, 0.90)),
        'top1_action_l2_mean': float(np.mean(top1_l2)),
        'top1_action_l2_p90': float(np.quantile(top1_l2, 0.90)),
        'set_oracle_action_l2_mean': float(np.mean(best_l2[finite_best])) if finite_best.any() else np.nan,
        'set_oracle_action_l2_p90': float(np.quantile(best_l2[finite_best], 0.90)) if finite_best.any() else np.nan,
        'action_l2_improvement_vs_top1': float(np.mean(top1_l2 - best_l2)) if finite_best.any() else np.nan,
        'mean_set_action_diameter': float(np.mean(set_diam)),
        'p90_set_action_diameter': float(np.quantile(set_diam, 0.90)),
        'empty_set_rate': float(np.mean(set_mask.sum(axis=1) == 0)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute continuous-action geometry diagnostics from saved prediction sets.')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--alpha', type=str, default=None, help='Filter alpha string, e.g. 0.10')
    args = ap.parse_args()
    run_dir = args.run_dir
    if args.data is None:
        import yaml
        cfg_files = sorted((run_dir.parent / 'matrix_configs').glob(f'config_{run_dir.name.replace("_", "_")}.yaml'))
        # More robust: load split config from train metadata if possible is not available, so require --data for ambiguity.
        raise SystemExit('Please pass --data path/to/dataset_tokens.npz')
    files = sorted(run_dir.glob('prediction_sets_*_alpha*.csv'))
    rows = []
    for f in files:
        if args.alpha and f'alpha{float(args.alpha):.2f}' not in f.name:
            continue
        # Skip compact files without membership columns.
        if 'compact' in f.name:
            continue
        try:
            row = action_geometry_for_file(f, args.data)
            row['file'] = f.name
            rows.append(row)
        except Exception as exc:
            print(f'[skip] {f}: {exc}')
    if not rows:
        raise SystemExit(f'No action-geometry rows produced in {run_dir}')
    df = pd.DataFrame(rows).sort_values(['alpha', 'method'])
    out = run_dir / 'action_geometry_metrics.csv'
    df.to_csv(out, index=False)
    print(f'Wrote {out}')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

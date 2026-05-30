#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def block_bootstrap_coverage(df: pd.DataFrame, seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    episodes = df['episode'].unique()
    by_ep = {ep: df[df['episode'] == ep]['covered'].astype(float).to_numpy() for ep in episodes}
    vals = []
    for _ in range(n_boot):
        eps = rng.choice(episodes, size=len(episodes), replace=True)
        arr = np.concatenate([by_ep[ep] for ep in eps])
        vals.append(float(arr.mean()))
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute episode-block diagnostics from compact prediction CSVs.')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--alpha', type=str, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rows = []
    for f in sorted(args.run_dir.glob('prediction_sets_compact_alpha*.csv')):
        if args.alpha and f'alpha{float(args.alpha):.2f}' not in f.name:
            continue
        df = pd.read_csv(f)
        if 'episode' not in df.columns:
            # Backward compatible fallback for old runs; new v3 runs include true episode ids.
            df['episode'] = (df['index'].to_numpy(dtype=int) // 100).astype(int)
        for method, sub in df.groupby('method'):
            sub = sub.copy()
            sub['covered'] = sub['covered'].astype(float)
            ep = sub.groupby('episode').agg(
                episode_len=('covered', 'size'),
                coverage=('covered', 'mean'),
                all_steps_covered=('covered', 'min'),
                mean_set_size=('set_size', 'mean'),
                first_index=('index', 'min'),
            ).reset_index()
            boot_mean, boot_lo, boot_hi = block_bootstrap_coverage(sub, seed=args.seed)
            rows.append({
                'alpha': float(sub['alpha'].iloc[0]) if 'alpha' in sub else np.nan,
                'method': method,
                'n_steps': int(len(sub)),
                'n_episodes': int(ep.shape[0]),
                'step_coverage': float(sub['covered'].mean()),
                'episode_mean_coverage': float(ep['coverage'].mean()),
                'episode_median_coverage': float(ep['coverage'].median()),
                'episode_all_steps_covered_rate': float(ep['all_steps_covered'].mean()),
                'mean_set_size': float(sub['set_size'].mean()),
                'episode_block_boot_mean': boot_mean,
                'episode_block_boot_lo95': boot_lo,
                'episode_block_boot_hi95': boot_hi,
                'file': f.name,
            })
    if not rows:
        raise SystemExit(f'No compact prediction files found in {args.run_dir}')
    out = args.run_dir / 'episode_metrics.csv'
    df = pd.DataFrame(rows).sort_values(['alpha', 'method'])
    df.to_csv(out, index=False)
    print(f'Wrote {out}')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

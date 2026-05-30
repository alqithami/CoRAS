#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


def read_all(root: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob(f'seed*_calib*/{pattern}')):
        try:
            df = pd.read_csv(p)
            df['run_dir'] = p.parent.name
            frames.append(df)
        except Exception as exc:
            print(f'[skip] {p}: {exc}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        row['n_runs'] = int(sub['run_dir'].nunique()) if 'run_dir' in sub else int(len(sub))
        for m in metric_cols:
            if m in sub.columns:
                vals = pd.to_numeric(sub[m], errors='coerce').to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    row[f'{m}_mean'] = float(vals.mean())
                    row[f'{m}_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                else:
                    row[f'{m}_mean'] = np.nan
                    row[f'{m}_std'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_latex_table(df: pd.DataFrame, out: Path, cols: list[str]) -> None:
    if df.empty:
        return
    lines = []
    lines.append('\\begin{tabular}{lrrrrr}')
    lines.append('\\toprule')
    lines.append('Method & Coverage & Gap & Set size & Top-1 & Action L2 \\\\')
    lines.append('\\midrule')
    for _, r in df.iterrows():
        method = str(r['method']).replace('_', '\\_')
        cov = r.get('coverage_mean', np.nan)
        gap = r.get('abs_coverage_gap_mean', np.nan)
        size = r.get('mean_set_size_mean', np.nan)
        top1 = r.get('top1_accuracy_mean', np.nan)
        al2 = r.get('set_oracle_action_l2_mean_mean', np.nan)
        lines.append(f'{method} & {cov:.3f} & {gap:.3f} & {size:.2f} & {top1:.3f} & {al2:.3f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    out.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description='Build publication aggregate tables from v3 diagnostics.')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--alpha', type=float, default=0.10)
    args = ap.parse_args()
    root = args.root
    agg_dir = root / 'paper_grade_aggregate'
    agg_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_all(root, f'metrics_summary_alpha{args.alpha:.2f}.csv')
    domains = read_all(root, f'metrics_by_domain_alpha{args.alpha:.2f}.csv')
    action = read_all(root, 'action_geometry_metrics.csv')
    episode = read_all(root, 'episode_metrics.csv')

    if not metrics.empty:
        metrics = metrics[np.isclose(pd.to_numeric(metrics['alpha'], errors='coerce'), args.alpha)]
        main = summarize(metrics, ['method'], ['coverage', 'abs_coverage_gap', 'mean_set_size', 'median_set_size', 'p90_set_size', 'top1_accuracy', 'fail_to_abstain_rate', 'ece_top1'])
    else:
        main = pd.DataFrame()
    if not action.empty:
        action = action[np.isclose(pd.to_numeric(action['alpha'], errors='coerce'), args.alpha)]
        act = summarize(action, ['method'], ['top1_action_l2_mean', 'set_oracle_action_l2_mean', 'action_l2_improvement_vs_top1', 'mean_set_action_diameter'])
        main = main.merge(act, on='method', how='left') if not main.empty else act
    if not episode.empty:
        episode = episode[np.isclose(pd.to_numeric(episode['alpha'], errors='coerce'), args.alpha)]
        ep = summarize(episode, ['method'], ['episode_mean_coverage', 'episode_all_steps_covered_rate', 'episode_block_boot_lo95', 'episode_block_boot_hi95'])
        main = main.merge(ep, on='method', how='left') if not main.empty else ep

    if not main.empty:
        main = main.sort_values(['abs_coverage_gap_mean', 'mean_set_size_mean'], na_position='last')
        main.to_csv(agg_dir / f'paper_main_table_alpha{args.alpha:.2f}.csv', index=False)
        make_latex_table(main, agg_dir / f'paper_main_table_alpha{args.alpha:.2f}.tex', [])
        print(main.to_string(index=False))
    else:
        print('No main metrics found')

    if not domains.empty:
        domains = domains[np.isclose(pd.to_numeric(domains['alpha'], errors='coerce'), args.alpha)]
        dom = summarize(domains, ['method', 'domain'], ['coverage', 'abs_coverage_gap', 'mean_set_size', 'top1_accuracy', 'fail_to_abstain_rate'])
        dom.to_csv(agg_dir / f'per_domain_table_alpha{args.alpha:.2f}.csv', index=False)
        print(f'Wrote {agg_dir / f"per_domain_table_alpha{args.alpha:.2f}.csv"}')
    if not action.empty:
        action.to_csv(agg_dir / f'action_geometry_all_alpha{args.alpha:.2f}.csv', index=False)
    if not episode.empty:
        episode.to_csv(agg_dir / f'episode_metrics_all_alpha{args.alpha:.2f}.csv', index=False)

if __name__ == '__main__':
    main()

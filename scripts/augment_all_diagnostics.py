#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description='Add action-geometry and episode-block diagnostics to every matrix run directory.')
    ap.add_argument('--root', type=Path, required=True, help='Experiment root, e.g. results/pusht_shift_k64')
    ap.add_argument('--alpha', type=str, default=None)
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    cfg_dir = args.root / 'matrix_configs'
    cfgs = {p.stem.replace('config_', ''): yaml.safe_load(p.read_text()) for p in cfg_dir.glob('config_seed*_calib*.yaml')}
    n = 0
    for run_dir in sorted(args.root.glob('seed*_calib*')):
        if not run_dir.is_dir():
            continue
        key = run_dir.name
        cfg = cfgs.get(key)
        if cfg is None:
            print(f'[skip] no config for {run_dir}')
            continue
        data_path = Path(cfg['data_path'])
        cmds = [
            [sys.executable, str(project / 'scripts' / 'evaluate_episode_metrics.py'), '--run-dir', str(run_dir)],
            [sys.executable, str(project / 'scripts' / 'evaluate_action_geometry.py'), '--run-dir', str(run_dir), '--data', str(data_path)],
        ]
        if args.alpha:
            cmds[0] += ['--alpha', args.alpha]
            cmds[1] += ['--alpha', args.alpha]
        for cmd in cmds:
            print('+', ' '.join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(project), check=True)
        n += 1
    print(f'Augmented {n} run directories under {args.root}')

if __name__ == '__main__':
    main()

#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description='Summarize an exported AutoEval NPZ.')
ap.add_argument('--data', type=Path, default=Path('data/autoeval_online_raw.npz'))
args = ap.parse_args()

z = np.load(args.data, allow_pickle=True)
print('path:', args.data)
print('keys:', sorted(z.files))
print('n:', len(z['images']))
print('image_shape:', z['images'].shape)
print('action_shape:', z['actions'].shape)
for key in ['domains', 'episodes', 'tasks', 'success', 'eval_ids', 'action_sources']:
    if key in z.files:
        arr = z[key]
        if key in ['domains', 'tasks', 'eval_ids', 'action_sources']:
            vals, counts = np.unique(arr.astype(str), return_counts=True)
            print(key, dict(zip(vals[:20].tolist(), counts[:20].astype(int).tolist())), 'unique=', len(vals))
        elif key == 'success':
            vals, counts = np.unique(arr[np.isfinite(arr)], return_counts=True)
            print(key, 'finite_n=', int(np.isfinite(arr).sum()), 'mean=', float(np.nanmean(arr)) if np.isfinite(arr).any() else 'nan', 'values=', dict(zip(vals.tolist(), counts.astype(int).tolist())))
        else:
            print(key, 'unique=', len(np.unique(arr)))

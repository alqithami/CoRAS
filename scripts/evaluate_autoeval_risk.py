#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def auc_binary(y_true, scores):
    y_raw = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y_raw)
    y = y_raw[mask].astype(int)
    s = s[mask]
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    for val in np.unique(s):
        idx = np.where(s == val)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser(description="Risk-correlation diagnostics for AutoEval online logs.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--alpha", type=str, default="0.10")
    ap.add_argument("--method", default="coras")
    args = ap.parse_args()

    pred = args.run_dir / f"prediction_sets_{args.method}_alpha{args.alpha}.csv"
    if not pred.exists():
        pred = args.run_dir / f"prediction_sets_compact_alpha{args.alpha}.csv"
    if not pred.exists():
        raise FileNotFoundError(f"No prediction set CSV found in {args.run_dir} for alpha {args.alpha}")
    df = pd.read_csv(pred)
    data = np.load(args.data, allow_pickle=True)
    if "success" not in data:
        print("No success field in data; skipping AutoEval risk diagnostics.")
        return
    success = data["success"].astype(float)
    episodes = data["episodes"].astype(int) if "episodes" in data else np.arange(len(success))

    n = min(len(df), len(success))
    df = df.iloc[:n].copy()
    success = success[:n]
    episodes = episodes[:n]
    valid_success = np.isfinite(success)
    if not valid_success.any():
        print("All success labels are NaN; skipping AutoEval risk diagnostics.")
        return

    set_size_col = "set_size" if "set_size" in df.columns else None
    if set_size_col is None:
        candidates = [c for c in df.columns if "set" in c.lower() and "size" in c.lower()]
        if candidates:
            set_size_col = candidates[0]
    if set_size_col is None:
        raise ValueError(f"Could not find set-size column in {pred}; columns={list(df.columns)}")

    df["success"] = success
    df["failure"] = 1.0 - success
    df["episode"] = episodes
    df["uncertainty"] = df[set_size_col].astype(float)
    df_valid = df[np.isfinite(df["success"].values)].copy()

    frame_auc = auc_binary(df_valid["failure"].values, df_valid["uncertainty"].values)
    ep = df_valid.groupby("episode").agg(
        success=("success", "mean"),
        failure=("failure", "mean"),
        mean_uncertainty=("uncertainty", "mean"),
        max_uncertainty=("uncertainty", "max"),
        n_steps=("uncertainty", "size"),
    ).reset_index()
    ep_failure = (ep["success"].values < 0.5).astype(int)
    ep_auc_mean = auc_binary(ep_failure, ep["mean_uncertainty"].values)
    ep_auc_max = auc_binary(ep_failure, ep["max_uncertainty"].values)

    out_dir = args.run_dir
    out = pd.DataFrame([{
        "method": args.method,
        "alpha": float(args.alpha),
        "n_frames": int(len(df_valid)),
        "n_episodes": int(len(ep)),
        "failure_rate_frame": float(df_valid["failure"].mean()),
        "failure_rate_episode": float(ep_failure.mean()) if len(ep_failure) else float('nan'),
        "frame_failure_auc_set_size": frame_auc,
        "episode_failure_auc_mean_set_size": ep_auc_mean,
        "episode_failure_auc_max_set_size": ep_auc_max,
        "mean_uncertainty_success": float(df_valid.loc[df_valid["success"] >= 0.5, "uncertainty"].mean()) if (df_valid["success"] >= 0.5).any() else float('nan'),
        "mean_uncertainty_failure": float(df_valid.loc[df_valid["success"] < 0.5, "uncertainty"].mean()) if (df_valid["success"] < 0.5).any() else float('nan'),
    }])
    out.to_csv(out_dir / f"autoeval_risk_{args.method}_alpha{args.alpha}.csv", index=False)
    ep.to_csv(out_dir / f"autoeval_episode_risk_{args.method}_alpha{args.alpha}.csv", index=False)
    print(out.to_string(index=False))
    print(f"Wrote AutoEval risk diagnostics to {out_dir}")

if __name__ == "__main__":
    main()

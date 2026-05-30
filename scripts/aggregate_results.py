#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def add_metric_aggs(rows: list[dict], df: pd.DataFrame, group_cols: list[str], metric_names: list[str]) -> pd.DataFrame:
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(g))
        for m in metric_names:
            if m in g.columns:
                vals = g[m].astype(float)
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def rget(row: pd.Series, name: str, fallback: str | None = None):
    if name in row:
        return row.get(name, np.nan)
    return row.get(fallback, np.nan) if fallback else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CoRAS result CSVs into paper-ready tables.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.rglob("metrics_summary_alpha*.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_summary_alpha*.csv found under {args.root}")
    df = pd.concat([pd.read_csv(f).assign(result_file=str(f)) for f in files], ignore_index=True)
    out_dir = args.root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    metric_names = [
        "coverage", "abs_coverage_gap", "mean_set_size", "mean_set_size_norm", "top1_accuracy",
        "fail_to_abstain_rate", "singleton_rate", "ece_top1",
        # v3 robot-facing metrics
        "episode_coverage_min", "episode_coverage_p10", "episodes_below_target_rate",
        "longest_miss_run_mean", "longest_miss_run_p95", "rare_action_coverage", "common_action_coverage",
        "execute_rate_k1", "execute_rate_k3", "execute_rate_k5", "execute_rate_k10",
        "execute_covered_rate_k5", "conditional_coverage_k5", "abstain_rate_k5",
        "top1_action_l2_mean", "oracle_set_action_l2_mean", "quantization_action_l2_mean",
        "set_diameter_mean", "set_action_radius_mean",
        # compatibility with earlier v3 draft metric names
        "execute_rate_set_le_1", "wrong_execute_rate_set_le_1", "execute_rate_set_le_2", "wrong_execute_rate_set_le_2",
        "execute_rate_set_le_4", "wrong_execute_rate_set_le_4", "top1_action_mse", "best_in_set_action_mse",
        "mean_action_set_diameter",
    ]
    group_cols = ["method", "alpha", "calib_frac"] if "calib_frac" in df.columns else ["method", "alpha"]
    agg = add_metric_aggs([], df, group_cols, metric_names)
    agg.to_csv(out_dir / "aggregate_metrics.csv", index=False)

    compact = agg.copy()
    if "alpha" in compact.columns:
        compact = compact[np.isclose(compact["alpha"].astype(float), 0.10)]
    if "calib_frac" in compact.columns and len(compact):
        max_cf = compact["calib_frac"].astype(float).max()
        compact = compact[np.isclose(compact["calib_frac"].astype(float), max_cf)]
    compact_rows = []
    for _, r in compact.iterrows():
        compact_rows.append({
            "Method": r["method"],
            "Coverage": fmt_mean_std(rget(r, "coverage_mean"), rget(r, "coverage_std")),
            "|Coverage gap|": fmt_mean_std(rget(r, "abs_coverage_gap_mean"), rget(r, "abs_coverage_gap_std")),
            "Mean set size": fmt_mean_std(rget(r, "mean_set_size_mean"), rget(r, "mean_set_size_std")),
            "Top-1 acc.": fmt_mean_std(rget(r, "top1_accuracy_mean"), rget(r, "top1_accuracy_std")),
            "Unsafe singleton error": fmt_mean_std(rget(r, "fail_to_abstain_rate_mean"), rget(r, "fail_to_abstain_rate_std")),
            "Worst episode cov.": fmt_mean_std(rget(r, "episode_coverage_min_mean"), rget(r, "episode_coverage_min_std")),
            "Rare-action cov.": fmt_mean_std(rget(r, "rare_action_coverage_mean"), rget(r, "rare_action_coverage_std")),
            "Exec@5": fmt_mean_std(rget(r, "execute_rate_k5_mean", "execute_rate_set_le_4_mean"), rget(r, "execute_rate_k5_std", "execute_rate_set_le_4_std")),
            "Oracle set L2/MSE": fmt_mean_std(rget(r, "oracle_set_action_l2_mean_mean", "best_in_set_action_mse_mean"), rget(r, "oracle_set_action_l2_mean_std", "best_in_set_action_mse_std")),
        })
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(out_dir / "main_table_alpha010.csv", index=False)
    (out_dir / "main_table_alpha010.tex").write_text(compact_df.to_latex(index=False, escape=True) if len(compact_df) else "", encoding="utf-8")

    domain_files = sorted(args.root.rglob("metrics_by_domain_alpha*.csv"))
    if domain_files:
        ddf = pd.concat([pd.read_csv(f).assign(result_file=str(f)) for f in domain_files], ignore_index=True)
        ddf.to_csv(out_dir / "all_domain_metrics.csv", index=False)
        agg_spec = {
            "coverage_mean": ("coverage", "mean"),
            "coverage_std": ("coverage", "std"),
            "mean_set_size_mean": ("mean_set_size", "mean"),
            "fail_to_abstain_rate_mean": ("fail_to_abstain_rate", "mean"),
            "runs": ("coverage", "size"),
        }
        optional_cols = [
            "episode_coverage_min", "rare_action_coverage", "execute_rate_k5", "oracle_set_action_l2_mean",
            "mean_set_size_norm", "best_in_set_action_mse",
        ]
        for c in optional_cols:
            if c in ddf.columns:
                agg_spec[f"{c}_mean"] = (c, "mean")
        dgroup = ddf.groupby(["method", "alpha", "domain"], as_index=False).agg(**agg_spec)
        dgroup.to_csv(out_dir / "aggregate_domain_metrics.csv", index=False)

        worst_rows = []
        for (method, alpha), g in dgroup.groupby(["method", "alpha"]):
            g2 = g.copy()
            worst_cov = g2.loc[g2["coverage_mean"].idxmin()]
            largest_size = g2.loc[g2["mean_set_size_mean"].idxmax()]
            worst_rows.append({
                "method": method,
                "alpha": alpha,
                "worst_coverage_domain": worst_cov["domain"],
                "worst_coverage": float(worst_cov["coverage_mean"]),
                "largest_set_domain": largest_size["domain"],
                "largest_mean_set_size": float(largest_size["mean_set_size_mean"]),
                "domains": int(len(g2)),
            })
        pd.DataFrame(worst_rows).to_csv(out_dir / "worst_domain_summary.csv", index=False)

    episode_files = sorted(args.root.rglob("metrics_by_episode_alpha*.csv"))
    if episode_files:
        edf = pd.concat([pd.read_csv(f).assign(result_file=str(f)) for f in episode_files], ignore_index=True)
        edf.to_csv(out_dir / "all_episode_metrics.csv", index=False)
        egroup = edf.groupby(["method", "alpha"], as_index=False).agg(
            episode_coverage_mean=("coverage", "mean"),
            episode_coverage_min=("coverage", "min"),
            episode_coverage_p10=("coverage", lambda x: float(np.quantile(x, 0.10))),
            longest_miss_run_mean=("longest_miss_run", "mean"),
            longest_miss_run_p95=("longest_miss_run", lambda x: float(np.quantile(x, 0.95))),
            episodes=("episode", "nunique"),
        )
        egroup.to_csv(out_dir / "aggregate_episode_metrics.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        for metric in ["coverage", "mean_set_size", "fail_to_abstain_rate", "episode_coverage_min", "oracle_set_action_l2_mean"]:
            if metric not in df.columns:
                continue
            fig = plt.figure(figsize=(8, 4.5))
            plot_df = df[df["alpha"].round(2) == 0.10].copy() if "alpha" in df.columns else df.copy()
            order = list(plot_df.groupby("method")[metric].mean().sort_values(ascending=(metric not in {"coverage", "episode_coverage_min"})).index)
            vals = [plot_df[plot_df["method"] == m][metric].dropna().to_numpy() for m in order]
            boxplot_kwargs = {"tick_labels" if "tick_labels" in inspect.signature(plt.boxplot).parameters else "labels": order}
            plt.boxplot(vals, vert=True, **boxplot_kwargs)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel(metric.replace("_", " "))
            plt.tight_layout()
            fig.savefig(out_dir / f"box_{metric}_alpha010.png", dpi=180)
            plt.close(fig)
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")

    print(f"Aggregated {len(files)} result files into {out_dir}")
    print(compact_df.to_string(index=False) if len(compact_df) else "No compact rows")


if __name__ == "__main__":
    main()

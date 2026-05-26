#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def mean_ci(series: pd.Series) -> tuple[float, float, float]:
    x = series.dropna().astype(float).to_numpy()
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, np.nan, np.nan
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


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

    group_cols = ["method", "alpha", "calib_frac"] if "calib_frac" in df.columns else ["method", "alpha"]
    metrics = ["coverage", "abs_coverage_gap", "mean_set_size", "top1_accuracy", "fail_to_abstain_rate", "singleton_rate", "ece_top1"]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(g))
        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    agg = pd.DataFrame(rows).sort_values(group_cols)
    agg.to_csv(out_dir / "aggregate_metrics.csv", index=False)

    # Main paper compact: alpha=0.10 and largest calibration fraction if available.
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
            "Coverage": fmt_mean_std(r.get("coverage_mean", np.nan), r.get("coverage_std", np.nan)),
            "|Coverage gap|": fmt_mean_std(r.get("abs_coverage_gap_mean", np.nan), r.get("abs_coverage_gap_std", np.nan)),
            "Mean set size": fmt_mean_std(r.get("mean_set_size_mean", np.nan), r.get("mean_set_size_std", np.nan)),
            "Top-1 acc.": fmt_mean_std(r.get("top1_accuracy_mean", np.nan), r.get("top1_accuracy_std", np.nan)),
            "Unsafe singleton error": fmt_mean_std(r.get("fail_to_abstain_rate_mean", np.nan), r.get("fail_to_abstain_rate_std", np.nan)),
        })
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(out_dir / "main_table_alpha010.csv", index=False)

    latex = compact_df.to_latex(index=False, escape=True) if len(compact_df) else ""
    (out_dir / "main_table_alpha010.tex").write_text(latex, encoding="utf-8")

    # Domain aggregation.
    domain_files = sorted(args.root.rglob("metrics_by_domain_alpha*.csv"))
    if domain_files:
        ddf = pd.concat([pd.read_csv(f).assign(result_file=str(f)) for f in domain_files], ignore_index=True)
        ddf.to_csv(out_dir / "all_domain_metrics.csv", index=False)
        dgroup = ddf.groupby(["method", "alpha", "domain"], as_index=False).agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            mean_set_size_mean=("mean_set_size", "mean"),
            fail_to_abstain_rate_mean=("fail_to_abstain_rate", "mean"),
            runs=("coverage", "size"),
        )
        dgroup.to_csv(out_dir / "aggregate_domain_metrics.csv", index=False)

    # Simple plots for appendix.
    try:
        import matplotlib.pyplot as plt
        for metric in ["coverage", "mean_set_size", "fail_to_abstain_rate"]:
            if metric not in df.columns:
                continue
            fig = plt.figure(figsize=(8, 4.5))
            plot_df = df[df["alpha"].round(2) == 0.10].copy() if "alpha" in df.columns else df.copy()
            order = list(plot_df.groupby("method")[metric].mean().sort_values(ascending=(metric != "coverage")).index)
            vals = [plot_df[plot_df["method"] == m][metric].dropna().to_numpy() for m in order]
            plt.boxplot(vals, labels=order, vert=True)
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

# CoRAS Complete Experiment Pipeline

This repository is a complete, runnable pipeline for **Conformal Robot Affordance Sets (CoRAS)**. It removes the previous placeholders and provides scripts that run full experiment matrices, aggregate results, and generate paper/appendix tables.

The pipeline supports three tracks:

1. **Self-contained sim-to-target-shift track**: no external data, complete runs on any Mac/Linux machine.
2. **Offline real-robot DROID_100 track**: exports the public `lerobot/droid_100` dataset, tokenizes actions, trains, calibrates, and aggregates.
3. **Offline LeRobot PushT track**: exports `lerobot/pusht`, tokenizes actions, trains, calibrates, and aggregates.

The method follows the CoRAS paper concept: train a robot action-token model, adapt a lightweight prompt/adapter on a target-domain tuning split, then build split-conformal prediction sets on a disjoint calibration split.

## Quick local validation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
bash scripts/run_fast_local_validation.sh
```

Expected outputs:

```text
results/sim_fast/model_base.pt
results/sim_fast/model_prompt.pt
results/sim_fast/metrics_summary_alpha0.10.csv
results/sim_fast/metrics_by_domain_alpha0.10.csv
results/sim_fast/aggregate/main_table_alpha010.csv
results/sim_fast/aggregate/main_table_alpha010.tex
results/sim_fast/aggregate/box_coverage_alpha010.png
```

## Full self-contained suite

This is the complete no-external-data run for the first empirical section and appendix sweeps.

```bash
bash scripts/run_complete_synthetic_suite.sh
```

Environment variables let you scale it down/up:

```bash
CORAS_SIM_N=6000 CORAS_SEEDS="0 1 2" CORAS_CALIB_FRACS="0.10 0.25" bash scripts/run_complete_synthetic_suite.sh
```

## DROID_100 public real-robot offline track

Install the optional robot-data dependencies and run:

```bash
pip install -r requirements-robotdata.txt
bash scripts/run_droid100_complete.sh
```

This script performs every step:

1. Export `lerobot/droid_100` frames/actions to NPZ.
2. Build a KMeans action-token codebook.
3. Automatically select held-out task domains.
4. Train source and prompt-adapted checkpoints.
5. Evaluate all methods over seeds, alphas, and calibration fractions.
6. Aggregate tables and figures.

## PushT offline track

```bash
pip install -r requirements-robotdata.txt
bash scripts/run_pusht_complete.sh
```

## Methods evaluated

- `top1_singleton`: uncalibrated singleton action-token prediction.
- `topk_calibrated`: top-k set with k calibrated from true-label ranks.
- `vanilla_conformal`: split conformal prediction on the base model.
- `aps_conformal`: adaptive prediction sets on the base model.
- `temperature_conformal`: temperature scaling on tune split, conformal calibration on calibration split.
- `prompt_only_top1`: prompt-adapted top-1 baseline.
- `coras`: prompt-adapted inverse-probability conformal sets.
- `coras_aps`: prompt-adapted APS sets.
- `mondrian_domain`: prompt-adapted domain-conditional conformal thresholds.

## Output format

Every run directory contains:

```text
split_indices.npz
split_summary.json
train_history.json
model_base.pt
model_prompt.pt
metrics_summary_alphaXX.csv
metrics_by_domain_alphaXX.csv
prediction_sets_compact_alphaXX.csv
prediction_sets_coras_alphaXX.csv
```

The aggregate directory contains:

```text
all_metrics.csv
aggregate_metrics.csv
main_table_alpha010.csv
main_table_alpha010.tex
aggregate_domain_metrics.csv
box_coverage_alpha010.png
box_mean_set_size_alpha010.png
box_fail_to_abstain_rate_alpha010.png
```

## Main paper vs appendix

Use the compact `main_table_alpha010` table for the main eight-page paper. Use the full aggregate metrics, per-domain CSVs, prediction-set CSVs, and box plots for the appendix.

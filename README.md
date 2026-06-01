# CoRAS: Conformal Robot Action Sets under Domain Shift

CoRAS is the reproducibility package for the paper **Conformal Robot Action Sets under Domain Shift**. The implementation studies a planner-facing uncertainty layer for robot action policies. A visual action-token model is trained on a source robot domain, a lightweight adapter is tuned on a small target-domain split, and split conformal calibration is applied on a disjoint calibration split to produce set-valued robot action predictions.

The package contains the supplementary PDF, implementation code, experiment configurations, aggregate result tables, and dataset metadata used by the paper and supplement. It does not include raw public datasets, model checkpoints, local NPZ exports, or raw AutoEval trajectory files. Public robot datasets are downloaded by the provided scripts.

## Method overview

CoRAS converts a policy's single action-token prediction into a calibrated action set. For a robot observation, the output is a finite set of candidate action tokens rather than a single top-1 action. The set is intended for use by downstream planning or monitoring logic:

- small sets indicate relatively sharp policy confidence;
- large sets indicate ambiguity under domain shift and can trigger replanning, additional observation, or conservative fallback;
- coverage and set size are reported separately, because valid coverage with very large sets indicates a weak candidate generator rather than a fully operational controller.

The implementation evaluates top-1 prediction, calibrated top-k sets, vanilla conformal prediction, adaptive prediction sets, temperature-scaled conformal prediction, prompt-adapted CoRAS, CoRAS-APS, and domain-conditional/Mondrian thresholds.

## Repository layout

```text
.
├── supplementary.pdf
├── README.md
├── REPRODUCE.md
├── ENVIRONMENT.md
├── DATA_LICENSES_AND_SOURCES.md
├── RESULTS_MANIFEST.md
├── coras/                         # Core package: data loading, models, conformal utilities
├── scripts/                       # Data export, training, evaluation, diagnostics, aggregation
├── configs/                       # Base and generated experiment configurations
├── docs/                          # Protocol notes and troubleshooting material
├── runpod/                        # CUDA/RunPod helper files
├── results/                       # Aggregate result tables and plots; no checkpoints
├── metadata/                      # Dataset summaries and file manifests
├── requirements-mac.txt
├── requirements-cuda.txt
├── requirements-robotdata.txt
└── requirements-robotdata-mac.txt
```

## Included and excluded artifacts

Included:

- source code for data export, visual-shift construction, action-codebook construction, model training, conformal evaluation, diagnostics, and aggregation;
- generated experiment configurations for the reported tracks;
- aggregate CSV/TeX tables and plots used in the paper and supplementary PDF;
- dataset summaries and result manifests;
- standalone supplementary material as `supplementary.pdf`.

Excluded:

- raw PushT, DROID, and AutoEval datasets;
- local `data/*.npz` exports;
- trained checkpoints such as `*.pt`, `*.pth`, and `*.ckpt`;
- raw AutoEval `traj_*.pkl` files;
- large per-frame prediction-set dumps;
- local virtual environments, caches, and machine-specific logs.

## Installation

Commands are run from the repository root.

### macOS / Apple Silicon / CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

For public robot datasets on macOS:

```bash
pip install -r requirements-robotdata-mac.txt
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_NUM_WORKERS=0
export CORAS_DEVICE=auto
```

### Linux / NVIDIA GPU / RunPod

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cuda.txt
pip install -r requirements-robotdata.txt
export CORAS_DEVICE=cuda
export CORAS_NUM_WORKERS=4
```

## Quick validation

The smoke test generates a small synthetic dataset, trains the source and adapter models, evaluates conformal methods, and aggregates results.

```bash
bash scripts/run_fast_local_validation.sh
```

Expected output directory:

```text
results/sim_fast/
```

## Main experiment tracks

### 1. Synthetic controlled-shift suite

```bash
export CORAS_NUM_WORKERS=0
bash scripts/run_complete_synthetic_suite.sh
```

A smaller development version:

```bash
CORAS_SIM_N=6000 \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25" \
CORAS_NUM_WORKERS=0 \
bash scripts/run_complete_synthetic_suite.sh
```

### 2. PushT visual-shift suite

```bash
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_NUM_WORKERS=0
export CORAS_PRETRAINED=1

CORAS_CODEBOOK_KS="32 64 128" \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25 0.40" \
bash scripts/run_pusht_paper_suite.sh
```

This script exports `lerobot/pusht`, constructs clean-source and shifted-target domains, fits action-token codebooks on the source domain, trains models, evaluates conformal methods, and writes aggregate tables.

### 3. DROID_100 visual-shift suite

CUDA is recommended for the full DROID matrix.

```bash
export CORAS_DEVICE=cuda
export CORAS_NUM_WORKERS=4
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_PRETRAINED=1

CORAS_CODEBOOK_KS="64 128" \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25 0.40" \
CORAS_BATCH_SIZE=128 \
CORAS_EPOCHS=8 \
CORAS_ADAPTER_EPOCHS=5 \
bash scripts/run_droid100_paper_suite.sh
```

Local preflight:

```bash
export CORAS_DEVICE=auto
export CORAS_NUM_WORKERS=0
export CORAS_LEROBOT_VIDEO_BACKEND=pyav

CORAS_CODEBOOK_KS="64" \
CORAS_SEEDS="0" \
CORAS_ALPHAS="0.10" \
CORAS_CALIB_FRACS="0.25" \
CORAS_BATCH_SIZE=64 \
CORAS_EPOCHS=4 \
CORAS_ADAPTER_EPOCHS=3 \
bash scripts/run_droid100_paper_suite.sh
```

### 4. AutoEval auxiliary online-log diagnostic

AutoEval is optional and network-sensitive. It is included as an auxiliary online real-robot log diagnostic, not as a closed-loop deployment.

```bash
CORAS_AUTOEVAL_FORCE_EXPORT=1 \
CORAS_AUTOEVAL_MAX_EVALS=24 \
CORAS_AUTOEVAL_SCAN_EVALS=600 \
CORAS_AUTOEVAL_TRAJS_PER_EVAL=2 \
CORAS_AUTOEVAL_MAX_TRAJ_DOWNLOADS=250 \
CORAS_AUTOEVAL_FRAME_STRIDE=4 \
CORAS_AUTOEVAL_MAX_FRAMES_PER_TRAJ=120 \
CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES=1000 \
CORAS_AUTOEVAL_PREFER_RECENT=1 \
CORAS_AUTOEVAL_ALLOW_ACTION_PROXY=1 \
CORAS_AUTOEVAL_K=64 \
CORAS_SEEDS="0" \
CORAS_ALPHAS="0.10" \
CORAS_CALIB_FRACS="0.25" \
CORAS_NUM_WORKERS=0 \
bash scripts/run_autoeval_public_logs_suite.sh
```

If the exporter cannot obtain the requested minimum number of frames, the script stops unless `CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1` is explicitly set.

## Included result tables

The included aggregate results are located in:

```text
results/sim_complete/aggregate/
results/pusht_paper_k32/paper_grade_aggregate/
results/pusht_paper_k64/paper_grade_aggregate/
results/pusht_paper_k128/paper_grade_aggregate/
results/droid100_paper_k64/paper_grade_aggregate/
results/droid100_paper_k128/paper_grade_aggregate/
results/autoeval_online_k32/
```

See `RESULTS_MANIFEST.md` and `metadata/included_files_manifest.txt` for file-level details.

## Dataset summaries

Dataset metadata are stored in:

```text
metadata/dataset_summaries.json
```

The summary records frame counts, image shapes, action dimensions, episode counts, domain counts, codebook sizes, and AutoEval action-source fields when applicable.

## Evaluation metrics

The main metrics are marginal coverage, absolute coverage gap, mean set size, top-1 action-token accuracy, fail-to-abstain diagnostics, per-domain coverage, calibration-size sweeps, action-codebook sweeps, continuous-action geometry diagnostics, and episode/block diagnostics.

These metrics separate statistical validity from operational sharpness. A method can achieve nominal coverage while returning large sets; such behavior is reported as evidence of uncertainty rather than hidden by a single aggregate score.

## Scope

This repository supports reproducibility for an offline robot-learning study. It does not provide a closed-loop physical-robot controller, a certified safety shield, or a new real-robot deployment. CoRAS is a set-valued uncertainty interface intended to expose calibrated action ambiguity to downstream planning or monitoring modules.

## Anonymous review artifact

For double-blind review, serve this repository through an anonymized mirror. Remove author-identifying repository metadata, local paths, private logs, raw data, checkpoints, and any non-anonymized commit metadata from the submitted artifact.

# CoRAS: Conformal Robot Action Sets under Domain Shift

CoRAS is a reproducibility package for the paper **Conformal Robot Action Sets under Domain Shift**. The package implements a planner-facing uncertainty layer for robot action policies: a base visual action-token model is trained on a source robot domain, a lightweight adapter is tuned on a small target-domain split, and split conformal calibration is applied on a disjoint calibration split to produce set-valued robot action predictions.

The repository is intended to support inspection and reproduction of the experiments reported in the main paper and supplementary material. It includes the supplementary PDF, implementation snapshot, experiment configurations, aggregate result tables, and dataset metadata. It does **not** include raw public datasets, trained model checkpoints, or large intermediate exports.

## Summary of the method

CoRAS converts a robot policy's single action-token prediction into a calibrated action set. Given an observation image, the model returns a finite candidate set of action tokens rather than only the top-1 action. These sets are designed to expose uncertainty to a downstream planner or execution monitor:

- small sets indicate relatively sharp policy confidence;
- large sets indicate ambiguity under domain shift and can trigger replanning, additional observation, or conservative fallback;
- empty or inconsistent planner interfaces are treated as operational failures rather than silently executed actions.

The implementation evaluates several set-valued methods, including top-1 prediction, calibrated top-k sets, vanilla conformal prediction, adaptive prediction sets, temperature-scaled conformal prediction, prompt-adapted CoRAS, CoRAS-APS, and domain-conditional/Mondrian thresholds.

## Repository contents

```text
.
├── supplementary.pdf                 # Standalone supplementary material
├── README.md                         # Repository overview and reproduction guide
├── REPRODUCE.md                      # Detailed reproduction commands
├── ENVIRONMENT.md                    # Environment and hardware notes
├── DATA_LICENSES_AND_SOURCES.md      # Dataset source and license notes
├── RESULTS_MANIFEST.md               # Included result-table manifest
├── code/
│   ├── coras/                        # Core data, model, conformal, and metric code
│   ├── scripts/                      # Data export, training, evaluation, diagnostics, aggregation
│   ├── configs/                      # Base and generated experiment configurations
│   ├── docs/                         # Additional protocol and troubleshooting notes
│   ├── runpod/                       # RunPod / CUDA helper files
│   ├── requirements-mac.txt          # Local CPU/MPS environment
│   ├── requirements-cuda.txt         # CUDA / RunPod environment
│   ├── requirements-robotdata.txt    # Public robot-dataset dependencies
│   └── requirements-robotdata-mac.txt# Mac-safe robot-data dependencies
├── results/                          # Aggregate result tables used in the paper/supplement
│   ├── sim_complete/
│   ├── pusht_paper_k32/
│   ├── pusht_paper_k64/
│   ├── pusht_paper_k128/
│   ├── droid100_paper_k64/
│   ├── droid100_paper_k128/
│   └── autoeval_online_k32/          # Auxiliary online-log diagnostic summary
└── metadata/
    ├── dataset_summaries.json        # Dataset sizes, domains, episodes, action dimensions
    ├── included_files_manifest.txt
    ├── result_tables_manifest.txt
    └── excluded_files.txt
```

The `results/` directory contains aggregate CSV and TeX tables, not raw experiment rollouts. The `metadata/` directory documents dataset sizes and exclusions so that the reported results can be audited without shipping large public datasets.

## What is included and excluded

Included:

- source code for data export, action-codebook construction, model training, conformal calibration, evaluation, diagnostics, and aggregation;
- generated experiment configurations for the reported tracks;
- aggregate result tables used by the paper and supplement;
- dataset summaries and manifests;
- the standalone supplementary PDF.

Excluded:

- raw PushT, DROID, and AutoEval data;
- large local `.npz` exports;
- trained model checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- raw AutoEval trajectory pickle files;
- large per-frame prediction-set dumps;
- local virtual environments and cache directories.

The public datasets are downloaded by the provided scripts. The synthetic track is generated locally from the included generator.

## Installation

### Local Mac / CPU / Apple Silicon

```bash
git clone <repository-url> coras
cd coras/code

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

For public LeRobot datasets on macOS, install the Mac-safe robot-data dependencies:

```bash
pip install -r requirements-robotdata-mac.txt
```

The Mac setup uses CPU or Apple MPS when available. For long DROID runs, a CUDA machine is recommended.

### CUDA / RunPod

```bash
git clone <repository-url> coras
cd coras/code

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cuda.txt
pip install -r requirements-robotdata.txt
```

If using RunPod, the helper files under `code/runpod/` provide a container-oriented setup path.

## Quick validation

A small synthetic smoke test checks the installation and end-to-end pipeline:

```bash
cd code
source .venv/bin/activate
bash scripts/run_fast_local_validation.sh
```

Expected outputs are written under `code/results/sim_fast/`, including a trained source model, prompt-adapted model, conformal evaluation CSVs, and aggregate tables.

## Reproducing the experiment tracks

All commands below are run from `code/` after activating the environment.

### 1. Synthetic controlled-shift suite

This track is self-contained and does not require external data.

```bash
bash scripts/run_complete_synthetic_suite.sh
```

A smaller development run can be launched as:

```bash
CORAS_SIM_N=6000 \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25" \
CORAS_NUM_WORKERS=0 \
bash scripts/run_complete_synthetic_suite.sh
```

### 2. PushT visual-shift suite

This track downloads `lerobot/pusht`, constructs real-frame visual-shift target domains, fits action-token codebooks on the source domain, and evaluates multiple conformal baselines.

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

The visual-shift split contains a clean source domain and target domains corresponding to lighting, crop, occlusion, and blur/noise perturbations.

### 3. DROID_100 visual-shift suite

This track downloads `lerobot/droid_100` and evaluates the same conformal action-set pipeline on a larger offline real-robot manipulation dataset. CUDA is recommended.

```bash
export CORAS_DEVICE=cuda
export CORAS_NUM_WORKERS=4
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

For a local preflight, use:

```bash
export CORAS_DEVICE=auto
export CORAS_NUM_WORKERS=0

CORAS_CODEBOOK_KS="64" \
CORAS_SEEDS="0" \
CORAS_ALPHAS="0.10" \
CORAS_CALIB_FRACS="0.25" \
CORAS_BATCH_SIZE=64 \
CORAS_EPOCHS=4 \
CORAS_ADAPTER_EPOCHS=3 \
bash scripts/run_droid100_paper_suite.sh
```

### 4. AutoEval online-log diagnostic

AutoEval is included as an auxiliary online real-robot log diagnostic. It is network-sensitive and is not required for the main paper tables. The exporter uses public AutoEval logs and a temporal source/target split when sufficient data can be downloaded.

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

The AutoEval diagnostics should be interpreted as auxiliary evidence on public online robot evaluation traces, not as a closed-loop deployment of CoRAS.

## Included result tables

The aggregate tables included in this repository are the tables used by the main paper and supplementary material. The most relevant directories are:

```text
results/sim_complete/aggregate/
results/pusht_paper_k32/paper_grade_aggregate/
results/pusht_paper_k64/paper_grade_aggregate/
results/pusht_paper_k128/paper_grade_aggregate/
results/droid100_paper_k64/paper_grade_aggregate/
results/droid100_paper_k128/paper_grade_aggregate/
results/autoeval_online_k32/
```

The supplementary PDF reports the full tables and diagnostics. The CSV files are included to support direct inspection and independent reformatting.

## Dataset and split summary

The dataset summary file is located at:

```text
metadata/dataset_summaries.json
```

It records, for each generated or exported dataset, the number of frames, image shape, action dimension, number of episodes, domain counts, codebook size, and relevant auxiliary fields such as AutoEval action sources. This file is intended to verify that the experiments used public robot observations and explicit source/target splits rather than undocumented local data.

## Evaluation metrics

The implementation reports:

- marginal coverage relative to the target level `1 - alpha`;
- absolute coverage gap;
- mean action-set size;
- top-1 action-token accuracy;
- fail-to-abstain / unsafe singleton error diagnostics;
- per-domain coverage and set size;
- calibration-size and action-codebook sweeps;
- continuous-action geometry diagnostics using codebook centers;
- episode/block diagnostics for temporally correlated trajectories;
- auxiliary AutoEval risk diagnostics when public online logs are available.

These metrics are designed to distinguish statistical validity from operational usefulness. A method may achieve target coverage with large sets; in that case, the set size itself is a diagnostic of policy or candidate-generator weakness.

## Reproducibility policy

The experimental protocol uses separate data roles:

1. source training data for the base action-token model;
2. target tuning data for the lightweight adapter;
3. target calibration data for split conformal threshold selection;
4. held-out target test data for reporting coverage and set efficiency.

The conformal calibration split is disjoint from the adapter tuning split. Action codebooks in the paper-grade public-data tracks are fit on source-domain actions, then applied to shifted target domains.

## Troubleshooting

### macOS video decoding

For LeRobot video datasets on macOS, use the PyAV backend:

```bash
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
pip install -r requirements-robotdata-mac.txt
```

This avoids common TorchCodec/FFmpeg dynamic-library issues on Apple Silicon.

### PyTorch DataLoader workers on macOS

Set workers to zero if multiprocessing causes serialization or memory issues:

```bash
export CORAS_NUM_WORKERS=0
```

### CUDA on local Mac

Do not set `CORAS_DEVICE=cuda` on macOS. Use:

```bash
export CORAS_DEVICE=auto
```

Use CUDA only on a Linux/NVIDIA environment such as RunPod.

### AutoEval download instability

AutoEval trajectory downloads can fail because of network or Hugging Face/Xet instability. The AutoEval scripts support bounded scans, strict minimum-frame checks, and explicit force-export behavior. AutoEval is auxiliary and can be omitted without affecting the main synthetic, PushT, and DROID result tables.

## Scope of the artifact

This repository supports reproducibility of an offline robot-learning study. It does not claim to provide a closed-loop physical-robot controller, a certified safety shield, or a new real-robot deployment. CoRAS is a set-valued uncertainty interface that exposes calibrated action ambiguity to downstream planning or monitoring modules.

## Anonymous review note

For anonymous review, this repository should be served through an anonymized mirror. The repository contents are written to avoid author-identifying information. Do not include local paths, raw private data, checkpoint files, or non-anonymized logs in the review artifact.

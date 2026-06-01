# Reproduction Instructions

All commands are run from the repository root after creating and activating the Python environment.

## 1. Environment setup

### macOS / CPU / Apple Silicon

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

For PushT, DROID, or AutoEval data export on macOS:

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

## 2. Smoke test

```bash
bash scripts/run_fast_local_validation.sh
```

This creates a small synthetic dataset, trains the action-token model, evaluates conformal methods, and writes aggregate tables under `results/sim_fast/`.

## 3. Synthetic controlled-shift suite

```bash
export CORAS_NUM_WORKERS=0
bash scripts/run_complete_synthetic_suite.sh
```

Optional smaller run:

```bash
CORAS_SIM_N=6000 \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25" \
CORAS_NUM_WORKERS=0 \
bash scripts/run_complete_synthetic_suite.sh
```

Aggregate tables are written to `results/sim_complete/aggregate/`.

## 4. PushT visual-shift suite

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

Outputs are written to:

```text
results/pusht_paper_k32/
results/pusht_paper_k64/
results/pusht_paper_k128/
```

## 5. DROID_100 visual-shift suite

Full GPU run:

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

Outputs are written to `results/droid100_paper_k64/` and `results/droid100_paper_k128/`.

## 6. AutoEval auxiliary online-log diagnostic

AutoEval is optional and may be affected by public download stability.

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

The strict exporter stops when fewer frames than `CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES` are obtained, unless `CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1` is explicitly set.

## 7. Regenerating aggregate tables

For a completed track:

```bash
python scripts/aggregate_paper_grade_results.py --root results/pusht_paper_k64 --alpha 0.10
```

For the synthetic track:

```bash
python scripts/aggregate_results.py --root results/sim_complete
```

## 8. Included aggregates

The repository includes aggregate tables for the runs reported in the paper and supplement. See:

```text
RESULTS_MANIFEST.md
metadata/dataset_summaries.json
metadata/included_files_manifest.txt
```

Raw public datasets and trained checkpoints are intentionally excluded.

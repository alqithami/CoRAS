# Reproduction Instructions

All commands below are run from the `code/` directory after extracting the archive.

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

For public robot datasets on macOS, install the robot-data dependencies and use the PyAV video backend:

```bash
pip install -r requirements-robotdata.txt
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_NUM_WORKERS=0
```

For RunPod or another Linux/NVIDIA environment, use the CUDA requirements and set `CORAS_DEVICE=cuda`.

## 2. Smoke test

```bash
bash scripts/run_fast_local_validation.sh
```

This generates a small synthetic dataset, trains the action-token predictor, calibrates the conformal methods, and writes aggregate tables under `results/sim_fast/`.

## 3. Synthetic controlled-shift suite

```bash
export CORAS_NUM_WORKERS=0
bash scripts/run_complete_synthetic_suite.sh
```

The reported aggregate tables included in this archive are under `../results/sim_complete/aggregate/`.

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

The script downloads `lerobot/pusht`, creates real-frame visual-shift target domains, builds action codebooks, trains models, evaluates conformal methods, and writes paper-grade aggregate tables.

## 5. DROID_100 visual-shift suite

```bash
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_NUM_WORKERS=0
export CORAS_PRETRAINED=1
CORAS_CODEBOOK_KS="64 128" \
CORAS_SEEDS="0 1 2" \
CORAS_ALPHAS="0.05 0.10 0.20" \
CORAS_CALIB_FRACS="0.10 0.25 0.40" \
bash scripts/run_droid100_paper_suite.sh
```

For GPU execution, set `CORAS_DEVICE=cuda` and increase batch size as appropriate.

## 6. AutoEval auxiliary online-log diagnostic

AutoEval is included as an auxiliary diagnostic. The strict exporter stops when fewer frames than the requested minimum are obtained, unless `CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1` is explicitly set.

```bash
CORAS_AUTOEVAL_FORCE_EXPORT=1 \
CORAS_AUTOEVAL_MAX_EVALS=24 \
CORAS_AUTOEVAL_SCAN_EVALS=600 \
CORAS_AUTOEVAL_TRAJS_PER_EVAL=2 \
CORAS_AUTOEVAL_MAX_TRAJ_DOWNLOADS=250 \
CORAS_AUTOEVAL_FRAME_STRIDE=4 \
CORAS_AUTOEVAL_MAX_FRAMES_PER_TRAJ=120 \
CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES=1000 \
CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1 \
CORAS_AUTOEVAL_K=64 \
CORAS_SEEDS="0" \
CORAS_ALPHAS="0.10" \
CORAS_CALIB_FRACS="0.25" \
bash scripts/run_autoeval_public_logs_suite.sh
```

## 7. Included result tables

The included `results/` directory contains aggregate result tables only. It does not include checkpoints or raw datasets. The configuration files in `code/configs/generated/` record the settings used to produce the uploaded results.

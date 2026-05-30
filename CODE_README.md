# CoRAS Implementation Snapshot

This directory contains an anonymized implementation of the CoRAS experimental pipeline. The code supports synthetic data generation, public robot-dataset export, action-codebook construction, model training, split conformal calibration, diagnostic evaluation, and table aggregation.

The implementation is organized around four experiment tracks:

1. Synthetic controlled domain shift.
2. PushT real-frame visual-shift evaluation.
3. DROID_100 real-frame visual-shift evaluation.
4. Optional AutoEval public online-log diagnostics.

Raw datasets and trained model checkpoints are not included. The scripts regenerate datasets and checkpoints locally from public dataset identifiers and fixed configuration files.

## Directory layout

- `coras/`: core conformal, dataset, model, and utility modules.
- `scripts/`: data export, training, evaluation, diagnostics, and aggregation scripts.
- `configs/`: base and generated experiment configuration files.
- `runpod/`: minimal container/setup files for Linux/NVIDIA execution.
- `requirements-*.txt`: environment files for local and GPU execution.

## Minimal validation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
bash scripts/run_fast_local_validation.sh
```

## Public robot datasets

For LeRobot datasets on macOS, the PyAV video backend is recommended:

```bash
pip install -r requirements-robotdata.txt
export CORAS_LEROBOT_VIDEO_BACKEND=pyav
export CORAS_NUM_WORKERS=0
```


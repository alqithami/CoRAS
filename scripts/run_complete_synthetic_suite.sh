#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
mkdir -p data results
python scripts/make_simulated_affordance_data.py --out data/sim_affordance_v2.npz --n ${CORAS_SIM_N:-9000} --size 96 --grid 4 --episode-len 20 --seed 123
python scripts/run_experiment_matrix.py \
  --base-config configs/sim_complete.yaml \
  --seeds ${CORAS_SEEDS:-0 1 2 3 4} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device ${CORAS_DEVICE:-auto}

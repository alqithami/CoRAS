#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p data results
python scripts/make_simulated_affordance_data.py --out data/sim_affordance_fast.npz --n 1200 --size 64 --grid 4 --episode-len 20 --seed 0
python scripts/train_action_model.py --config configs/sim_fast.yaml --device auto
python scripts/evaluate_methods.py --config configs/sim_fast.yaml --device auto --alpha 0.10
python scripts/aggregate_results.py --root results/sim_fast

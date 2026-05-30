#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CORAS_NUM_WORKERS=${CORAS_NUM_WORKERS:-0}
export CORAS_SIM_N=${CORAS_SIM_N:-1800}
export CORAS_SEEDS=${CORAS_SEEDS:-0}
export CORAS_ALPHAS=${CORAS_ALPHAS:-0.10}
export CORAS_CALIB_FRACS=${CORAS_CALIB_FRACS:-0.10}
export CORAS_EPOCHS=${CORAS_EPOCHS:-1}
export CORAS_ADAPTER_EPOCHS=${CORAS_ADAPTER_EPOCHS:-1}
bash scripts/run_complete_synthetic_suite.sh

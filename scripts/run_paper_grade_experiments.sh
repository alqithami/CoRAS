#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# This is the recommended complete empirical package.
# On Mac, use smaller env vars. On RunPod, leave defaults or increase epochs/seeds.
: "${CORAS_NUM_WORKERS:=0}"
export CORAS_NUM_WORKERS
bash scripts/run_complete_synthetic_suite.sh
bash scripts/run_pusht_shift_complete.sh
bash scripts/run_droid100_task_complete_v3.sh

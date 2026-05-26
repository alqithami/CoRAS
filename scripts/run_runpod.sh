#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cuda.txt
bash scripts/run_fast_local_validation.sh

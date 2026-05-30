#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
# Avoid accidental TorchCodec selection on macOS. The exporters use PyAV for LeRobot videos.
pip uninstall -y torchcodec || true
pip install -r requirements-robotdata.txt
python - <<'PY'
import av
import lerobot
print('robotdata dependencies OK:', 'av', av.__version__, 'lerobot', getattr(lerobot, '__version__', 'unknown'))
PY

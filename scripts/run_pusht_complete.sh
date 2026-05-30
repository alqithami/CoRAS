#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "WARNING: run_pusht_complete.sh is a within-dataset sanity check with a constant PushT domain. For paper-grade evidence use scripts/run_pusht_shift_complete.sh."
mkdir -p data results configs/generated
export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
python scripts/export_lerobot_to_npz.py \
  --repo-id lerobot/pusht \
  --camera observation.image \
  --action-key action \
  --domain lerobot_pusht \
  --domain-strategy constant \
  --resize 96 \
  --max-samples ${CORAS_PUSHT_MAX_SAMPLES:-50000} \
  --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
  --out data/pusht_raw.npz
python scripts/build_action_codebook.py --input data/pusht_raw.npz --out data/pusht_tokens_k64.npz --num-codes ${CORAS_CODEBOOK_K:-64} --seed 0
python scripts/write_config_from_npz.py \
  --data data/pusht_tokens_k64.npz \
  --out configs/generated/pusht_auto.yaml \
  --output-dir results/pusht_complete \
  --seed 0 \
  --target-domain lerobot_pusht \
  --image-size 96 \
  --batch-size ${CORAS_BATCH_SIZE:-128} \
  --encoder ${CORAS_ENCODER:-resnet18} \
  --epochs ${CORAS_EPOCHS:-5} \
  --adapter-epochs ${CORAS_ADAPTER_EPOCHS:-3} \
  --calib-frac 0.25 \
  --num-workers ${CORAS_NUM_WORKERS:-0} \
  --include-remaining-target-in-train
python scripts/run_experiment_matrix.py \
  --base-config configs/generated/pusht_auto.yaml \
  --seeds ${CORAS_SEEDS:-0 1 2} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device ${CORAS_DEVICE:-auto}

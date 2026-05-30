#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated
export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
export CORAS_NUM_WORKERS="${CORAS_NUM_WORKERS:-0}"

if [[ ! -f data/pusht_raw.npz ]]; then
  python scripts/export_lerobot_to_npz.py \
    --repo-id lerobot/pusht \
    --camera observation.image \
    --action-key action \
    --domain pusht_clean \
    --domain-strategy constant \
    --resize 96 \
    --max-samples ${CORAS_PUSHT_MAX_SAMPLES:-50000} \
    --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
    --out data/pusht_raw.npz
fi

if [[ ! -f data/pusht_tokens_k64.npz ]]; then
  python scripts/build_action_codebook.py \
    --input data/pusht_raw.npz \
    --out data/pusht_tokens_k64.npz \
    --num-codes ${CORAS_CODEBOOK_K:-64} \
    --seed 0
fi

python scripts/make_pusht_shift_suite.py \
  --input data/pusht_tokens_k64.npz \
  --out data/pusht_shift_tokens_k${CORAS_CODEBOOK_K:-64}.npz \
  --source-domain pusht_clean \
  --max-base-samples ${CORAS_PUSHT_SHIFT_MAX_BASE_SAMPLES:-20000} \
  --seed 0

python scripts/write_config_from_npz.py \
  --data data/pusht_shift_tokens_k${CORAS_CODEBOOK_K:-64}.npz \
  --out configs/generated/pusht_shift_auto.yaml \
  --output-dir results/pusht_shift_complete \
  --seed 0 \
  --target-domain pusht_brightness_low \
  --target-domain pusht_brightness_high \
  --target-domain pusht_contrast_low \
  --target-domain pusht_occlusion \
  --target-domain pusht_crop_shift \
  --target-domain pusht_gaussian_noise \
  --image-size 96 \
  --batch-size ${CORAS_BATCH_SIZE:-128} \
  --encoder ${CORAS_ENCODER:-resnet18} \
  --epochs ${CORAS_EPOCHS:-8} \
  --adapter-epochs ${CORAS_ADAPTER_EPOCHS:-4} \
  --calib-frac 0.25 \
  --num-workers ${CORAS_NUM_WORKERS:-0} \
  --include-remaining-target-in-train \
  --exclude-source-with-target-eval-episodes

python scripts/run_experiment_matrix.py \
  --base-config configs/generated/pusht_shift_auto.yaml \
  --seeds ${CORAS_SEEDS:-0 1 2} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device ${CORAS_DEVICE:-auto}

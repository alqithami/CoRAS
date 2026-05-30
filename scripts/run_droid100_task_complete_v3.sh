#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated
export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
PRETRAINED_ARGS=()
if [[ "${CORAS_PRETRAINED:-1}" == "1" ]]; then PRETRAINED_ARGS+=(--pretrained); fi

if [[ ! -f data/droid100_raw.npz || "${CORAS_FORCE_REEXPORT:-0}" == "1" ]]; then
  python scripts/export_lerobot_to_npz.py \
    --repo-id lerobot/droid_100 \
    --camera observation.images.exterior_image_1_left \
    --action-key action \
    --domain-strategy task \
    --resize 128 \
    --max-samples ${CORAS_DROID_MAX_SAMPLES:-32000} \
    --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
    --out data/droid100_raw.npz
fi

python scripts/build_action_codebook.py \
  --input data/droid100_raw.npz \
  --out data/droid100_tokens_k${CORAS_CODEBOOK_K:-64}.npz \
  --num-codes ${CORAS_CODEBOOK_K:-64} \
  --seed 0

python scripts/write_config_from_npz.py \
  --data data/droid100_tokens_k${CORAS_CODEBOOK_K:-64}.npz \
  --out configs/generated/droid100_task_auto.yaml \
  --output-dir results/droid100_task_complete \
  --seed 0 \
  --n-target-domains ${CORAS_TARGET_DOMAINS:-5} \
  --min-target-count ${CORAS_MIN_TARGET_COUNT:-300} \
  --image-size 128 \
  --batch-size ${CORAS_BATCH_SIZE:-96} \
  --encoder ${CORAS_ENCODER:-resnet18} \
  "${PRETRAINED_ARGS[@]}" \
  --epochs ${CORAS_EPOCHS:-8} \
  --adapter-epochs ${CORAS_ADAPTER_EPOCHS:-4} \
  --calib-frac 0.25 \
  --num-workers ${CORAS_NUM_WORKERS:-0}

python scripts/run_experiment_matrix.py \
  --base-config configs/generated/droid100_task_auto.yaml \
  --seeds ${CORAS_SEEDS:-0 1 2} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device ${CORAS_DEVICE:-auto}

python scripts/audit_split_integrity.py \
  --config results/droid100_task_complete/matrix_configs/config_seed0_calib0.25.yaml \
  --out results/droid100_task_complete/aggregate/split_audit_seed0_calib025 || true

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated

export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
export CORAS_NUM_WORKERS="${CORAS_NUM_WORKERS:-0}"

RAW="data/droid100_raw.npz"
SHIFTED="data/droid100_visual_shift_v3.npz"
MODES_DEFAULT="lighting camera_crop occlusion blur_noise"
MODES=${CORAS_DROID_SHIFT_MODES:-$MODES_DEFAULT}
TARGET_ARGS=()
for m in $MODES; do
  TARGET_ARGS+=(--target-domain "droid_shift_${m}")
done

DEVICE="${CORAS_DEVICE:-auto}"
if [ "$DEVICE" = "cuda" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "[warn] CORAS_DEVICE=cuda requested, but this PyTorch build has no CUDA. Falling back to --device auto."
    DEVICE="auto"
  fi
fi

if [ ! -f "$RAW" ]; then
  python scripts/export_lerobot_to_npz.py \
    --repo-id lerobot/droid_100 \
    --camera ${CORAS_DROID_CAMERA:-observation.images.exterior_image_1_left} \
    --action-key ${CORAS_DROID_ACTION_KEY:-action} \
    --domain droid100 \
    --domain-strategy constant \
    --resize ${CORAS_IMAGE_SIZE:-96} \
    --max-samples ${CORAS_DROID_MAX_SAMPLES:-60000} \
    --stride ${CORAS_DROID_STRIDE:-1} \
    --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
    --out "$RAW"
else
  echo "Using existing $RAW"
fi

python scripts/apply_real_robot_shifts.py \
  --input "$RAW" \
  --out "$SHIFTED" \
  --modes $MODES \
  --target-frac ${CORAS_DROID_TARGET_FRAC:-0.35} \
  --severity ${CORAS_DROID_SHIFT_SEVERITY:-0.80} \
  --seed ${CORAS_SHIFT_SEED:-0} \
  --source-domain-name real_clean_source \
  --target-prefix droid_shift

PRETRAIN_FLAG=""
if [ "${CORAS_PRETRAINED:-0}" = "1" ]; then
  PRETRAIN_FLAG="--pretrained"
fi

for K in ${CORAS_CODEBOOK_KS:-64 128}; do
  TOK="data/droid100_visual_shift_k${K}.npz"
  OUTROOT="results/droid100_paper_k${K}"
  CFG="configs/generated/droid100_paper_k${K}.yaml"
  python scripts/build_action_codebook_domain_fit.py \
    --input "$SHIFTED" \
    --out "$TOK" \
    --num-codes "$K" \
    --fit-domain-regex 'real_clean_source' \
    --seed 0
  python scripts/write_config_from_npz.py \
    --data "$TOK" \
    --out "$CFG" \
    --output-dir "$OUTROOT" \
    --seed 0 \
    "${TARGET_ARGS[@]}" \
    --image-size ${CORAS_IMAGE_SIZE:-96} \
    --batch-size ${CORAS_BATCH_SIZE:-128} \
    --encoder ${CORAS_ENCODER:-resnet18} \
    $PRETRAIN_FLAG \
    --epochs ${CORAS_EPOCHS:-8} \
    --adapter-epochs ${CORAS_ADAPTER_EPOCHS:-5} \
    --calib-frac 0.25 \
    --num-workers ${CORAS_NUM_WORKERS:-0}

  python scripts/run_experiment_matrix.py \
    --base-config "$CFG" \
    --seeds ${CORAS_SEEDS:-0 1 2} \
    --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
    --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
    --device "$DEVICE"

  python scripts/augment_all_diagnostics.py --root "$OUTROOT" --alpha 0.10
  python scripts/aggregate_paper_grade_results.py --root "$OUTROOT" --alpha 0.10
done

echo "DROID100 paper-grade suite complete. See results/droid100_paper_k*/paper_grade_aggregate/."

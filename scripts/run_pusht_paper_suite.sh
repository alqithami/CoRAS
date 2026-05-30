#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated

export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
export CORAS_NUM_WORKERS="${CORAS_NUM_WORKERS:-0}"

RAW="data/pusht_raw.npz"
SHIFTED="data/pusht_visual_shift_v3.npz"
MODES_DEFAULT="lighting camera_crop occlusion blur_noise"
MODES=${CORAS_PUSHT_SHIFT_MODES:-$MODES_DEFAULT}
TARGET_ARGS=()
for m in $MODES; do
  TARGET_ARGS+=(--target-domain "pusht_shift_${m}")
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
    --repo-id lerobot/pusht \
    --camera observation.image \
    --action-key action \
    --domain lerobot_pusht \
    --domain-strategy constant \
    --resize ${CORAS_IMAGE_SIZE:-96} \
    --max-samples ${CORAS_PUSHT_MAX_SAMPLES:-50000} \
    --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
    --out "$RAW"
else
  echo "Using existing $RAW"
fi

python scripts/apply_real_robot_shifts.py \
  --input "$RAW" \
  --out "$SHIFTED" \
  --modes $MODES \
  --target-frac ${CORAS_PUSHT_TARGET_FRAC:-0.40} \
  --severity ${CORAS_PUSHT_SHIFT_SEVERITY:-0.85} \
  --seed ${CORAS_SHIFT_SEED:-0} \
  --source-domain-name real_clean_source \
  --target-prefix pusht_shift

PRETRAIN_FLAG=""
if [ "${CORAS_PRETRAINED:-0}" = "1" ]; then
  PRETRAIN_FLAG="--pretrained"
fi

for K in ${CORAS_CODEBOOK_KS:-32 64 128}; do
  TOK="data/pusht_visual_shift_k${K}.npz"
  OUTROOT="results/pusht_paper_k${K}"
  CFG="configs/generated/pusht_paper_k${K}.yaml"
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

echo "PushT paper-grade suite complete. See results/pusht_paper_k*/paper_grade_aggregate/."

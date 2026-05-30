#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated

export CORAS_LEROBOT_VIDEO_BACKEND="${CORAS_LEROBOT_VIDEO_BACKEND:-pyav}"
export CORAS_NUM_WORKERS="${CORAS_NUM_WORKERS:-0}"

# 1) Export public PushT data if needed. This is real public robot-learning data,
# real-frame shifted domains. The stress domains are created from these frames.
if [ ! -f data/pusht_raw.npz ]; then
  python scripts/export_lerobot_to_npz.py \
    --repo-id lerobot/pusht \
    --camera observation.image \
    --action-key action \
    --domain lerobot_pusht \
    --domain-strategy constant \
    --resize 96 \
    --max-samples ${CORAS_PUSHT_MAX_SAMPLES:-0} \
    --video-backend ${CORAS_LEROBOT_VIDEO_BACKEND} \
    --out data/pusht_raw.npz
else
  echo "Using existing data/pusht_raw.npz"
fi

# 2) Quantize continuous actions into an action-token codebook.
K="${CORAS_CODEBOOK_K:-64}"
TOK="data/pusht_tokens_k${K}.npz"
if [ ! -f "$TOK" ]; then
  python scripts/build_action_codebook.py \
    --input data/pusht_raw.npz \
    --out "$TOK" \
    --num-codes "$K" \
    --seed 0
else
  echo "Using existing $TOK"
fi

# 3) Create stress-domain public-data benchmark. Default mode keeps the dataset
# near the size of PushT; duplicate_target is heavier and better for appendix.
STRESS="data/pusht_stress_k${K}_${CORAS_PUSHT_STRESS_MODE:-single}.npz"
if [ ! -f "$STRESS" ]; then
  python scripts/make_pusht_stress_domains.py \
    --input "$TOK" \
    --out "$STRESS" \
    --seed ${CORAS_STRESS_SEED:-0} \
    --resize 96 \
    --source-episode-frac ${CORAS_SOURCE_EPISODE_FRAC:-0.60} \
    --max-base-samples ${CORAS_PUSHT_STRESS_MAX_BASE_SAMPLES:-0} \
    --mode ${CORAS_PUSHT_STRESS_MODE:-single}
else
  echo "Using existing $STRESS"
fi

# 4) Configure source-to-target evaluation. Training uses source_clean episodes;
# tune/calib/test are target stress-domain episodes, episode-blocked.
python scripts/write_config_from_npz.py \
  --data "$STRESS" \
  --out configs/generated/pusht_stress_auto.yaml \
  --output-dir results/pusht_stress_complete \
  --seed 0 \
  --target-domain target_lighting \
  --target-domain target_crop \
  --target-domain target_blur \
  --target-domain target_noise \
  --target-domain target_occlusion \
  --target-domain target_hard \
  --image-size 96 \
  --batch-size ${CORAS_BATCH_SIZE:-128} \
  --encoder ${CORAS_ENCODER:-resnet18} \
  --epochs ${CORAS_EPOCHS:-12} \
  --adapter-epochs ${CORAS_ADAPTER_EPOCHS:-5} \
  --tune-frac ${CORAS_TUNE_FRAC:-0.15} \
  --calib-frac ${CORAS_CALIB_FRAC:-0.25} \
  --test-frac ${CORAS_TEST_FRAC:-0.25} \
  --num-workers ${CORAS_NUM_WORKERS:-0}

python scripts/run_experiment_matrix.py \
  --base-config configs/generated/pusht_stress_auto.yaml \
  --seeds ${CORAS_SEEDS:-0 1 2 3 4} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device ${CORAS_DEVICE:-auto}

python scripts/summarize_experiment_quality.py --root results/pusht_stress_complete || true

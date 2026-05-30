#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data results configs/generated

export CORAS_NUM_WORKERS="${CORAS_NUM_WORKERS:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

DEVICE="${CORAS_DEVICE:-auto}"
if [ "$DEVICE" = "cuda" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "[warn] CORAS_DEVICE=cuda requested, but CUDA is unavailable. Falling back to --device auto."
    DEVICE="auto"
  fi
fi

RAW="data/autoeval_online_raw.npz"
K="${CORAS_AUTOEVAL_K:-64}"
TOK="data/autoeval_online_k${K}.npz"
OUTROOT="results/autoeval_online_k${K}"
CFG="configs/generated/autoeval_online_k${K}.yaml"
MIN_FRAMES="${CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES:-200}"
FORCE_EXPORT="${CORAS_AUTOEVAL_FORCE_EXPORT:-0}"

needs_export=0
if [ ! -f "$RAW" ]; then
  needs_export=1
else
  raw_n=$(python - "$RAW" <<'PY'
import sys, numpy as np
p=sys.argv[1]
try:
    with np.load(p, allow_pickle=True) as z:
        print(int(len(z["images"])))
except Exception:
    print(-1)
PY
)
  echo "Using existing $RAW with n=$raw_n frames"
  if [ "$FORCE_EXPORT" = "1" ]; then
    echo "CORAS_AUTOEVAL_FORCE_EXPORT=1, re-exporting AutoEval raw file."
    needs_export=1
  elif [ "$raw_n" -lt "$MIN_FRAMES" ]; then
    echo "Existing $RAW has only $raw_n frames, below requested CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES=$MIN_FRAMES. Re-exporting."
    needs_export=1
  fi
fi

if [ "$needs_export" = "1" ]; then
  rm -f "$RAW"
  PREFER_ARGS=()
  if [ "${CORAS_AUTOEVAL_PREFER_RECENT:-1}" = "1" ]; then
    PREFER_ARGS+=(--prefer-recent)
  fi
  PROXY_ARGS=()
  if [ "${CORAS_AUTOEVAL_ALLOW_ACTION_PROXY:-1}" = "1" ]; then
    PROXY_ARGS+=(--allow-action-proxy)
  else
    PROXY_ARGS+=(--no-allow-action-proxy)
  fi
  UNDER_MIN_ARGS=()
  if [ "${CORAS_AUTOEVAL_ALLOW_UNDER_MIN:-0}" = "1" ]; then
    echo "[warn] CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1: AutoEval export may continue with fewer frames than CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES. Use this only for exploratory logs."
    UNDER_MIN_ARGS+=(--allow-under-min)
  fi
  python scripts/export_autoeval_to_npz.py \
    --repo-id "${CORAS_AUTOEVAL_REPO:-zhouzypaul/auto_eval}" \
    --out "$RAW" \
    --resize "${CORAS_IMAGE_SIZE:-96}" \
    --max-evals "${CORAS_AUTOEVAL_MAX_EVALS:-24}" \
    --scan-evals "${CORAS_AUTOEVAL_SCAN_EVALS:-300}" \
    --max-trajs-per-eval "${CORAS_AUTOEVAL_TRAJS_PER_EVAL:-2}" \
    --max-traj-downloads "${CORAS_AUTOEVAL_MAX_TRAJ_DOWNLOADS:-500}" \
    --frame-stride "${CORAS_AUTOEVAL_FRAME_STRIDE:-4}" \
    --max-frames-per-traj "${CORAS_AUTOEVAL_MAX_FRAMES_PER_TRAJ:-120}" \
    --min-exported-frames "$MIN_FRAMES" \
    --max-consecutive-download-errors "${CORAS_AUTOEVAL_MAX_CONSECUTIVE_DOWNLOAD_ERRORS:-40}" \
    --source-frac "${CORAS_AUTOEVAL_SOURCE_FRAC:-0.60}" \
    "${PREFER_ARGS[@]}" \
    "${PROXY_ARGS[@]}" \
    "${UNDER_MIN_ARGS[@]}"
fi

python scripts/check_autoeval_npz.py --data "$RAW" || true

python - "$RAW" "$MIN_FRAMES" <<'PY'
import sys, numpy as np
p=sys.argv[1]; min_n=int(sys.argv[2])
with np.load(p, allow_pickle=True) as z:
    n=len(z["images"])
print(f"[AutoEval raw check] n={n}; requested_min={min_n}")
if n < min_n and "CORAS_AUTOEVAL_ALLOW_UNDER_MIN" not in __import__('os').environ:
    # This should usually be impossible because export is strict, but protect reuse of old small NPZs.
    raise SystemExit(f"Existing {p} has n={n} < requested_min={min_n}. Use CORAS_AUTOEVAL_FORCE_EXPORT=1, lower CORAS_AUTOEVAL_MIN_EXPORTED_FRAMES, or set CORAS_AUTOEVAL_ALLOW_UNDER_MIN=1 for exploratory runs.")
PY

python scripts/build_action_codebook_domain_fit.py \
  --input "$RAW" \
  --out "$TOK" \
  --num-codes "$K" \
  --fit-domain-regex 'autoeval_online_source' \
  --seed 0

PRETRAIN_FLAG=()
if [ "${CORAS_PRETRAINED:-1}" = "1" ]; then
  PRETRAIN_FLAG+=(--pretrained)
fi

python scripts/write_config_from_npz.py \
  --data "$TOK" \
  --out "$CFG" \
  --output-dir "$OUTROOT" \
  --seed 0 \
  --target-domain autoeval_online_target_late \
  --image-size "${CORAS_IMAGE_SIZE:-96}" \
  --batch-size "${CORAS_BATCH_SIZE:-64}" \
  --encoder "${CORAS_ENCODER:-resnet18}" \
  "${PRETRAIN_FLAG[@]}" \
  --epochs "${CORAS_EPOCHS:-6}" \
  --adapter-epochs "${CORAS_ADAPTER_EPOCHS:-4}" \
  --calib-frac 0.25 \
  --num-workers "${CORAS_NUM_WORKERS:-0}"

python scripts/run_experiment_matrix.py \
  --base-config "$CFG" \
  --seeds ${CORAS_SEEDS:-0 1 2} \
  --alphas ${CORAS_ALPHAS:-0.05 0.10 0.20} \
  --calib-fracs ${CORAS_CALIB_FRACS:-0.10 0.25 0.40} \
  --device "$DEVICE"

python scripts/augment_all_diagnostics.py --root "$OUTROOT" --alpha 0.10 || true
python scripts/aggregate_paper_grade_results.py --root "$OUTROOT" --alpha 0.10 || python scripts/aggregate_results.py --root "$OUTROOT"

for d in "$OUTROOT"/seed*_calib*; do
  [ -d "$d" ] || continue
  python scripts/evaluate_autoeval_risk.py --run-dir "$d" --data "$TOK" --alpha 0.10 --method coras || true
done

echo "AutoEval online-log suite complete. See $OUTROOT and $OUTROOT/paper_grade_aggregate/."

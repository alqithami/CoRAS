#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results

# Codebook sweep for appendix: shows that conclusions are not an artifact of
# K=64. Use fewer seeds by default because this is expensive.
for K in ${CORAS_SWEEP_KS:-16 32 64 128}; do
  echo "=== PushT stress codebook sweep K=$K ==="
  CORAS_CODEBOOK_K=$K \
  CORAS_SEEDS="${CORAS_SWEEP_SEEDS:-0 1 2}" \
  CORAS_ALPHAS="${CORAS_SWEEP_ALPHAS:-0.10}" \
  CORAS_CALIB_FRACS="${CORAS_SWEEP_CALIB_FRACS:-0.25}" \
  CORAS_EPOCHS="${CORAS_SWEEP_EPOCHS:-10}" \
  CORAS_ADAPTER_EPOCHS="${CORAS_SWEEP_ADAPTER_EPOCHS:-4}" \
  bash scripts/run_pusht_stress_complete.sh
  if [ -d results/pusht_stress_complete ]; then
    mv results/pusht_stress_complete "results/pusht_stress_k${K}" || true
  fi
  rm -f configs/generated/pusht_stress_auto.yaml
  echo "Finished K=$K"
done

python - <<'PY'
from pathlib import Path
import pandas as pd
roots = sorted(Path('results').glob('pusht_stress_k*/aggregate/aggregate_metrics.csv'))
frames=[]
for p in roots:
    df=pd.read_csv(p)
    k=p.parts[1].replace('pusht_stress_k','')
    df['codebook_k']=k
    frames.append(df)
if frames:
    out=pd.concat(frames, ignore_index=True)
    Path('results/pusht_codebook_sweep').mkdir(parents=True, exist_ok=True)
    out.to_csv('results/pusht_codebook_sweep/aggregate_codebook_sweep.csv', index=False)
    print('Wrote results/pusht_codebook_sweep/aggregate_codebook_sweep.csv')
PY

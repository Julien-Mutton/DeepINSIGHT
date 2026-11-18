#!/bin/bash
# Run every step in order for one sample rate; stops at the first step whose checks fail.
#   bash pipeline/run_all.sh 0.001            # optionally: FROM=s06_embed bash pipeline/run_all.sh 0.001
# Same as:  python -m deepinsight run --rate 0.001 [--from s06_embed]
# Uses $DEEPINSIGHT_PY if set, else the `python` on PATH (activate the `deepinsight` env first).
set -euo pipefail
RATE=${1:-0.001}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${DEEPINSIGHT_PY:-python}
STEPS="s00_inputs s01_benchmark s02_sites s03_liftover s04_features s05_embedder s06_embed s06b_composition s07_sae s08_design s09_train s09b_mle s09c_groundtruth s10_evaluate s11_report"
started=${FROM:+no}
for s in $STEPS; do
  if [ "${started:-yes}" = no ]; then [ "$s" = "$FROM" ] && started=yes || continue; fi
  echo "### $s (rate $RATE)"
  "$PY" "$ROOT/pipeline/run_step.py" "$s" --rate "$RATE"
done
echo "### pipeline finished for rate $RATE"

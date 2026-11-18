#!/bin/bash
# C++ baseline: fit LINSIGHT with the authors' own programs on ALL INSIGHT sites.
#   bash cpp_baseline/run_linsight_cpp.sh        (or: python -m deepinsight cpp-baseline)
# Needs ~45 GB disk for the binary site files and up to 80 GB RAM; the fit is single-threaded and
# slow (our SLURM job allowed 400 h). Outputs go to $DEEPINSIGHT_OUT/cpp_baseline/.
# The recorded result of our run is results/cpp_baseline/ (parameter.txt + AUROCs).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DEEPINSIGHT_DATA:-$ROOT/data}
OUT=${DEEPINSIGHT_OUT:-$ROOT/outputs}/cpp_baseline
BIN=$ROOT/third_party/linsight_cpp
make -C "$BIN" >/dev/null
mkdir -p "$OUT/binary_dir"
if ! ls "$OUT"/binary_dir/chr*.fit >/dev/null 2>&1; then
  echo "[1/2] LINSIGHT-prep: INSIGHT database -> per-chromosome binary files"
  "$BIN/LINSIGHT-prep" -i "$DATA/linsight/INSIGHT_database.ins" -o "$OUT/binary_dir/"
fi
echo "[2/2] LINSIGHT-fit: 20 epochs (beta values from the LINSIGHT README)"
"$BIN/LINSIGHT-fit" -d "$OUT/binary_dir/" -f "$DATA/linsight/genomic_feature.bed" \
  -o "$OUT/parameter.txt" -n 20 -b1 0.7567672441 -b2 0.2053143291
echo "wrote $OUT/parameter.txt ($(wc -l < "$OUT/parameter.txt") epochs)"
echo "next: python cpp_baseline/evaluate_cpp.py   (needs pipeline steps s01 and s04 of a run)"

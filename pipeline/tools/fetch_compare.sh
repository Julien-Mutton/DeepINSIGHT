#!/bin/bash
# Download an upstream CSHL .gz with resume/retries until complete, test it, compare with the local original.
#   bash pipeline/tools/fetch_compare.sh genomic_feature.bed "$DEEPINSIGHT_DATA/linsight/genomic_feature.bed"
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=${DEEPINSIGHT_OUT:-$ROOT/outputs}/upstream; mkdir -p "$D"; cd "$D"
N=$1; LOC=$2; URL=http://compgen.cshl.edu/LINSIGHT/$N.gz
EXP=$(curl -sI -m 60 $URL | grep -i content-length | tr -dc 0-9)
for i in $(seq 1 200); do
  [ -f $N.gz ] && [ "$(stat -c %s $N.gz)" = "$EXP" ] && break
  curl -sS -C - --retry 5 -m 3600 -o $N.gz $URL; sleep 5
done
{ echo "file $N expected_bytes $EXP got_bytes $(stat -c %s $N.gz)"
  gunzip -t $N.gz && echo "gzip_integrity ok"
  gunzip -c $N.gz | cmp - "$LOC" && echo "IDENTICAL to $LOC" || echo "DIFFERENT from $LOC"; } > $N.compare.txt 2>&1

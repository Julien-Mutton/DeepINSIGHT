#!/bin/bash
# sha256 of the decompressed upstream .gz (streamed, never stored) vs the local original.
#   bash pipeline/tools/verify_upstream.sh      -> $DEEPINSIGHT_OUT/upstream/verify.tsv
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${DEEPINSIGHT_DATA:-$ROOT/data}; OUTD=${DEEPINSIGHT_OUT:-$ROOT/outputs}/upstream
mkdir -p "$OUTD"; OUT=$OUTD/verify.tsv; G=$DATA/linsight
echo -e "file\tupstream_sha256\tlocal_sha256\tequal" > "$OUT"
for f in genomic_feature.bed INSIGHT_database.ins; do
  up=$(curl -sS -m 7200 http://compgen.cshl.edu/LINSIGHT/$f.gz | gunzip -c | sha256sum | cut -d' ' -f1)
  loc=$(sha256sum "$G/$f" | cut -d' ' -f1)
  eq=$([ "$up" == "$loc" ] && echo yes || echo NO)
  echo -e "$f\t$up\t$loc\t$eq" >> "$OUT"
done
echo done >> "$OUT"

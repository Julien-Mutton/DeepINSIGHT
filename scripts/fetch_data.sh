#!/bin/bash
# Download every PUBLIC original input into $DEEPINSIGHT_DATA (default <repo>/data), in the layout
# that src/lctx/config.py expects (data/README.md). About 95 GB after decompression.
#   bash scripts/fetch_data.sh            (or: python -m deepinsight fetch)
# The HGMD benchmark panel is licensed and NOT downloaded; see data/README.md.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DEEPINSIGHT_DATA:-$ROOT/data}
mkdir -p "$DATA"/{linsight,genomes,hyenadna/hyenadna-medium-160k-seqlen,benchmark}

get() {  # url dest [gunzip]
  local url=$1 dest=$2
  if [ -s "$dest" ]; then echo "have  $dest"; return; fi
  echo "fetch $url"
  if [ "${3:-}" = gunzip ]; then
    curl -fL --retry 5 -C - -o "$dest.gz" "$url" && gunzip -f "$dest.gz"
  else
    curl -fL --retry 5 -C - -o "$dest" "$url"
  fi
}

# LINSIGHT (Huang, Gulko & Siepel 2017): INSIGHT database, 48 annotations, published scores
get http://compgen.cshl.edu/LINSIGHT/INSIGHT_database.ins.gz "$DATA/linsight/INSIGHT_database.ins" gunzip
get http://compgen.cshl.edu/LINSIGHT/genomic_feature.bed.gz  "$DATA/linsight/genomic_feature.bed" gunzip
get http://compgen.cshl.edu/LINSIGHT/LINSIGHT.bw             "$DATA/linsight/LINSIGHT.bw"

# genomes and liftover chain
get https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz "$DATA/genomes/hg19.fa" gunzip
# hg38 exactly as HyenaDNA was trained on it (HyenaDNA README, data instructions)
get https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz "$DATA/genomes/hg38.ml.fa" gunzip
get https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz "$DATA/genomes/hg19ToHg38.over.chain.gz"

# HyenaDNA medium-160k weights (Nguyen et al. 2023)
H=https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen/resolve/main
get $H/weights.ckpt "$DATA/hyenadna/hyenadna-medium-160k-seqlen/weights.ckpt"
get $H/config.json  "$DATA/hyenadna/hyenadna-medium-160k-seqlen/config.json"

echo
echo "Public inputs are in $DATA. Step s00 checks sizes and checksums against upstream."
n=$(ls "$DATA"/benchmark/*.pandas 2>/dev/null | wc -l)
[ "$n" = 4 ] || echo "STILL MISSING: the HGMD benchmark panel (4 .pandas files) in $DATA/benchmark/ (see data/README.md)"

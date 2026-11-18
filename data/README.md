# data/: the original inputs (not in git)

The pipeline reads **only** these files. They total about 95 GB, so they are not in the repository.
`python -m deepinsight fetch` (= `bash scripts/fetch_data.sh`) downloads every public one into this
folder, in this layout. To keep the data elsewhere, set `DEEPINSIGHT_DATA=/path/to/data` (same layout
below it). Symlinks are fine.

```
data/
├── linsight/
│   ├── INSIGHT_database.ins        58 GB   compgen.cshl.edu/LINSIGHT/INSIGHT_database.ins.gz (gunzipped)
│   ├── genomic_feature.bed         28 GB   compgen.cshl.edu/LINSIGHT/genomic_feature.bed.gz (gunzipped)
│   └── LINSIGHT.bw                 1.4 GB  compgen.cshl.edu/LINSIGHT/LINSIGHT.bw (published scores)
├── genomes/
│   ├── hg19.fa                     3.2 GB  UCSC hg19 (soft-masked; lower case = repeats)
│   ├── hg38.ml.fa                  3.1 GB  the hg38 HyenaDNA was trained on (storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz)
│   └── hg19ToHg38.over.chain.gz    0.2 MB  UCSC liftOver chain
├── hyenadna/
│   └── hyenadna-medium-160k-seqlen/
│       ├── weights.ckpt                    huggingface.co/LongSafari/hyenadna-medium-160k-seqlen
│       └── config.json
└── benchmark/                      LICENSED, NOT DISTRIBUTED (see below)
    ├── hgmd_reg.pandas             1,614 HGMD non-coding regulatory variants (positives)
    ├── unmatched.pandas            161,400 common 1000 Genomes variants
    ├── matched_by_tss.pandas       16,140 TSS-distance-matched negatives
    └── matched_by_region.pandas    5,027 region-matched (≤ 1 kb) negatives
```

Step `s00_inputs` checks every file against upstream: SHA-256 of the HyenaDNA weights (HuggingFace LFS
id), MD5 of the chain (UCSC), byte size of `LINSIGHT.bw`, and autosome lengths of both genomes (UCSC
chrom.sizes). `pipeline/tools/verify_upstream.sh` also compares the two large CSHL files with
their upstream copies by SHA-256. Our results: `results/verify.tsv`, `results/genomic_feature.bed.compare.txt`.

## The HGMD benchmark

The evaluation panel is the one of Ritchie et al. (2014, GWAVA), which Huang et al. (2017) used to
evaluate LINSIGHT: HGMD regulatory variants plus three sets of common-variant negatives, stored as
Python-2 pandas 0.13 pickles. The positives come from HGMD (Stenson et al.), which is licensed by
QIAGEN, so the files cannot be redistributed. To reproduce the HGMD evaluation:

1. obtain the four pickles from the GWAVA authors or under your own HGMD licence;
2. put them in `data/benchmark/` with the names above;
3. create the Python-2 environment that reads them: `conda env create -f environment-legacy.yml`.

Step `s01_benchmark` checks the counts above and that the published LINSIGHT track reproduces the
paper's AUROCs (0.897 / 0.759 / 0.660) within 0.01. Later steps need the panel too: `s02` removes every
site within 1 kb of a benchmark variant, so the training data cannot be rebuilt exactly without it.
Only `s10_evaluate` reads the labels to score models (once, under a lock); the held-out likelihood test
(`s09b`) and the power simulations (`s09c`) do not use HGMD labels.

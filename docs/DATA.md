# Inputs — the only files this pipeline reads

Everything else under `outputs/` (`$DEEPINSIGHT_OUT`) is produced by the pipeline. How to obtain each file: `data/README.md` and `scripts/fetch_data.sh`. Paths are set in
`src/lctx/config.py`; `pipeline/steps/s00_inputs.py` records size and sha256 of each.

| name | path (under `data/` = `$DEEPINSIGHT_DATA`) | origin | verified against upstream |
|---|---|---|---|
| benchmark | `data/benchmark/{hgmd_reg,unmatched,matched_by_tss,matched_by_region}.pandas` | GWAVA / LINSIGHT benchmark (Ritchie et al. 2014; Huang et al. 2017), Python-2 pandas pickles | read with the `legacy_pandas` env (pandas 0.13.1) |
| INSIGHT data | `data/linsight/INSIGHT_database.ins` | compgen.cshl.edu/LINSIGHT/INSIGHT_database.ins.gz, gunzipped | sha256 of decompressed upstream stream (s00, optional `--upstream`) |
| features | `data/linsight/genomic_feature.bed` | compgen.cshl.edu/LINSIGHT/genomic_feature.bed.gz, gunzipped | as above |
| published scores | `data/linsight/LINSIGHT.bw` | compgen.cshl.edu/LINSIGHT/LINSIGHT.bw | byte size = upstream (1,395,672,914) |
| LINSIGHT code | `third_party/linsight_cpp/*.cpp` (in the repository) | github.com/CshlSiepelLab/LINSIGHT | read as the specification; `LINSIGHT-prep` used once as a parser cross-check |
| hg19 | `data/genomes/hg19.fa` | UCSC hg19 | chromosome lengths = UCSC chromInfo |
| hg38 | `data/genomes/hg38.ml.fa` | the hg38 FASTA from the HyenaDNA repository's data instructions (the assembly HyenaDNA was trained on) | chromosome lengths = UCSC chromInfo |
| chain | `data/genomes/hg19ToHg38.over.chain.gz` | UCSC liftOver | md5 = UCSC md5sum.txt (`35887f73…`) |
| HyenaDNA weights | `data/hyenadna/hyenadna-medium-160k-seqlen/` | huggingface.co/LongSafari/hyenadna-medium-160k-seqlen | sha256 = HF LFS oid (`02729e5f…`) |
| HyenaDNA code | `third_party/hyena_dna/` (in the repository) | github.com/HazyResearch/hyena-dna @ d553021 (`standalone_hyenadna.py`, `huggingface.py` loader logic) | vendored verbatim, BSD-3 licence kept |

Not used (derived in earlier work, deliberately ignored): the `LINSIGHT-prep` binary files `chr*.fit`
(except as an independent comparison in a data test), every output of the earlier, superseded pipelines
(not in this repository), the published `LINSIGHT_hg38.bw`, and the C++ fit's `parameter.txt`.

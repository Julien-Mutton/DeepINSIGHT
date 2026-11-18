# Earlier work that is not in this repository

The canonical pipeline here is the last of nine experiments (March–November 2026). The earlier ones
were exploratory, superseded or negative. Their code stays in the project archive and is not
needed to reproduce the report. Most design rules of the pipeline answer a failure found in one of them.

| # | Question | Outcome | What it changed here |
|---|---|---|---|
| 01 | Can INSIGHT and LINSIGHT be run from source? | The authors' C++ LINSIGHT reproduces the published track exactly | kept as the C++ baseline (`cpp_baseline/`) and as reference binaries in the tests |
| 02 | First hybrid: 48 annotations + HyenaDNA (also NT-v2, soft prompts, LoRA) | near chance; traced to a likelihood error (a missing −Fρ term) and to a ClinVar stand-in benchmark | the INSIGHT likelihood is now tested term by term against `LINSIGHT-fit`; only the HGMD panel of the paper is used |
| 03 | Can HyenaDNA regress LINSIGHT scores from sequence alone (hg19 windows)? | Spearman 0.22 frozen, 0.38 fine-tuned | negative |
| 04 | The same on hg38, per base | Spearman 0.46 | negative; settled hg38 + liftover for the embedding |
| 05 | GPN-MSA / GPN-star as the sequence encoder | set up only; needs a GPU | listed as future work |
| 06 | Coordinate-exact hg38 hybrid with the corrected likelihood | sequence added ≤ 0.004 AUROC; the epoch chosen moved AUROC by up to 0.29 | final epoch only, several seeds, and an exact-likelihood estimator |
| 07 | First paper-strict pipeline with embedding and autoencoder models | the embedding lowered AUROC | rebuilt from scratch as this pipeline with matched controls and gates |
| 08 | Is the INSIGHT likelihood the bottleneck? | the information sits in about 0.5% of sites, but the data budget does not limit ranking | the power simulation (`s09c`) |
| 09 | **this repository** | canonical | — |

Benchmark warning: experiment 02 also built a ClinVar-based stand-in panel, on which LINSIGHT scores
0.94–0.98. Its AUROCs cannot be compared with the HGMD panel (0.66–0.90) and are not used anywhere.

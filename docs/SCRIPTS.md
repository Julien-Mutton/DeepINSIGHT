# What every script does

A guide to every file with code in this repository, grouped by folder. For the data that flows between
steps (file names, array shapes, checks), see `docs/DATA_FLOW.md`. For the formulas, see `docs/MATH.md`.

## The entry point: `src/deepinsight/cli.py`

`python -m deepinsight <command>` (or `deepinsight <command>` after `pip install -e .`).

| Command | What it runs |
|---|---|
| `paths` | prints where the inputs, outputs and Python environments are, and lists any missing input |
| `fetch` | `scripts/fetch_data.sh`: downloads every public input into `data/` |
| `build` | compiles the authors' LINSIGHT C++ (`third_party/linsight_cpp`, used as reference binaries in tests and for the baseline) and our C kernel (`src/lctx/model/_kernel.c`) |
| `test [--data]` | `pytest`: fast synthetic unit tests; `--data` adds the tests on the real input files |
| `steps` | lists the pipeline steps with one line each |
| `run --rate R` | runs the pipeline steps in order for a sample of `R` of all INSIGHT sites; `--from`, `--to`, `--only` select steps; `s10_evaluate` runs only with `--read-benchmark` |
| `cpp-baseline` | `cpp_baseline/run_linsight_cpp.sh`, then `cpp_baseline/evaluate_cpp.py` |
| `paper` | compiles `paper/main.tex` with `tectonic` |
| `all --rate R` | build, test, run, paper |

## `pipeline/`: the steps

Each step is a module `pipeline/steps/sNN_name.py` with a function `run(run)`. A step first calls
`run.gate(...)`, which stops unless the `checks.json` of every upstream step it needs says `"ok": true`.
It then writes its outputs to `outputs/rate_<R>/sNN_name/`, plus a `checks.json` that lists every check
(name, value, threshold, pass/fail) and every control (a deliberately wrong input that must fail).

| File | Role |
|---|---|
| `run_step.py` | runs one step: `python pipeline/run_step.py s04_features --rate 0.001` |
| `run_all.sh` | runs all steps in order (shell equivalent of `deepinsight run`; it includes `s10`) |
| `common.py` | the `Run` object (run directory, paths, gates), logging, saving arrays |
| `arms.py` | **the only place where model inputs are built.** `ARMS` lists every model; `Data.inputs(arm, seed)` returns its matrix `x` = 48 annotations + its added block, standardised with training-site statistics |
| `steps/s00_inputs.py` | every original exists and is the upstream file (checksums, sizes, genome lengths); runs the unit tests |
| `steps/s01_benchmark.py` | reads the four HGMD/GWAVA pickles (through the Python-2 env) into one panel; checks counts, reference alleles, and that the published LINSIGHT track reproduces the paper's AUROCs |
| `steps/s02_sites.py` | parses the 58-GB INSIGHT database in parallel, as `LINSIGHT-prep` does; keeps a reproducible hash-based sample of sites; removes sites within 1 kb of a benchmark variant; splits whole 1-Mb blocks into train / select / eval |
| `steps/s03_liftover.py` | hg19 → hg38 for every site and variant (HyenaDNA reads hg38); checks that the base and the 1,001-bp window agree |
| `steps/s04_features.py` | joins the 48 LINSIGHT annotations onto every site and variant; recovers the published weights from the published scores as a check |
| `steps/s05_embedder.py` | loads HyenaDNA and tests it: next-base accuracy above chance, deterministic, batch-invariant, causal |
| `steps/s06_embed.py` | the 256-number HyenaDNA embedding (mean of the last layer over the 1,001-bp window) for every site and variant; 100 CPU processes |
| `steps/s06b_composition.py` | four composition numbers per window (GC, CpG observed/expected, repeat fraction, longest run): a simple control input |
| `steps/s07_sae.py` | TopK sparse autoencoders on training-site embeddings (32/8, 64/16, 128/32 × 5 seeds) and a feature autoencoder; writes codes for every row |
| `steps/s08_design.py` | builds every model's `x` and re-derives 600 random rows from the original files with independent code; they must be equal |
| `steps/s09_train.py` | fits LINSIGHT and every variant exactly as the authors' C++ does (AdaGrad, mini-batches of 100, 20 epochs) |
| `steps/s09b_mle.py` | fits every model by exact penalised maximum likelihood (L-BFGS, prior widths chosen on `select` sites, two starts per fit); the held-out likelihood test with a block bootstrap |
| `steps/s09c_groundtruth.py` | power: simulates genomes from LINSIGHT's likelihood with a planted sequence effect of known size, then reruns `s09b` in each |
| `steps/s10_evaluate.py` | the HGMD read: writes the analysis plan and its hash to a lock **before** opening labels, refuses a second read of the same plan, computes AUROC and paired bootstrap differences. The file implements plan v2 (see README, "What reproduces what") |
| `steps/s11_report.py` | `REPORT.md` / `REPORT.html` of the run, built only from files on disk |
| `steps/dataflow_figure.py` | the plain-language walkthrough panel used inside `REPORT.html` |
| `tools/verify_upstream.sh` | SHA-256 of the two large CSHL files, streamed from upstream, vs your local copies |
| `tools/fetch_compare.sh` | downloads one upstream CSHL file with resume and compares it byte by byte with the local copy |

## `src/lctx/`: the library

| Module | Contents |
|---|---|
| `config.py` | every path (from `DEEPINSIGHT_DATA` / `DEEPINSIGHT_OUT`), the paper's constants (β, learning rates, epochs) and the design constants (window 1,001 bp, exclusion 1 kb, 1-Mb blocks, autoencoder grid, seeds, bootstrap size, prior grids) |
| `checks.py` | `CheckReport` (checks + controls → `checks.json`) and `require` (the gate) |
| `io/ins.py` | INSIGHT database parser with `LINSIGHT-prep`'s semantics (tested equal to it on real data) |
| `io/features.py` | streaming and random-access lookup in the 48-annotation BED |
| `io/genome.py` | FASTA access (0-based) and bigWig lookup |
| `io/benchmark.py` | reads the Python-2 pickles through the legacy env; `tier_mask` = positives + one negative set |
| `coords/liftover.py` | hg19 → hg38 with the UCSC chain; unique, same-chromosome, plus-strand only |
| `embed/hyena.py` | HyenaDNA tokenizer (official ids, no special tokens) and embedder (official model code in `third_party/hyena_dna`) |
| `seqcomp.py` | the four composition numbers |
| `sites.py` | exclusion around benchmark variants; block split into train / select / eval |
| `sae/topk.py` | TopK sparse autoencoder (Gao et al. 2024) with auxiliary loss for dead latents |
| `model/insight.py` | INSIGHT per-site likelihood terms, as in `L-INSIGHT-fit.cpp` |
| `model/linsight.py` + `_kernel.c` | LINSIGHT's model and the authors' training procedure (C kernel; tested equal to `LINSIGHT-fit` epoch by epoch) |
| `model/mle.py` | exact penalised maximum likelihood (float64 L-BFGS) |
| `model/simulate.py` | draws INSIGHT observations from a known ρ (for the power study) |
| `stats.py` | held-out likelihood comparisons, block bootstrap, AUROC difference bootstrap |
| `eval/metrics.py` | tie-aware AUROC, locus clusters, paired bootstrap, `SingleReadLock` |

## Other folders

| File | Role |
|---|---|
| `tests/unit/*.py` | fast tests on synthetic data, including our kernel vs the C++ `LINSIGHT-fit` and our parser vs `LINSIGHT-prep` |
| `tests/data/test_real_inputs.py` | the same comparisons on excerpts of the real files (`deepinsight test --data`) |
| `cpp_baseline/run_linsight_cpp.sh` | the authors' `LINSIGHT-prep` + `LINSIGHT-fit` on all sites (days of CPU) |
| `cpp_baseline/evaluate_cpp.py` | applies the C++ fit to the benchmark variants and computes AUROC per negative set |
| `scripts/fetch_data.sh` | downloads the public inputs |
| `paper/main.tex` | the report; `paper/NUMBERS.md` gives the source file of every number in it |
| `third_party/hyena_dna/` | official HyenaDNA model code (BSD-3), vendored verbatim |
| `third_party/linsight_cpp/` | official LINSIGHT C++ (BSD-2), vendored verbatim at commit f28b163 |

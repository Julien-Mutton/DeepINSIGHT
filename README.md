# DeepINSIGHT

**Does a genomic language model add information to LINSIGHT?**

LINSIGHT (Huang, Gulko & Siepel, *Nat. Genet.* 2017) scores purifying selection at each non-coding site
of the human genome from 48 genomic annotations. Its weights are fitted to human polymorphism and
human–chimpanzee divergence with the INSIGHT likelihood, without disease labels. This repository asks
whether adding a HyenaDNA reading of the surrounding DNA improves that model. The reading is added in
one of two forms: the 256-number embedding itself, or a short sparse-autoencoder code of it. "Improve"
is judged in two ways: (1) ranking HGMD non-coding disease variants above common variants (the benchmark
LINSIGHT was published on), and (2) the likelihood of withheld parts of the genome. Every added input
has a control of the same width (random columns, shuffled rows).

## Result (0.1% of sites)

No sequence input beat LINSIGHT on either test.

| HGMD AUROC | Unmatched | TSS-matched | Region-matched |
|---|---|---|---|
| Reported by Huang et al. 2017 | 0.897 | 0.759 | 0.660 |
| Published LINSIGHT track (reproduced) | 0.898 | 0.762 | 0.665 |
| Authors' C++ program, rerun by us on all sites | 0.898 | 0.762 | 0.665 |
| LINSIGHT (48 annotations), retrained at 0.1% | 0.887 | 0.748 | 0.664 |
| + HyenaDNA embedding (256) | 0.725 | 0.683 | 0.647 |
| + random columns (256), control | 0.827 | 0.719 | 0.654 |
| + autoencoder code (64) | 0.824 | 0.731 | 0.665 |
| + random columns (64), control | 0.849 | 0.730 | 0.657 |

- The embedding scores worse than random columns of the same width on two of the three negative sets.
- On withheld genome, no sequence input fits better than LINSIGHT.
- The power simulation shows that the withheld-genome test is underpowered at 0.1% of sites. The 1% run
  is the next step.

The sources are `results/rate_0.001/s10_evaluate/results.json`, `results/rate_0.001/s09b_mle/checks.json`,
`results/rate_0.001/s09c_groundtruth/checks.json` and `results/cpp_baseline/results.json`.
[`paper/NUMBERS.md`](paper/NUMBERS.md) traces every number in the report to its file.

## Quick start

```bash
git clone <this repo> DeepInsight && cd DeepInsight
conda env create -f environment.yml && conda activate deepinsight && pip install -e .
conda env create -f environment-legacy.yml      # Python 2 env that reads the benchmark pickles

python -m deepinsight fetch          # download the public inputs (~95 GB) into data/
#   then place the licensed HGMD benchmark pickles in data/benchmark/  (data/README.md)
python -m deepinsight paths          # check that every input is found
python -m deepinsight build          # compile the LINSIGHT C++ reference and our C kernel
python -m deepinsight test           # unit tests (~8 min)
python -m deepinsight run --rate 0.001                  # the whole pipeline, s00 … s09c, s11
python -m deepinsight run --rate 0.001 --only s10_evaluate --read-benchmark   # the locked HGMD read
python -m deepinsight paper          # compile the report (needs tectonic)
```

`python -m deepinsight --help` lists every command, and `python -m deepinsight steps` lists the
pipeline steps. Data and outputs can live outside the repository: set `DEEPINSIGHT_DATA` and
`DEEPINSIGHT_OUT`.

To rebuild the report without rerunning anything, run `python -m deepinsight paper`. Its numbers come
from the recorded results of our run in `results/`.

### Compute

Everything runs on CPU. Our run used one 112-core node with no GPU.

- At 0.1% of sites the pipeline needs about 35 GB of output space.
- The slow steps are the HyenaDNA embedding (`s06`, about 10–18 windows per second per process, with
  100 processes) and the exact-likelihood fits (`s09b`, `s09c`).
- The C++ baseline on all sites (`python -m deepinsight cpp-baseline`) is single-threaded and takes
  days.

## How the pipeline works

```
 original inputs (data/)        step                      output (outputs/rate_<r>/<step>/)
 benchmark pickles (HGMD)   ─► s01_benchmark    ─► panel.csv.gz        positives + 3 negative sets, hg19
 INSIGHT_database.ins       ─► s02_sites        ─► sites.npz           sampled sites, evolutionary terms, block split
 chain + hg19 + hg38        ─► s03_liftover     ─► *_lift.npz          hg38 position of every site / variant
 genomic_feature.bed        ─► s04_features     ─► *_features.npz      the 48 LINSIGHT annotations
 HyenaDNA weights           ─► s05_embedder     ─► checks only         the model loads and behaves
                            ─► s06_embed        ─► *_mean.npy          256-d embedding of the 1,001-bp window
 hg19                       ─► s06b_composition ─► *_comp.npy          4 composition numbers (control)
                            ─► s07_sae          ─► sae*.pt, codes_*    TopK sparse autoencoders (train sites only)
                            ─► s08_design       ─► checks only         every model's input re-derived from originals
                            ─► s09_train        ─► params, panel_rho   authors' procedure (AdaGrad, 20 epochs)
                            ─► s09b_mle         ─► params, eval_rho    exact penalised ML + withheld-genome test
                            ─► s09c_groundtruth ─► checks              power with a planted effect
                            ─► s10_evaluate     ─► results.json        the single locked HGMD read
                            ─► s11_report       ─► REPORT.md / .html
```

- Every step writes a `checks.json` and refuses to start unless the checks of the steps before it
  passed.
- HGMD is used only for evaluation. It is never used for training, tuning or model choice.
- The benchmark is read once per frozen analysis plan, and a lock file enforces this.

Documentation:

| Read | For |
|---|---|
| [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md) | every step: input files, what it does, output files with shapes, checks, and how results are evaluated |
| [`docs/SCRIPTS.md`](docs/SCRIPTS.md) | what every script and module does |
| [`data/README.md`](data/README.md) | the original inputs, where to get them, and the HGMD licence |
| [`docs/SPEC.md`](docs/SPEC.md), [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/MATH.md`](docs/MATH.md) | design, every decision in order (and why), and the formulas |
| [`docs/DATA.md`](docs/DATA.md) | provenance and upstream verification of each input |

## What reproduces what

| Report item | Command | Our recorded result |
|---|---|---|
| Published AUROCs reproduced from the published track | `run --to s01_benchmark` | `results/rate_0.001/s01_benchmark/checks.json` |
| Authors' C++ fit equals the published model | `cpp-baseline` | `results/cpp_baseline/` |
| Withheld-genome likelihood test (Table 2) | `run` (s09b) | `results/rate_0.001/s09b_mle/checks.json` → `info.heldout` |
| Power (Table 3) | `run` (s09c) | `results/rate_0.001/s09c_groundtruth/checks.json` |
| HGMD AUROCs and differences (Table 1) | see note below | `results/rate_0.001/s10_evaluate/results.json`, `lock.json` |

**Note on the HGMD read.** The HGMD numbers in the report come from the first frozen plan ("plan v1").
That plan used the authors' training procedure, a split with 10% of 1-Mb blocks held out, 3 seeds, and
7 score columns. It was read once, and the full plan is recorded in
`results/rate_0.001/s10_evaluate/lock.json`.

After that read, the design was extended into plan v2: a 50% split, 5 seeds, exact-likelihood fits, more
controls and Holm-adjusted families. `pipeline/steps/s10_evaluate.py` now implements plan v2, which has
not been read yet. The repository's defaults (`HELDOUT_FRAC = 0.50`, five seeds) are those of plan v2 and
of the withheld-genome analysis.

To rebuild the models behind the plan-v1 read, set `DEEPINSIGHT_HELDOUT_FRAC=0.10` and
`DEEPINSIGHT_SEEDS=0,1,2` before running `s02`–`s09`. Scoring them with the v1 plan would need the v1
evaluation code, which is not in this repository. Its plan is fully specified in `lock.json`.

## Repository layout

```
DeepInsight/
├── src/deepinsight/      the command-line runner (python -m deepinsight)
├── src/lctx/             the library: parsers, liftover, HyenaDNA embedder, autoencoder, LINSIGHT model, statistics
├── pipeline/             steps s00 … s11, arms.py (model inputs), run_step.py, run_all.sh, tools/
├── tests/                unit tests (synthetic) and data tests (real files)
├── cpp_baseline/         authors' C++ LINSIGHT on all sites + its HGMD evaluation
├── third_party/          HyenaDNA model code (BSD-3) and LINSIGHT C++ (BSD-2), vendored verbatim
├── paper/                main.tex, NUMBERS.md
├── results/              small result files of our 0.1% run and of the C++ baseline
├── docs/                 DATA_FLOW, SCRIPTS, SPEC, DECISIONS, MATH, DATA
├── data/                 the original inputs (not in git; data/README.md)
└── outputs/              pipeline outputs (not in git)
```

## What is not here

This repository holds only the analysis the report rests on. Earlier exploratory work is not included:

- first hybrid models with a likelihood error;
- HyenaDNA fine-tuning to regress LINSIGHT scores;
- a GPN-MSA set-up that needed a GPU;
- a first version of this pipeline;
- likelihood diagnostics.

Each of these was superseded or gave a negative result. [`docs/HISTORY.md`](docs/HISTORY.md) explains
what each one taught and how it shaped the current design.

## Still to do

- The 1% run (`run --rate 0.01`). Widen the prior grid for the feature weights first: the chosen width
  is at the edge of the grid.
- The plan-v2 HGMD read (one read allowed).
- An alignment-aware sequence encoder, which needs a GPU.

## Licence and citation

The code is under the MIT licence (`LICENSE`), and the third-party code keeps its own licence.

If you use this work, please cite the report and the methods it builds on:

- LINSIGHT: Huang, Gulko & Siepel 2017.
- INSIGHT: Gronau et al. 2013.
- HyenaDNA: Nguyen et al. 2023.
- TopK sparse autoencoders: Gao et al. 2024.
- The GWAVA benchmark: Ritchie et al. 2014.

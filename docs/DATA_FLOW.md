# DeepINSIGHT data flow: which file goes where, and what happens to it

Written for someone joining the project and updated with the v4 report (8 pages),
which leaves every path, step code, file format and check to this document. It traces the canonical pipeline
(this repository) from the original input files to the numbers in the report
(`paper/main.tex`), plus the C++ baseline check. Every path below was checked with `ls`,
and every shape was read from the file itself.

**Root.** All paths are relative to the repository root. The original inputs live under `data/`
(or `$DEEPINSIGHT_DATA`) in the layout of `data/README.md`; every path is defined once in
`src/lctx/config.py`. Pipeline outputs go to `outputs/` (or `$DEEPINSIGHT_OUT`). The small result files of
our 0.1% run (every `checks.json`, the HGMD `results.json` and `lock.json`) are shipped in
`results/rate_0.001/`, so `R/...` below can be read there without re-running anything.

**Run directory.** `R = outputs/rate_0.001/` holds every output of a 0.1%
run (`results/rate_0.001/` for our recorded run). Each step `sNN_*` has its own folder in `R` with the step's data files and a `checks.json`.

## 1. The question in one paragraph

LINSIGHT scores each non-coding site with a probability of being under purifying selection,
`rho = exp(-3 exp(-(b + w·x)))`, where `x` is a vector of 48 genomic annotations. The weights are fitted
to population-genetic data (human polymorphism and human–chimpanzee divergence) with the INSIGHT
likelihood; no disease labels are used. We ask whether appending a HyenaDNA reading of the
surrounding DNA to `x` makes LINSIGHT better. The reading is added in one of two alternative forms,
never both: the 256-number embedding itself, **or** a short sparse-autoencoder code of it (32, 64 or
128 numbers), which replaces the embedding. Each model therefore gets the 48 annotations plus at most
one added block (`pipeline/arms.py`, `Data.inputs`: `emb` = 48 + 256 columns; `sae<d>k<k>` = 48 + d). "Better" is measured two ways: (k) AUROC on the HGMD regulatory-variant benchmark that
LINSIGHT was published on, and (m) the likelihood of withheld genome. Every added input has a control
of the same width (random columns, shuffled rows, composition numbers).

## 2. End-to-end design

One shared architecture. A frozen HyenaDNA model reads the 1,001-bp window around
each site (the site and 500 bp on each side) and its last layer, averaged over the window, gives the
256-dimensional embedding **h**. **h** enters the model in one of two ways: directly (`x` = 48 + 256 = 304
columns) or compressed by a TopK sparse autoencoder into a code **z** of d = 32, 64 or 128 dimensions with
k = d/4 nonzero entries (`x` = 48 + d). The **baseline** is LINSIGHT with the 48 annotations alone, and each
added block has **control** models in which it is replaced by random or shuffled values of the same width.
No model uses both h and z. Every model is fitted with the INSIGHT likelihood to polymorphism (54 genomes)
and human-chimp divergence, with no disease labels, and evaluated on the HGMD benchmark (1,564 disease
variants vs. common variants in three negative sets; AUROC, and the difference from the baseline with a 95%
bootstrap interval). The withheld-genome likelihood test (§8) is a second evaluation.

## 3. Original inputs (read-only)

| Input | Path (as the code sees it) | Origin | Content / size | Check before use |
|---|---|---|---|---|
| INSIGHT database | `data/linsight/INSIGHT_database.ins` | compgen.cshl.edu/LINSIGHT/INSIGHT_database.ins.gz, gunzipped | text, 58,195,469,085 bytes. Header `samples 108`; `block <chr:start-end> theta <t> lambda <l>` lines (neutral rates); `site <chr:pos> <class> <pZmaj> [<pZmin>]` lines (polymorphism class M, L or H and ancestral-allele probabilities). hg19, 0-based positions | SHA-256 of the decompressed upstream stream equals the local file (`results/verify.tsv`, row INSIGHT_database.ins: `yes`) |
| 48 features | `data/linsight/genomic_feature.bed` | compgen.cshl.edu/LINSIGHT/genomic_feature.bed.gz, gunzipped | BED, 28,220,283,397 bytes; 48 annotation values per interval (hg19) | byte comparison with the fully downloaded upstream file: `IDENTICAL` (`results/genomic_feature.bed.compare.txt`). The `NO` row in `verify.tsv` is from an earlier truncated download, superseded by that comparison |
| Published scores | `data/linsight/LINSIGHT.bw` | compgen.cshl.edu/LINSIGHT/LINSIGHT.bw | bigWig, 1,395,672,914 bytes; genome-wide published `rho` | byte size equals upstream (`R/s00_inputs/checks.json`) |
| LINSIGHT C++ | `third_party/linsight_cpp/` (`L-INSIGHT-prep.cpp`, `L-INSIGHT-fit.cpp`; binaries `LINSIGHT-prep`, `LINSIGHT-fit` built by `python -m deepinsight build`) | github.com/CshlSiepelLab/LINSIGHT, vendored at commit f28b163 | read as the specification; binaries used as references in tests | — |
| HGMD benchmark | `data/benchmark/hgmd_reg.pandas`, `unmatched.pandas`, `matched_by_tss.pandas`, `matched_by_region.pandas` | GWAVA / LINSIGHT benchmark (Ritchie et al. 2014; Huang et al. 2017), Python-2 pandas pickles | 1,614 HGMD regulatory variants; negatives 161,400 / 16,140 / 5,027 | counts equal the published counts (`R/s01_benchmark/checks.json`) |
| hg19 | `data/genomes/hg19.fa` | UCSC | FASTA, 3,199,905,909 bytes; soft-masked (lower case = repeats) | autosome lengths equal UCSC chrom.sizes |
| hg38 | `data/genomes/hg38.ml.fa` | the hg38 FASTA from HyenaDNA's data instructions (the assembly it was trained on) | FASTA, 3,091,663,428 bytes; chr1–22, chrX | autosome lengths equal UCSC |
| Liftover chain | `data/genomes/hg19ToHg38.over.chain.gz` | UCSC | 227,698 bytes | MD5 `35887f73…` equals UCSC |
| HyenaDNA | weights: `data/hyenadna/hyenadna-medium-160k-seqlen/weights.ckpt` (+ `config.json`: 8 layers, `d_model` 256); code: `third_party/hyena_dna/` | huggingface.co/LongSafari/hyenadna-medium-160k-seqlen; github.com/HazyResearch/hyena-dna @ d553021 | — | SHA-256 `02729e5f…` equals the HuggingFace LFS id |

Paths and checksums are also written by step `s00` to `R/s00_inputs/checks.json`.
The list of originals and their provenance is `docs/DATA.md`.

## 4. How to run

```bash
conda activate deepinsight          # environment.yml; then: pip install -e .
python -m deepinsight paths         # are all inputs in place?
python -m deepinsight test          # unit tests (also run inside s00)
python -m deepinsight run --rate 0.001                   # every step in order; stops at the first failed gate
python -m deepinsight run --rate 0.001 --from s06_embed  # resume from a step
```

`pipeline/run_step.py <step> --rate <r>` runs a single step. Each step calls `run.gate(...)`, which
refuses to start unless the named upstream `checks.json` files report `"ok": true`. Each step writes
its own `checks.json` with every check (name, value, threshold, pass/fail) and its log to `R/logs/<step>.log`.

**`s10_evaluate` is skipped unless `--read-benchmark` is given.** It opens the HGMD benchmark under a lock that allows one read per
analysis plan (the lock lives in your run directory). The file now on disk defines plan v2 (models fitted by exact likelihood), which has not been read
yet and is reserved for the final analysis.

## 5. Step by step

**Names.** The report calls each fitted variant a *model* (e.g. "the LINSIGHT + autoencoder-code
model") and the random, shuffled and composition variants *control models*. The code keeps its older
identifier for a model variant, `arm` (`pipeline/arms.py`, `ARMS`, `PAPER_ARMS`, `Data.inputs(arm, seed)`,
array names such as `repl_s0`); where this document quotes such an identifier it is in backticks.

Notation: *n_s* = 1,362,112 sampled sites (rows of every `sites_*` array), *n_p* = 184,181 benchmark
rows (rows of every `panel_*` array). Site rows are in the order of `R/s02_sites/sites.npz`; panel rows
are in the order of `R/s01_benchmark/panel.csv.gz` and carry its `key`. Both sets pass through the same
functions.

### s00 inputs: are the originals the upstream files?
- **Script:** `pipeline/steps/s00_inputs.py`
- **In:** every original in §3.
- **Does:** checks existence; hashes HyenaDNA weights (SHA-256) and the chain (MD5); compares the
  bigWig byte size and genome autosome lengths with UCSC; runs the unit tests.
- **Out:** `R/s00_inputs/checks.json` (paths, byte sizes). Gate: 7/7 checks passed.

### s01 benchmark: HGMD positives and three negative sets
- **Script:** `pipeline/steps/s01_benchmark.py` (reader: `src/lctx/io/benchmark.py`; the pickles are
  read through the Python-2-compatible env `legacy_pandas`).
- **In:** the four `data/benchmark/*.pandas` pickles; `hg19.fa` (allele check); `LINSIGHT.bw`.
- **Does:** concatenates positives (`hgmd_reg`, label 1) and the three negative sets (label 0) into
  one panel; keeps the 0-based hg19 `start` as `pos19`; looks up the published LINSIGHT score of
  every variant; flags one negative that shares a coordinate with a positive (`label_conflict`).
- **Out:** `R/s01_benchmark/panel.csv.gz`: 184,181 rows × 11 columns
  (`key, source, row_id, chrom, pos19, end, cls, label, ref, label_conflict, linsight_published`).
- **Gate (15/15, 2 controls):** counts 1,614 / 161,400 / 16,140 / 5,027; reference allele equals hg19 at
  `pos19` for 100% of rows (control: 24.6% one base off); the published track at `pos19` reproduces the
  paper's AUROCs within 0.01 (0.8977 / 0.7618 / 0.6643 vs 0.897 / 0.759 / 0.660).

### s02 sites: which genomic sites are used and how they are split
- **Script:** `pipeline/steps/s02_sites.py` (parser `src/lctx/io/ins.py`, split `src/lctx/sites.py`).
- **In:** `INSIGHT_database.ins`; `R/s01_benchmark/panel.csv.gz` (for the exclusion); `hg19.fa`.
- **Does:** parses the 58-GB file in 200 parallel chunks with the semantics of `LINSIGHT-prep`; keeps a
  site if a hash of (chromosome, position) is below the rate 0.001, so samples are reproducible and
  nested; turns each site record into 5 numbers (allele-frequency class 0 / 0.05 / 0.5, P(ancestral =
  major), P(ancestral = minor), θ·a and λ of its block, clamped at 1e-5); 1,542,881 autosomal sites
  sampled. (θ·a uses Watterson's constant a = Σ_{k=1}^{107} 1/k for the 108 sampled chromosomes.) Tests the coordinate convention (0-based: 0.0% CpG at offset 0 vs 0.73% / 0.79% at ∓1).
  Removes sites within 1,000 bp of any benchmark variant (11.7%), leaving 1,362,112. Splits whole 1-Mb
  hg19 blocks: 49% train; held-out blocks are divided into `select` (10% of held-out, model choice
  only) and `eval` (tests only).
- **Out:** `R/s02_sites/sites.npz` with `chrom` (int8, n_s), `pos_ins` (int64, n_s), `pos19` (int64,
  n_s), `evol` (float32, n_s × 5), `role` (str, n_s: train 670,261 / select 70,332 / eval 621,519).
- **Gate (22/22, 2 controls):** sample fraction / rate = 1.0014; site-class fractions equal the whole
  database; no block on both sides of the split; every autosome in every role; nearest kept site is
  1,006 bp from any benchmark variant.
- **Note:** the HGMD read (s10) used an earlier split of the same sites with 10% of blocks held out
  (1,212,074 / 72,029 / 78,009; `R/REPORT.md`). The file on disk now holds the 51% split, which the
  exact-likelihood analysis uses.

### s03 liftover: hg19 → hg38
- **Script:** `pipeline/steps/s03_liftover.py` (`src/lctx/coords/liftover.py`).
- **In:** `sites.npz`, `panel.csv.gz`, chain, `hg19.fa`, `hg38.ml.fa`.
- **Does:** lifts every site and variant with pyliftover and the UCSC chain; keeps unique, same-chromosome,
  plus-strand mappings. Rejected sites: 895 minus strand, 76 other chromosome, 19 unmapped. Rejected
  variants: 320, 57 and 123 for the same reasons, plus 2 on a chromosome absent from `hg38.ml.fa`.
- **Out:** `R/s03_liftover/sites_lift.npz`: `pos38` (int64, n_s), `status` (str, n_s; `ok` for 1,361,122);
  `R/s03_liftover/panel_lift.npz`: `key`, `pos38`, `status` (n_p; `ok` for 183,679).
- **Gate (17/17, 4 controls):** lifted 99.93% of sites, 99.73% of variants; same base in hg19 and hg38
  for 100% / 99.85%; 1,001-bp window identity 0.9998 / 0.9992; the off-by-one control (pos38+1) fails.

### s04 features: the 48 LINSIGHT annotations
- **Script:** `pipeline/steps/s04_features.py` (`src/lctx/io/features.py`).
- **In:** `genomic_feature.bed`, `sites.npz` (at `pos_ins`, as `LINSIGHT-fit` does), `panel.csv.gz` (at
  `pos19`), `LINSIGHT.bw`.
- **Does:** streams the 28-GB BED once and joins the 48 values onto every site and variant. Then
  inverts the link (`u = -log(-log(bw)/3)`) and regresses `u` on the features over benchmark variants:
  this recovers the published weights exactly, without any label.
- **Out:** `R/s04_features/sites_features.npz`: `F` (float32, n_s × 48), `found` (bool, n_s; 98.6%);
  `R/s04_features/panel_features.npz`: `key`, `F` (n_p × 48), `found` (98.8%);
  `R/s04_features/published_weights.npz`: `rho_coef` (49 = intercept + 48 weights).
- **Gate (10/10, 3 controls):** linear fit R² = 1.000; the recovered weights reproduce the published
  score to 2e-3 at 100% of variants and sites (controls with other rows' features: 21% and 18%;
  score 5 kb away: 32%).

### s05 embedder: does HyenaDNA load and behave?
- **Script:** `pipeline/steps/s05_embedder.py` (`src/lctx/embed/hyena.py`).
- **In:** HyenaDNA weights and code, `hg38.ml.fa`, 64 lifted sites.
- **Out:** `R/s05_embedder/checks.json` only.
- **Gate (10/10, 2 controls):** 227 tensors loaded; next-base accuracy 43.5% on hg38 windows (control:
  25.3% on random sequence); deterministic; batch-invariant; causal (mutating base 500 does not change
  states before 500; mutating base 501 does not change state 500).

### s06 embed: the 256-number sequence representation
- **Script:** `pipeline/steps/s06_embed.py`.
- **In:** `sites_lift.npz`, `panel_lift.npz`, `hg38.ml.fa`, HyenaDNA.
- **Does:** for every lifted site and variant, takes the 1,001-bp hg38 window with the site at index
  500, maps bases to the official token ids (A, C, G, T, N = 7–11, case ignored, any other character 6;
  no special tokens, as in HyenaDNA's hg38 pretraining), runs HyenaDNA in evaluation mode on CPU (100
  processes, 16 windows per batch), and averages the 1,001 final hidden states. The centre state is
  stored too (diagnostic only). Embeddings are reused only if the SHA-256 of all windows, weights and
  window length is unchanged (`targets.sha256`).
- **Out:** `R/s06_embed/sites_mean.npy` (float32, n_s × 256, 1.39 GB); `R/s06_embed/panel_mean.npy`
  (float32, n_p × 256); `sites_center.npy`, `panel_center.npy` (same shapes); rows that did not lift
  are NaN.
- **Gate (13/13, 2 controls):** every lifted row finite; a fresh process re-embeds 100 random rows with
  cosine 1.0 (control: shuffled pairing matches 1%).

### s06b composition: four simple numbers per window (control input)
- **Script:** `pipeline/steps/s06b_composition.py` (`src/lctx/seqcomp.py`).
- **In:** `sites.npz`, `panel.csv.gz`, soft-masked `hg19.fa`.
- **Out:** `R/s06b_composition/sites_comp.npy` (float32, n_s × 4, from the 1,001-bp hg19 window centred on
  `pos19`: GC fraction among A/C/G/T; CpG observed/expected n_CG·L/(n_C·n_G) with L = 1,001, 0 if n_C or n_G is
  0; repeat fraction = share of lower-case bases; longest single-base run / L); `R/s06b_composition/panel_comp.npy` (n_p × 4).
- **Gate (13/13, 2 controls):** genome-typical means (GC 0.404); random rows recomputed equal; a window
  5 kb away differs.

### s07 sae: sparse autoencoder codes
- **Script:** `pipeline/steps/s07_sae.py` (`src/lctx/sae/topk.py`).
- **In:** `sites_mean.npy` (train rows only), `panel_mean.npy`, `sites.npz` (roles),
  `sites_features.npz` and `panel_features.npz` (for the feature autoencoder).
- **Does:** fits TopK autoencoders on train-site embeddings only (669,704 rows; no labels, no benchmark
  rows; cap 1,000,000 rows, not reached). Each embedding is centred on the training mean and divided by one
  scalar (RMS norm of the centred rows / √256). Encoder: z = TopK_k(ReLU(W_enc(e − b_dec) + b_enc));
  decoder: ê = W_decᵀ z + b_dec; decoder rows kept at unit norm; encoder initialised as the decoder transpose.
  Loss: mean squared reconstruction error + 1/32 × auxiliary loss in which latents silent for 500 batches
  reconstruct the residual with their top 2k. Adam, lr 1e-3, 10 epochs, batch 4,096; the seed sets the
  initial weights and batch order. Grid: for
  (latents, active) = (32, 8), (64, 16), (128, 32) × seeds 0–4; encodes every site and variant. Also
  fits a (32, 8) autoencoder on the 48 features (`fsae32k8`) for the feature-autoencoder models.
- **Out:** models `R/s07_sae/sae{32k8,64k16,128k32}_s{0..4}.pt`, `fsae32k8_s{0..4}.pt`; codes
  `R/s07_sae/codes_<name>_s<seed>_sites.npy` (float32, n_s × d) and `..._panel.npy` (n_p × d), with
  d = 32, 64, 128 (and 32 for `fsae32k8`).
- **Gate (90/90):** at most k active latents per row; no dead latents; held-out fraction of variance
  unexplained (FVU) below PCA with k components for every model and seed (mean of 5 seeds: 0.049 /
  0.028 / 0.016 vs PCA 0.074 / 0.041 / 0.021); overfit (held-out − train FVU) ≤ 0.05.
- **Note:** the files on disk were fitted on the training rows of the current 51% split. The HGMD read
  used autoencoders fitted the same way on the training rows of the earlier 10% split; those files were
  replaced when the step was re-run.

### s08 design: one function builds every model's input `x`
- **Script:** `pipeline/steps/s08_design.py`; the inputs are built only in `pipeline/arms.py`
  (`Data.inputs(arm, seed)`, where `arm` names the model variant).
- **In:** everything above.
- **Does:** defines each model's `x` as 48 features followed by its added columns; added columns are
  standardised with means and SDs of the train rows, and the same statistics are applied to held-out
  sites and benchmark variants. A row is usable if it has features, a valid liftover, a finite
  embedding and composition (sites: train 660,061 / select 69,412 / eval 612,686; benchmark 181,397
  of 184,181). Then, for 300 random training sites and 300 benchmark variants, rebuilds every row of
  every model from the original files with independent code (byte-bisection BED lookup, fresh liftover,
  fresh HyenaDNA, re-loaded autoencoders) and requires equality.
- **Models** (name in the report ← name in code; added width):

| Report name | Code name | Added columns |
|---|---|---|
| LINSIGHT (48 features) | `repl` | 0 |
| + HyenaDNA embedding (256) | `emb` | 256 (`s06` mean embedding) |
| + autoencoder code (32 / 64 / 128) | `sae32k8`, `sae64k16`, `sae128k32` | 32 / 64 / 128 (`s07` codes) |
| + random columns (256 / 32 / 64 / 128) | `noise_emb`, `noise32`, `noise64`, `noise128` | i.i.d. N(0,1), fixed seeds |
| + shuffled embedding / code | `shuf_emb`, `shuf_sae*` | real values permuted across usable rows |
| + composition (4) | `comp` | `s06b` numbers |
| feature autoencoder: 32 codes replace the features | `fsae_codes` | 48 features removed, 32 codes of `fsae32k8` |
| feature autoencoder: features + 32 codes | `fsae_plus` | 32 codes of `fsae32k8` |
| + positive control | `posctl` | 47 features + the 48th (largest published weight) through the added channel |

- **Out:** `R/s08_design/checks.json` only (the matrices are rebuilt in memory by `arms.py` in every
  later step). **Gate (296/296, 74 controls):** re-derived rows equal stored rows (100%; ≥ 99% for
  autoencoder-code models because of TopK ties); swapped-row controls fail; train and benchmark share columns
  and standardisation.

### s09 train: LINSIGHT as its authors fit it
- **Script:** `pipeline/steps/s09_train.py` (`src/lctx/model/linsight.py` + C kernel `_kernel.c`,
  site terms `src/lctx/model/insight.py`).
- **In:** model inputs from `arms.py`; the `evol` terms of `sites.npz`.
- **Does:** re-implements `LINSIGHT-fit`: AdaGrad, mini-batches of 100 sites, learning rates 1e-3 (rho)
  and 1e-2 (gamma'), zero start, 20 epochs, final epoch kept; seed 0 visits sites in the C++ order,
  other seeds shuffle. Models: `repl, emb, sae32k8, sae64k16, sae128k32, noise_emb, noise32, noise64,
  noise128, shuf_emb, posctl`. Scores held-out sites each epoch and every benchmark variant at the end.
- **Details:** AdaGrad offset 1e-6; β₀ = 0.7567672441, β₁ = 0.2053143291; polymorphism-class threshold
  0.15; `x` enters both rho and gamma'.
- **Out (current files: 51% split, 5 seeds):** `R/s09_train/params.npz`: 55 arrays `<arm>_s<seed>`
  of shape (20 epochs × 2(1 + width)), e.g. `repl_s0` 20 × 98, `emb_s0` 20 × 610;
  `R/s09_train/panel_rho.npz`: 55 arrays of length n_p (rho of every benchmark variant).
- **Gate (24/24):** finite parameters and the expected shape (20 epochs × 2(1 + width)) for all 11
  models; two further checks compare scores with the published track and are not used in the report.
  LINSIGHT's 48 weights vs published: Pearson 0.856, norm ratio 0.143 (`info` of `checks.json`).
- **Note on the HGMD read:** the benchmark was read with this procedure on the earlier 10% split and
  three seeds. Those `params.npz` / `panel_rho.npz` were replaced when the step was re-run on the 51%
  split; what remains of that run is `R/s09_train/selection.json` (autoencoder chosen: `sae64k16`,
  because it gave the highest likelihood on `select` sites of the three sizes), `R/REPORT.md` (held-out
  likelihood per model), and the read itself in `R/s10_evaluate/`. On that split's `eval` sites, every
  sequence and random-column model fitted held-out data about as well as LINSIGHT; the positive control
  fitted them slightly better although it carries no new information, which shows that this procedure
  is sensitive to how a column is scaled. That sensitivity is why every model was also fitted by exact
  penalised maximum likelihood (s09b). These held-out values were not tested statistically and are not in
  the v4 report.

### s09b mle: exact penalised maximum likelihood (the held-out likelihood test)
- **Script:** `pipeline/steps/s09b_mle.py` (`src/lctx/model/mle.py`, `src/lctx/stats.py`).
- **In:** model inputs from `arms.py` (all 17 models); `sites.npz`; `LINSIGHT.bw` (read for a diagnostic that the report does not use).
- **Does:** fits each model by L-BFGS on the full training set (660,061 sites) in float64 to max gradient
  1e-6, with Gaussian priors: width for the 48 feature weights chosen once by LINSIGHT on `select`
  sites from {0.03, 0.1, 0.3, 1, 3, 10} (chosen: 10, at the edge of the grid), then fixed; width for
  added columns chosen per model and seed on `select` from {0.003, 0.01, 0.03, 0.1, 0.3, 1, 10}. Every
  fit runs from two starts (zeros; LINSIGHT's optimum with the new weights at 0) and keeps the lower
  penalised training objective (the zero start won 293 times, the LINSIGHT start 177). Added columns
  enter rho only. The latent b + w·x is bounded at ±30. Then scores the 612,686 `eval` sites and
  compares, for each pair of models, how probable each makes the observed data of every site, averaged
  over sites (and over five seeds for seeded models), with a bootstrap over 1,187 withheld 1-Mb blocks
  (2,000 replicates).
- **Out:** `R/s09b_mle/params.npz` (69 arrays, e.g. `repl_s0` length 98, `emb_s0` 354);
  `R/s09b_mle/panel_rho.npz` (69 arrays of length n_p; for the pending HGMD read);
  `R/s09b_mle/eval_rho.npz` (69 arrays of length 612,686); `R/s09b_mle/selection.json` (chosen widths;
  autoencoder chosen: `sae32k8`); every individual fit in `R/s09b_mle/fits_7b47ab16b66a/` (1,332 files,
  14 GB). Results are in `R/s09b_mle/checks.json` → `info.heldout`.
- **Gate (1,041/1,041):** every selectable fit converged; models that contain LINSIGHT never fit training data worse
  than LINSIGHT; positive control equals LINSIGHT's training objective to 1e-9 and is indistinguishable from it on
  held-out sites.

### s09c groundtruth: how much power does the held-out test have?
- **Script:** `pipeline/steps/s09c_groundtruth.py` (`src/lctx/model/simulate.py`).
- **In:** the real sites, features, embeddings and neutral rates; LINSIGHT's exact fit from `s09b`.
- **Does:** simulates new INSIGHT records from LINSIGHT's own likelihood with a true latent score =
  LINSIGHT's + s × (fixed random projection of the standardised embedding), s ∈ {0, 0.025, 0.05, 0.1,
  0.2, 0.4} (20 worlds at s = 0, 5 otherwise), and runs the `s09b` procedure in every world.
- **Out:** `R/s09c_groundtruth/checks.json` (`info.summary`, `info.rows`: 85 world × model results);
  fits in `R/s09c_groundtruth/fits_c60a673719b9/` (1,244 files, 5.7 GB).
- **Gate (178/180):** the two failing checks are the result: power at s = 0.4 is 0.2 (target 0.8),
  and the embedding model does not recover the true rho better than LINSIGHT.

### s10 evaluate: the single HGMD read
- **Script:** `pipeline/steps/s10_evaluate.py` (`src/lctx/eval/metrics.py`). **Do not run** (see §4).
- **In (for the read that was made):** the 10%-split, 3-seed authors'-procedure scores of 7 columns
  (published, `repl`, `emb`, `sae64k16`, `noise_emb`, `noise64`, `posctl`) on 181,397 usable variants
  (98.49% of the panel).
- **Does:** writes the plan (columns, comparisons, seeds, bootstrap) and its hash to the lock before
  opening the labels; refuses a second read of the same plan; computes, per negative set, AUROC of the
  seed-averaged rho (Mann–Whitney, ties one half; equals scikit-learn to 1e-9) and paired differences
  with a bootstrap over 100-kb loci (2,000 replicates).
- **Out:** `R/s10_evaluate/results.json` (per negative set: `n`, `n_pos`, `auroc` per column,
  `deltas` with `lo`/`hi`/`p_two_sided`, `auroc_per_seed`); `R/s10_evaluate/lock.json` (one read,
  plan `89b069b68c299e39`).
- **Gate (3/3):** paired rows 0.9849 of the panel; AUROC equals scikit-learn; exactly one read.

### s11 report
- **Script:** `pipeline/steps/s11_report.py` → `R/REPORT.md`, `R/REPORT.html` (tables of `results.json`,
  held-out likelihood per model of the 10%-split run).

## 6. C++ baseline check (outside the pipeline)

Purpose: show that the authors' own program, run by us on all sites, reproduces the published model.

| Item | Detail |
|---|---|
| Step 1 | `cpp_baseline/run_linsight_cpp.sh`: `LINSIGHT-prep -i data/linsight/INSIGHT_database.ins -o outputs/cpp_baseline/binary_dir/` → 22 binary files `chr*.fit` (41 GB) |
| Step 2 | same script: `LINSIGHT-fit -d binary_dir/ -f data/linsight/genomic_feature.bed -o outputs/cpp_baseline/parameter.txt -n 20 -b1 0.7567672441 -b2 0.2053143291` → 20 rows (epochs) × 98 numbers (rho intercept + 48 weights, gamma' intercept + 48 weights) |
| Evaluation | `cpp_baseline/evaluate_cpp.py`: final-epoch weights applied to the benchmark variants' features (`R/s04_features/panel_features.npz`), AUROC per negative set |
| Result | our run (March 2026): `results/cpp_baseline/parameter.txt` and `results/cpp_baseline/results.json`: 0.8977 / 0.7617 / 0.6645, equal to the published track to four decimals. (Originally computed by the benchmark step of an earlier pipeline version, which is not in this repository.) |
| One command | `python -m deepinsight cpp-baseline` (both steps + evaluation) |

## 7. File → produced by → consumed by

| File (in `R/` unless a full path is given) | Produced by | Consumed by |
|---|---|---|
| `data/benchmark/*.pandas` | original (licensed, not distributed) | s01 |
| `data/linsight/INSIGHT_database.ins` | original | s02; C++ `LINSIGHT-prep` |
| `data/linsight/genomic_feature.bed` | original | s04, s08; C++ `LINSIGHT-fit` |
| `data/linsight/LINSIGHT.bw` | original | s01, s04, s09, s09b |
| `data/genomes/hg19.fa` | original | s01, s02, s03, s06b, s08 |
| `data/genomes/hg38.ml.fa` | original | s03, s05, s06, s08 |
| `data/genomes/hg19ToHg38.over.chain.gz` | original | s03, s08 |
| HyenaDNA weights + `third_party/hyena_dna/` | original | s05, s06, s08 |
| `s01_benchmark/panel.csv.gz` | s01 | s02, s03, s04, s06, s06b, s08–s10 (`arms.Data`) |
| `s02_sites/sites.npz` | s02 | s03, s04, s05, s06, s06b, s07, s08–s09c |
| `s03_liftover/sites_lift.npz`, `panel_lift.npz` | s03 | s05, s06, s08 (usable rows) |
| `s04_features/sites_features.npz`, `panel_features.npz` | s04 | s07 (feature autoencoder), s08–s10 |
| `s04_features/published_weights.npz` | s04 | s08 (positive-control feature), s09 (weight comparison) |
| `s06_embed/sites_mean.npy`, `panel_mean.npy` | s06 | s07, s08–s09c |
| `s06b_composition/sites_comp.npy`, `panel_comp.npy` | s06b | s08–s09b (`comp` model, usable rows) |
| `s07_sae/*.pt`, `s07_sae/codes_*.npy` | s07 | s08 (re-derivation), s09, s09b, s09c |
| `s09_train/params.npz`, `panel_rho.npz` | s09 | s10 (plan v2, pending) |
| `s09_train/selection.json` | s09 (10% split) | the HGMD read (choice of `sae64k16`) |
| `s09b_mle/params.npz`, `eval_rho.npz`, `selection.json`, `checks.json` | s09b | s09c, s10 (plan v2), report Table 2 |
| `s09b_mle/panel_rho.npz` | s09b | s10 (plan v2, pending) |
| `s10_evaluate/results.json`, `lock.json` | s10 (plan v1, one read) | s11, report Table 1 |
| `s09c_groundtruth/checks.json` | s09c | report Table 3 (power) |
| `REPORT.md` | s11 | this document (s09 note); not in the v4 report |
| `outputs/cpp_baseline/parameter.txt` (ours: `results/cpp_baseline/`) | C++ `LINSIGHT-fit` | `cpp_baseline/evaluate_cpp.py` |
| `outputs/cpp_baseline/results.json` (ours: `results/cpp_baseline/results.json`) | `cpp_baseline/evaluate_cpp.py` | report (baseline check) |

## 8. Evaluation and results

### (k) HGMD benchmark: AUROC

- **Positives:** 1,564 usable HGMD regulatory variants. **Negatives:** 159,765 unmatched, 15,346
  TSS-matched, 4,722 region-matched (usable rows of the three sets).
- **Metric:** AUROC of each model's seed-averaged rho (Mann–Whitney; ties one half).
- **Test:** paired difference between two models on the same variants; 95% interval and two-sided p
  from a bootstrap that resamples 100-kb loci (2,000 replicates).
- **Which models:** authors' procedure, 10% split, three seeds, one read (plan `89b069b68c299e39`).
- **Source:** `R/s10_evaluate/results.json`.

| Model | Unmatched | TSS-matched | Region-matched |
|---|---|---|---|
| Reported by Huang et al. 2017 | 0.897 | 0.759 | 0.660 |
| Published LINSIGHT track | 0.898 | 0.762 | 0.665 |
| LINSIGHT (48 features), retrained | 0.887 | 0.748 | 0.664 |
| + positive control | 0.884 | 0.745 | 0.663 |
| + HyenaDNA embedding (256) | 0.725 | 0.683 | 0.647 |
| + random columns (256) | 0.827 | 0.719 | 0.654 |
| + autoencoder code (64) | 0.824 | 0.731 | 0.665 |
| + random columns (64) | 0.849 | 0.730 | 0.657 |

| Difference (95% CI) | Unmatched | TSS-matched | Region-matched |
|---|---|---|---|
| Retrained LINSIGHT − published | −0.011 [−0.015, −0.007] | −0.013 [−0.019, −0.008] | −0.0003 [−0.008, +0.007] |
| Embedding − LINSIGHT | −0.162 [−0.182, −0.143] | −0.065 [−0.083, −0.048] | −0.018 [−0.031, −0.004] |
| Embedding − random (256) | −0.103 [−0.123, −0.082] | −0.036 [−0.055, −0.016] | −0.008 [−0.023, +0.007] |
| Autoencoder − LINSIGHT | −0.063 [−0.075, −0.052] | −0.017 [−0.027, −0.008] | +0.001 [−0.006, +0.008] |
| Autoencoder − random (64) | −0.025 [−0.037, −0.015] | +0.001 [−0.009, +0.011] | +0.008 [−0.0003, +0.016] |

**Result:** no model with a sequence input beats LINSIGHT on any negative set; the embedding is worse than random
columns of its size on two sets; the autoencoder code is level with random columns on the two matched
sets.

### (m) Held-out likelihood (withheld genome)

- **Data:** 612,686 `eval` sites in 1,187 withheld 1-Mb blocks (51% split).
- **What is measured:** each model is fitted on the training blocks and then asked how probable the
  real polymorphism and divergence data of every `eval` site are. A model that has learned something
  true about selection gives the withheld data a higher probability. Two models are compared by the
  average over sites of the difference (paired: same sites), after averaging over seeds for seeded
  models. The stored values (`mean`, `lo`, `hi` in `info.heldout`) are per-site negative
  log-likelihood differences; they are not reported as numbers, only as a direction and a p-value.
- **Test:** bootstrap over blocks (2,000 replicates); 95% percentile interval; two-sided p from the
  bootstrap standard error (normal approximation); not adjusted for the 22 comparisons (about one
  p < 0.05 is expected among them by chance).
- **Which models:** exact penalised ML (`s09b`), five seeds for seeded models.
- **Source:** `R/s09b_mle/checks.json` → `info.heldout` (key `"<model> - <compared with>"`).

| Model | Compared with | Estimate | p | Verdict (5% level) |
|---|---|---|---|---|
| + HyenaDNA embedding (256) | LINSIGHT | worse | 0.12 | no difference |
| + HyenaDNA embedding (256) | random columns (256) | worse | 0.04 | worse |
| + HyenaDNA embedding (256) | shuffled embedding | worse | 0.29 | no difference |
| + HyenaDNA embedding (256) | composition (4) | worse | 0.08 | no difference |
| + autoencoder code (32) | LINSIGHT | better | 0.99 | no difference |
| + autoencoder code (32) | random columns (32) | worse | 0.46 | no difference |
| + autoencoder code (32) | shuffled code (32) | better | 0.53 | no difference |
| + autoencoder code (64) | LINSIGHT | better | 0.29 | no difference |
| + autoencoder code (128) | LINSIGHT | better | 0.26 | no difference |
| + positive control | LINSIGHT | worse | 0.25 | no difference |
| feature autoencoder: 32 codes replace the 48 features | LINSIGHT | better | < 0.0001 | better |
| 48 features + 32 feature codes | LINSIGHT | better | 0.27 | no difference |

The other 10 comparisons in `info.heldout` (each random, shuffled and composition control against
LINSIGHT, and features + codes against codes only) are not in the report; none of the nine controls differs
from LINSIGHT except the shuffled embedding (worse, p = 0.04), and features + codes is worse than codes
only (p < 0.0001), the mirror image of the feature-autoencoder result.

**Result:** no sequence input improves the held-out fit. The feature autoencoder improves it on `eval`
sites, but on the `select` sites used for model choice it was worse than LINSIGHT (`info.select_nll`:
`fsae_codes` above `repl`), so it is a secondary result to re-test.

**Size of effects.** A relative size (for example, as a share of the improvement LINSIGHT's annotations
give over a model with no annotations) would need a fit without annotations, which was never run;
the v4 report therefore gives direction and p-value only.

### Power (from `s09c`)

The planted effect s is the weight of a fixed random unit projection of the standardised embedding
(scaled to unit variance) added to LINSIGHT's latent score; one polymorphism/divergence outcome per site
is drawn from the INSIGHT likelihood (five outcome classes) and every model is refitted as in s09b.
In plain terms, s is expressed as a share of the spread (SD) of LINSIGHT's own latent score across
sites, 1.71 (`info["R0 latent sd (train)"]`).

| Planted effect s | 0 | 0.025 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|
| % of the SD of LINSIGHT's score | 0 | 1.5 | 2.9 | 5.8 | 12 | 23 |
| Simulated genomes | 20 | 5 | 5 | 5 | 5 | 5 |
| Embedding called better (p < 0.05 and better) | 2 | 0 | 1 | 0 | 1 | 1 |

In the 20 genomes without an effect, 256 random columns were called better in 4 (20%) and the
autoencoder code (32) in 2 (10%), against 1 in 40 (2.5%) expected for one direction of a two-sided 5%
test. The fitted embedding model did not track the true rho better than LINSIGHT (change in correlation
−0.020 to +0.003 across s > 0). The test is underpowered at 0.1% of sites; the 1% run is the next step.

### Baseline (from §6)

The authors' C++ fit on all sites gives HGMD AUROC 0.8977 / 0.7617 / 0.6645, equal to the published
track; our 0.1% retrain is 0.011 / 0.013 / 0.0003 below it.

## 9. What is not done yet

- The 1% run (`bash pipeline/run_all.sh 0.01`); widen the feature prior grid first (chosen width 10
  is at its edge).
- The second HGMD read (plan v2 in the current `s10_evaluate.py`: models fitted by exact likelihood, five seeds,
  Holm-adjusted families). One read allowed.
- An alignment-aware encoder (GPN-MSA), which needs a GPU.

## 10. Detail moved out of the report (v4)

The v4 report (8 pages) keeps the steps in prose and leaves the following here:

- **Processing-steps table** (step, script, input files, output files with shapes): it is §5 of this
  document, step by step, and §7 (file → produced by → consumed by).
- **File-level Methods of v3** (byte sizes, checksums, file names, column names, coordinate fields,
  per-step check counts): §3 and §5. Check counts per step: s00 7/7, s01 15/15, s02 22/22, s03 17/17,
  s04 10/10, s05 10/10, s06 13/13, s06b 13/13, s07 90/90, s08 296/296, s09 24/24, s09b 1,041/1,041,
  s09c 178/180 (the two failures are the power result), s10 3/3.
- **Settings table of v3:** sample 0.1% of autosomal INSIGHT sites, 1-kb exclusion around benchmark
  variants; 1-Mb blocks with 10% (HGMD analysis) or 51% (likelihood analysis) held out; autoencoders
  (d, k) ∈ {(32, 8), (64, 16), (128, 32)}, chosen on `select`: 64/16 (HGMD), 32/8 (likelihood);
  authors' procedure: LINSIGHT and its embedding, code, random-column (256, 32, 64, 128),
  shuffled-embedding and positive-control variants, 3 seeds (HGMD) or 5 seeds (likelihood); exact likelihood: all of these plus
  shuffled codes, composition and the feature autoencoder, 1 or 5 seeds; simulation s ∈ {0, 0.025,
  0.05, 0.1, 0.2, 0.4}, 20 worlds at s = 0 and 5 otherwise; prior widths: features {0.03, 0.1, 0.3, 1,
  3, 10} (chosen 10, at the edge), added columns {0.003, 0.01, 0.03, 0.1, 0.3, 1, 10}.
- **Bootstrap details:** HGMD loci are (chromosome, ⌊pos19 / 10⁵⌋); AUROC p-values are twice the
  smaller bootstrap tail fraction (a stored 0.0 means p < 1/2,000); held-out p-values use the normal
  approximation from the bootstrap standard error.
- **The held-out comparison under the authors' procedure** (10% split): described in words in the s09
  note above; no statistical test was made.
- **Held-out log-likelihood differences as numbers** (the held-out column of v3 Table 1): removed; the
  values are in `info.heldout` of `R/s09b_mle/checks.json`.


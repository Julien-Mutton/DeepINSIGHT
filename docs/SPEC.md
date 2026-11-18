# Specification — LINSIGHT + HyenaDNA sequence context

## Question

LINSIGHT (Huang, Gulko & Siepel 2017) scores negative selection on noncoding sites as
ρ = f(48 genomic features), fitted with the INSIGHT likelihood on population-genetic
data. **Does adding HyenaDNA sequence context to those 48 features improve LINSIGHT's
prediction of noncoding disease variants (HGMD)?**

Two ways of adding context:

| arm | x (model input) |
|---|---|
| **R0** `repl` | the 48 LINSIGHT features (replication of LINSIGHT) |
| **R1** `emb` | 48 features ⊕ the 256-d HyenaDNA embedding of the site's sequence window |
| **R2** `sae` | 48 features ⊕ the codes of a sparse autoencoder trained on those embeddings |

Controls, trained with the identical procedure:

| arm | x | what it tells us |
|---|---|---|
| `noise_emb` | 48 ⊕ 256 iid N(0,1) columns | cost of adding 256 uninformative inputs; R1 must be compared against this |
| `noise_sae` | 48 ⊕ d iid N(0,1) columns (d = chosen SAE width) | same, for R2 |
| `posctl` | 47 features ⊕ the 48th through the standardised extra-column channel | the extra-column path can carry a real signal (must ≈ R0) |

## Hard rules

1. **Only original inputs** (see `DATA.md`). Nothing derived in earlier projects is read.
2. **HGMD is evaluation only.** No model is trained, tuned or selected on benchmark labels.
   All choices are frozen before the single benchmark read (`eval.lock`).
3. **Paper procedure for every arm** (`L-INSIGHT-fit.cpp`, read in full):
   ρ = exp(−3·exp(−(b + w·x))), γ′ = σ(c + v·x), γ = β₀γ′, η = 0; INSIGHT site likelihood
   with the C++ site terms (incl. clamping θ·a, λ ≥ 1e-5); AdaGrad, mini-batches of 100
   sites with summed gradients, lr 1e-3 (ρ) and 1e-2 (γ′), ε 1e-6, zero init, 20 epochs,
   final epoch used; β₁ = 0.7567672441, β₂ = 0.2053143291, β₃ = 1 − β₁ − β₂.
   The C++ γ′-gradient is reproduced as written (it omits a factor ρ and divides by β₀),
   and the unfinished mini-batch gradient carries over into the next epoch, as in the C++.
4. **Every block has a gate** (`checks.json`); a step refuses to run unless its inputs'
   gates passed. Every sensitive check has a **control that must fail** on purpose-broken
   input (off-by-one, shuffled pairing), which shows the check can detect the error.
5. **No caches from earlier work.** Every artefact is produced by this pipeline from the
   originals, under `outputs/`.

## Flow

```
 ORIGINAL INPUTS                                        BLOCKS (pipeline/steps)
 ───────────────                                        ───────────────────────
 GWAVA/LINSIGHT benchmark (.pandas pickles) ──► s01 benchmark  panel of HGMD + 3 negative tiers (hg19)
 INSIGHT_database.ins (58 GB text)          ──► s02 sites      random r of INSIGHT sites + evol terms (hg19),
                                                               train / select / eval split by 1 Mb blocks
 hg19→hg38 chain, hg19.fa, hg38.fa          ──► s03 liftover   hg38 position for every site + panel variant
 genomic_feature.bed (48 features)          ──► s04 features   48 features at hg19 position
 LINSIGHT.bw (published scores)                 (checks)       f(features) reproduces LINSIGHT.bw exactly
 HyenaDNA weights + official model code     ──► s05 embedder   model loads + behaves (LM, causality, centring)
                                            ──► s06 embed      mean of 1001 hidden states of the hg38 window
                                            ──► s07 sae        TopK SAEs fit on TRAIN-site embeddings only
                                            ──► s08 design     x for every arm, rows re-derived from source
                                            ──► s09 train      LINSIGHT C++ procedure, every arm × 3 seeds
                                            ──► s10 evaluate   frozen plan → one read → AUROC per tier, Δ + CI
                                            ──► s11 report     REPORT.md / REPORT.html
```

Sites and benchmark variants pass through the **same** functions for liftover, feature
lookup, windowing, embedding, SAE encoding and standardisation (statistics from training
sites only). Design-matrix rows are keyed (`site|chrom:pos19`, `bench|source|row_id`), never
positional.

## Blocks and their checks

| step | produces | key checks (each with a failing control where meaningful) |
|---|---|---|
| s00 inputs | `inputs.json` (paths, sizes, sha256) | every input present; upstream checksums (HF weights, UCSC chain, CSHL sizes); unit tests pass |
| s01 benchmark | `panel.parquet` | counts 1614/161400/16140/5027; ref base = hg19[start] (off-by-one control fails); keys unique; published LINSIGHT.bw AUROC = paper ±0.01 |
| s02 sites | `sites.npz` | parser = original `LINSIGHT-prep` on an excerpt (bit-equal); per-chrom sampled fraction ≈ r; coordinate convention test (CpG enrichment of polymorphic sites) ; exclusion ≥1 kb from every benchmark variant; train/select/eval disjoint 1 Mb blocks, all autosomes on both sides |
| s03 liftover | `lift_*.npz` | ref base hg19 = hg38 (≥99.5%), 21-mer identity, 1001-bp window identity; pos+1 control fails |
| s04 features | `features_*.npz` | random access = streaming join (C++ semantics); LINSIGHT.bw is an exact function of the features (identical vectors → identical score; Gompertz-linear fit R² ≥ 0.999); shifted-features control fails |
| s05 embedder | `checks.json` | weights sha = HuggingFace; next-base LM accuracy ≫ chance; deterministic; batch-invariant; causal: mutating base 500 leaves states < 500 unchanged, mutating 501 leaves state 500 unchanged |
| s06 embed | `emb_*.npy` | all keys embedded, finite; re-embedding in a fresh process identical; site and panel windows built by the same function |
| s07 sae | `sae/*.pt`, codes | FVU on held-out sites, dead latents, L0 = k; fit on train sites only |
| s08 design | arm specs | ≥500 rows per matrix re-derived from originals by key; train/panel column names + standardisation stats identical; swapped-row control fails |
| s09 train | params per epoch | kernel = numpy transcription of the C++ (bit-level), ρ-gradient = autograd; R0 ρ vs LINSIGHT.bw Spearman ≥ 0.9 on eval sites; R0 weights vs weights recovered from LINSIGHT.bw; finite, non-constant |
| s10 evaluate | `results.json` | plan hash logged; exactly one read; AUROC = sklearn; paired cluster bootstrap (100 kb) |

## Decisions fixed in advance

- **Sample rate** r = 0.001 first (run `rate_0.001`), then r = 0.01 (`rate_0.01`).
  Sampling is a deterministic hash of (chrom, pos) < r, so the 0.1% sample is a subset of the 1% sample.
- **Held-out sites**: 10% of 1 Mb hg19 blocks (seed 0), halved by block into `select`
  (SAE-config choice) and `eval` (genome-wide test). All autosomes on both sides.
- **Exclusion**: training sites within 1,000 bp (hg19) of any benchmark variant are removed
  (their 1001-bp windows would overlap).
- **Window**: 1001 bp of hg38 centred on the site (index 500); representation = mean over
  the 1001 final hidden states (256-d). The centre state is stored too, for diagnostics.
- **SAE grid**: (d, k) ∈ {(32, 8), (64, 16), (128, 32)}, seeds 0–2, fitted on train-site
  embeddings. Chosen config = lowest mean final-epoch INSIGHT NLL of R2 on `select` sites.
- **Seeds**: training seed 0 visits sites in the C++ order (chromosome files in string
  order, positions ascending); seeds 1, 2 use a fresh random permutation each epoch.
  Arm score = mean ρ over the 3 seeds. SAE seed s is used with training seed s.
- **Primary comparisons** (paired cluster bootstrap, 2000 reps, 100 kb hg19 clusters):
  R1 − R0, R2 − R0, R1 − noise_emb, R2 − noise_sae; plus posctl − R0 and R0 − published.

# Decisions and findings (in the order they were made)

## Inputs
- **Originals only.** The pipeline reads the files in DATA.md and nothing produced by earlier
  work. Earlier caches, converted CSVs, `binary_dir/*.fit`, `LINSIGHT_hg38.bw` and
  `cpp_results/parameter.txt` are not used.
- **Upstream identity verified** (s00): HyenaDNA weights sha256 = HuggingFace LFS oid; chain md5
  = UCSC; LINSIGHT.bw byte size = CSHL; hg19/hg38 autosome lengths = UCSC chrom.sizes.
  `pipeline/tools/verify_upstream.sh` compares the sha256 of the decompressed CSHL `.ins.gz`
  and `genomic_feature.bed.gz` with the local files (result: `outputs/upstream/verify.tsv`).
- **Benchmark read from the original Python-2 pickles** through the `legacy_pandas` env.
- **LINSIGHT C++ vendored from git** (`f28b163`, clean tree) and rebuilt with its makefile; the
  binaries are used as references in tests.

## Reading the sources
- **The C++ clamps θ·a and λ at 1e-5** (both in LINSIGHT-prep and again in LINSIGHT-fit). The
  earlier pipeline's site-term code said "no clamping, as the C++", which was wrong.
- **Float vs double.** Some C++ site-term sub-expressions are evaluated in float (all operands
  float), others in double; `model/insight.py` follows the same promotion rules.
- **The C++ γ′ gradient** omits a factor ρ and divides by β₀ where the chain rule multiplies;
  reproduced on purpose (tested against autograd: exact for ρ, and γ′ = true / (ρ·β₀²)).
- **Unfinished mini-batch** gradient carries over into the next epoch (the C++ resets only the
  counter); reproduced and tested.
- **Verified against the original binaries:** our `.ins` parser = `LINSIGHT-prep` bit for bit
  (several chunks, chrX counted but not sampled); our trainer = `LINSIGHT-fit` parameters at
  every epoch (synthetic data, 3 chromosomes in string order, sites without features skipped).
- **AdaGrad gives every coordinate about the same step size**, so an uninformative 0/1 input
  drifts with the intercept early in training (seen in a unit test, same in the binary). This
  is the mechanism by which extra inputs can perturb ρ under the paper's procedure, and why
  every added-input arm has a matched noise control.

## Coordinates
- **`.ins` positions are 0-based hg19 indices.** INSIGHT removes CpG sites: read at offset 0 the
  sample has 0.0% CpG and equal C/G content; at ±1 it has ~0.75% CpG genome-wide (~2.6% on chr1 near the start) and C/G asymmetry (s02
  check, with the ±1 offsets as controls). LINSIGHT-fit matches these positions to 0-based BED
  intervals, so the published model, the features and the sequence all refer to the same base.
- **Benchmark `start` is the 0-based hg19 position** (ref allele = hg19[start] for 100%,
  start±1 ≈ 25%), and published LINSIGHT.bw at `start` reproduces the paper's AUROCs.
- **hg38 = HyenaDNA's own FASTA** (`hg38.ml.fa`, chr1–22 + X). Rows on other chromosomes get
  status `no_hg38_chrom` and are not embedded.

## Design
- **Sample rate** 0.001 first, then 0.01. Sampling = splitmix64 hash of (autosome, position)
  < rate, so samples are nested and reproducible in any process.
- **Tokens without special tokens**, as in HyenaDNA's hg38 pretraining loader (`add_eos=False`).
- **SAE grid** (32,8), (64,16), (128,32). The earlier (512,32) diverged under the C++ procedure
  (standardised rare latents push the Gompertz latent past exp overflow). SAEs are fit on
  training-site embeddings (not on a separate background set), so encoder and LINSIGHT see the
  same distribution.
- **Positive control** feature = the feature with the largest |published ρ-weight| (weights
  recovered exactly from LINSIGHT.bw; no benchmark data involved).

## Upstream verification (results)
- `INSIGHT_database.ins`: sha256 of the decompressed CSHL stream = local file (`7dd76bfd…`).
- `genomic_feature.bed`: byte-identical to the decompressed CSHL `.gz` (cmp). A first streamed
  hash differed only because that download was truncated (curl exit 18); the tool now resumes
  until the full upstream size arrives (`pipeline/tools/fetch_compare.sh`).

## Design v2 (after the first 0.1% run) — ground truth, power, more controls
The first design could say "up or down" but not whether the pipeline would have found a real effect
or how small an effect it could detect. v2 adds:
- **Ground-truth simulator** (`model/simulate.py`, docs/MATH.md §2): INSIGHT observations drawn from a
  known ρ with LINSIGHT's own likelihood. Five outcome classes; probabilities sum to 1 + ε with
  |ε| ≤ 1.31·θa·λ. *A first version counted intermediate-frequency SNPs twice (total 1 + θ·β₂); the
  estimator test exposed it (biased intercept, γ′ → 0) and it was fixed.*
- **Finding: the paper procedure is far from the likelihood maximum at our sizes.** On simulated
  data it ranks correctly (corr ≈ 0.9 with the true ρ) but weights reach ~10% of their true size after
  20 epochs at 4·10⁵ sites; more epochs barely help. So comparisons of arms trained this way mix
  information with optimisation shortfall. It is kept for the replication only.
- **Exact maximum likelihood** (`model/mle.py`) for all information tests: recovers planted weights
  (±0.12 at 3 M sites), same optimum from different starts.
- **Finding: γ′ is barely identified.** Its fitted intercept is +1.5 at 60k and −0.8 at 300k sites
  for a true 0. Extra columns in the γ′ head inflate the null likelihood-ratio statistic to ~1.7× its
  χ² mean (30% false positives at 300k sites); in the ρ head only it is calibrated (mean 3.3 for df 3,
  KS p 0.53). New inputs therefore enter the ρ head (the LINSIGHT score) in the exact fit; T1 has df = d.
  This also explains the earlier paper-procedure observation that the embedding's γ′ weights were
  13× its ρ weights.
- **Tests:** T1 likelihood ratio (calibrated by simulation), T2 held-out NLL with a 1 Mb-block
  bootstrap, T3 HGMD ΔAUROC with a bootstrap over 100-kb loci AND training seeds; Holm within the
  primary (18) and secondary families; every result reported with its MDE = 2.80·SE.
- **Controls:** noise, shuffled embedding/SAE rows, composition (GC, CpG o/e, repeats, homopolymer
  from soft-masked hg19), feature-only SAE (codes only, and added to the features), positive control
  (must equal R0 exactly under the exact fit — an invariant).
- **Split:** 50% of 1 Mb blocks held out (10% of those for SAE selection), 5 training seeds.
- **Benchmark reads:** plan v1 (first design) was read once; plan v2 was fixed after seeing v1 and is
  read once. The report states both.
- **s06 reuse:** embeddings are reused only when the sha256 of every (chrom, hg38 position) window,
  the weights and the window length equal those of the stored result; checks are re-run regardless.
- **Finding: on real data the unpenalised maximum does not exist (quasi-separation).** Rare binary
  features (feature 18 at 0.038% of training sites, 17 at 0.016%, 19 at 0.31%) got weights of
  ~1e5 and the γ′ intercept 1,779; the replication gate (exact-fit R0 vs LINSIGHT.bw Spearman 0.26)
  caught it. The synthetic ground truth had no rare features, so it could not. Unit test added
  (`test_separation_breaks_unpenalised_fit_and_prior_fixes_it`).
- **Fix: penalised maximum likelihood.** N(0, sd²) priors on all weights (intercepts free), sd chosen
  per arm and seed from {0.03, 0.1, 0.3, 1, 3} by held-out `select` NLL. T1 (LRT) is not valid with
  a penalty and is not used on real data; T2 (held-out) is the information test, its false-positive
  rate and power measured in s09c with the same estimator.
- **Finding: the better a model fits the INSIGHT data, the less it ranks like published LINSIGHT.**
  R0 by penalised ML: prior sd 0.03 → Spearman with LINSIGHT.bw 0.94; 0.1 → 0.86; 0.3 → 0.64;
  1 → 0.40, while held-out select NLL improves (0.0846098 → 0.0845261). Published LINSIGHT behaves like
  a strongly regularised (early-stopped) fit. Hence two primary families: A = LINSIGHT as its authors
  train it (paper procedure), B = best fit to the evolutionary data (penalised ML), Holm separately.
- **Replication gate** is on the paper procedure (≥ 0.85 at 50% training blocks: 0.872; it was 0.923
  with 90%); the implementation's correctness is proven exactly elsewhere (binary equality, R² = 1).
- **Finding: warm-starting along the prior grid lands in worse local optima.** The likelihood is not
  convex; at prior sd 3 a cold start reached select NLL 0.0844875 vs 0.0844944 for the warm-started
  path, and the positive control and R0 (same information) then ranked the benchmark at Spearman 0.889.
  Cold starts are reproducible (two different starts: Spearman 1.0000, identical NLL) and posctl vs R0
  is ≥ 0.9956 at every width. Every (arm, seed, width) is now fitted from a cold start as its own job.
  The grid gained sd 10 because R0 still preferred the widest width (3).

## Design v3 (after s09b v2 failed its positive-control gate) — split prior widths

- **Finding: choosing one prior width per arm confounds every "arm − R0" comparison.** s09b v2
  (414 cold fits) failed two gates: positive control − R0 on held-out NLL −6.35e-5 vs MDE 4.45e-5,
  and posctl-vs-R0 benchmark Spearman 0.947 (< 0.99). R0 chose sd 10 and posctl sd 3, from `select`
  NLL differences of ~1e-6 (0.0844866 vs 0.0844875), i.e. noise. At a COMMON width posctl equals R0
  (sd ≥ 1: held-out Δ +1.2e-6 / +5e-7 / +7e-7, p 0.44 / 0.30 / 0.77; Spearman 0.998 / 0.9997 / 0.996).
  Under per-arm widths every control (noise, shuffled, comp, fsae) "beat" R0 by ~1e-4 because the
  extra-column arms chose smaller widths for all 48 feature weights too. At a common width every
  256-column arm (embedding included) is WORSE than R0 on held-out data, and emb vs shuffled is n.s.
  (exploratory, cached v2 fits; not a reported result). HGMD was not read (s10 never ran).
- **Fix: two widths.** sdF (48 feature weights) is chosen once by R0 on `select`, then fixed in every
  arm; sdE (new columns, both heads where they enter) is chosen per arm and seed on `select` from
  {0.003, 0.01, 0.03, 0.1, 0.3, 1, 10}. As sdE → 0 an arm is R0 exactly (unit test
  `test_tiny_extra_prior_recovers_the_nested_r0_fit`), so "arm − R0" measures only the new columns.
  Penalty split verified by the stationarity condition per column (`test_extra_prior_only_touches_extra_columns`);
  the whole selection-then-held-out procedure is calibrated and powerful on planted data
  (`test_split_prior_heldout_test_is_calibrated_and_powerful`). s09c re-chooses sdE inside every
  simulated world, so its false-positive rate describes the procedure actually applied.
- The convergence gate covers every fit that can be chosen (R0 at sdF, every arm's whole sdE grid);
  v2's only non-converged fit (shuf_sae64k16 s1 at sd 10) was a width v3 no longer uses for features.

## Design v4 (after s09b v3 failed one gate) — local optima: two starts per fit

- **s09b v3 result:** 548/549 checks. Held-out positive control now within MDE (Δ −3.8e-6, p 0.49,
  MDE 1.5e-5). Failed: posctl-vs-R0 benchmark ranking Spearman 0.977 (< 0.99).
- **Two causes, both diagnosed without benchmark labels:**
  1. posctl's moved feature had its own width sdE (chosen 0.3), a *different prior* from R0's sdF on
     that feature, so equality was not even expected (Spearman by sdE: 0.945 at 0.003 … 0.988 at 10).
  2. **The penalised likelihood has several local optima.** R0 refitted from posctl's solution (mapped
     into R0's parametrisation) converged (grad < 1e-6) to a different point: objective 0.08246977 vs
     0.08246938 for R0's cold start, weights of rare features (0.2–1.5% of sites) up to 2.3 apart,
     benchmark Spearman 0.989 between the two R0 optima. Not score saturation (0.6% of benchmark ρ = 0,
     none = 1; latent Pearson 0.997): 161k benchmark variants share 45k distinct feature vectors, so
     small weight shifts reorder whole groups.
- **Fix:** (a) every width of every nested arm is fitted from two starts — cold (zeros) and R0's optimum
  with the new weights at 0 (`arms.r0_start`) — and the lower **penalised training objective** is the
  fit (`mle.fit_multistart`; the estimator is the posterior mode, so this is its definition, and it uses
  no held-out or benchmark data). Consequence, now a gate: a nested arm's training objective ≤ R0's.
  (b) posctl's moved column gets exactly R0's prior (sdE = sdF × sd(feature)), making it an exact
  reparametrisation: gate = its training objective equals R0's to 1e-9. Verified on the real data
  before the run: objective at the mapped R0 point − R0's = 0.0 exactly for posctl, emb, sae32k8, comp.
  (c) s09c uses the same two-start estimator in every simulated world.
- Unit test `test_multistart_keeps_the_lower_objective_and_nested_arm_never_fits_worse_than_r0`.
  The penalty is one function (`mle.penalty_weights`) shared by the fit and `mle.objective`.
- Cost: the cold fits of v3 are reused (same code path for cold starts; objective recomputed from
  the stored parameters); only the R0-start fits are new.

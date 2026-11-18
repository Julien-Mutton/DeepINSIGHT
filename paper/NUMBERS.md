# NUMBERS.md: source of every number in `main.tex` (v4)

Every value below was re-read from disk during the v4 revision. Paths are relative to
the original (private) project folder, of which this repository holds the `experiments/09_thesis_v2` pipeline;
`R = experiments/09_thesis_v2/outputs/rate_0.001` (here: `outputs/rate_0.001`, small copies in `results/rate_0.001`).
"Report" is what the report prints; "Stored" is the value in the file.

Two runs of the same pipeline on the same 1,362,112 sites are used:
* **10% split** (10% of 1-Mb blocks held out; authors' procedure, 3 seeds): the one locked HGMD read
  (`R/s10_evaluate/results.json`, `lock.json`, plan `89b069b68c299e39`; `R/REPORT.md`; `R/s09_train/selection.json`).
* **51% split** (likelihood analysis): every other file in `R`.

v4 prints **no log-likelihood values or differences**. The withheld-genome result is given as the sign
of the stored mean (Estimate), the stored `p`, and a verdict (better/worse only if p < 0.05).

## Abstract and Introduction
No numbers except the fixed facts "48 annotations", "256-number embedding", "32, 64 or 128" (code
sizes), "1,001 bases" (window): `pipeline/arms.py` (`width`), `R/s07_sae/checks.json` keys, `src/lctx/config.py`.

## Study design (window, widths, sample)

| Item | Report | Source |
|---|---|---|
| Window | 1,001 bases | `R/s05_embedder/checks.json` `hidden_shape` [4, 1001, 256] |
| Embedding width | 256 | same |
| Code widths, one per model | 32, 64 or 128 | `pipeline/arms.py` `Data.inputs` (one extra block `Z` per model: `emb` 256, `sae<d>k<k>` d) |
| Polymorphism sample | 54 people | `R/s02_sites/checks.json` `info.n_samples` = 108 chromosomes |
| Withheld blocks | 1-Mb | `src/lctx/config.py` block size |

## Results §1: rebuilt LINSIGHT

| Claim | Report | Stored | Source |
|---|---|---|---|
| Parser identical to `LINSIGHT-prep`; trainer = `LINSIGHT-fit` every epoch | text | text | `experiments/09_thesis_v2/docs/DECISIONS.md` |
| C++ on all sites reproduces published-track AUROCs to four decimals | text | `LINSIGHT_cpp_refit` 0.8976765 / 0.76174 / 0.66454 vs `LINSIGHT_published` 0.8976765 / … | `experiments/07_thesis_pipeline_v1/outputs/benchmark/results.json` |
| Retrained matches published on region-matched; ~0.01 lower on the others | about 0.01 | −0.00026 [−0.0080, +0.0069]; −0.01071; −0.01335 | `R/s10_evaluate/results.json` `deltas["R0 - published"]` |
| Weights 14% of published size | 14% | 0.1431 | `R/s09_train/checks.json` `info.R0_rho_weight_norm_ratio_to_published` (Pearson 0.856: "point the same way") |
| 20 passes | 20 | `EPOCHS` | `src/lctx/config.py` |

## Table 1 (HGMD): `R/s10_evaluate/results.json`

| Row | Report | Stored |
|---|---|---|
| Positives | 1,564 | `n_pos` |
| Negatives | 159,765 / 15,346 / 4,722 | `n` − `n_pos`: 161,329, 16,910, 6,286 − 1,564 |
| Reported by Huang et al. | 0.897 / 0.759 / 0.660 | `paper` |
| Published track, scored by us | 0.898 / 0.762 / 0.665 | `auroc.published` 0.89768 / 0.76174 / 0.66454 |
| LINSIGHT retrained | 0.887 / 0.748 / 0.664 | `R0 repl` 0.88697 / 0.74840 / 0.66427 |
| + positive control | 0.884 / 0.745 / 0.663 | `posctl` |
| + HyenaDNA embedding (256) | 0.725 / 0.683 / 0.647 | `R1 emb` |
| + random columns (256) | 0.827 / 0.719 / 0.654 | `noise_emb` |
| + autoencoder code (64) | 0.824 / 0.731 / 0.665 | `R2 sae64k16` |
| + random columns (64) | 0.849 / 0.730 / 0.657 | `noise64` |
| Positive control within 0.004 | 0.004 | max \|Δ\| 0.0032 (`posctl - R0`) |
| Embedding lower on all three sets | text | `R1 - R0` −0.162, −0.065, −0.018, all CIs < 0 |
| Embedding below random (256) on two sets | text | `R1 - noise_emb` −0.103, −0.036 (CIs < 0); region −0.008 [−0.023, +0.007] |
| Code lower on unmatched, TSS; level on region | text | `R2 - R0` −0.063, −0.017 (CIs < 0); +0.0009 [−0.006, +0.008] |
| Code better than embedding on every set | text | `R2 - R1` +0.100, +0.048, +0.019 (CIs > 0) |
| Code not better than random (64) on any set | text | `R2 - noise64` −0.025 (CI < 0); +0.001 [−0.009, +0.011]; +0.008 [−0.0003, +0.016] |
| Code size 64 chosen on select | 64 | `R/s09_train/selection.json` (sae64k16 highest select likelihood of the three) |

## Table 2 (withheld genome): `R/s09b_mle/checks.json` → `info.heldout`

| Row | Estimate (sign of `mean`; + = model worse) | p (stored) | Verdict |
|---|---|---|---|
| emb − repl | worse (+2.25e-5) | 0.1249 → 0.12 | no difference |
| emb − noise_emb | worse | 0.0384 → 0.04 | worse |
| emb − shuf_emb | worse | 0.2886 → 0.29 | no difference |
| emb − comp | worse | 0.0782 → 0.08 | no difference |
| sae32k8 − repl | better (−6.5e-8) | 0.9877 → 0.99 | no difference |
| sae32k8 − noise32 | worse | 0.4632 → 0.46 | no difference |
| sae32k8 − shuf_sae32k8 | better | 0.5337 → 0.53 | no difference |
| sae64k16 − repl | better | 0.2934 → 0.29 | no difference |
| sae128k32 − repl | better | 0.2627 → 0.26 | no difference |
| posctl − repl | worse (+2.7e-8) | 0.2542 → 0.25 | no difference |
| fsae_codes − repl | better | 6.5e-6 → < 0.0001 | better |
| fsae_plus − repl | better | 0.2692 → 0.27 | no difference |
| Sites / blocks | 612,686 / 1,187 | `n_eval`, `n_eval_blocks` | |
| 22 comparisons | 22 | `len(heldout)` | |
| Feature code worse than LINSIGHT on select sites | text | `select_nll` fsae_codes 0.0845154 > repl 0.0844866 | |
| Five fits for seeded models | 5 | `info.fits` / `params.npz` (69 fits) | |

## Table 3 and power text: `R/s09c_groundtruth/checks.json` → `info`

| Item | Report | Stored |
|---|---|---|
| Planted effect as % of SD of LINSIGHT's score | 0, 1.5, 2.9, 5.8, 12, 23 | s / 1.7100 (`R0 latent sd (train)`): 0.025→1.46%, 0.05→2.92%, 0.1→5.85%, 0.2→11.7%, 0.4→23.4% |
| Simulated genomes | 20; 5 each | `summary[*].n_worlds` |
| Embedding called better | 2, 0, 1, 0, 1, 1 | `T2_reject_rate` × `n_worlds`: 0.1×20, 0, 0.2×5, 0, 0.2×5, 0.2×5 |
| Random columns called better, null | 4 of 20 | `summary["s=0.0 noise_emb"]` 0.2 |
| 1 in 40 expected | 2.5% | one direction of a two-sided 5% test |
| Two failing checks | 2 | 178/180 checks passed |

## Methods

| Claim | Report | Source |
|---|---|---|
| Autosomal sites in the INSIGHT database | 1,540,791,804 | `R/s02_sites/checks.json` `info.counts_all_sites` (chr1–22 sum) |
| Benchmark counts | 1,614; 161,400 / 16,140 / 5,027 | `R/s01_benchmark/checks.json` |
| Panel rows | 184,181 | `R/s08_design/checks.json` `info.rows.panel_all` |
| One excluded common variant | 1 | `R/s01_benchmark/checks.json` `info.n_label_conflict` |
| Published AUROCs within 0.01 | 0.01 | `R/s01_benchmark/checks.json` (0.8977 / 0.7618 / 0.6643 vs 0.897 / 0.759 / 0.660) |
| Sampling rate; kept sites | 0.001; 1,542,881 | `info.n_sampled` |
| Kept fraction within 0.2% | 0.2% | sample fraction / rate 1.0014 |
| Exclusion 1,000 bases; 11.7%; 1,362,112 | | `info.excluded_frac` 0.1172; `sites.npz` length |
| Nearest remaining site | 1,006 bases | `R/s02_sites/checks.json` |
| Split: half held out; select one tenth | | `HELDOUT_FRAC` 0.5, `SELECT_SHARE` 0.1 (`src/lctx/config.py`) |
| 51% split sizes | 670,261 / 70,332 / 621,519 | `info.n_by_role` |
| 10% split training sites | 1,212,074 | `R/REPORT.md` header |
| hg38 conversion accepted | 99.93% / 99.73% | `R/s03_liftover/checks.json` (0.99927, 0.99727) |
| Same base | all sites / 99.85% of variants | 1.0 / 0.99845 |
| Annotations found | 98.6% / 98.8% | `R/s04_features/checks.json` (0.98605, 0.98757) |
| Recovered weights reproduce score: every variant vs ~20% control | 100%; 21% | 1.0; 0.2136 |
| HyenaDNA 8 layers, 256 | | `…/hyenadna-medium-160k-seqlen/config.json` |
| Next-base accuracy | 43.5% vs 25% | `R/s05_embedder/checks.json` 0.4348 / 0.2531 |
| Autoencoder training rows | 669,704 | `R/s07_sae/checks.json` `n_train` |
| Sizes (d,k); five starts | (32,8), (64,16), (128,32); 5 | `R/s07_sae/` model files s0–s4 |
| Chosen sizes | 64 (HGMD); 32 (exact fit) | `R/s09_train/selection.json`; `R/s09b_mle/checks.json` `sae_chosen` |
| Unexplained variance, 32 numbers | 4.9% vs 7.4% (PCA) | mean over 5 seeds of `heldout.fvu` 0.0489 vs `pca_fvu_same_k` 0.0739 |
| Parameters 98 → 610; code adds 64–256 | | 2(1+m): m = 48, 304; 2d for d = 32…128 (`R/s09_train/checks.json` param shapes 20×98, 20×610, 20×162, 20×354) |
| Usable rows | 660,061 / 69,412 / 612,686; 181,397 | `R/s08_design/checks.json` `info.rows` |
| Rebuilt rows | 300 sites, 300 variants | `R/s08_design/checks.json` |
| Batches of 100; 20 passes; three runs (HGMD) | | `MINIBATCH`, `EPOCHS` (`src/lctx/config.py`); `results.json` `auroc_per_seed` s0–s2 |
| Disease variants used | 1,564 | `n_pos` |
| Simulation: 20 null genomes, five at each of five effects | | `EFFECTS`, `REPS` in `pipeline/steps/s09c_groundtruth.py`; `summary` |
| Bootstrap 2,000; 100-kb regions; 1-Mb blocks | | `R/s10_evaluate/lock.json` `bootstrap` [2000, 100000]; `src/lctx/config.py` |
| 112-core server, no GPU | | `nproc` |

## Removed in v4 (still on disk, described in `docs/DATA_FLOW.md`)
Held-out log-likelihood values and differences (v3 Table 1 column, Results §3); the
authors'-procedure held-out values (`R/REPORT.md`); byte sizes, checksums, file names, step codes;
liftover rejection counts; CpG coordinate-test percentages; tokenisation; prior grids and chosen
widths; start counts (293 / 177); per-step check counts; the change-in-correlation range of the
simulations; the settings table.

## Added in v5 Methods: settings read from code/config
| Number in text | Value | Source |
|---|---|---|
| INSIGHT constants beta | (0.757, 0.205, 0.038) | `experiments/09_thesis_v2/src/lctx/config.py` BETA = (0.7567672441, 0.2053143291, 1 - both) |
| rho link shift | -3 | config.py RHO_SHIFT |
| Authors' procedure: batch, lr rho, lr gamma, passes | 100, 1e-3, 1e-2, 20 | config.py MINIBATCH, LR_RHO, LR_GAMMA, EPOCHS |
| HyenaDNA medium-160k: layers, state width, max length | 8, 256, 160,002 | `Insight/linsight_project/checkpoints/hyenadna-medium-160k-seqlen/config.json` (n_layer, d_model, l_max) |
| Tokens A,C,G,T,N, no special tokens; mean of final hidden states | - | `src/lctx/embed/hyena.py` docstring |
| SAE: Adam lr, batch, epochs, AuxK k_aux, coefficient, dead steps | 1e-3, 4,096, 10, 2k, 1/32, 500 | `src/lctx/sae/topk.py` |
| SAE grid, seeds | (32,8),(64,16),(128,32); 5 | config.py SAE_GRID, SEEDS |
| Prior widths: annotations; added block | {0.03,0.1,0.3,1,3,10}; {0.003,0.01,0.03,0.1,0.3,1,10} | config.py PRIOR_SDS, PRIOR_SDS_EXTRA |
| Bootstrap resamples, cluster size | 2,000; 100 kb | config.py BOOTSTRAP_REPS, CLUSTER_BP |
| Block size; held-out share; select share | 1 Mb; 0.50; 0.10 | config.py BLOCK_BP, HELDOUT_FRAC, SELECT_SHARE |
| Parameter counts 98 / 610 | 2(48+1); 2(304+1) | formula 2(m+1) |

# Math behind every model, estimator and test

Notation: site s, input vector x_s (48 features, optionally extra columns), parameters
θ = (b, w, c, v).

## 1. Model (LINSIGHT, L-INSIGHT-fit.cpp)

    ρ_s  = exp(−3·exp(−(b + w·x_s)))                 probability the site is under selection (Gompertz)
    γ'_s = σ(c + v·x_s),   γ_s = β₀ γ'_s             selected low-frequency polymorphism factor
    P_s(θ) = (A_s (B_s + D_s γ_s) − F_s) ρ_s + F_s    INSIGHT likelihood of the site's record (η = 0)

A, B, D, F depend only on the site's record (allele-frequency class, P(ancestral = major/minor))
and its block rates θ·a (polymorphism) and λ (divergence), clamped at 10⁻⁵ as in the C++.
Training objective: NLL(θ) = −Σ_s log P_s(θ).

## 2. Ground-truth simulator (`model/simulate.py`)

With the ancestral allele known, the observable outcomes of a site are five classes; with
multiplicity m_k (number of specific bases giving that record):

| class | record (af, pZmaj, pZmin) | m |
|---|---|---|
| monomorphic, human = ancestral | (0, 1, 0) | 1 |
| monomorphic, human ≠ ancestral | (0, 0, 0) | 3 |
| low-freq SNP, major = ancestral | (0.05, 1, 0) | 3 |
| low-freq SNP, minor = ancestral | (0.05, 0, 1) | 3 |
| intermediate-freq SNP | (0.5, 1, 0) | 3 |

p_k = m_k · P(record_k | ρ, γ'). Summing the C++ terms gives Σ_k p_k = 1 + ε with
|ε| ≤ 1.31·θa·λ (measured; INSIGHT ignores double events), independent of γ'. The simulator draws
from p_k / Σ p_k, so its law equals the fitted likelihood up to that bounded factor.
Checked: probabilities sum to 1, empirical class frequencies match (χ² test), and each
simulated record's model likelihood equals its class probability / m_k.

## 3. Estimators

* **Paper procedure** (`model/linsight.py`): AdaGrad, mini-batches of 100 with summed gradients,
  lr 1e-3 / 1e-2, 20 epochs, final epoch; identical to the LINSIGHT-fit binary (unit test).
  On simulated data it ranks sites correctly (corr > 0.9 with the true ρ) but shrinks weights
  ~10× at 10⁵–10⁶ sites: it is far from the likelihood maximum. Used for the replication only.
* **Exact maximum likelihood** (`model/mle.py`): full-batch L-BFGS on NLL(θ) with the exact
  gradient, float64, converged to max|∇ mean NLL| < 10⁻⁶. On simulated data it recovers the
  planted parameters (within 0.12 at 3 M sites) and reaches the same optimum from different
  starts. Used for every information test.

## 4. Tests

**T1 — likelihood-ratio test (does the extra input carry ANY signal?).** For nested models
(R0 ⊂ R0 + extra, d extra columns entering both ρ and γ' heads):

    Λ = 2 (NLL_R0(θ̂₀) − NLL_R1(θ̂₁)) ~ χ²(2d)  under H0 (extra columns have no effect)

Wilks' theorem requires a correctly specified model; this is CHECKED rather than assumed:
under simulated nulls the p-values are tested for uniformity (KS) and the false-positive rate
at α = 0.05 is reported. Because 10⁶ sites detect tiny effects, T1 answers "is there any
signal", never "is it useful".

**T2 — held-out NLL (does it generalise?).** On eval sites (whole held-out 1 Mb blocks),
d_s = NLL_s(R1) − NLL_s(R0). Mean d̄ with SE from a block bootstrap (resampling blocks keeps
the within-block correlation; SE ≥ iid SE when blocks are correlated — unit-tested). d̄ < 0 means
the extra input improves prediction of unseen genome.

**T3 — HGMD AUROC (does it help disease prediction?).** AUROC = P(score_pos > score_neg)
(Mann-Whitney, ties ½). Δ = AUROC(B) − AUROC(A) on the same variants. Bootstrap: resample
100-kb loci with replacement AND training seeds with replacement (score = mean over the
resampled seeds), so the CI includes benchmark sampling and training randomness. Two-sided
p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)).

**Multiplicity.** The primary family (each new input vs its baseline and vs its matched
controls, × 3 tiers) is Holm-adjusted: sort p ascending, p̃₍ᵢ₎ = max_{j≤i} min(1, (m − j + 1) p₍ⱼ₎).

**Power.** For an estimate with standard error SE, the minimum detectable effect at
two-sided α = 0.05 and 80 % power is MDE = (z₀.₉₇₅ + z₀.₈₀)·SE = 2.80·SE. Every result is
reported with its MDE, so "not significant" is always quantified ("effects larger than MDE
would have been detected"). For T1/T2 power is also measured directly by simulation: planted
sequence effects of known size s are detected in what fraction of simulated datasets.

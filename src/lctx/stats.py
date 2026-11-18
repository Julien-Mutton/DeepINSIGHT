"""Statistical tests used to accept or reject every claim (formulas in docs/MATH.md).

  lrt                      nested likelihood-ratio test: 2 (NLL_0 - NLL_1) ~ chi2(df) under H0
  block_bootstrap_mean     mean of a per-site difference with an SE from resampling genome blocks
  auroc_delta_bootstrap    AUROC(b) - AUROC(a) on the same variants; resamples 100-kb loci AND
                           training seeds (score = mean over the resampled seeds), so the CI
                           includes both benchmark sampling and training randomness
  holm                     Holm step-down adjustment of a family of p-values
  mde                      minimum detectable effect = (z_{1-alpha/2} + z_power) * SE
"""
import numpy as np
from scipy.stats import chi2, norm, rankdata

from .config import ALPHA, POWER


def lrt(nll_null_sum, nll_alt_sum, df):
    stat = 2.0 * (nll_null_sum - nll_alt_sum)
    return {"stat": float(stat), "df": int(df), "p": float(chi2.sf(max(stat, 0.0), df))}


def mde(se, alpha=ALPHA, power=POWER):
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def block_bootstrap_mean(d, groups, reps=2000, seed=0):
    """Mean of per-site values d (sites grouped into blocks); SE, 95% CI and two-sided p for mean = 0."""
    d = np.asarray(d, np.float64)
    ug, inv = np.unique(groups, return_inverse=True)
    s, n = np.bincount(inv, d), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, len(ug), len(ug))
        bs[r] = s[i].sum() / n[i].sum()
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"mean": m, "se": se, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "p": float(2 * norm.sf(abs(m) / se)) if se > 0 else float("nan"), "mde": mde(se),
            "n_sites": int(len(d)), "n_blocks": int(len(ug))}


def _auc(s, y):
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_delta_bootstrap(y, A, B, clusters, reps=2000, seed=0):
    """A, B: (n_seeds, n) score matrices (one row per training seed; 1 row for fixed scores).
    Delta = AUROC(mean_seeds B) - AUROC(mean_seeds A)."""
    y = np.asarray(y).astype(bool)
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    d0 = _auc(B.mean(0), y) - _auc(A.mean(0), y)
    uc, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(reps):
        pick = rng.integers(0, len(uc), len(uc))
        idx = np.concatenate([order[starts[c]:starts[c + 1]] for c in pick])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        sa = A[rng.integers(0, len(A), len(A))].mean(0)[idx]
        sb = B[rng.integers(0, len(B), len(B))].mean(0)[idx]
        ds.append(_auc(sb, yy) - _auc(sa, yy))
    ds = np.array(ds)
    se = float(ds.std(ddof=1))
    p = float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))
    return {"delta": float(d0), "se": se, "lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5)),
            "p": p, "mde": mde(se), "reps": int(len(ds))}


def holm(pvals):
    """Holm-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, np.float64)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj

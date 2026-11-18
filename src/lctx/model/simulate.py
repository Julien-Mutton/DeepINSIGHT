"""Ground truth: simulate INSIGHT observations from a KNOWN rho, using LINSIGHT's own likelihood.

For a site with neutral rates (theta*a, lambda) and an ancestral allele known with certainty,
the INSIGHT likelihood (model/insight.py, eta = 0) assigns a probability to each observable
outcome. Five outcome classes cover every observation LINSIGHT-prep can produce:

    class            evol record (af, pZeqXmaj, pZeqXmin)   multiplicity (specific bases)
    M_anc            (0,    1, 0)   human = ancestral        1
    M_div            (0,    0, 0)   human != ancestral       3
    L_majanc         (0.05, 1, 0)   low-freq SNP, major = ancestral   3
    L_minanc         (0.05, 0, 1)   low-freq SNP, minor = ancestral   3
    H                (0.5,  1, 0)   intermediate-freq SNP              3

An intermediate-frequency SNP is ONE outcome: its INSIGHT term depends only on pZeqXmaj + pZeqXmin
(which allele is "major" carries no information at ~50%), so splitting it into two classes would
count its probability twice (total 1 + theta*beta2 instead of 1).

p_class = multiplicity * P(record | rho, gamma').  Their sum is 1 + O(theta*lambda) (INSIGHT ignores
double events; checked by tests/unit/test_simulate.py); `outcome_probs` returns the normalised probabilities and the excess, so the
simulator's generative law equals the fitted likelihood up to that (checked, bounded) constant.
"""
import numpy as np

from .insight import site_terms, site_prob

CLASSES = ("M_anc", "M_div", "L_majanc", "L_minanc", "H")
RECORD = {"M_anc": (0.0, 1.0, 0.0), "M_div": (0.0, 0.0, 0.0), "L_majanc": (0.05, 1.0, 0.0),
          "L_minanc": (0.05, 0.0, 1.0), "H": (0.5, 1.0, 0.0)}
MULT = {"M_anc": 1, "M_div": 3, "L_majanc": 3, "L_minanc": 3, "H": 3}


def _record(cls, theta_a, lam):
    af, pmaj, pmin = RECORD[cls]
    n = len(theta_a)
    return np.column_stack([np.full(n, af), np.full(n, pmaj), np.full(n, pmin), theta_a, lam]).astype(np.float32)


def outcome_probs(theta_a, lam, rho, gamma_prime):
    """(n, 6) normalised class probabilities and the (n,) unnormalised excess sum - 1."""
    theta_a, lam = np.asarray(theta_a, np.float32), np.asarray(lam, np.float32)
    raw = np.column_stack([MULT[c] * site_prob(site_terms(_record(c, theta_a, lam)), rho, gamma_prime)
                           for c in CLASSES])
    if (raw < -1e-12).any():
        raise ValueError("negative outcome probability")
    raw = np.clip(raw, 0, None)
    tot = raw.sum(1)
    return raw / tot[:, None], tot - 1.0


def simulate_evol(theta_a, lam, rho, gamma_prime, rng):
    """Draw one outcome per site; returns (evol float32 (n, 5), class index (n,), excess (n,))."""
    p, excess = outcome_probs(theta_a, lam, rho, gamma_prime)
    u = rng.random(len(p))[:, None]
    cls = (u > np.cumsum(p, 1)).sum(1)
    cls = np.minimum(cls, len(CLASSES) - 1)
    rec = np.array([RECORD[c] for c in CLASSES], np.float32)[cls]
    evol = np.column_stack([rec, np.asarray(theta_a, np.float32), np.asarray(lam, np.float32)]).astype(np.float32)
    return evol, cls, excess

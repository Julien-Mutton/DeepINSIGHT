"""INSIGHT site likelihood terms as computed in L-INSIGHT-fit.cpp::updateGradient (eta = 0).

    P(site) = (A (B + D gamma) - F) rho + F,      gamma = beta0 * gamma'

evol columns (float32, as written by LINSIGHT-prep): alleleFreq, pZeqXmaj, pZeqXmin,
thetaTimesA, lambda. lambda and thetaTimesA are clamped to >= 1e-5 (float), as in the C++.
The C++ evaluates some sub-expressions in float (all operands float) and others in double;
`site_terms` follows the same promotion rules so the terms match the binary.
"""
import numpy as np

from ..config import BETA, ALLELE_FREQ_CUTOFF, MIN_THETA_LAMBDA

f32 = np.float32


def site_terms(evol, beta=BETA, cutoff=ALLELE_FREQ_CUTOFF):
    e = np.asarray(evol, f32)
    af, pmaj, pmin = e[:, 0], e[:, 1], e[:, 2]
    th = np.where(e[:, 3] < f32(MIN_THETA_LAMBDA), f32(MIN_THETA_LAMBDA), e[:, 3]).astype(f32)
    lam = np.where(e[:, 4] < f32(MIN_THETA_LAMBDA), f32(MIN_THETA_LAMBDA), e[:, 4]).astype(f32)
    d = lambda v: np.asarray(v, np.float64)
    b0, b1, b2 = beta
    M = af == 0
    L = (af > 0) & (af < f32(cutoff))
    H = (af >= f32(cutoff)) & (af <= 1)
    if not (M | L | H).all():
        raise ValueError("unknown site type (allele frequency outside [0, 1])")
    one = f32(1)
    # monomorphic: float expressions
    A_m = np.ones(len(e))
    B_m = d(pmaj)
    D_m = d(-pmaj * th)
    F_m = d((one - th) * ((one - lam) * pmaj + lam / f32(3) * (one - pmaj)))
    # low frequency: valA = (thetaTimesA / 3.0) * pZeqXmaj in double
    A_l = d(th) / 3.0 * d(pmaj)
    F_l = (d(lam / f32(3)) * (b0 * d(one - pmaj) + b2 * d(one - pmin)) * d(th) / 3
           + d(one - lam) * (b0 * d(pmaj) + b2 * d(pmin)) * d(th) / 3)
    # high frequency
    F_h = ((1.0 - 2.0 * d(lam) / 3.0) * d(pmaj + pmin) + (2.0 * d(lam) / 3.0) * d(one - pmaj - pmin)) \
        * b1 * d(th) / 3.0
    A = np.where(M, A_m, np.where(L, A_l, 0.0))
    B = np.where(M, B_m, 0.0)
    D = np.where(M, D_m, np.where(L, 1.0, 0.0))
    F = np.where(M, F_m, np.where(L, F_l, F_h))
    return tuple(np.ascontiguousarray(v, np.float64) for v in (A, B, D, F))


def site_prob(terms, rho, gamma_prime, beta0=BETA[0]):
    A, B, D, F = terms
    return (A * (B + D * (gamma_prime * beta0)) - F) * rho + F


def nll(terms, rho, gamma_prime, beta0=BETA[0]):
    return -np.log(site_prob(terms, rho, gamma_prime, beta0))

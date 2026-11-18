"""Sequence-composition summary of a window (a deliberately simple, interpretable control).

From the soft-masked hg19 window (UCSC RepeatMasker lower case) centred on the site:
    gc          fraction of G/C among A/C/G/T
    cpg_oe      CpG observed / expected = n_CG * L / (n_C * n_G)   (0 when n_C or n_G is 0)
    repeat      fraction of soft-masked (lower-case) bases
    homopolymer longest single-base run / L
If HyenaDNA's effect on LINSIGHT is mostly composition, an arm with these 4 numbers reproduces it.
"""
import numpy as np

NAMES = ("gc", "cpg_oe", "repeat", "homopolymer")


def composition(seq):
    L = len(seq)
    up = seq.upper()
    nA, nC, nG, nT = (up.count(b) for b in "ACGT")
    acgt = nA + nC + nG + nT
    gc = (nC + nG) / acgt if acgt else np.nan
    ncg = up.count("CG")
    cpg = ncg * L / (nC * nG) if nC and nG else 0.0
    rep = sum(1 for c in seq if c.islower()) / L
    run = best = 1
    for a, b in zip(up, up[1:]):
        run = run + 1 if a == b and a in "ACGT" else 1
        best = max(best, run)
    return np.array([gc, cpg, rep, best / L], np.float32)

"""s06b composition: 4 interpretable numbers per window (GC, CpG o/e, repeat fraction, longest
homopolymer) from the soft-masked hg19 1001-bp window centred on every site / benchmark variant.
The `comp` arm uses them to test whether HyenaDNA's effect is explained by simple composition.

Checks: shape (n, 4), finite, values in range; genome-typical means (GC ~0.4, repeats ~0.4-0.5,
CpG o/e ~0.2-0.3); independent recomputation of random rows; a shifted-window control differs.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import C, CheckReport, finish, load_npz, log
from lctx import seqcomp

_R = {}


def _init():
    import pyfaidx
    _R["fa"] = pyfaidx.Fasta(C.INPUTS.hg19, as_raw=True, rebuild=False)   # keeps lower case


def _window(chrom, pos):
    s = pos - C.CENTER
    w = _R["fa"][chrom][max(0, s):s + C.CONTEXT]
    return "N" * (C.CONTEXT - len(w)) + w if s < 0 else w + "N" * (C.CONTEXT - len(w))


def _work(task):
    chroms, pos = task
    return np.stack([seqcomp.composition(_window(c, int(p))) for c, p in zip(chroms, pos)])


def compute(chroms, pos, n_proc=96, chunk=20000):
    tasks = [(chroms[i:i + chunk], pos[i:i + chunk]) for i in range(0, len(pos), chunk)]
    with mp.get_context("fork").Pool(n_proc, initializer=_init) as pool:
        return np.concatenate(pool.map(_work, tasks)).astype(np.float32)


def run(run):
    run.gate("s02_sites", "s04_features")
    rep = CheckReport("s06b_composition")
    S = load_npz(run.path("s02_sites", "sites.npz"))
    pan = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"), usecols=["chrom", "pos19"])
    sets = {"sites": (np.array(C.AUTOSOMES)[S["chrom"]], S["pos19"]),
            "panel": (pan.chrom.to_numpy().astype(str), pan.pos19.to_numpy())}
    _init()
    for name, (chroms, pos) in sets.items():
        X = compute(chroms, pos)
        np.save(run.path("s06b_composition", f"{name}_comp.npy"), X)
        rep.eq(f"{name}: shape", list(X.shape), [len(pos), 4])
        rep.true(f"{name}: finite", bool(np.isfinite(X).all()))
        rep.true(f"{name}: values in range", bool(((X[:, [0, 2, 3]] >= 0) & (X[:, [0, 2, 3]] <= 1)).all() and (X[:, 1] >= 0).all()))
        m = X.mean(0)
        rep.info[f"{name}: mean " + "/".join(seqcomp.NAMES)] = m.round(4).tolist()
        if name == "sites":
            rep.close("sites: mean GC (genome ~0.41)", round(float(m[0]), 3), 0.41, 0.04)
            rep.close("sites: mean CpG o/e (genome ~0.25)", round(float(m[1]), 3), 0.25, 0.1)
            rep.close("sites: mean repeat fraction (hg19 ~0.5)", round(float(m[2]), 3), 0.45, 0.15)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pos), 300, replace=False)
        re = np.stack([seqcomp.composition(_window(chroms[i], int(pos[i]))) for i in idx])
        rep.eq(f"{name}: random rows recomputed equal", int((np.abs(re - X[idx]) > 1e-6).any(1).sum()), 0)
        sh = np.stack([seqcomp.composition(_window(chroms[i], int(pos[i]) + 5000)) for i in idx])
        rep.ge(f"{name}: CONTROL window 5 kb away differs (frac rows)", round(float((np.abs(sh - X[idx]) > 1e-6).any(1).mean()), 3),
               0.9, control=True)
        log(f"{name}: composition {X.shape}, means {m.round(3)}")
    finish(run, "s06b_composition", rep)

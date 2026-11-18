"""The LINSIGHT trainer reproduces L-INSIGHT-fit.cpp: numpy transcription, autograd, and the
original binary built from the vendored source."""
import os
import subprocess

import numpy as np
import pytest

from conftest import synthetic_evol
from lctx.config import BETA, ROOT
from lctx.model import insight, linsight as LS

CPP_DIR = os.path.join(ROOT, "third_party", "linsight_cpp")


def _data(rng, n=3000, nf=6, nz=3):
    evol = synthetic_evol(rng, n)
    F = (rng.random((n, nf)) < 0.3).astype(np.float32)
    Z = rng.normal(size=(n, nz)).astype(np.float32)
    mu, sd = Z.mean(0).astype(np.float64), Z.std(0).astype(np.float64)
    return evol, F, Z, mu, sd


def test_kernel_equals_numpy_transcription(rng):
    evol, F, Z, mu, sd = _data(rng)
    terms = insight.site_terms(evol)
    inp = LS.Inputs(F, Z, mu, sd)
    tr = LS.Trainer(inp, terms, mb=7)            # 7 does not divide n -> exercises the carry-over
    X = inp.dense(np.arange(len(F)))
    ref = {"para": np.zeros_like(tr.para), "grad": np.zeros_like(tr.para), "hist": np.zeros_like(tr.para)}
    for ep in range(2):
        order = rng.permutation(len(F))
        tr.epoch(order)
        LS.fit_epoch_reference(ref, X, terms, order, mb=7)
        np.testing.assert_allclose(tr.para, ref["para"], rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(tr.grad, ref["grad"], rtol=1e-10, atol=1e-13)   # carried gradient


def test_rho_gradient_is_exact_and_gamma_gradient_is_the_cpp_quirk(rng):
    torch = pytest.importorskip("torch")
    evol, F, Z, mu, sd = _data(rng, n=100)
    terms = insight.site_terms(evol)
    inp = LS.Inputs(F, Z, mu, sd)
    X = inp.dense(np.arange(len(F)))
    p0 = rng.normal(scale=0.3, size=2 * (1 + X.shape[1]))
    st = {"para": p0.copy(), "grad": np.zeros_like(p0), "hist": np.zeros_like(p0)}
    LS.fit_epoch_reference(st, X, terms, np.arange(len(F)), mb=10 ** 9)   # accumulate only
    H = 1 + X.shape[1]
    p = torch.tensor(p0, requires_grad=True)
    Xt = torch.tensor(X)
    A, B, D, Fv = (torch.tensor(t) for t in terms)
    rho = torch.exp(-3 * torch.exp(-(p[0] + Xt @ p[1:H])))
    gp = torch.sigmoid(p[H] + Xt @ p[H + 1:])
    P = (A * (B + D * gp * BETA[0]) - Fv) * rho + Fv
    (-torch.log(P).sum()).backward()
    g = p.grad.numpy()
    np.testing.assert_allclose(st["grad"][:H], g[:H], rtol=1e-9, atol=1e-12)
    # gamma' head: per site the C++ gradient = true gradient / (rho * beta0^2)
    rho_n = rho.detach().numpy()
    true_site = []
    for s in range(len(F)):
        p.grad = None
        rr = torch.exp(-3 * torch.exp(-(p[0] + Xt[s] @ p[1:H])))
        gg = torch.sigmoid(p[H] + Xt[s] @ p[H + 1:])
        Ps = (A[s] * (B[s] + D[s] * gg * BETA[0]) - Fv[s]) * rr + Fv[s]
        (-torch.log(Ps)).backward()
        true_site.append(p.grad.numpy()[H:].copy() / (rho_n[s] * BETA[0] ** 2))
    np.testing.assert_allclose(st["grad"][H:], np.sum(true_site, 0), rtol=1e-8, atol=1e-12)


def test_gompertz_link():
    assert LS.gompertz_rho(0.0) == pytest.approx(np.exp(-3))
    assert LS.gompertz_rho(50.0) == pytest.approx(1.0)
    assert LS.gompertz_rho(-5.0) < 1e-100


def test_site_terms_clamp_and_types(rng):
    evol = synthetic_evol(rng, 50)
    evol[:, 4] = 0.0
    a = insight.site_terms(evol)
    evol2 = evol.copy()
    evol2[:, 4] = np.float32(1e-5)
    b = insight.site_terms(evol2)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)                       # lambda clamped to 1e-5
    bad = evol.copy()
    bad[0, 0] = 2.0
    with pytest.raises(ValueError):
        insight.site_terms(bad)


def test_trainer_recovers_a_planted_constraint(rng):
    """Sites with feature 0 = 1 are made strongly constrained (no polymorphism, no divergence);
    the trainer must learn a positive rho weight for it and ~0 for a useless feature."""
    n = 60000
    evol = synthetic_evol(rng, n)
    F = np.zeros((n, 2), np.float32)
    F[:, 0] = rng.random(n) < 0.3
    F[:, 1] = rng.random(n) < 0.5
    sel = F[:, 0] == 1
    evol[sel, 0] = 0.0                      # constrained sites: monomorphic ...
    evol[sel, 1] = 0.99999                  # ... and matching the ancestral allele
    tr = LS.Trainer(LS.Inputs(F), insight.site_terms(evol))
    for ep in range(5):
        tr.epoch(rng.permutation(n))
    w = tr.para[1:3]
    # AdaGrad gives every coordinate ~the same step size, so an uninformative 0/1 feature drifts
    # with the intercept early in training (same in the C++ binary); the planted feature must
    # still dominate and raise rho.
    assert w[0] > 0.05 and w[0] > abs(w[1])
    rho, _, _ = LS.score(LS.Inputs(F), tr.para)
    assert rho[sel].mean() > rho[~sel].mean()


def _cpp_binary():
    exe = os.path.join(CPP_DIR, "LINSIGHT-fit")
    if not os.path.exists(exe):
        subprocess.run(["make", "-s", "-C", CPP_DIR], check=True)
    return exe


def test_kernel_matches_original_linsight_fit_binary(rng, tmp_path):
    """Write a synthetic INSIGHT binary directory + feature BED, run the ORIGINAL LINSIGHT-fit
    for 3 epochs and compare every epoch's parameters with our kernel run in the C++ order."""
    exe = _cpp_binary()
    chroms = ["chr1", "chr10", "chr2"]                       # string order != numeric order
    recs, feats = [], []
    for c in chroms:
        pos = np.sort(rng.choice(np.arange(1000, 400000), 1500, replace=False))
        evol = synthetic_evol(rng, len(pos))
        recs.append((c, pos, evol))
    # feature intervals tile [0, 300000) per chromosome in 1000-bp steps; sites beyond are unmatched
    with open(tmp_path / "f.bed", "w") as fh:
        fh.write("#chr\tstart\tend" + "\tf" * 4 + "\n")
        for c in sorted(chroms):
            for s in range(0, 300000, 1000):
                v = (rng.random(4) < 0.4).astype(int)
                if rng.random() < 0.2:
                    v[3] = 0.5                               # exactly representable non-binary value
                fh.write(f"{c}\t{s}\t{s + 1000}\t" + "\t".join(str(x) for x in v) + "\n")
                feats.append((c, s, v))
    d = tmp_path / "fit"
    d.mkdir()
    rec_dtype = np.dtype([("pos", "<u8"), ("e", "<f4", (5,))])
    for c, pos, evol in recs:
        a = np.zeros(len(pos), rec_dtype)
        a["pos"], a["e"] = pos, evol
        a.tofile(d / f"{c}.fit")
    out = tmp_path / "param.txt"
    subprocess.run([exe, "-d", str(d), "-f", str(tmp_path / "f.bed"), "-o", str(out), "-n", "3",
                    "-b1", str(BETA[0]), "-b2", str(BETA[1])], check=True, capture_output=True)
    cpp = np.loadtxt(out)
    # our side: matched sites, in the C++ order
    fmap = {(c, s): v for c, s, v in feats}
    C, P, E, Fm = [], [], [], []
    for c, pos, evol in recs:
        for p, e in zip(pos, evol):
            s = (p // 1000) * 1000
            if (c, s) in fmap:
                C.append(c); P.append(p); E.append(e); Fm.append(fmap[(c, s)])
    order = LS.cpp_order(C, P)
    tr = LS.Trainer(LS.Inputs(np.array(Fm, np.float32)), insight.site_terms(np.array(E, np.float32)))
    for ep in range(3):
        tr.epoch(order)
        np.testing.assert_allclose(tr.para, cpp[ep], rtol=2e-5, atol=1e-9,
                                   err_msg=f"epoch {ep + 1} differs from LINSIGHT-fit")

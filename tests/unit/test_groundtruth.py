"""Ground truth for the estimators and tests: data simulated from a KNOWN rho with LINSIGHT's own
likelihood. These tests are what justify trusting the real-data results."""
import numpy as np
import pytest
from scipy.stats import chisquare, kstest

from lctx import seqcomp, stats
from lctx.model import insight, linsight as LS, mle, simulate as SIM


def _rates(rng, n):
    return rng.uniform(1e-3, 0.02, n).astype(np.float32), rng.uniform(1e-3, 0.02, n).astype(np.float32)


# ------------------------------------------------------------------ simulator
def test_outcome_probabilities_normalise_up_to_theta_lambda():
    rng = np.random.default_rng(0)
    th, lam = rng.uniform(1e-4, 0.05, 50000), rng.uniform(1e-4, 0.05, 50000)
    for rho in (0.0, 0.3, 0.9, 1.0):
        for gp in (0.0, 0.5, 1.0):
            p, ex = SIM.outcome_probs(th, lam, np.full(len(th), rho), np.full(len(th), gp))
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
            assert (p >= 0).all()
            assert np.all(np.abs(ex) <= 1.5 * th * lam + 1e-12)        # INSIGHT ignores double events


def test_simulated_class_frequencies_match_probabilities(rng):
    n = 400_000
    th, lam = _rates(rng, n)
    rho, gp = np.full(n, 0.4), np.full(n, 0.6)
    p, _ = SIM.outcome_probs(th, lam, rho, gp)
    _, cls, _ = SIM.simulate_evol(th, lam, rho, gp, rng)
    obs = np.bincount(cls, minlength=len(SIM.CLASSES))
    assert chisquare(obs, p.sum(0) * n / p.sum(0).sum() * (obs.sum() / n)).pvalue > 1e-3


def test_simulated_records_have_the_likelihood_the_model_assigns(rng):
    """Each simulated record's INSIGHT probability equals its class probability / multiplicity."""
    n = 20000
    th, lam = _rates(rng, n)
    rho, gp = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    ev, cls, ex = SIM.simulate_evol(th, lam, rho, gp, rng)
    p, _ = SIM.outcome_probs(th, lam, rho, gp)
    lik = insight.site_prob(insight.site_terms(ev), rho, gp)
    mult = np.array([SIM.MULT[c] for c in SIM.CLASSES])[cls]
    np.testing.assert_allclose(lik * mult / (1 + ex), p[np.arange(n), cls], rtol=1e-6)


# ------------------------------------------------------------------ estimators
def _planted(rng, n, w, v, nf=4):
    th, lam = _rates(rng, n)
    F = (rng.random((n, nf)) < 0.3).astype(np.float32)
    rho = LS.gompertz_rho(w[0] + F @ w[1:])
    gp = 1 / (1 + np.exp(-(v[0] + F @ v[1:])))
    ev, _, _ = SIM.simulate_evol(th, lam, rho, gp, rng)
    return F, insight.site_terms(ev), rho


def test_mle_recovers_planted_parameters():
    rng = np.random.default_rng(1)
    w = np.array([-0.2, 1.5, 0.8, 0.0, -0.5])
    v = np.array([0.3, -0.5, 0.0, 0.0, 0.0])
    F, T, _ = _planted(rng, 3_000_000, w, v)
    r = mle.fit(LS.Inputs(F), T, np.arange(len(F)), threads=16)
    assert r["converged"]
    np.testing.assert_allclose(r["para"][:5], w, atol=0.12)            # rho head
    r2 = mle.fit(LS.Inputs(F), T, np.arange(len(F)), init=np.random.default_rng(2).normal(0, .5, 10), threads=16)
    np.testing.assert_allclose(r2["para"][:5], r["para"][:5], atol=2e-3)   # same optimum from another start


def test_paper_procedure_shrinks_weights_at_small_n():
    """Documents WHY the exact fit is used for information tests: 20 C++ epochs under-fit."""
    rng = np.random.default_rng(3)
    w = np.array([-0.2, 1.5, 0.8, 0.0, -0.5])
    F, T, rho = _planted(rng, 400_000, w, np.zeros(5))
    tr = LS.Trainer(LS.Inputs(F), T)
    for _ in range(20):
        tr.epoch(rng.permutation(len(F)))
    assert tr.para[1] < 0.5 * w[1]                                     # strongly shrunk ...
    r, _, _ = LS.score(LS.Inputs(F), tr.para)
    assert np.corrcoef(r, rho)[0, 1] > 0.85                            # ... but ranks well (0.89 measured)


def test_mle_score_matches_fit_nll(rng):
    F, T, _ = _planted(rng, 50_000, np.array([0, 1, 0, 0, 0.]), np.zeros(5))
    r = mle.fit(LS.Inputs(F), T, np.arange(len(F)))
    _, _, nll = mle.score(LS.Inputs(F), r["para"], terms=T)
    assert nll.sum() == pytest.approx(r["nll_sum"], rel=1e-10)


# ------------------------------------------------------------------ the likelihood-ratio test
def _lrt_once(args):
    seed, n, effect = args
    rng = np.random.default_rng(seed)
    th, lam = _rates(rng, n)
    F = (rng.random((n, 2)) < 0.3).astype(np.float32)
    Z = rng.normal(size=(n, 3)).astype(np.float32)
    lat = -0.2 + F @ np.array([1.0, 0.5]) + effect * Z[:, 0]
    ev, _, _ = SIM.simulate_evol(th, lam, LS.gompertz_rho(lat), np.full(n, 0.5), rng)
    T = insight.site_terms(ev)
    idx = np.arange(n)
    r0 = mle.fit(LS.Inputs(F), T, idx, threads=1)
    r1 = mle.fit(LS.Inputs(F, Z, np.zeros(3), np.ones(3)), T, idx, threads=1, gamma_cols=2)   # Z in rho head only
    return stats.lrt(r0["nll_sum"], r1["nll_sum"], df=3)["p"]


def _many(args):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(40) as pool:
        return np.array(pool.map(_lrt_once, args))


def test_lrt_is_calibrated_under_the_null():
    """No effect of Z (rho head): LRT p-values must be ~Uniform(0,1) (chi-square approximation holds)."""
    p = _many([(s, 300_000, 0.0) for s in range(60)])
    assert kstest(p, "uniform").pvalue > 0.01
    assert (p < 0.05).mean() <= 0.15


def test_lrt_power_grows_with_effect_size():
    """Measured at 300k sites: effect 0.1 (latent units) is detected ~30% of the time, 0.3 almost always.
    The per-site information in INSIGHT data is small; this is why every result reports its MDE."""
    small = (_many([(100 + s, 300_000, 0.1) for s in range(20)]) < 0.05).mean()
    large = (_many([(200 + s, 300_000, 0.3) for s in range(20)]) < 0.05).mean()
    assert large >= 0.8 and large > small


def test_extra_columns_in_gamma_head_break_the_lrt():
    """Documents the design decision: with Z also in the gamma' head the null LRT is inflated."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(40) as pool:
        st = np.array(pool.map(_lrt_stat_both_heads, [(s, 300_000) for s in range(40)]))
    assert st.mean() > 1.3 * 6                                     # chi2(6) mean is 6


def _lrt_stat_both_heads(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    th, lam = _rates(rng, n)
    F = (rng.random((n, 2)) < 0.3).astype(np.float32)
    Z = rng.normal(size=(n, 3)).astype(np.float32)
    ev, _, _ = SIM.simulate_evol(th, lam, LS.gompertz_rho(-0.2 + F @ np.array([1.0, 0.5])), np.full(n, 0.5), rng)
    T = insight.site_terms(ev)
    r0 = mle.fit(LS.Inputs(F), T, np.arange(n), threads=1)
    r1 = mle.fit(LS.Inputs(F, Z, np.zeros(3), np.ones(3)), T, np.arange(n), threads=1)
    return 2 * (r0["nll_sum"] - r1["nll_sum"])


# ------------------------------------------------------------------ bootstrap, Holm, MDE
def test_block_bootstrap_se_matches_iid_theory(rng):
    d = rng.normal(0.1, 1.0, 40000)
    g = np.repeat(np.arange(4000), 10)
    r = stats.block_bootstrap_mean(d, g, reps=1000)
    assert r["se"] == pytest.approx(1 / np.sqrt(40000), rel=0.15)
    assert r["lo"] < 0.1 < r["hi"]


def test_block_bootstrap_widens_for_correlated_blocks(rng):
    g = np.repeat(np.arange(400), 100)
    d = rng.normal(size=400)[g] + rng.normal(size=40000) * 0.1        # strong within-block correlation
    r = stats.block_bootstrap_mean(d, g, reps=1000)
    assert r["se"] > 3 / np.sqrt(40000)


def test_auroc_bootstrap_includes_seed_variance(rng):
    n = 4000
    y = rng.random(n) < 0.1
    base = rng.normal(size=n) + y
    cl = np.arange(n) // 20
    same = stats.auroc_delta_bootstrap(y, base[None], base[None], cl, reps=300)
    assert same["delta"] == 0 and same["se"] == 0
    seeds = base[None] + rng.normal(scale=1.0, size=(5, n))            # noisy training seeds
    one = stats.auroc_delta_bootstrap(y, base[None], seeds.mean(0)[None], cl, reps=300)
    many = stats.auroc_delta_bootstrap(y, base[None], seeds, cl, reps=300)
    assert many["se"] > one["se"]


def test_holm_and_mde():
    np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert stats.mde(1.0) == pytest.approx(1.959964 + 0.841621, rel=1e-5)


def test_composition_features():
    gc, cpg, rep, hom = seqcomp.composition("ACGTacgtNN" * 10 + "AAAAA")
    assert gc == pytest.approx(40 / 85) and rep == pytest.approx(40 / 105) and hom == pytest.approx(5 / 105)   # 85 ACGT letters, 40 of them C/G
    assert cpg == pytest.approx(20 * 105 / (20 * 20))


def test_separation_breaks_unpenalised_fit_and_prior_fixes_it():
    """A rare feature whose sites all show no variation: the unpenalised maximum is at infinity
    (what happens on the real data); the penalised fit is finite and still recovers the others."""
    rng = np.random.default_rng(5)
    n = 300_000
    w = np.array([-0.2, 1.0, 0.5, 0.0])
    F = (rng.random((n, 3)) < np.array([0.3, 0.3, 0.0005])).astype(np.float32)
    th, lam = _rates(rng, n)
    ev, _, _ = SIM.simulate_evol(th, lam, LS.gompertz_rho(w[0] + F @ w[1:]), np.full(n, 0.5), rng)
    rare = F[:, 2] == 1
    ev[rare, 0] = 0.0                                          # rare-feature sites: no variation at all
    ev[rare, 1] = 1.0
    T = insight.site_terms(ev)
    free = mle.fit(LS.Inputs(F), T, np.arange(n), threads=8)
    pen = mle.fit(LS.Inputs(F), T, np.arange(n), threads=8, prior_sd=1.0)
    assert abs(free["para"][3]) > 20                           # runs off towards infinity
    assert abs(pen["para"][3]) < 5 and np.isfinite(pen["para"]).all()
    np.testing.assert_allclose(pen["para"][1:3], w[1:3], atol=0.2)


# ------------------------------------------------------------------ split prior (features vs extra columns)
def _with_noise(rng, n, nz, effect=0.0):
    """Planted world: rho from 4 features (+ effect * Z[:,0]); Z = nz standard-normal extra columns."""
    th, lam = _rates(rng, n)
    F = (rng.random((n, 4)) < 0.3).astype(np.float32)
    Z = rng.normal(size=(n, nz)).astype(np.float32)
    w = np.array([-0.2, 1.5, 0.8, 0.0, -0.5])
    rho = LS.gompertz_rho(w[0] + F @ w[1:] + effect * Z[:, 0])
    ev, _, _ = SIM.simulate_evol(th, lam, rho, np.full(n, 0.5), rng)
    return F, Z, insight.site_terms(ev)


def test_tiny_extra_prior_recovers_the_nested_r0_fit():
    """Invariant used by every 'arm - R0' comparison: as prior_sd_extra -> 0 the arm IS R0 (same
    feature weights, extra weights ~0, same held-out NLL), whatever the feature prior."""
    rng = np.random.default_rng(11)
    F, Z, T = _with_noise(rng, 200_000, 8)
    idx = np.arange(len(F))
    r0 = mle.fit(LS.Inputs(F), T, idx, threads=8, prior_sd=3.0)
    rz = mle.fit(LS.Inputs(F, Z, np.zeros(8), np.ones(8)), T, idx, threads=8, gamma_cols=4,
                 prior_sd=3.0, prior_sd_extra=1e-5)
    assert r0["converged"] and rz["converged"]
    np.testing.assert_allclose(rz["para"][:5], r0["para"][:5], atol=1e-4)
    assert np.abs(rz["para"][5:13]).max() < 1e-4
    assert abs(rz["nll_mean"] - r0["nll_mean"]) < 1e-8


def test_extra_prior_only_touches_extra_columns():
    """The penalty on the 48-feature analogue does not change with prior_sd_extra, and vice versa:
    the objective's penalty equals sum w_f^2/(2n sdF^2) + sum w_z^2/(2n sdE^2) exactly."""
    rng = np.random.default_rng(12)
    F, Z, T = _with_noise(rng, 50_000, 3)
    inp = LS.Inputs(F, Z, np.zeros(3), np.ones(3))
    idx = np.arange(len(F))
    r = mle.fit(inp, T, idx, threads=4, gamma_cols=4, prior_sd=2.0, prior_sd_extra=0.5)
    p = r["para"]
    k = 1 + inp.nx
    # stationarity of the penalised objective, column by column: d(mean NLL)/dw = -w / (n sd^2)
    import torch
    X = torch.from_numpy(mle._design(inp, idx))
    A, B, D, Fv = (torch.from_numpy(np.ascontiguousarray(t)) for t in T)
    pt = torch.tensor(p, requires_grad=True)
    mle._nll_torch(pt, X, A, B, D, Fv, mle.BETA[0], k).mean().backward()
    g = pt.grad.numpy()
    n = len(idx)
    np.testing.assert_allclose(g[1:5], -p[1:5] / (n * 2.0 ** 2), atol=2e-7)       # features: sd 2
    np.testing.assert_allclose(g[5:8], -p[5:8] / (n * 0.5 ** 2), atol=2e-7)       # extras:   sd 0.5
    assert abs(g[0]) < 1e-6 and abs(g[k]) < 1e-6                                  # intercepts free


def _split_once(args):
    seed, effect = args
    rng = np.random.default_rng(seed)
    F, Z, T = _with_noise(rng, 300_000, 16, effect)
    perm = rng.permutation(len(F))
    tr, se, ev = perm[:150_000], perm[150_000:225_000], perm[225_000:]
    r0 = mle.fit(LS.Inputs(F), T, tr, threads=1, prior_sd=3.0)
    inp = LS.Inputs(F, Z, np.zeros(16), np.ones(16))
    best = None
    for sde in (0.003, 0.03, 0.3):
        r = mle.fit(inp, T, tr, threads=1, gamma_cols=4, prior_sd=3.0, prior_sd_extra=sde)
        s = mle.score(inp, r["para"], se, T)[2].mean()
        if best is None or s < best[0]:
            best = (s, r)
    d = mle.score(inp, best[1]["para"], ev, T)[2] - mle.score(LS.Inputs(F), r0["para"], ev, T)[2]
    return d.mean() / (d.std() / np.sqrt(len(d)))


def test_split_prior_heldout_test_is_calibrated_and_powerful():
    """The B_bestfit design end to end on planted data: extra prior chosen on `select`, test on
    held-out sites. Null (16 useless columns): the held-out gain is not significant more than ~5%
    of the time (one-sided z < -1.645). Measured power at 150k training / 75k held-out sites: effect
    0.3 detected 4/10, effect 0.6 6/10 with all ten z < 0 (held-out is weaker than the in-sample LRT,
    which reaches >= 80% at 0.3 on 300k sites). The guarantee here is calibration plus power that grows
    with the effect; the power on the REAL sites is measured in s09c, not assumed."""
    import multiprocessing as mp
    with mp.get_context("fork").Pool(60) as pool:
        z0 = np.array(pool.map(_split_once, [(300 + s, 0.0) for s in range(40)]))
        z1 = np.array(pool.map(_split_once, [(400 + s, 0.3) for s in range(10)]))
        z2 = np.array(pool.map(_split_once, [(500 + s, 0.6) for s in range(10)]))
    assert (z0 < -1.645).mean() <= 0.15
    assert (z1 < -1.645).mean() > (z0 < -1.645).mean()
    assert (z2 < -1.645).mean() >= max(0.5, (z1 < -1.645).mean())
    assert z2.mean() < z1.mean() < z0.mean() and (z2 < 0).mean() >= 0.9


def test_multistart_keeps_the_lower_objective_and_nested_arm_never_fits_worse_than_r0():
    """fit_multistart returns the start with the lowest penalised objective; a nested arm started at
    R0's optimum (extra weights 0) has exactly R0's objective there, so its fit can only be lower."""
    rng = np.random.default_rng(13)
    F, Z, T = _with_noise(rng, 150_000, 6)
    idx = np.arange(len(F))
    r0 = mle.fit(LS.Inputs(F), T, idx, threads=8, prior_sd=3.0)
    inp = LS.Inputs(F, Z, np.zeros(6), np.ones(6))
    start = np.r_[r0["para"][:5], np.zeros(6), r0["para"][5:]]
    assert abs(mle.objective(inp, start, T, idx, 3.0, 0.1) - r0["objective"]) < 1e-12
    best, objs = mle.fit_multistart(inp, T, idx, {"cold": None, "r0": start}, threads=8, gamma_cols=4,
                                    prior_sd=3.0, prior_sd_extra=0.1)
    assert best["objective"] == min(objs.values())
    assert best["objective"] <= r0["objective"] + 1e-12
    assert abs(mle.objective(inp, best["para"], T, idx, 3.0, 0.1) - best["objective"]) < 1e-10

"""Exact maximum-likelihood fit of the LINSIGHT model (the statistically sound estimator).

Model of the C++, parameters [b, w(nx), c, v(ng)]:
    rho = exp(-3 exp(-clamp(b + w.x)))      gamma' = sigmoid(c + v.x[:ng])
    NLL = - sum_s log( (A_s (B_s + D_s beta0 gamma'_s) - F_s) rho_s + F_s )
minimised over all parameters with full-batch L-BFGS (float64, exact autograd gradient - not the
C++ gamma' gradient). Convergence is declared on the max-abs gradient of the MEAN NLL.

Why this estimator: on simulated data with a known rho the C++ AdaGrad procedure (20 epochs) ranks
correctly but shrinks the weights ~10x at our data sizes, while this fit recovers the planted
weights and reaches the same optimum from different starts (tests/unit/test_groundtruth.py).

`gamma_cols` (= ng) lets extra columns enter the rho head only. Measured on simulated nulls: gamma'
is barely identified, and extra columns in the gamma' head inflate the likelihood-ratio statistic to
~1.7x its chi2 mean (30% false positives at 300k sites), while extra columns in the rho head only
give a calibrated test. rho is the LINSIGHT score, so the information tests put new inputs there.
The latent of rho is clamped to +-LATENT_CLAMP (rho is 1 - 1e-13 or 0 beyond it), which keeps the
gradient finite for extreme standardised inputs; `score` applies the same clamp.
"""
import numpy as np
import torch

from ..config import BETA, RHO_SHIFT, LATENT_CLAMP


def _design(inputs, idx):
    X = inputs.dense(np.asarray(idx))
    return np.hstack([np.ones((len(X), 1)), X])


def penalty_weights(inputs, k, kg, n, prior_sd=None, prior_sd_extra=None):
    """Per-parameter weight c such that penalty = sum c p^2 over [b, w(k-1), c, v(kg-1)]: 1/(2 n sd^2),
    0 for intercepts; columns >= inputs.nf use prior_sd_extra when it is set. Also returns the mask
    of extra-column parameters."""
    col = np.r_[-1, np.arange(k - 1), -1, np.arange(kg - 1)]       # input column of each parameter
    extra = (col >= inputs.nf) if prior_sd_extra is not None else np.zeros(len(col), bool)
    pw = lambda sd: 0.0 if sd is None else 1.0 / (2.0 * n * sd ** 2)
    w = np.where(col < 0, 0.0, pw(prior_sd))                       # intercepts are not penalised
    w[extra] = pw(prior_sd_extra) if extra.any() else 0.0
    return w, extra


def objective(inputs, para, terms, idx, prior_sd=None, prior_sd_extra=None):
    """The penalised training objective `fit` minimises, for any parameter vector."""
    k = 1 + inputs.nx
    w, _ = penalty_weights(inputs, k, len(para) - k, len(idx), prior_sd, prior_sd_extra)
    return float(score(inputs, para, idx, terms)[2].mean() + (w * np.asarray(para) ** 2).sum())


def _nll_torch(p, X, A, B, D, F, beta0, k):
    lat = torch.clamp(X @ p[:k], -LATENT_CLAMP, LATENT_CLAMP)
    rho = torch.exp(RHO_SHIFT * torch.exp(-lat))
    g = torch.sigmoid(X[:, :len(p) - k] @ p[k:])
    return -torch.log((A * (B + D * g * beta0) - F) * rho + F)


def fit(inputs, terms, idx, init=None, beta0=BETA[0], tol=1e-9, max_rounds=40, threads=8, gamma_cols=None,
        prior_sd=None, prior_sd_extra=None):
    """gamma_cols: number of input columns entering gamma' (default all).
    prior_sd: if set, maximise the posterior under independent N(0, prior_sd^2) priors on every weight
    (intercepts unpenalised) = mean NLL + (w.w + v.v) / (2 n prior_sd^2). Needed on real data: rare
    binary features are quasi-separated and the unpenalised maximum does not exist (weights -> inf).
    prior_sd_extra: if set, the weights of the EXTRA columns (input columns >= inputs.nf, in both
    heads) get N(0, prior_sd_extra^2) instead, so the 48 features keep R0's prior in every arm and only
    the new columns' shrinkage is chosen per arm (prior_sd_extra -> 0 recovers R0 exactly).
    Returns dict(para, nll_sum (unpenalised), nll_mean, objective (penalised, what is minimised),
    grad_max, rounds, converged)."""
    torch.set_num_threads(threads)
    X = torch.from_numpy(_design(inputs, idx))
    A, B, D, F = (torch.from_numpy(np.ascontiguousarray(t[np.asarray(idx)])) for t in terms)
    k = X.shape[1]
    kg = k if gamma_cols is None else 1 + gamma_cols
    # extra-column weights are optimised in units of their prior width (w = sdE * u): the maximum is
    # the same, but a small sdE no longer makes the problem stiff (measured: sdE 1e-4 did not converge)
    wp, extra = penalty_weights(inputs, k, kg, len(X), prior_sd, prior_sd_extra)
    scale = torch.from_numpy(np.where(extra, prior_sd_extra or 1.0, 1.0))
    p0 = torch.zeros(k + kg, dtype=torch.float64) if init is None else torch.tensor(np.asarray(init, np.float64))
    u = (p0 / scale).requires_grad_(True)
    opt = torch.optim.LBFGS([u], lr=1, max_iter=500, tolerance_grad=tol, tolerance_change=1e-16,
                            history_size=50, line_search_fn="strong_wolfe")

    mask = torch.from_numpy(wp) * scale ** 2                        # penalty weights in u units

    def closure():
        opt.zero_grad()
        loss = _nll_torch(u * scale, X, A, B, D, F, beta0, k).mean() + (mask * u * u).sum()
        loss.backward()
        return loss

    prev, rounds, gmax = None, 0, np.inf
    for rounds in range(1, max_rounds + 1):
        loss = float(opt.step(closure).detach())
        gmax = float(u.grad.abs().max())
        if gmax < tol or (prev is not None and abs(prev - loss) < 1e-15):
            break
        prev = loss
    with torch.no_grad():
        p = u * scale
        per = _nll_torch(p, X, A, B, D, F, beta0, k)
        obj = float(per.mean() + (mask * u * u).sum())
    return {"para": p.numpy().copy(), "nll_sum": float(per.sum()), "nll_mean": float(per.mean()), "objective": obj,
            "grad_max": gmax, "rounds": rounds, "converged": bool(gmax < 1e-6), "n": int(len(per))}


def score(inputs, para, idx=None, terms=None, beta0=BETA[0], batch=500_000):
    """(rho, gamma', per-site NLL or None) with the same clamp as `fit`."""
    idx = np.arange(len(inputs.F)) if idx is None else np.asarray(idx)
    k = 1 + inputs.nx
    rho, gp, nll = np.empty(len(idx)), np.empty(len(idx)), (np.empty(len(idx)) if terms is not None else None)
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        X = _design(inputs, j)
        lat = np.clip(X @ para[:k], -LATENT_CLAMP, LATENT_CLAMP)
        r = np.exp(RHO_SHIFT * np.exp(-lat))
        g = 1 / (1 + np.exp(-(X[:, :len(para) - k] @ para[k:])))
        rho[i:i + len(j)], gp[i:i + len(j)] = r, g
        if terms is not None:
            A, B, D, F = (t[j] for t in terms)
            nll[i:i + len(j)] = -np.log((A * (B + D * g * beta0) - F) * r + F)
    return rho, gp, nll


def fit_path(inputs, terms, train, select, prior_sds, gamma_cols=None, threads=8):
    """Fit each prior width from a COLD start (zero init) and keep the one with the lowest mean NLL on
    `select` sites. Cold starts, not warm starts: the likelihood is not convex, and warm-starting from
    the previous width's solution was measured to end in worse local optima (docs/DECISIONS.md).
    Returns (best fit dict + 'prior_sd', {sd: select NLL})."""
    best, path = None, {}
    for sd in sorted(prior_sds):
        r = fit(inputs, terms, train, threads=threads, gamma_cols=gamma_cols, prior_sd=sd)
        _, _, nll = score(inputs, r["para"], select, terms)
        path[sd] = float(nll.mean())
        if best is None or path[sd] < path[best["prior_sd"]]:
            best = {**r, "prior_sd": sd}
    return best, path


def fit_multistart(inputs, terms, idx, inits, **kw):
    """Fit from each start in `inits` (None = cold zero start) and keep the lowest penalised training
    objective. The likelihood is not convex: on the real data two fits with identical information
    converged (gradient < 1e-6) to different optima, objective 0.08246938 vs 0.08246977, ranking the
    benchmark at Spearman 0.989 (docs/DECISIONS.md #10). Choosing by the training objective is the
    definition of the estimator (the posterior mode); it never looks at held-out or benchmark data.
    Returns (best fit dict with 'start', {start: objective})."""
    best, objs = None, {}
    for name, init in inits.items():
        for _ in range(4):                                       # continue a fit that hit the iteration cap
            r = fit(inputs, terms, idx, init=init, **kw)
            if r["converged"]:
                break
            init = r["para"]
        objs[name] = r["objective"]
        if best is None or r["objective"] < best["objective"]:
            best = {**r, "start": name}
    return best, objs

"""s09b mle: penalised maximum-likelihood fit of EVERY arm ("best fit to the evolutionary data").

Why penalised: on the real data rare binary features are quasi-separated (e.g. a feature present at
0.04% of sites, all without variation), so the unpenalised maximum does not exist - weights run to
~1e5 (measured; docs/DECISIONS.md). Each fit maximises the posterior under Gaussian priors on the
weights (intercepts free). Two prior widths (design v3, docs/DECISIONS.md #9):
  sdF  the 48 feature weights: chosen ONCE, by R0, from C.PRIOR_SDS (lowest mean NLL on `select`
       sites), then the same in every arm;
  sdE  the new columns' weights: chosen per arm and seed from C.PRIOR_SDS_EXTRA on `select` sites.
       sdE -> 0 is R0 exactly, so "arm - R0" measures only what the new columns add.
(v2 chose one width per arm for all weights: the positive control then differed from R0 by more than
the test's MDE purely through the width choice - every "arm - R0" was confounded.)
New inputs enter the rho head only (model/mle.py). Benchmark data are never used here.

Test computed here:
  T2 held-out NLL on `eval` sites (whole held-out 1 Mb blocks), per-site differences averaged over
     seeds, 1 Mb-block bootstrap; vs R0 and vs the matched controls. (The likelihood-ratio test T1
     is not used on real data: with a penalty Wilks' theorem does not apply.)
Checks: every fit converged, finite and bounded; the positive control (same information as R0) is
within the test's detectable effect of R0 on held-out data and ranks the benchmark like R0.
"""
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.stats import spearmanr

from arms import ARMS, SAE_NAMES, SEEDED, Data, gamma_cols, r0_start, width
from common import C, CheckReport, finish, log
from lctx import stats
from lctx.io.genome import BigWig
from lctx.model import insight, mle

G = {}
MULTI_SEED = SEEDED | {"noise_emb", "shuf_emb"} | {f"noise{d}" for d, _ in C.SAE_GRID}


def seeds_of(arm):
    return C.SEEDS if arm in MULTI_SEED else (0,)


def _fit(job):
    """One (arm, seed, sdF, sdE, start) fit; every width and start is an independent job (sdE None:
    R0, no extra columns). start "cold" = zeros; "r0" = R0's optimum at sdF with the new columns at 0
    (arms.r0_start). Each finished fit is saved under fits/ and reused by a restart of THIS step (same
    code, same inputs - the directory carries the sha256 of the design-matrix stamp)."""
    arm, seed, sd, sde, start = job
    path = G["fit_path"](*job)
    D, T = G["D"], G["terms"]
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            r = z["r"].item()
        r.setdefault("start", start)                              # fits saved before starts were recorded
        if "objective" not in r:                                  # fits saved before the objective was stored
            inp, _ = D.inputs(arm, seed)
            r["objective"] = mle.objective(inp, r["para"], T, D.idx["train"], sd, sde)
        if r["converged"]:
            return job, r
        init = r["para"]                                          # hit the iteration cap: continue below
    else:
        init = None
    t = time.time()
    inp, pinp = D.inputs(arm, seed)
    if init is None and start == "r0":
        init = r0_start(arm, inp, G["p_r0"], D.posctl_feature)
    # a fit that hits the iteration cap is continued from where it stopped (same objective), at most
    # 3 more times; the convergence gate then judges it
    for _ in range(4):
        r = mle.fit(inp, T, D.idx["train"], init=init, threads=2, gamma_cols=gamma_cols(arm, inp), prior_sd=sd, prior_sd_extra=sde)
        if r["converged"]:
            break
        init = r["para"]
    _, _, nll_sel = mle.score(inp, r["para"], D.idx["select"], T)
    rho_ev, _, nll_ev = mle.score(inp, r["para"], D.idx["eval"], T)
    rho_p, _, _ = mle.score(pinp, r["para"])
    log(f"  fit {arm} s{seed} sdF {sd} sdE {sde} {start}: objective {r['objective']:.9f}, select NLL {nll_sel.mean():.7f}, grad {r['grad_max']:.1e} ({time.time() - t:.0f}s)")
    out = {**r, "prior_sd": sd, "prior_sd_extra": sde, "start": start, "select_nll": float(nll_sel.mean()), "eval_nll_sites": nll_ev,
           "rho_eval": rho_ev, "rho_panel": rho_p}
    np.savez(path + ".tmp.npz", r=np.array(out, dtype=object))
    os.replace(path + ".tmp.npz", path)                      # atomic: a killed job leaves no partial file
    return job, out


def run(run):
    run.gate("s08_design")
    rep = CheckReport("s09b_mle")
    D = Data(run)
    G["D"], G["terms"] = D, insight.site_terms(D.evol)
    # fits are keyed by the inputs they were built from: the s08 checks file (passed, design verified)
    tag = hashlib.sha256(open(run.checks_path("s08_design"), "rb").read() + open(run.path("s02_sites", "sites.npz"), "rb").read(1 << 20)).hexdigest()[:12]
    fdir = os.path.join(os.path.dirname(run.checks_path("s09b_mle")), f"fits_{tag}")
    os.makedirs(fdir, exist_ok=True)
    G["fit_path"] = lambda a, s, sd, sde, st: os.path.join(fdir, f"{a}_s{s}_F{sd}_E{sde}" + ("" if st == "cold" else f"_{st}") + ".npz")
    rep.info["fit_cache_dir"] = fdir
    log(f"penalised ML on {len(D.idx['train'])} training sites; phase 1: R0 feature prior width")
    allfits = {}
    with mp.get_context("fork").Pool(56) as pool:
        for job, r in pool.imap_unordered(_fit, [("repl", 0, sd, None, "cold") for sd in C.PRIOR_SDS]):
            allfits[job] = r
        path_f = {sd: allfits[("repl", 0, sd, None, "cold")]["select_nll"] for sd in C.PRIOR_SDS}
        sdF = min(path_f, key=path_f.get)
    G["p_r0"] = allfits[("repl", 0, sdF, None, "cold")]["para"]
    log(f"  R0 feature prior width sdF = {sdF} (select NLL by width: {path_f})")
    with mp.get_context("fork").Pool(56) as pool:                  # forked after p_r0 is set: workers see it
        # phase 2: every other arm at sdF, extra-column width on a grid, from two starts (cold, R0's
        # optimum); the lower penalised TRAINING objective is the fit (design v4); widest arms first
        # posctl gets exactly R0's prior on its moved (standardised) feature: sdE = sdF * sd(feature), so
        # it is an exact reparametrisation of R0 and must reach R0's training objective
        sde_pc = float(sdF * D.inputs("posctl", 0)[0].sd[0])
        egrid = lambda a: (sde_pc,) if a == "posctl" else C.PRIOR_SDS_EXTRA
        jobs = sorted([(a, s, sdF, e, st) for a in ARMS if a != "repl" for s in seeds_of(a) for e in egrid(a)
                       for st in (("cold",) if a == "fsae_codes" else ("cold", "r0"))], key=lambda j: (-width(j[0]), j[4]))
        log(f"phase 2: {len(jobs)} fits (arm x seed x extra prior width x start)")
        for job, r in pool.imap_unordered(_fit, jobs):
            allfits[job] = r
    best_of = lambda a, s, e: min((allfits[k] for k in allfits if k[:4] == (a, s, sdF, e)), key=lambda r: r["objective"])
    res = {("repl", 0): {**allfits[("repl", 0, sdF, None, "cold")], "select_path": path_f}}
    starts_won = {}
    for a in ARMS:
        for s in (seeds_of(a) if a != "repl" else ()):
            fits = {e: best_of(a, s, e) for e in egrid(a)}
            path = {e: f["select_nll"] for e, f in fits.items()}
            best = min(path, key=path.get)
            res[(a, s)] = {**fits[best], "select_path": path}
            starts_won.update({f"{a}_s{s}_E{e}": f["start"] for e, f in fits.items()})
    # convergence is required of every fit that can be chosen: R0 at sdF and every arm's whole extra grid
    for (a, s, sd, e, st), r in sorted(allfits.items(), key=str):
        if sd == sdF:
            rep.true(f"converged, finite [{a} s{s} sdF {sd} sdE {e} {st}]", r["converged"] and bool(np.isfinite(r["para"]).all()),
                     f"grad {r['grad_max']:.1e}")
    # nesting invariant: an arm containing R0 cannot fit the TRAINING data worse than R0 (penalised)
    o_r0 = res[("repl", 0)]["objective"]
    for (a, s), r in sorted(res.items()):
        if a not in ("repl", "fsae_codes", "posctl"):
            rep.le(f"nested: training objective of {a} s{s} <= R0's (+1e-10)", round(r["objective"] - o_r0, 12), 1e-10)
    rep.close("POSITIVE CONTROL: training objective equals R0's (exact reparametrisation)",
              round(res[("posctl", 0)]["objective"] - o_r0, 12), 0.0, 1e-9)
    rep.info["starts_won"] = {st: sum(v == st for v in starts_won.values()) for st in ("cold", "r0")}
    rep.info["starts_won_detail"] = starts_won
    rep.info["sdF"] = sdF
    rep.info["sdF_at_grid_edge"] = sdF in (min(C.PRIOR_SDS), max(C.PRIOR_SDS))
    rep.info["sdE_at_grid_edge"] = sorted(f"{a}_s{s}:{r['prior_sd_extra']}" for (a, s), r in res.items()
                                          if a != "repl" and r["prior_sd_extra"] in (min(C.PRIOR_SDS_EXTRA), max(C.PRIOR_SDS_EXTRA)))

    for (a, s), r in sorted(res.items()):
        w = np.abs(r["para"][1:])
        rep.true(f"chosen fit bounded [{a} s{s}] (max|w| < 100)", w.max() < 100, f"max|w| {w.max():.2f}, sd {r['prior_sd']}")

    # ---- T2 held-out NLL
    ev = D.idx["eval"]
    blocks = np.char.add(D.names[ev], (D.pos19[ev] // C.BLOCK_BP).astype(str))
    nll_ev = {a: np.mean([res[(a, s)]["eval_nll_sites"] for s in seeds_of(a)], 0) for a in ARMS}
    sel = {a: float(np.mean([res[(a, s)]["select_nll"] for s in seeds_of(a)])) for a in ARMS}
    chosen = min(SAE_NAMES, key=lambda n: sel[n])
    d_ch = width(chosen)
    pairs = [(a, "repl") for a in ARMS if a != "repl"] + [
        ("emb", "noise_emb"), ("emb", "shuf_emb"), (chosen, f"noise{d_ch}"), (chosen, f"shuf_{chosen}"),
        ("emb", "comp"), ("fsae_plus", "fsae_codes")]
    heldout = {f"{a} - {b}": stats.block_bootstrap_mean(nll_ev[a] - nll_ev[b], blocks, reps=C.BOOTSTRAP_REPS)
               for a, b in pairs}

    # ---- positive control and descriptive agreement with published LINSIGHT
    pc = heldout["posctl - repl"]
    rep.le("POSITIVE CONTROL: |held-out NLL posctl - R0| below the test's MDE", round(abs(pc["mean"]), 8), round(pc["mde"], 8))
    rep.ge("POSITIVE CONTROL: benchmark rho of posctl vs R0, Spearman",
           round(float(spearmanr(res[("posctl", 0)]["rho_panel"], res[("repl", 0)]["rho_panel"])[0]), 4), 0.99)
    sub = np.random.default_rng(0).choice(len(ev), min(50000, len(ev)), replace=False)
    bw = BigWig(C.INPUTS.linsight_bw).values_at(D.names[ev][sub].tolist(), D.pos_ins[ev][sub].tolist())
    okb = np.isfinite(bw)
    agree = {a: float(spearmanr(np.mean([res[(a, s)]["rho_eval"] for s in seeds_of(a)], 0)[sub][okb], bw[okb])[0]) for a in ARMS}
    rep.ge("held-out eval sites", len(ev), 1000)
    log(f"SAE config chosen by held-out select NLL: {chosen} ({ {n: round(sel[n], 8) for n in SAE_NAMES} })")
    for k in ("emb - repl", f"{chosen} - repl", "emb - noise_emb", f"{chosen} - noise{d_ch}", "emb - shuf_emb", "comp - repl"):
        h = heldout[k]
        log(f"  T2 {k}: {h['mean']:+.2e} per site (SE {h['se']:.1e}, p {h['p']:.3g}, MDE {h['mde']:.1e})")
    log(f"  R0 (penalised ML, sd {res[('repl', 0)]['prior_sd']}) vs LINSIGHT.bw Spearman {agree['repl']:.3f}")

    rep.info.update({"sae_chosen": chosen, "noise_for_chosen": f"noise{d_ch}", "select_nll": sel,
                     "heldout": heldout, "spearman_vs_published": agree, "n_train": int(len(D.idx["train"])),
                     "n_eval": int(len(ev)), "n_eval_blocks": int(len(np.unique(blocks))),
                     "prior_sd": {f"{a}_s{s}": r["prior_sd"] for (a, s), r in res.items()},
                     "prior_sd_extra": {f"{a}_s{s}": r["prior_sd_extra"] for (a, s), r in res.items()},
                     "select_path": {f"{a}_s{s}": r["select_path"] for (a, s), r in res.items()},
                     "fits": {f"{a}_s{s}": {k: r[k] for k in ("nll_mean", "grad_max", "rounds", "converged", "n")}
                              for (a, s), r in res.items()}})
    np.savez(run.path("s09b_mle", "params.npz"), **{f"{a}_s{s}": r["para"] for (a, s), r in res.items()})
    np.savez(run.path("s09b_mle", "panel_rho.npz"), **{f"{a}_s{s}": r["rho_panel"] for (a, s), r in res.items()})
    np.savez(run.path("s09b_mle", "eval_rho.npz"), **{f"{a}_s{s}": r["rho_eval"] for (a, s), r in res.items()})
    with open(run.path("s09b_mle", "selection.json"), "w") as f:
        json.dump({"sae_chosen": chosen, "noise_for_chosen": f"noise{d_ch}", "select_nll": sel,
                   "prior_sd": {f"{a}_s{s}": r["prior_sd"] for (a, s), r in res.items()},
                   "prior_sd_extra": {f"{a}_s{s}": r["prior_sd_extra"] for (a, s), r in res.items()}}, f, indent=1)
    finish(run, "s09b_mle", rep)

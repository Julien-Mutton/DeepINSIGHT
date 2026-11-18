"""s09c groundtruth: the tests of s09b re-run on data where the answer is KNOWN.

On the REAL sites, features, embeddings and neutral rates, new INSIGHT observations are simulated
(model/simulate.py) from
    latent_true = latent_R0 + s * z,      z = unit-variance projection of the standardised HyenaDNA
                                              embedding on a fixed random direction
with R0 = the exact-ML fit of s09b (so s = 0 is a world where the sequence adds nothing).
Each world is fitted exactly like the real data: R0 at the real feature prior width sdF; every other
arm at sdF with its extra-column width sdE re-chosen IN THAT WORLD on `select` sites from
C.PRIOR_SDS_EXTRA, each width fitted from two starts (cold, and that world's R0 optimum) keeping the
lower training objective (the selection and the starts are part of the procedure whose error rates
are measured); then
T2 (held-out NLL, block bootstrap).
Measured:
  * false-positive rate of T2 in null worlds (s = 0), also for R2 and the noise arm
  * power of T2 and recovery of the planted signal as a function of s
  * the paper procedure in null worlds: does adding 256 columns degrade it with NO true effect?
  * the planted size s* whose held-out gain equals the REAL embedding's (calibration of the result)
"""
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from arms import Data, gamma_cols, r0_start, width
from common import C, CheckReport, finish, load_npz, log
from lctx import stats
from lctx.model import insight, linsight as LS, mle, simulate as SIM

EFFECTS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4)
REPS = {0.0: 20, 0.025: 5, 0.05: 5, 0.1: 5, 0.2: 5, 0.4: 5}
G = {}


def _world(s, rep):
    """Simulated evol + true rho for every site (deterministic in (s, rep)); one world cached per worker."""
    if G.get("wkey") != (s, rep):
        D = G["D"]
        rng = np.random.default_rng(int(1e6 * s) * 100 + rep)
        lat = G["lat0"] + s * G["z"]
        rho = np.exp(C.RHO_SHIFT * np.exp(-np.clip(lat, -C.LATENT_CLAMP, C.LATENT_CLAMP)))
        evol, _, ex = SIM.simulate_evol(D.evol[:, 3], D.evol[:, 4], rho, G["gp0"], rng)
        G["wkey"], G["world"] = (s, rep), (insight.site_terms(evol), rho, float(np.abs(ex).max()))
    return G["world"]


def _cached(name, fn):
    path = os.path.join(G["fdir"], name + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:
            return z["r"].item()
    out = fn()
    np.savez(path + ".tmp.npz", r=np.array(out, dtype=object))
    os.replace(path + ".tmp.npz", path)
    return out


def _fit(job):
    """One fit in one world: (s, rep, arm, sdE, start). R0 (sdE None) is fitted cold; other arms from
    'cold' or 'r0' (that world's R0 optimum, passed in G['p_r0'][(s, rep)])."""
    s, rep, arm, sde, start = job

    def go():
        D = G["D"]
        T, rho_true, ex = _world(s, rep)
        inp, _ = D.inputs(arm, 0)
        init = None if start == "cold" else r0_start(arm, inp, G["p_r0"][(s, rep)], D.posctl_feature)
        for _ in range(4):                                       # continue a fit that hit the iteration cap
            r = mle.fit(inp, T, D.idx["train"], init=init, threads=2, gamma_cols=gamma_cols(arm, inp),
                        prior_sd=G["sdF"], prior_sd_extra=sde)
            if r["converged"]:
                break
            init = r["para"]
        ev = D.idx["eval"]
        rho_ev, _, nll_ev = mle.score(inp, r["para"], ev, T)
        return {"para": r["para"], "objective": r["objective"], "converged": r["converged"],
                "grad_max": r["grad_max"], "excess_max": ex,
                "select_nll": float(mle.score(inp, r["para"], D.idx["select"], T)[2].mean()),
                "nll_ev": nll_ev, "corr_truth_eval": float(np.corrcoef(rho_ev, rho_true[ev])[0, 1])}
    return job, _cached(f"w{s}_{rep}_{arm}_E{sde}_{start}", go)


def _paper(job):
    """The paper (C++) procedure in a null world: does it degrade with extra columns and no effect?"""
    s, rep, arm = job

    def go():
        D = G["D"]
        T, rho_true, _ = _world(s, rep)
        inp, _ = D.inputs(arm, 0)
        tr = LS.Trainer(inp, T)
        for ep in range(C.EPOCHS):
            tr.epoch(np.random.default_rng(1000 + ep).permutation(D.idx["train"]))
        r_ev, _, _ = LS.score(inp, tr.para, D.idx["eval"])
        return {"corr_truth_eval": float(np.corrcoef(r_ev, rho_true[D.idx["eval"]])[0, 1])}
    return job, _cached(f"paper_w{s}_{rep}_{arm}", go)


def _assemble(worlds, fits):
    """Per world and arm: per sdE the lower training objective over starts, then the sdE with the
    lowest select NLL - exactly the rule of s09b."""
    res = {}
    for (s, rep), arms in worlds.items():
        out = {"excess_max": fits[(s, rep, "repl", None, "cold")]["excess_max"]}
        for arm in arms:
            grid = (None,) if arm == "repl" else C.PRIOR_SDS_EXTRA
            per_e = {e: min((fits[k] for k in fits if k[:4] == (s, rep, arm, e)), key=lambda r: r["objective"]) for e in grid}
            e_best = min(per_e, key=lambda e: per_e[e]["select_nll"])
            r = per_e[e_best]
            ok = all(fits[k]["converged"] for k in fits if k[:3] == (s, rep, arm))
            out[(arm, 0)] = {"converged": ok, "nll_ev": r["nll_ev"], "sdE": e_best, "corr_truth_eval": r["corr_truth_eval"],
                             "objective": r["objective"]}
        res[(s, rep)] = out
    return res


def run(run):
    run.gate("s09b_mle")
    rep = CheckReport("s09c_groundtruth")
    D = Data(run)
    sel = json.load(open(run.path("s09b_mle", "selection.json")))
    chosen = sel["sae_chosen"]
    P = load_npz(run.path("s09b_mle", "params.npz"))
    inp0, _ = D.inputs("repl", 0)
    idx_all = np.arange(len(D.F))
    k = 1 + inp0.nx
    X0 = np.hstack([np.ones((len(idx_all), 1)), inp0.dense(idx_all)])
    G["lat0"] = X0 @ P["repl_s0"][:k]
    G["gp0"] = 1 / (1 + np.exp(-(X0 @ P["repl_s0"][k:])))
    embi, _ = D.inputs("emb", 0)
    Ez = (np.asarray(D.E, np.float64) - embi.mu) / embi.sd
    u = np.random.default_rng(99).normal(size=C.D_MODEL)
    z = np.nan_to_num(Ez @ (u / np.linalg.norm(u)))
    tr = D.idx["train"]
    G["z"] = (z - z[tr].mean()) / z[tr].std()
    G["D"] = D
    G["sdF"] = sel["prior_sd"]["repl_s0"]
    rep.info["R0 latent sd (train)"] = float(G["lat0"][tr].std())

    worlds = {(s, r): ["repl", "emb"] + ([chosen, "noise_emb"] if s == 0.0 else []) for s in EFFECTS for r in range(REPS[s])}
    tag = hashlib.sha256(open(run.checks_path("s09b_mle"), "rb").read()).hexdigest()[:12]
    G["fdir"] = os.path.join(os.path.dirname(run.checks_path("s09c_groundtruth")), f"fits_{tag}")
    os.makedirs(G["fdir"], exist_ok=True)
    rep.info["fit_cache_dir"] = G["fdir"]
    t = time.time()
    fits = {}
    # phase A: every world's R0 (its optimum is the second start of the other arms)
    with mp.get_context("fork").Pool(min(len(worlds), 48)) as pool:
        for job, r in pool.imap_unordered(_fit, [(s, r, "repl", None, "cold") for (s, r) in worlds]):
            fits[job] = r
    G["p_r0"] = {(s, r): fits[(s, r, "repl", None, "cold")]["para"] for (s, r) in worlds}
    log(f"phase A: {len(worlds)} world R0 fits in {time.time() - t:.0f}s")
    # phase B: every (world, arm, sdE, start) as its own job, widest arms first; paper procedure jobs
    jobs = sorted([(s, r, a, e, st) for (s, r), arms in worlds.items() for a in arms if a != "repl"
                   for e in C.PRIOR_SDS_EXTRA for st in ("cold", "r0")], key=lambda j: (-width(j[2]), j[0], j[1]))
    pjobs = [(0.0, r, a) for r in range(3) for a in ("repl", "emb", "noise_emb")]
    log(f"phase B: {len(jobs)} fits + {len(pjobs)} paper-procedure fits")
    with mp.get_context("fork").Pool(56) as pool:
        paper_async = pool.map_async(_paper, pjobs)
        for i, (job, r) in enumerate(pool.imap_unordered(_fit, jobs)):
            fits[job] = r
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(jobs)} fits ({time.time() - t:.0f}s)")
        paper_res = dict(paper_async.get())
    res = _assemble(worlds, fits)
    for (s, r, a), o in paper_res.items():
        res[(s, r)][f"paper_{a}"] = o
    log(f"{len(worlds)} simulated worlds fitted in {time.time() - t:.0f}s")

    ev = D.idx["eval"]
    blocks = np.char.add(D.names[ev], (D.pos19[ev] // C.BLOCK_BP).astype(str))
    rows = []
    for (s, r), o in sorted(res.items()):
        rep.true(f"world s={s} rep {r}: all fits converged", all(o[a]["converged"] for a in o if isinstance(a, tuple)))
        rep.le(f"world s={s} rep {r}: simulator excess <= 1.5 theta*lambda bound (max)", round(o["excess_max"], 6), 0.005)
        for a in [a for a in o if isinstance(a, tuple) and a[0] != "repl"]:
            rep.le(f"world s={s} rep {r}: nested {a[0]} training objective <= R0's (+1e-10)",
                   round(o[a]["objective"] - o[("repl", 0)]["objective"], 12), 1e-10)
        for arm in [a for a in o if isinstance(a, tuple) and a[0] != "repl"]:
            t2 = stats.block_bootstrap_mean(o[arm]["nll_ev"] - o[("repl", 0)]["nll_ev"], blocks, reps=500, seed=r)
            rows.append({"s": s, "rep": r, "arm": arm[0], "heldout_mean": t2["mean"],
                         "heldout_se": t2["se"], "heldout_p": t2["p"],
                         "corr_truth": o[arm]["corr_truth_eval"], "corr_truth_R0": o[("repl", 0)]["corr_truth_eval"]})
    summary = {}
    for s in EFFECTS:
        for arm in sorted({x["arm"] for x in rows if x["s"] == s}):
            xs = [x for x in rows if x["s"] == s and x["arm"] == arm]
            summary[f"s={s} {arm}"] = {
                "n_worlds": len(xs),
                "T2_reject_rate": float(np.mean([(x["heldout_p"] < C.ALPHA) and (x["heldout_mean"] < 0) for x in xs])),
                "T2_heldout_gain_mean": float(np.mean([-x["heldout_mean"] for x in xs])),
                "corr_truth_arm_minus_R0": float(np.mean([x["corr_truth"] - x["corr_truth_R0"] for x in xs]))}
    paper = {a: [o[f"paper_{a}"]["corr_truth_eval"] for (s, r), o in res.items() if f"paper_{a}" in o]
             for a in ("repl", "emb", "noise_emb")}

    # gates: tests must be calibrated in null worlds and powerful for the largest planted effect
    null_emb = summary["s=0.0 emb"]
    rep.le("NULL worlds: T2 false-positive rate (emb)", null_emb["T2_reject_rate"], 0.3,
           "20 worlds; expected 0.05; 0.3 = clearly miscalibrated")
    for arm in [k.split()[1] for k in summary if k.startswith("s=0.0 ") and not k.endswith(" emb")]:
        rep.le(f"NULL worlds: T2 false-positive rate ({arm})", summary[f"s=0.0 {arm}"]["T2_reject_rate"], 0.3)
    big = summary[f"s={EFFECTS[-1]} emb"]
    rep.ge(f"PLANTED s={EFFECTS[-1]}: T2 power", big["T2_reject_rate"], 0.8)
    rep.ge(f"PLANTED s={EFFECTS[-1]}: R1 recovers truth better than R0 (corr gain)", round(big["corr_truth_arm_minus_R0"], 5), 0.0)

    # calibrate the REAL result: planted s whose mean held-out gain equals the real one
    real = json.load(open(run.checks_path("s09b_mle")))["info"]["heldout"]["emb - repl"]
    gains = np.array([summary[f"s={s} emb"]["T2_heldout_gain_mean"] for s in EFFECTS])
    s_star = float(np.interp(-real["mean"], gains, EFFECTS)) if -real["mean"] > gains[0] else 0.0
    rep.info.update({"effects": list(EFFECTS), "summary": summary, "rows": rows, "paper_null_corr_truth": paper,
                     "real_emb_heldout_gain": -real["mean"], "equivalent_planted_s": s_star, "sae_used": chosen})
    for kname, v in summary.items():
        log(f"  {kname}: T2 reject {v['T2_reject_rate']:.2f}  gain {v['T2_heldout_gain_mean']:+.2e}  "
            f"corr gain {v['corr_truth_arm_minus_R0']:+.4f}")
    log(f"paper procedure in null worlds, corr(rho, truth): {paper}")
    log(f"real embedding held-out gain {-real['mean']:+.2e} ~ planted s = {s_star:.4f}")
    finish(run, "s09c_groundtruth", rep)

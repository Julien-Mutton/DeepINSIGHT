"""s09 train (paper procedure): the LINSIGHT C++ procedure for the arms in PAPER_ARMS x 5 seeds.

seed 0 visits training sites in the C++ order (chromosome string order, pos_ins ascending);
other seeds use a fresh random permutation each epoch. 20 epochs; the final epoch is used.
Recorded per epoch: parameters and mean INSIGHT NLL on held-out `select` and `eval` sites.
This is the REPLICATION estimator; information tests use the exact fit of s09b.
"""
import multiprocessing as mp
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

from arms import PAPER_ARMS, Data
from common import C, CheckReport, finish, load_npz, log
from lctx.io.genome import BigWig
from lctx.model import insight, linsight as LS

G = {}


def _job(job):
    from threadpoolctl import threadpool_limits
    arm, seed = job
    D, terms = G["D"], G["terms"]
    with threadpool_limits(1):
        inp, pinp = D.inputs(arm, seed)
        tr = LS.Trainer(inp, terms)
        train = D.idx["train"]
        cpp = train[LS.cpp_order(D.names[train], D.pos_ins[train])]
        hist = {"select_nll": [], "eval_nll": []}
        paras = []
        t = time.time()
        for ep in range(C.EPOCHS):
            order = cpp if seed == 0 else np.random.default_rng(1000 * seed + ep).permutation(train)
            tr.epoch(order)
            for part in ("select", "eval"):
                idx = D.idx[part]
                rho, gp, _ = LS.score(inp, tr.para, idx)
                hist[f"{part}_nll"].append(float(np.mean(insight.nll(tuple(x[idx] for x in terms), rho, gp))))
            paras.append(tr.para.copy())
        rho_eval, _, _ = LS.score(inp, tr.para, D.idx["eval"])
        rho_panel, _, lat_panel = LS.score(pinp, tr.para)
    log(f"  paper {arm} s{seed}: final eval-NLL {hist['eval_nll'][-1]:.7f} ({time.time() - t:.0f}s)")
    return job, {"hist": hist, "para": np.array(paras), "rho_eval": rho_eval, "rho_panel": rho_panel,
                 "lat_extra_sd_panel": float(np.std(lat_panel))}


def run(run):
    run.gate("s08_design")
    rep = CheckReport("s09_train")
    D = Data(run)
    G["D"], G["terms"] = D, insight.site_terms(D.evol)
    jobs = [(a, s) for a in PAPER_ARMS for s in C.SEEDS]
    log(f"paper procedure: {len(jobs)} jobs on {len(D.idx['train'])} training sites")
    res = {}
    with mp.get_context("fork").Pool(min(len(jobs), 100)) as pool:
        for job, r in pool.imap_unordered(_job, jobs):
            res[job] = r
    ev = D.idx["eval"]
    sub = np.random.default_rng(0).choice(len(ev), min(50000, len(ev)), replace=False)
    bw = BigWig(C.INPUTS.linsight_bw).values_at(D.names[ev][sub].tolist(), D.pos_ins[ev][sub].tolist())
    ok = np.isfinite(bw)
    out = {}
    for a in PAPER_ARMS:
        fin = all(np.isfinite(res[(a, s)]["para"][-1]).all() for s in C.SEEDS)
        r = np.mean([res[(a, s)]["rho_eval"] for s in C.SEEDS], 0)
        out[a] = {"finite": fin, "eval_nll": float(np.mean([res[(a, s)]["hist"]["eval_nll"][-1] for s in C.SEEDS])),
                  "select_nll": float(np.mean([res[(a, s)]["hist"]["select_nll"][-1] for s in C.SEEDS])),
                  "spearman_vs_published": float(spearmanr(r[sub][ok], bw[ok])[0]) if fin else float("nan"),
                  "lat_extra_sd_panel": float(np.mean([res[(a, s)]["lat_extra_sd_panel"] for s in C.SEEDS])),
                  "param_shape": list(res[(a, C.SEEDS[0])]["para"].shape)}
        rep.true(f"finite parameters [{a}]", fin)
        rep.eq(f"parameter shape [{a}] = (epochs, 2(1+nx))", out[a]["param_shape"][0], C.EPOCHS)
    # The paper procedure under-converges as the training set shrinks (docs/DECISIONS.md, ground truth):
    # 0.923 with 90% of 0.1% sites, 0.872 with 50%. Correctness of the implementation is proven exactly
    # elsewhere (kernel == LINSIGHT-fit binary; features reproduce LINSIGHT.bw with R^2 = 1); the
    # >= 0.9 replication gate is on the converged exact-ML R0 in s09b.
    rep.ge(f"paper-procedure R0 vs LINSIGHT.bw, Spearman on eval sites ({len(D.idx['train']):,} training sites)",
           round(out["repl"]["spearman_vs_published"], 4), 0.85)
    rep.ge("POSITIVE CONTROL: posctl rho vs LINSIGHT.bw, Spearman", round(out["posctl"]["spearman_vs_published"], 4), 0.85)
    pub = load_npz(run.path("s04_features", "published_weights.npz"))["rho_coef"]
    w = np.mean([res[("repl", s)]["para"][-1][:1 + C.N_FEATURES] for s in C.SEEDS], 0)
    rep.info.update({"arms": out, "R0_rho_weights_vs_published_pearson": float(pearsonr(w[1:], pub[1:])[0]),
                     "R0_rho_weight_norm_ratio_to_published": float(np.linalg.norm(w[1:]) / np.linalg.norm(pub[1:])),
                     "history": {f"{a}_s{s}": res[(a, s)]["hist"] for a in PAPER_ARMS for s in C.SEEDS},
                     "n_rows": {k: int(len(v)) for k, v in D.idx.items()}})
    np.savez(run.path("s09_train", "params.npz"), **{f"{a}_s{s}": res[(a, s)]["para"] for a in PAPER_ARMS for s in C.SEEDS})
    np.savez(run.path("s09_train", "panel_rho.npz"), **{f"{a}_s{s}": res[(a, s)]["rho_panel"] for a in PAPER_ARMS for s in C.SEEDS})
    finish(run, "s09_train", rep)

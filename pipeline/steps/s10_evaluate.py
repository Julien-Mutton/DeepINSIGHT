"""s10 evaluate: T3, the single read of the HGMD benchmark under a frozen plan.

Plan v2 (this file). A first plan (v1: paper procedure, 3 seeds, 7 columns) was read once
for run rate_0.001; v2 was designed AFTER seeing v1 (more controls, exact-ML estimator,
seed-aware CIs, Holm, MDE). The report states this. Every arm and comparison is fixed here, before
the v2 read; nothing is chosen with benchmark labels.

Scores: each arm's rho on every usable benchmark variant; arms with several seeds are kept as a
(seeds x variants) matrix so the bootstrap resamples seeds as well as 100-kb loci.
Two primary families, Holm-adjusted separately (each answers its own question):
  A_paper   LINSIGHT as its authors train it (C++ procedure, 5 seeds): R1-R0, R2-R0, R1-noise,
            R2-noise, R1-shuffled
  B_bestfit best fit to the evolutionary data (penalised ML): the same + R2-shuffled
Secondary (own Holm adjustment): composition, feature-SAE, positive controls, replication.
"""
import json
import multiprocessing as mp
import zlib

import numpy as np

from arms import ARMS, PAPER_ARMS, Data
from common import C, CheckReport, finish, load_npz, log
from lctx import stats
from lctx.eval.metrics import SingleReadLock, auroc, locus_clusters
from lctx.io.benchmark import tier_mask
from steps.s09b_mle import seeds_of

G = {}


def _boot(job):
    name, tier, a, b = job
    m = G["masks"][tier]
    return job, stats.auroc_delta_bootstrap(G["y"][m], G["S"][a][:, m], G["S"][b][:, m], G["cl"][tier],
                                            reps=C.BOOTSTRAP_REPS, seed=zlib.crc32(f"{name}|{tier}".encode()))


def run(run):
    run.gate("s09_train", "s09b_mle", "s09c_groundtruth")
    rep = CheckReport("s10_evaluate")
    D = Data(run)
    sel = json.load(open(run.path("s09b_mle", "selection.json")))
    R2, N2 = sel["sae_chosen"], sel["noise_for_chosen"]
    fam_b = [("R1 - R0", "mle:repl", "mle:emb"), ("R2 - R0", "mle:repl", f"mle:{R2}"),
             ("R1 - noise", "mle:noise_emb", "mle:emb"), ("R2 - noise", f"mle:{N2}", f"mle:{R2}"),
             ("R1 - shuffled", "mle:shuf_emb", "mle:emb"), ("R2 - shuffled", f"mle:shuf_{R2}", f"mle:{R2}")]
    fam_a = [("R1 - R0", "paper:repl", "paper:emb"), ("R2 - R0", "paper:repl", f"paper:{R2}"),
             ("R1 - noise", "paper:noise_emb", "paper:emb"), ("R2 - noise", f"paper:{N2}", f"paper:{R2}"),
             ("R1 - shuffled", "paper:shuf_emb", "paper:emb")]
    secondary = [("comp - R0", "mle:repl", "mle:comp"), ("R1 - comp", "mle:comp", "mle:emb"),
                 ("feature-SAE codes - R0", "mle:repl", "mle:fsae_codes"), ("R0 + feature-SAE - R0", "mle:repl", "mle:fsae_plus"),
                 ("posctl - R0 (best fit)", "mle:repl", "mle:posctl"), ("posctl - R0 (paper)", "paper:repl", "paper:posctl"),
                 ("R0 best fit - published", "published", "mle:repl"), ("R0 paper - published", "published", "paper:repl"),
                 ("R0 best fit - R0 paper", "paper:repl", "mle:repl")]
    families = {"A_paper": fam_a, "B_bestfit": fam_b, "secondary": secondary}
    plan = {"version": 2, "families": families, "seeds": list(C.SEEDS),
            "estimators": {"mle": "penalised ML, prior width chosen on select sites (s09b)",
                           "paper": "C++ procedure, final epoch (s09)"},
            "rows": "benchmark variants usable in every column", "bootstrap": [C.BOOTSTRAP_REPS, C.CLUSTER_BP],
            "multiplicity": "Holm within each family (A_paper, B_bestfit, secondary)", "sae_chosen_without_benchmark": sel}
    lock = SingleReadLock(run.path("s10_evaluate", "lock.json"))
    h = lock.open(plan)
    log(f"benchmark opened for plan v2 {h}; earlier reads in the log: {len(lock.log['reads']) - 1}")

    PM, PP = load_npz(run.path("s09b_mle", "panel_rho.npz")), load_npz(run.path("s09_train", "panel_rho.npz"))
    S = {"published": D.panel.linsight_published.to_numpy()[None]}
    for a in ARMS:
        S[f"mle:{a}"] = np.stack([PM[f"{a}_s{s}"] for s in seeds_of(a)])
    for a in PAPER_ARMS:
        S[f"paper:{a}"] = np.stack([PP[f"{a}_s{s}"] for s in C.SEEDS])
    common = D.panel_usable & np.all([np.isfinite(v).all(0) for v in S.values()], axis=0)
    rep.ge("paired rows / all benchmark rows", round(float(common.mean()), 4), 0.95)
    y = D.panel.label.to_numpy().astype(bool)
    masks = {t: tier_mask(D.panel, t) & common for t in C.TIERS}
    G.update(S=S, y=y, masks=masks,
             cl={t: locus_clusters(D.panel.chrom[m], D.panel.pos19[m], C.CLUSTER_BP) for t, m in masks.items()})
    res = {"plan_hash": h, "plan_version": 2, "n_reads_logged": len(lock.log["reads"]), "tiers": {}}
    for t, m in masks.items():
        res["tiers"][t] = {"paper": C.TIERS[t], "n": int(m.sum()), "n_pos": int(y[m].sum()),
                           "auroc": {k: auroc(v.mean(0)[m], y[m]) for k, v in S.items()},
                           "auroc_seed_sd": {k: float(np.std([auroc(r[m], y[m]) for r in v])) for k, v in S.items() if len(v) > 1}}
    jobs = [(f"{fn}:{n}", t, a, b) for fn, fam in families.items() for n, a, b in fam for t in C.TIERS]
    with mp.get_context("fork").Pool(min(len(jobs), 96)) as pool:
        out = dict(pool.map(_boot, jobs))
    for fn, fam in families.items():
        keys = [(f"{fn}:{n}", t, a, b) for n, a, b in fam for t in C.TIERS]
        adj = stats.holm([out[k]["p"] for k in keys])
        for k, pa in zip(keys, adj):
            res["tiers"][k[1]].setdefault(fn, {})[k[0].split(":", 1)[1]] = {**out[k], "p_holm": float(pa), "a": k[2], "b": k[3]}
    m = masks["unmatched"]
    from sklearn.metrics import roc_auc_score
    rep.close("auroc implementation equals sklearn", round(auroc(S["mle:repl"][0][m], y[m]) - roc_auc_score(y[m], S["mle:repl"][0][m]), 12), 0.0, 1e-9)
    from scipy.stats import spearmanr
    rep.ge("posctl ranks the benchmark like R0 (best fit), Spearman", round(float(spearmanr(S["mle:posctl"][0][common], S["mle:repl"][0][common])[0]), 4), 0.99)
    rep.eq("benchmark reads for this plan", sum(r["hash"] == h for r in lock.log["reads"]), 1)
    for fn in ("A_paper", "B_bestfit"):
        for t in C.TIERS:
            for n, d in res["tiers"][t][fn].items():
                log(f"{fn:9s} {t:18s} {n:14s} {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] p_holm {d['p_holm']:.3g} MDE {d['mde']:.4f}")
    with open(run.path("s10_evaluate", "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    finish(run, "s10_evaluate", rep)

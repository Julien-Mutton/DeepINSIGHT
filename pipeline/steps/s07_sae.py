"""s07 sae: TopK sparse autoencoders on the HyenaDNA mean embeddings of TRAINING sites only
(no labels, no benchmark rows), grid C.SAE_GRID x seeds; validated on held-out `select` sites.
Codes are then computed for every site and every benchmark variant with the same encoder."""
import multiprocessing as mp
import os

import numpy as np

from common import C, CheckReport, finish, load_npz, log
from lctx.sae import topk

MAX_TRAIN = 1_000_000


def _fit(args):
    import torch
    torch.set_num_threads(5)
    d, k, seed, path = args
    S = load_npz(path["sites"])
    E = np.load(path["emb"], mmap_mode="r")
    ok = np.isfinite(np.asarray(E[:, 0]))
    tr = np.where(ok & (S["role"] == "train"))[0]
    va = np.where(ok & (S["role"] == "select"))[0]
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(tr, min(MAX_TRAIN, len(tr)), replace=False))
    cfg = topk.SAEConfig(d_sae=d, k=k, seed=seed)
    m = topk.fit(np.asarray(E[tr]), cfg)
    name = f"sae{d}k{k}"
    topk.save(m, cfg, os.path.join(path["out"], f"{name}_s{seed}.pt"))
    mt = topk.metrics(m, np.asarray(E[tr[:200000]]))
    mv = topk.metrics(m, np.asarray(E[va]))
    for which, X in (("sites", E), ("panel", np.load(path["pemb"], mmap_mode="r"))):
        Xa = np.asarray(X)
        okr = np.isfinite(Xa[:, 0])
        Z = np.full((len(Xa), d), np.nan, np.float32)
        Z[okr] = topk.encode(m, Xa[okr])
        np.save(os.path.join(path["out"], f"codes_{name}_s{seed}_{which}.npy"), Z)
    return name, seed, mt, mv, len(tr), len(va)


def _fit_features(args):
    """SAE on the 48 LINSIGHT features (no sequence): the `fsae_*` arms' codes."""
    import torch
    torch.set_num_threads(5)
    seed, path = args
    S = load_npz(path["sites"])
    SF = load_npz(path["feat"])
    PF = load_npz(path["pfeat"])
    tr = np.where(SF["found"] & (S["role"] == "train"))[0]
    va = np.where(SF["found"] & (S["role"] == "select"))[0]
    tr = np.sort(np.random.default_rng(seed).choice(tr, min(MAX_TRAIN, len(tr)), replace=False))
    cfg = topk.SAEConfig(d_sae=32, k=8, seed=seed)
    m = topk.fit(SF["F"][tr], cfg)
    topk.save(m, cfg, os.path.join(path["out"], f"fsae32k8_s{seed}.pt"))
    mt, mv = topk.metrics(m, SF["F"][tr[:200000]]), topk.metrics(m, SF["F"][va])
    for which, F in (("sites", SF["F"]), ("panel", PF["F"])):
        np.save(os.path.join(path["out"], f"codes_fsae32k8_s{seed}_{which}.npy"), topk.encode(m, F))
    return "fsae32k8", seed, mt, mv, len(tr), len(va)


def run(run):
    run.gate("s02_sites", "s06_embed")
    rep = CheckReport("s07_sae")
    path = {"sites": run.path("s02_sites", "sites.npz"), "emb": run.path("s06_embed", "sites_mean.npy"),
            "pemb": run.path("s06_embed", "panel_mean.npy"), "out": os.path.dirname(run.checks_path("s07_sae"))}
    path.update({"feat": run.path("s04_features", "sites_features.npz"),
                 "pfeat": run.path("s04_features", "panel_features.npz")})
    jobs = [(d, k, s, path) for d, k in C.SAE_GRID for s in C.SEEDS]
    with mp.get_context("fork").Pool(len(jobs) + len(C.SEEDS)) as pool:
        fres = pool.map_async(_fit_features, [(s, path) for s in C.SEEDS])
        res = pool.map(_fit, jobs)
        fres = fres.get()
    for name, seed, mt, mv, ntr, nva in fres:
        tag = f"{name}_s{seed}"
        rep.le(f"{tag} (48 features): L0 <= 8", round(mv["l0"], 4), 8)
        rep.le(f"{tag} (48 features): held-out FVU", round(mv["fvu"], 4), 0.2)
        rep.le(f"{tag} (48 features): overfit", round(mv["fvu"] - mt["fvu"], 4), 0.05)
        rep.info[tag] = {"train": mt, "heldout": mv, "n_train": ntr, "n_heldout": nva}
        log(f"{tag}: held-out FVU {mv['fvu']:.4f} (feature SAE)")
    E = np.load(path["emb"], mmap_mode="r")
    Xv = np.asarray(E[np.where(np.isfinite(np.asarray(E[:, 0])) & (load_npz(path["sites"])["role"] == "select"))[0]], np.float64)
    Xc = Xv - Xv.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False) ** 2
    for name, seed, mt, mv, ntr, nva in res:
        k = int(name.split("k")[1])
        tag = f"{name}_s{seed}"
        pca_fvu = float(1 - sv[:k].sum() / sv.sum())
        rep.le(f"{tag}: L0 <= k", round(mv["l0"], 4), k)
        rep.le(f"{tag}: held-out FVU below PCA with k components", round(mv["fvu"] - pca_fvu, 4), 0.0)
        rep.le(f"{tag}: held-out FVU", round(mv["fvu"], 4), 0.5)
        rep.le(f"{tag}: held-out FVU - train FVU (overfit)", round(mv["fvu"] - mt["fvu"], 4), 0.05)
        rep.le(f"{tag}: dead latent frac", round(mv["dead_frac"], 4), 0.5)
        rep.info[tag] = {"train": mt, "heldout": mv, "pca_fvu_same_k": pca_fvu, "n_train": ntr, "n_heldout": nva}
        log(f"{tag}: held-out FVU {mv['fvu']:.4f} (PCA with {k} comps {pca_fvu:.4f}), dead {mv['dead_frac']:.2f}")
    finish(run, "s07_sae", rep)

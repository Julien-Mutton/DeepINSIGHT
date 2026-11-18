"""s06 embed: HyenaDNA mean embedding of the 1001-bp hg38 window centred on every lifted site
and every lifted benchmark variant. Both sets go through the same `windows -> embed` worker.

Rows that did not lift are NaN. Checks: every lifted row finite; a fresh embedder in the parent
process reproduces random rows exactly (and a shuffled pairing does not); rows not degenerate.
"""
import hashlib
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from common import C, CheckReport, finish, load_npz, log

N_PROC, CHUNK, BATCH = 100, 512, 16
_W = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from lctx.embed.hyena import HyenaEmbedder
    from lctx.io.genome import Reference
    _W["emb"] = HyenaEmbedder(C.INPUTS.hyena_dir)
    _W["ref"] = Reference(C.INPUTS.hg38)


def windows(ref, chroms, pos38):
    return [ref.window(c, int(p), C.CONTEXT) for c, p in zip(chroms, pos38)]


def _work(task):
    i0, chroms, pos38 = task
    seqs = windows(_W["ref"], chroms, pos38)
    mean, center = [], []
    for j in range(0, len(seqs), BATCH):
        e = _W["emb"].embed(seqs[j:j + BATCH])
        mean.append(e["mean"]); center.append(e["center"])
    return i0, np.concatenate(mean), np.concatenate(center)


def embed_all(chroms, pos38):
    n = len(pos38)
    M = np.full((n, C.D_MODEL), np.nan, np.float32)
    Z = np.full((n, C.D_MODEL), np.nan, np.float32)
    tasks = [(i, chroms[i:i + CHUNK], pos38[i:i + CHUNK]) for i in range(0, n, CHUNK)]
    t, done = time.time(), 0
    with mp.get_context("fork").Pool(N_PROC, initializer=_init) as pool:
        for i0, m, c in pool.imap_unordered(_work, tasks):
            M[i0:i0 + len(m)], Z[i0:i0 + len(c)] = m, c
            done += len(m)
            if done // CHUNK % 100 == 0:
                log(f"  embedded {done}/{n} ({done / (time.time() - t):.0f}/s)")
    return M, Z


def run(run):
    run.gate("s03_liftover", "s05_embedder")
    rep = CheckReport("s06_embed")
    S = load_npz(run.path("s02_sites", "sites.npz"))
    SL = load_npz(run.path("s03_liftover", "sites_lift.npz"))
    PL = load_npz(run.path("s03_liftover", "panel_lift.npz"))
    pan = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"), usecols=["key", "chrom"])
    rep.true("panel lift keys aligned with panel", bool((PL["key"] == pan.key.to_numpy()).all()))
    snames = np.array(C.AUTOSOMES)[S["chrom"]]
    pnames = pan.chrom.to_numpy().astype(str)
    sok, pok = SL["status"] == "ok", PL["status"] == "ok"
    # one combined job so both sets share the same workers and code path
    chroms = np.concatenate([snames[sok], pnames[pok]])
    pos38 = np.concatenate([SL["pos38"][sok], PL["pos38"][pok]])
    # content address of the work: every window (chrom, hg38 position), the model weights, the length
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(chroms.astype("U5")).tobytes()); h.update(np.ascontiguousarray(pos38).tobytes())
    h.update(C.HYENA_WEIGHTS_SHA256.encode()); h.update(str(C.CONTEXT).encode())
    target = h.hexdigest()
    rep.info["targets_sha256"] = target
    stamp = run.path("s06_embed", "targets.sha256")
    ns = int(sok.sum())
    if os.path.exists(stamp) and open(stamp).read() == target and os.path.exists(run.path("s06_embed", "panel_center.npy")):
        log(f"windows identical to the existing embeddings (sha256 {target[:12]}): reusing them; checks re-run below")
        rep.info["reused_identical_targets"] = True
        M = np.concatenate([np.load(run.path("s06_embed", "sites_mean.npy"))[sok], np.load(run.path("s06_embed", "panel_mean.npy"))[pok]])
        Z = np.concatenate([np.load(run.path("s06_embed", "sites_center.npy"))[sok], np.load(run.path("s06_embed", "panel_center.npy"))[pok]])
    else:
        log(f"embedding {sok.sum()} sites + {pok.sum()} benchmark variants with {N_PROC} processes")
        M, Z = embed_all(chroms, pos38)
    out = {}
    for name, ok, a, b in (("sites", sok, M[:ns], Z[:ns]), ("panel", pok, M[ns:], Z[ns:])):
        mean = np.full((len(ok), C.D_MODEL), np.nan, np.float32)
        cen = mean.copy()
        mean[ok], cen[ok] = a, b
        out[name] = (mean, cen, ok)
        rep.true(f"{name}: every lifted row finite", bool(np.isfinite(mean[ok]).all() and np.isfinite(cen[ok]).all()))
        rep.ge(f"{name}: min row std (not degenerate)", round(float(mean[ok].std(1).min()), 5), 1e-3)
        np.save(run.path("s06_embed", f"{name}_mean.npy"), mean)
        np.save(run.path("s06_embed", f"{name}_center.npy"), cen)
        rep.info[f"shape {name}_mean"] = list(mean.shape)
        rep.eq(f"{name}: rows == rows of the set", len(mean), len(ok))
        rep.true(f"{name}: non-lifted rows are NaN", bool(np.isnan(mean[~ok]).all()))
    with open(stamp, "w") as f:
        f.write(target)

    # fresh embedder in this process must reproduce random rows (both sets)
    _init()
    rng = np.random.default_rng(0)
    for name, names_, lift in (("sites", snames, SL), ("panel", pnames, PL)):
        mean, _, ok = out[name]
        idx = rng.choice(np.where(ok)[0], 100, replace=False)
        e = _W["emb"].embed(windows(_W["ref"], names_[idx], lift["pos38"][idx]))["mean"]
        cos = (e * mean[idx]).sum(1) / np.linalg.norm(e, axis=1) / np.linalg.norm(mean[idx], axis=1)
        rep.ge(f"{name}: re-embedding cos (min over 100 rows)", round(float(cos.min()), 6), 0.9999)
        sh = rng.permutation(idx)
        cs = (e * mean[sh]).sum(1) / np.linalg.norm(e, axis=1) / np.linalg.norm(mean[sh], axis=1)
        rep.le(f"{name}: CONTROL frac shuffled rows with cos >= 0.9999", round(float((cs >= 0.9999).mean()), 4), 0.05,
               control=True)
    finish(run, "s06_embed", rep)

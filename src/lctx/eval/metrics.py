"""Benchmark evaluation: tie-aware AUROC, locus-clustered paired bootstrap, single-read lock."""
import hashlib
import json
import os
import time

import numpy as np
from scipy.stats import rankdata


def auroc(scores, labels):
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels).astype(bool)
    if np.isnan(s).any():
        raise ValueError("NaN scores")
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def locus_clusters(chroms, pos, bp):
    """Cluster id = (chrom, pos // bp)."""
    keys = [f"{c}:{p // bp}" for c, p in zip(chroms, np.asarray(pos, np.int64))]
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def paired_bootstrap(y, a, b, clusters, reps, seed=0):
    """Delta = AUROC(b) - AUROC(a); clusters resampled with replacement."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    members = [np.where(clusters == c)[0] for c in uc]
    d0 = auroc(b, y) - auroc(a, y)
    ds = []
    for _ in range(reps):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uc), len(uc))])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        ds.append(auroc(b[idx], yy) - auroc(a[idx], yy))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return {"delta": d0, "lo": float(lo), "hi": float(hi), "p_two_sided": float(min(1.0, p)),
            "reps": int(len(ds))}


class SingleReadLock:
    """The benchmark may be read once per frozen plan. `open(plan)` hashes the plan, refuses a
    second read of the same hash and logs every read."""

    def __init__(self, path):
        self.path = path
        self.log = json.load(open(path)) if os.path.exists(path) else {"reads": []}

    @staticmethod
    def plan_hash(plan):
        return hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def open(self, plan):
        h = self.plan_hash(plan)
        if any(r["hash"] == h for r in self.log["reads"]):
            raise RuntimeError(f"benchmark already read for plan {h}")
        self.log["reads"].append({"hash": h, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": plan})
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.log, f, indent=1, default=str)
        return h

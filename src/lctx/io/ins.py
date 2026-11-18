"""INSIGHT database (.ins text) -> per-site evolutionary terms, with the semantics of
LINSIGHT's own converter (`third_party/linsight_cpp/L-INSIGHT-prep.cpp`):

    samples <n>
    block <chr:start-end> theta <t> lambda <l>      theta*a_n and lambda clamped to >= 1e-5
    site  <chr:pos> M <pZmaj>                      alleleFreq 0
    site  <chr:pos> L <pZmaj> <pZmin> ...          alleleFreq 0.05
    site  <chr:pos> H <pZmaj> <pZmin> ...          alleleFreq 0.5

Each site becomes float32 (alleleFreq, pZeqXmaj, pZeqXmin, thetaTimesA, lambda), exactly the
5 floats LINSIGHT-prep writes after the uint64 position. Positions are kept as written in the
file (`pos_ins`); LINSIGHT matches them to 0-based BED intervals.

Sampling is a deterministic hash of (chromosome, position) below the rate, so a smaller
rate's sample is a subset of a larger rate's sample, and any process gets the same answer.
"""
import os

import numpy as np

from ..config import AUTOSOMES, MIN_THETA_LAMBDA

CHROM_INDEX = {c: i for i, c in enumerate(AUTOSOMES)}
AF = {b"M": 0.0, b"L": 0.05, b"H": 0.5}
_M64 = (1 << 64) - 1


def wattersons_a(n_samples):
    """a_n = sum_{k=1}^{n-1} 1/k, accumulated in the same order as computeWattersons_a()."""
    a = 0.0
    for k in range(1, n_samples):
        a += 1.0 / k
    return a


def _splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def site_uniform(chrom_idx, pos):
    """Deterministic U[0,1) per (autosome index, position)."""
    return _splitmix64((int(chrom_idx) << 40) | int(pos)) / 2.0 ** 64


def site_uniform_np(chrom_idx, pos):
    x = (np.asarray(chrom_idx, np.uint64) << np.uint64(40)) | np.asarray(pos, np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x.astype(np.float64) / 2.0 ** 64


def read_n_samples(path):
    with open(path, "rb") as f:
        items = f.readline().split()
    if items[0] != b"samples":
        raise ValueError("first line must be 'samples <n>'")
    return int(items[1])


def chunk_offsets(path, n_chunks):
    size = os.path.getsize(path)
    return [(size * i) // n_chunks for i in range(n_chunks)] + [size]


def scan_chunk(path, start, end, rate, a_n):
    """Parse the blocks whose 'block' line starts in [start, end).

    Returns dict: chrom (int8), pos (int64), evol (float32 n x 5), counts {chrom: n sites},
    type_counts {chrom: {M,L,H}} over ALL sites of those blocks (sampled or not).
    """
    chrom, pos, evol = [], [], []
    counts, tcounts = {}, {}
    theta_a = lam = None
    cur = None
    with open(path, "rb") as f:
        f.seek(start)
        off = start
        if start > 0:
            off += len(f.readline())          # finish the partial line
        started = False
        for line in f:
            lo = off
            off += len(line)
            if line.startswith(b"block"):
                if lo >= end:
                    break
                started = True
                it = line.split()
                theta = float(it[3])
                lam = float(it[5])
                theta_a = np.float32(max(theta * a_n, MIN_THETA_LAMBDA))
                lam = np.float32(max(lam, MIN_THETA_LAMBDA))
                cur = None
            elif started and line.startswith(b"site"):
                it = line.split()
                c, p = it[1].rsplit(b":", 1)
                if c != cur:
                    cur = c
                    cs = c.decode()
                    ci = CHROM_INDEX.get(cs, -1)
                    cnt = counts.setdefault(cs, 0)
                    tc = tcounts.setdefault(cs, {"M": 0, "L": 0, "H": 0})
                p = int(p)
                counts[cs] += 1
                t = it[2]
                tc[t.decode()] += 1
                if ci < 0 or site_uniform(ci, p) >= rate:
                    continue
                if t == b"M":
                    pmaj, pmin = float(it[3]), 0.0
                elif t in (b"L", b"H"):
                    pmaj, pmin = float(it[3]), float(it[4])
                else:
                    raise ValueError(f"unknown site type {t!r}")
                chrom.append(ci)
                pos.append(p)
                evol.append((AF[t], pmaj, pmin, theta_a, lam))
    return {"chrom": np.array(chrom, np.int8), "pos": np.array(pos, np.int64),
            "evol": np.array(evol, np.float32).reshape(-1, 5), "counts": counts, "type_counts": tcounts}


def merge_chunks(parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("chrom", "pos", "evol")}
    counts, tcounts = {}, {}
    for p in parts:
        for c, n in p["counts"].items():
            counts[c] = counts.get(c, 0) + n
        for c, d in p["type_counts"].items():
            t = tcounts.setdefault(c, {"M": 0, "L": 0, "H": 0})
            for k, v in d.items():
                t[k] += v
    order = np.lexsort((out["pos"], out["chrom"]))
    out = {k: v[order] for k, v in out.items()}
    out["counts"], out["type_counts"] = counts, tcounts
    return out

"""The 48 LINSIGHT features from genomic_feature.bed (sorted by chromosome string, then start).

A position p on chromosome c gets the features of the interval with start <= p < end
(the matching rule of L-INSIGHT-fit.cpp); positions in no interval are 'not found'.

Two independent implementations:
  * `lookup_sorted`  - streaming merge of sorted queries with one chromosome's byte range
                       (what the pipeline uses; same semantics as the C++ loop)
  * `RandomAccessBed.lookup` - byte bisection per query (used only to cross-check)
"""
import os

import numpy as np

from ..config import N_FEATURES


def _parse(line):
    it = line.rstrip(b"\n").split(b"\t")
    return it[0].decode(), int(it[1]), int(it[2]), it


class RandomAccessBed:
    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        self.fh = open(path, "rb")

    def close(self):
        self.fh.close()

    def _line_at(self, off):
        """First complete data line starting at or after byte offset `off`."""
        self.fh.seek(off)
        if off > 0:
            self.fh.readline()
        while True:
            start = self.fh.tell()
            raw = self.fh.readline()
            if not raw:
                return start, None
            if not raw.startswith(b"#"):
                return start, raw

    def first_offset(self, key):
        """Byte offset of the first line whose (chrom, end) > key=(chrom, pos)."""
        lo, hi = 0, self.size
        while hi - lo > 1:
            mid = (lo + hi) // 2
            _, raw = self._line_at(mid)
            if raw is None:
                hi = mid
                continue
            c, _, e, _ = _parse(raw)
            if (c, e) <= key:
                lo = mid
            else:
                hi = mid
        start, _ = self._line_at(lo)
        # step forward until the line's (chrom, end) > key
        self.fh.seek(start)
        while True:
            off = self.fh.tell()
            raw = self.fh.readline()
            if not raw:
                return off
            if raw.startswith(b"#"):
                continue
            c, _, e, _ = _parse(raw)
            if (c, e) > key:
                return off

    def lookup(self, chrom, pos):
        off = self.first_offset((chrom, pos))
        self.fh.seek(off)
        raw = self.fh.readline()
        if not raw:
            return None
        c, s, e, it = _parse(raw)
        if c == chrom and s <= pos < e:
            return np.array([float(x) for x in it[3:]], np.float32)
        return None

    def chrom_range(self, chrom):
        """[first, last) byte offsets of the chromosome's lines."""
        a = self.first_offset((chrom, -1))
        b = self.first_offset((chrom, 1 << 62))
        return a, b


def lookup_sorted(path, chrom, positions, n_features=N_FEATURES):
    """Features for ascending `positions` on one chromosome, by one streaming pass over the
    chromosome's lines. Returns (F float32 n x 48 with NaN where not found, found bool)."""
    positions = np.asarray(positions, np.int64)
    if len(positions) and np.any(np.diff(positions) < 0):
        raise ValueError("positions must be ascending")
    F = np.full((len(positions), n_features), np.nan, np.float32)
    found = np.zeros(len(positions), bool)
    if not len(positions):
        return F, found
    rab = RandomAccessBed(path)
    a, b = rab.chrom_range(chrom)
    rab.fh.seek(a)
    i, n = 0, len(positions)
    off = a
    prev_end = -1
    for raw in rab.fh:
        if off >= b:
            break
        off += len(raw)
        if raw.startswith(b"#"):
            continue
        tab1 = raw.index(b"\t")
        tab2 = raw.index(b"\t", tab1 + 1)
        tab3 = raw.index(b"\t", tab2 + 1)
        s, e = int(raw[tab1 + 1:tab2]), int(raw[tab2 + 1:tab3])
        if s < prev_end:
            raise ValueError(f"BED not sorted / overlapping at {chrom}:{s}")
        prev_end = e
        while i < n and positions[i] < s:
            i += 1                                   # in a gap: not found
        j = i
        while j < n and positions[j] < e:
            j += 1
        if j > i:
            v = np.array(raw[tab3 + 1:].split(b"\t"), np.float32)
            if len(v) != n_features:
                raise ValueError(f"{len(v)} features at {chrom}:{s}")
            F[i:j] = v
            found[i:j] = True
            i = j
        if i >= n:
            break
    rab.close()
    return F, found


def lookup_many(path, chroms, positions, n_proc=22):
    """Any order of (chrom, pos): grouped by chromosome, one streaming pass each, in parallel."""
    import multiprocessing as mp
    chroms = np.asarray(chroms).astype(str)
    positions = np.asarray(positions, np.int64)
    F = np.full((len(positions), N_FEATURES), np.nan, np.float32)
    found = np.zeros(len(positions), bool)
    jobs = []
    for c in np.unique(chroms):
        idx = np.where(chroms == c)[0]
        idx = idx[np.argsort(positions[idx], kind="stable")]
        jobs.append((c, idx))
    with mp.get_context("fork").Pool(min(n_proc, len(jobs))) as pool:
        res = pool.starmap(lookup_sorted, [(path, c, positions[idx]) for c, idx in jobs])
    for (c, idx), (f, ok) in zip(jobs, res):
        F[idx], found[idx] = f, ok
    return F, found

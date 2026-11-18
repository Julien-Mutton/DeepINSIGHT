"""hg19 -> hg38 with the UCSC chain (pyliftover). Only unique, same-chromosome,
plus-strand mappings are accepted; everything else is reported with a status."""
import numpy as np

STATUS = ("ok", "unmapped", "multiple", "other_chrom", "minus_strand")


class Lifter:
    def __init__(self, chain_path):
        from pyliftover import LiftOver
        self.lo = LiftOver(chain_path)

    def lift(self, chrom, pos):
        """(pos38 or -1, status) for a 0-based hg19 position."""
        hits = self.lo.convert_coordinate(chrom, int(pos))
        if not hits:
            return -1, "unmapped"
        if len(hits) > 1:
            return -1, "multiple"
        c, p, strand, _ = hits[0]
        if c != chrom:
            return -1, "other_chrom"
        if strand != "+":
            return -1, "minus_strand"
        return int(p), "ok"

    def lift_many(self, chroms, positions):
        pos38 = np.full(len(positions), -1, np.int64)
        status = np.empty(len(positions), dtype="<U16")
        for i, (c, p) in enumerate(zip(chroms, positions)):
            pos38[i], status[i] = self.lift(c, p)
        return pos38, status


def _lift_chunk(args):
    chain, chroms, positions = args
    return Lifter(chain).lift_many(chroms, positions)


def lift_parallel(chain, chroms, positions, n_proc=32):
    import multiprocessing as mp
    chroms = np.asarray(chroms).astype(str)
    positions = np.asarray(positions, np.int64)
    parts = np.array_split(np.arange(len(positions)), n_proc)
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(_lift_chunk, [(chain, chroms[p], positions[p]) for p in parts])
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def sequence_agreement(ref19, ref38, chroms, pos19, pos38, flank=10, window=None):
    """Per row: hg19 base == hg38 base; (2*flank+1)-mer identical; optional window identity frac."""
    base = np.zeros(len(pos19), bool)
    kmer = np.zeros(len(pos19), bool)
    win = np.full(len(pos19), np.nan)
    for i, (c, a, b) in enumerate(zip(chroms, pos19, pos38)):
        if b < 0:
            continue
        s19, s38 = ref19.window(c, a, 2 * flank + 1), ref38.window(c, b, 2 * flank + 1)
        base[i] = s19[flank] == s38[flank] and s19[flank] != "N"
        kmer[i] = s19 == s38 and "N" not in s19
        if window:
            w19, w38 = ref19.window(c, a, window), ref38.window(c, b, window)
            win[i] = np.mean(np.frombuffer(w19.encode(), np.uint8) == np.frombuffer(w38.encode(), np.uint8))
    return base, kmer, win

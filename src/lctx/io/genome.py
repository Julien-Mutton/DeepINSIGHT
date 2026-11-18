"""Reference genomes (0-based coordinates) and bigWig tracks."""
import numpy as np


class Reference:
    """FASTA access. All positions are 0-based; windows outside the chromosome are N-padded."""

    def __init__(self, path):
        import pyfaidx
        self.fa = pyfaidx.Fasta(path, as_raw=True, sequence_always_upper=True, rebuild=False)
        self.lengths = {k: len(v) for k, v in self.fa.items()}

    def has(self, chrom):
        return chrom in self.lengths

    def base(self, chrom, pos):
        return self.fa[chrom][int(pos):int(pos) + 1]

    def seq(self, chrom, start, end):
        """[start, end) with N padding outside the chromosome."""
        L = self.lengths[chrom]
        a, b = max(0, start), min(L, end)
        s = self.fa[chrom][a:b] if b > a else ""
        return "N" * (a - start) + s + "N" * (end - b)

    def window(self, chrom, center, length):
        """`length` bases with `center` at index length // 2."""
        start = int(center) - length // 2
        return self.seq(chrom, start, start + length)


class BigWig:
    def __init__(self, path):
        import pyBigWig
        self.bw = pyBigWig.open(path)
        self.chroms = self.bw.chroms()

    def values_at(self, chroms, positions):
        """Value covering each 0-based position (NaN where undefined)."""
        out = np.full(len(positions), np.nan)
        for i, (c, p) in enumerate(zip(chroms, positions)):
            if c in self.chroms and 0 <= p < self.chroms[c]:
                v = self.bw.values(c, int(p), int(p) + 1)[0]
                out[i] = np.nan if v is None else v
        return out

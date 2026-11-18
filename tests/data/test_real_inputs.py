"""Block tests on the REAL original files (pytest -m data)."""
import os
import subprocess

import numpy as np
import pytest

from lctx.config import INPUTS, AUTOSOMES, ROOT, CONTEXT
from lctx.io import ins
from lctx.io.features import RandomAccessBed, lookup_sorted
from lctx.io.genome import Reference

pytestmark = pytest.mark.data


def test_ins_parser_equals_linsight_prep_on_real_excerpt(tmp_path):
    """First ~30 MB of the real .ins (cut at a block boundary) through LINSIGHT-prep and ours."""
    ex = tmp_path / "ex.ins"
    with open(INPUTS.ins, "rb") as f, open(ex, "wb") as o:
        data = f.read(30_000_000)
        cut = data.rfind(b"\nblock")
        o.write(data[:cut + 1])
    exe = os.path.join(ROOT, "third_party", "linsight_cpp", "LINSIGHT-prep")
    od = tmp_path / "bin"
    od.mkdir()
    subprocess.run([exe, "-i", str(ex), "-o", str(od)], check=True, capture_output=True)
    got = ins.scan_chunk(str(ex), 0, os.path.getsize(ex), 1.0, ins.wattersons_a(ins.read_n_samples(str(ex))))
    ref = np.fromfile(od / "chr1.fit", np.dtype([("pos", "<u8"), ("e", "<f4", (5,))]))
    np.testing.assert_array_equal(got["pos"], ref["pos"].astype(np.int64))
    np.testing.assert_array_equal(got["evol"], ref["e"])


def test_feature_lookup_streaming_equals_bisection_on_real_bed():
    rng = np.random.default_rng(3)
    rab = RandomAccessBed(INPUTS.feature_bed)
    for c in ("chr1", "chr21"):
        pos = np.sort(rng.integers(1_000_000, 40_000_000, 300))
        F, found = lookup_sorted(INPUTS.feature_bed, c, pos)
        for p, f, ok in zip(pos, F, found):
            r = rab.lookup(c, int(p))
            assert (r is None) == (not ok)
            if ok:
                np.testing.assert_array_equal(f, r)


def test_liftover_on_random_genome_positions():
    from lctx.coords.liftover import Lifter, sequence_agreement
    rng = np.random.default_rng(4)
    ref19, ref38 = Reference(INPUTS.hg19), Reference(INPUTS.hg38)
    chroms = rng.choice(AUTOSOMES, 1000)
    pos = np.array([rng.integers(1_000_000, ref19.lengths[c] - 1_000_000) for c in chroms])
    keep = np.array([ref19.base(c, p) != "N" for c, p in zip(chroms, pos)])
    chroms, pos = chroms[keep], pos[keep]
    p38, st = Lifter(INPUTS.chain).lift_many(chroms, pos)
    ok = st == "ok"
    assert ok.mean() > 0.95
    b, k, _ = sequence_agreement(ref19, ref38, chroms[ok], pos[ok], p38[ok])
    assert b.mean() > 0.99 and k.mean() > 0.95


def test_hyena_predicts_real_dna_better_than_random():
    from lctx.embed.hyena import HyenaEmbedder
    ref38 = Reference(INPUTS.hg38)
    rng = np.random.default_rng(5)
    seqs = [ref38.window("chr2", int(p), CONTEXT) for p in rng.integers(50_000_000, 150_000_000, 16)]
    seqs = [s for s in seqs if "N" not in s]
    e = HyenaEmbedder(INPUTS.hyena_dir)
    rand = ["".join(rng.choice(list("ACGT"), CONTEXT)) for _ in range(4)]
    assert e.next_base_accuracy(seqs) > e.next_base_accuracy(rand) + 0.08

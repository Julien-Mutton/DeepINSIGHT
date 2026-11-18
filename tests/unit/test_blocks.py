"""Unit tests (synthetic data) for every block except the LINSIGHT model."""
import os
import subprocess

import numpy as np
import pytest

from lctx.config import ROOT
from lctx.eval import metrics
from lctx.io import features as FT
from lctx.io import ins
from lctx import sites

CPP_DIR = os.path.join(ROOT, "third_party", "linsight_cpp")


# ---------------------------------------------------------------- .ins parser
def _write_ins(path, rng, n_blocks=40, sites_per=200, chroms=("chr1", "chr2", "chrX")):
    lines = ["samples\t108"]
    for c in chroms:
        p = 10_000
        for b in range(n_blocks):
            th, lam = rng.uniform(0, 3e-3), rng.choice([rng.uniform(0, 1e-2), 1e-7])
            lines.append(f"block\t{c}:{p}-{p + 5000}\ttheta\t{th:.9e}\tlambda\t{lam:.9e}")
            for s in range(sites_per):
                p += int(rng.integers(1, 5))
                t = rng.choice(["M", "L", "H"], p=[0.9, 0.07, 0.03])
                if t == "M":
                    lines.append(f"site\t{c}:{p}\tM\t{rng.uniform(0.9, 1):.10e}")
                else:
                    lines.append(f"site\t{c}:{p}\t{t}\t{rng.uniform(0.9, 1):.10e}\t{rng.uniform(0, .05):.10e}\t107\t1")
    path.write_text("\n".join(lines) + "\n")


def test_ins_parser_equals_original_linsight_prep(rng, tmp_path):
    """Our parser at rate 1.0 must reproduce LINSIGHT-prep's binary records bit for bit."""
    f = tmp_path / "x.ins"
    _write_ins(f, rng)
    exe = os.path.join(CPP_DIR, "LINSIGHT-prep")
    if not os.path.exists(exe):
        subprocess.run(["make", "-s", "-C", CPP_DIR], check=True)
    od = tmp_path / "bin"
    od.mkdir()
    subprocess.run([exe, "-i", str(f), "-o", str(od)], check=True, capture_output=True)
    a_n = ins.wattersons_a(108)
    offs = ins.chunk_offsets(str(f), 7)                        # several chunks: tests the boundaries
    got = ins.merge_chunks([ins.scan_chunk(str(f), offs[i], offs[i + 1], 1.0, a_n) for i in range(7)])
    rec = np.dtype([("pos", "<u8"), ("e", "<f4", (5,))])
    for ci, c in enumerate(("chr1", "chr2")):
        ref = np.fromfile(od / f"{c}.fit", rec)
        m = got["chrom"] == ci
        np.testing.assert_array_equal(got["pos"][m], ref["pos"].astype(np.int64))
        np.testing.assert_array_equal(got["evol"][m], ref["e"])
    assert got["counts"] == {"chr1": 8000, "chr2": 8000, "chrX": 8000}
    assert not (got["chrom"] < 0).any()                          # chrX counted, not sampled


def test_ins_sampling_is_deterministic_nested_and_at_rate(rng, tmp_path):
    f = tmp_path / "x.ins"
    _write_ins(f, rng, n_blocks=100, sites_per=300, chroms=("chr1",))
    a_n = ins.wattersons_a(108)
    size = os.path.getsize(f)
    s1 = ins.scan_chunk(str(f), 0, size, 0.01, a_n)
    s2 = ins.scan_chunk(str(f), 0, size, 0.1, a_n)
    assert set(s1["pos"]) <= set(s2["pos"])
    assert abs(len(s2["pos"]) / 30000 - 0.1) < 0.01
    u = ins.site_uniform_np(s2["chrom"], s2["pos"])
    assert np.all(u < 0.1)
    assert all(abs(ins.site_uniform(c, p) - v) < 1e-15 for c, p, v in zip(s2["chrom"][:50], s2["pos"][:50], u[:50]))


# ---------------------------------------------------------------- features
def _write_bed(path, rng, chroms=("chr1", "chr10", "chr2"), n=400):
    rows = []
    for c in sorted(chroms):
        s = 0
        for _ in range(n):
            s += int(rng.integers(0, 3))                      # occasional gaps
            e = s + int(rng.integers(1, 50))
            rows.append((c, s, e, (rng.random(48) < 0.3).astype(int)))
            s = e
    with open(path, "w") as fh:
        fh.write("#chr\tstart\tend" + "\t1" * 48 + "\n")
        for c, s, e, v in rows:
            fh.write(f"{c}\t{s}\t{e}\t" + "\t".join(map(str, v)) + "\n")
    return rows


def _brute(rows, c, p):
    for cc, s, e, v in rows:
        if cc == c and s <= p < e:
            return v
    return None


def test_feature_lookup_three_implementations_agree(rng, tmp_path):
    f = tmp_path / "f.bed"
    rows = _write_bed(f, rng)
    rab = FT.RandomAccessBed(str(f))
    for c in ("chr1", "chr10", "chr2"):
        hi = max(e for cc, s, e, v in rows if cc == c)
        pos = np.sort(rng.integers(0, hi + 20, 300))
        F, found = FT.lookup_sorted(str(f), c, pos)
        for p, fr, ok in zip(pos, F, found):
            b = _brute(rows, c, p)
            r = rab.lookup(c, int(p))
            assert (b is None) == (not ok) == (r is None)
            if ok:
                np.testing.assert_array_equal(fr, b)
                np.testing.assert_array_equal(r, b)


def test_feature_lookup_many_restores_input_order(rng, tmp_path):
    f = tmp_path / "f.bed"
    rows = _write_bed(f, rng)
    chroms = rng.choice(["chr1", "chr10", "chr2", "chr3"], 500)
    pos = rng.integers(0, 4000, 500)
    F, found = FT.lookup_many(str(f), chroms, pos, n_proc=3)
    for c, p, fr, ok in zip(chroms, pos, F, found):
        b = _brute(rows, c, p)
        assert (b is None) == (not ok)
        if ok:
            np.testing.assert_array_equal(fr, b)


def test_feature_lookup_rejects_unsorted_queries(tmp_path, rng):
    f = tmp_path / "f.bed"
    _write_bed(f, rng)
    with pytest.raises(ValueError):
        FT.lookup_sorted(str(f), "chr1", np.array([5, 3]))


# ---------------------------------------------------------------- sites
def test_min_distance_and_exclusion(rng):
    ref_c = np.array(["chr1"] * 3 + ["chr2"])
    ref_p = np.array([100, 5000, 9000, 50])
    d = sites.min_distance(["chr1", "chr1", "chr2", "chr3"], [1100, 7000, 51, 7], ref_c, ref_p)
    np.testing.assert_array_equal(d, [1000, 2000, 1, np.inf])


def test_block_split_properties(rng):
    chroms = np.repeat([f"chr{i}" for i in range(1, 23)], 3000)
    pos = np.concatenate([np.sort(rng.integers(0, 80_000_000, 3000)) for _ in range(22)])
    role, blk = sites.block_split(chroms, pos, 1_000_000, 0.1, seed=0)
    for c in np.unique(chroms):
        r = role[chroms == c]
        assert (r == "train").any() and (r == "select").any() and (r == "eval").any()
    for b in np.unique(blk):                                     # whole blocks
        assert len(set(role[blk == b])) == 1
    assert abs((role != "train").mean() - 0.1) < 0.03
    assert (role == "select").any() and (role == "eval").any()
    role2, _ = sites.block_split(chroms, pos, 1_000_000, 0.1, seed=0)
    r3, _ = sites.block_split(chroms, pos, 1_000_000, 0.5, seed=0, select_share=0.1)
    assert abs((r3 != "train").mean() - 0.5) < 0.05
    assert abs((r3 == "select").sum() / (r3 != "train").sum() - 0.1) < 0.05
    np.testing.assert_array_equal(role, role2)


# ---------------------------------------------------------------- metrics
def test_auroc_equals_sklearn_with_ties(rng):
    from sklearn.metrics import roc_auc_score
    y = rng.random(2000) < 0.1
    s = np.round(rng.normal(size=2000) + y, 1)
    assert metrics.auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_paired_bootstrap_identical_scores_give_zero(rng):
    y = rng.random(3000) < 0.2
    a = rng.normal(size=3000) + y
    cl = metrics.locus_clusters(["chr1"] * 3000, np.arange(3000) * 1000, 100_000)
    r = metrics.paired_bootstrap(y, a, a, cl, reps=200)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0
    r2 = metrics.paired_bootstrap(y, a, a + 3 * y, cl, reps=200)
    assert r2["lo"] > 0


def test_single_read_lock(tmp_path):
    L = metrics.SingleReadLock(str(tmp_path / "lock.json"))
    h = L.open({"arms": ["a"]})
    with pytest.raises(RuntimeError):
        metrics.SingleReadLock(str(tmp_path / "lock.json")).open({"arms": ["a"]})
    assert metrics.SingleReadLock(str(tmp_path / "lock.json")).open({"arms": ["b"]}) != h


# ---------------------------------------------------------------- SAE + tokenizer
def test_topk_sae_learns_and_respects_k(rng):
    from lctx.sae import topk
    D = rng.normal(size=(16, 24))
    codes = np.zeros((6000, 16))
    for i in range(6000):
        j = rng.choice(16, 2, replace=False)
        codes[i, j] = rng.uniform(1, 3, 2)
    X = (codes @ D).astype(np.float32)
    m = topk.fit(X, topk.SAEConfig(d_sae=16, k=2, epochs=30, batch=256, seed=0))
    met = topk.metrics(m, X)
    assert met["l0"] <= 2 + 1e-9 and met["fvu"] < 0.2


def test_tokenizer_matches_official_ids():
    """The official CharacterTokenizer cannot be instantiated with the installed transformers,
    so check our ids against the mapping written in its source."""
    from lctx.embed.hyena import tokenize
    src = open(os.path.join(ROOT, "third_party", "hyena_dna", "standalone_hyenadna.py")).read()
    assert '"[UNK]": 6' in src and "**{ch: i + 7 for i, ch in enumerate(characters)}" in src
    ours = tokenize(["ACGTNacgtnR"])[0].tolist()
    assert ours == [7, 8, 9, 10, 11, 7, 8, 9, 10, 11, 6]      # characters = A,C,G,T,N; R -> [UNK]

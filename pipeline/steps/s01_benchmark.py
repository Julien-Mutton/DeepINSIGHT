"""s01 benchmark: HGMD regulatory positives + 3 GWAVA negative tiers from the original pickles.

Checks: counts as published, unique keys, valid reference one-hot, the one-hot base equals
hg19 at `start` (0-based) and NOT at start+1 / start-1 (controls), and the published
LINSIGHT.bw at `start` reproduces the paper's AUROCs (so coordinates and tiers are right).
"""
import numpy as np

from common import C, CheckReport, finish, log
from lctx.eval.metrics import auroc
from lctx.io.benchmark import load_panel, tier_mask
from lctx.io.genome import Reference, BigWig


def run(run):
    rep = CheckReport("s01_benchmark")
    pan = load_panel(C.INPUTS.bench_dir)
    for src, n in C.EXPECTED_COUNTS.items():
        rep.eq(f"count[{src}]", int((pan.source == src).sum()), n)
    rep.true("keys_unique", pan.key.is_unique)
    rep.true("end_equals_start_plus_1", bool((pan.end == pan.pos19 + 1).all()))
    rep.ge("ref_onehot_valid_frac", float((pan.ref != "N").mean()), 0.99)

    ref = Reference(C.INPUTS.hg19)
    def match(off):
        ok = [ref.has(c) and ref.base(c, p + off) == r for c, p, r in zip(pan.chrom, pan.pos19, pan.ref)]
        return float(np.mean(ok))
    rep.ge("ref_equals_hg19[start] (0-based)", match(0), 0.99)
    rep.le("CONTROL ref_equals_hg19[start+1]", match(1), 0.5, control=True)
    rep.le("CONTROL ref_equals_hg19[start-1]", match(-1), 0.5, control=True)

    # the same coordinate labelled both positive and negative cannot be scored meaningfully
    pos_coords = set(zip(pan.chrom[pan.label == 1], pan.pos19[pan.label == 1]))
    conflict = np.array([(c, p) in pos_coords and l == 0 for c, p, l in zip(pan.chrom, pan.pos19, pan.label)])
    pan["label_conflict"] = conflict
    rep.le("label_conflict_frac", float(conflict.mean()), 0.01)
    rep.info["n_label_conflict"] = int(conflict.sum())
    rep.info["chromosomes"] = sorted(set(pan.chrom))

    bw = BigWig(C.INPUTS.linsight_bw).values_at(pan.chrom.tolist(), pan.pos19.tolist())
    pan["linsight_published"] = bw
    rep.ge("published_linsight_defined_frac", float(np.isfinite(bw).mean()), 0.97)
    for tier, paper in C.TIERS.items():
        m = tier_mask(pan, tier) & np.isfinite(bw)
        a = auroc(bw[m], pan.label[m])
        rep.close(f"published_linsight_auroc[{tier}] = paper", round(a, 4), paper, 0.01)
    pan.to_csv(run.path("s01_benchmark", "panel.csv.gz"), index=False)
    log(f"panel: {len(pan)} rows")
    finish(run, "s01_benchmark", rep)

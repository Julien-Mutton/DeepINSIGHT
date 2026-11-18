"""s03 liftover: hg38 position for every training site and benchmark variant.

Sites and benchmark variants go through the same `lift_parallel` and the same sequence checks:
base and 21-mer identical in hg19 and hg38, 1001-bp window identity; a pos38+1 control must fail.
For benchmark variants the reference allele recorded in the benchmark must equal hg38 too.
"""
import numpy as np
import pandas as pd

from common import C, CheckReport, finish, load_npz, log, save_npz
from lctx.coords.liftover import lift_parallel, sequence_agreement
from lctx.io.genome import Reference


def _mark_missing_chrom(ref38, chroms, status):
    """hg38.ml.fa (HyenaDNA's assembly) has chr1-22 and chrX only."""
    status = status.copy()
    miss = np.array([not ref38.has(c) for c in chroms]) & (status == "ok")
    status[miss] = "no_hg38_chrom"
    return status


def _check_set(rep, name, ref19, ref38, chroms, pos19, pos38, status, rng):
    ok = status == "ok"
    rep.ge(f"{name}: lift_ok_frac", round(float(ok.mean()), 5), 0.98)
    rep.info[f"{name}: status_counts"] = {s: int((status == s).sum()) for s in np.unique(status)}
    idx = np.where(ok)[0]
    sub = rng.choice(idx, min(20000, len(idx)), replace=False)
    b, k, _ = sequence_agreement(ref19, ref38, chroms[sub], pos19[sub], pos38[sub])
    bc, kc, _ = sequence_agreement(ref19, ref38, chroms[sub], pos19[sub], pos38[sub] + 1)
    rep.ge(f"{name}: hg19 base == hg38 base", round(float(b.mean()), 5), 0.995)
    rep.ge(f"{name}: 21-mer identical", round(float(k.mean()), 5), 0.97)
    rep.le(f"{name}: CONTROL base at pos38+1", round(float(bc.mean()), 4), 0.5, control=True)
    rep.le(f"{name}: CONTROL 21-mer at pos38+1", round(float(kc.mean()), 4), 0.05, control=True)
    w = rng.choice(idx, min(2000, len(idx)), replace=False)
    _, _, win = sequence_agreement(ref19, ref38, chroms[w], pos19[w], pos38[w], window=C.CONTEXT)
    rep.ge(f"{name}: mean 1001-bp window identity", round(float(np.nanmean(win)), 4), 0.95)
    rep.ge(f"{name}: frac windows >= 0.99 identical", round(float(np.mean(win >= 0.99)), 4), 0.9)
    has38 = np.array([ref38.has(c) for c in chroms])
    rep.true(f"{name}: every ok row's chromosome is in hg38.fa", bool(has38[ok].all()))


def run(run):
    run.gate("s01_benchmark", "s02_sites")
    rep = CheckReport("s03_liftover")
    ref19, ref38 = Reference(C.INPUTS.hg19), Reference(C.INPUTS.hg38)
    rng = np.random.default_rng(0)

    S = load_npz(run.path("s02_sites", "sites.npz"))
    names = np.array(C.AUTOSOMES)[S["chrom"]]
    p38, st = lift_parallel(C.INPUTS.chain, names, S["pos19"])
    st = _mark_missing_chrom(ref38, names, st)
    _check_set(rep, "sites", ref19, ref38, names, S["pos19"], p38, st, rng)
    save_npz(run.path("s03_liftover", "sites_lift.npz"), pos38=p38, status=st.astype("U16"))
    log(f"sites lifted: {np.mean(st == 'ok'):.5f} ok")

    pan = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"))
    pc = pan.chrom.to_numpy().astype(str)
    pp38, pst = lift_parallel(C.INPUTS.chain, pc, pan.pos19.to_numpy())
    pst = _mark_missing_chrom(ref38, pc, pst)
    _check_set(rep, "benchmark", ref19, ref38, pc, pan.pos19.to_numpy(), pp38, pst, rng)
    ok = pst == "ok"
    same = np.array([ref38.base(c, p) == r for c, p, r in zip(pc[ok], pp38[ok], pan.ref[ok])])
    rep.ge("benchmark: recorded ref allele == hg38 base", round(float(same.mean()), 5), 0.99)
    save_npz(run.path("s03_liftover", "panel_lift.npz"), key=pan.key.to_numpy().astype("U40"),
             pos38=pp38, status=pst.astype("U16"))
    finish(run, "s03_liftover", rep)

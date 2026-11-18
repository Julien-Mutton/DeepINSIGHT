"""s02 sites: a random fraction `rate` of all INSIGHT sites, read from INSIGHT_database.ins.

Coordinates: `pos_ins` is the position as written in the .ins file. LINSIGHT-fit matches it to
0-based BED intervals, so features and LINSIGHT.bw are looked up at pos_ins. Whether pos_ins is
also the 0-based hg19 index of the site's base is decided from the data: INSIGHT removes CpG
sites, so at the right offset the sample contains no CpG and has equal C and G content.
pos19 = pos_ins + offset is used for sequence, liftover and distances.
Then: drop sites within EXCLUDE_BP of any benchmark variant; split the rest by 1 Mb blocks.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import C, CheckReport, finish, log, save_npz
from lctx import sites as ST
from lctx.io import ins
from lctx.io.genome import Reference

N_CHUNKS = 200


def _scan(args):
    return ins.scan_chunk(*args)


def coordinate_offset(rep, ref, names, pos_ins, rng):
    smp = rng.choice(len(pos_ins), 40000, replace=False)
    cpg, cg_asym = {}, {}
    for off in (-1, 0, 1):
        w = [ref.seq(c, int(p) + off - 1, int(p) + off + 2) for c, p in zip(names[smp], pos_ins[smp])]
        cpg[off] = float(np.mean([(x[1] == "C" and x[2] == "G") or (x[0] == "C" and x[1] == "G") for x in w]))
        cg_asym[off] = float(abs(np.mean([x[1] == "C" for x in w]) - np.mean([x[1] == "G" for x in w])))
    offset = min(cpg, key=cpg.get)
    rep.eq("ins position is the 0-based hg19 index (offset with no CpG)", offset, 0)
    rep.le("CpG fraction at chosen offset", round(cpg[offset], 5), 1e-3)
    rep.le("|C - G| at chosen offset (strand symmetry)", round(cg_asym[offset], 4), 0.01)
    for off in (-1, 1):
        rep.ge(f"CONTROL CpG fraction at offset {off:+d}", round(cpg[off], 5), 0.002, control=True)
    nfrac = float(np.mean([ref.base(c, p + offset) == "N" for c, p in zip(names[smp], pos_ins[smp])]))
    rep.le("N base fraction at sites", nfrac, 1e-3)
    rep.info.update({"cpg_by_offset": cpg, "c_g_asymmetry_by_offset": cg_asym, "ins_to_0based_offset": offset})
    log(f"CpG fraction by offset {cpg}; chosen offset {offset}")
    return offset


def window_cover_fraction(pan, lengths):
    """Genome fraction covered by the union of [p - EXCLUDE_BP, p + EXCLUDE_BP] windows."""
    cover = 0
    for c in C.AUTOSOMES:
        p = np.sort(pan.pos19[pan.chrom == c].to_numpy())
        if not len(p):
            continue
        lo, hi = p - C.EXCLUDE_BP, p + C.EXCLUDE_BP + 1
        end = np.maximum.accumulate(hi)
        start = np.r_[True, lo[1:] >= end[:-1]]
        first = np.where(start)[0]
        last = np.r_[first[1:] - 1, len(p) - 1]
        cover += int((end[last] - lo[first]).sum())
    return cover / sum(lengths[c] for c in C.AUTOSOMES)


def run(run):
    run.gate("s00_inputs", "s01_benchmark")
    rep = CheckReport("s02_sites")
    path = C.INPUTS.ins
    n_samples = ins.read_n_samples(path)
    a_n = ins.wattersons_a(n_samples)
    offs = ins.chunk_offsets(path, N_CHUNKS)
    log(f".ins: samples={n_samples}, a_n={a_n:.6f}; scanning {N_CHUNKS} chunks at rate {run.rate}")
    with mp.get_context("fork").Pool(100) as pool:
        parts = pool.map(_scan, [(path, offs[i], offs[i + 1], run.rate, a_n) for i in range(N_CHUNKS)])
    S = ins.merge_chunks(parts)
    chrom, pos_ins, evol = S["chrom"], S["pos"], S["evol"]
    names = np.array(C.AUTOSOMES)[chrom]
    log(f"sampled {len(pos_ins)} of {sum(S['counts'].values())} sites")
    rep.info.update({"n_samples": n_samples, "wattersons_a": a_n, "counts_all_sites": S["counts"],
                     "type_counts_all_sites": S["type_counts"], "n_sampled": int(len(pos_ins))})

    # --- sampling
    rel = {}
    for i, c in enumerate(C.AUTOSOMES):
        tot = S["counts"].get(c, 0)
        rel[c] = (int((chrom == i).sum()) / tot / run.rate - 1) if tot else float("nan")
    rep.eq("every_autosome_present", [c for c in C.AUTOSOMES if S["counts"].get(c, 0) == 0], [])
    rep.le("max per-chromosome relative error of the sample rate", round(max(abs(v) for v in rel.values()), 4), 0.1)
    rep.info["per_chrom_rel_error"] = rel
    tot_auto = sum(S["counts"][c] for c in C.AUTOSOMES)
    rep.close("overall sample fraction / rate", round(len(pos_ins) / tot_auto / run.rate, 4), 1.0, 0.02)
    pop = {t: sum(S["type_counts"][c][t] for c in C.AUTOSOMES) / tot_auto for t in "MLH"}
    smp = {"M": float((evol[:, 0] == 0).mean()), "L": float((evol[:, 0] == np.float32(0.05)).mean()),
           "H": float((evol[:, 0] == np.float32(0.5)).mean())}
    for t in "MLH":
        rep.close(f"site type fraction [{t}]: sample - all sites", round(smp[t] - pop[t], 5), 0.0,
                  round(max(0.002, 0.1 * pop[t]), 5))
    key = chrom.astype(np.int64) * (1 << 40) + pos_ins
    rep.true("positions unique", len(np.unique(key)) == len(key))
    rep.true("evol values in range", bool(np.isin(evol[:, 0], np.float32([0, 0.05, 0.5])).all()
                                         and ((evol[:, 1:3] >= 0) & (evol[:, 1:3] <= 1)).all()
                                         and (evol[:, 3:] >= np.float32(C.MIN_THETA_LAMBDA)).all()))

    # --- coordinate convention
    ref = Reference(C.INPUTS.hg19)
    rng = np.random.default_rng(0)
    offset = coordinate_offset(rep, ref, names, pos_ins, rng)
    pos19 = pos_ins + offset

    # --- exclusion around benchmark variants (hg19, 0-based)
    pan = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"), usecols=["chrom", "pos19"])
    d = ST.min_distance(names, pos19, pan.chrom.to_numpy(), pan.pos19.to_numpy())
    keep = d > C.EXCLUDE_BP
    cover = window_cover_fraction(pan, ref.lengths)
    rep.info.update({"excluded_frac": float(1 - keep.mean()), "window_cover_frac": cover})
    rep.close("excluded fraction / genome fraction covered by benchmark windows",
              round(float((1 - keep.mean()) / cover), 3), 1.0, 0.5)
    by_chrom = {c: np.sort(pan.pos19[pan.chrom == c].to_numpy()) for c in C.AUTOSOMES}
    chk = rng.choice(np.where(keep)[0], 3000, replace=False)
    brute = min(np.abs(by_chrom[names[i]] - pos19[i]).min() if len(by_chrom[names[i]]) else np.inf for i in chk)
    rep.ge("brute-force min distance of kept sites to any benchmark variant", float(brute), C.EXCLUDE_BP + 1)

    chrom, names, pos_ins, pos19, evol = chrom[keep], names[keep], pos_ins[keep], pos19[keep], evol[keep]
    role, block = ST.block_split(names, pos19, C.BLOCK_BP, C.HELDOUT_FRAC, seed=0, select_share=C.SELECT_SHARE)
    for r in ("train", "select", "eval"):
        miss = [c for c in C.AUTOSOMES if not ((names == c) & (role == r)).any()]
        rep.eq(f"autosomes missing from [{r}]", miss, [])
    rep.eq("blocks in both train and held-out", len(set(block[role == "train"]) & set(block[role != "train"])), 0)
    rep.close("held-out fraction", round(float((role != "train").mean()), 4), C.HELDOUT_FRAC, 0.05)
    rep.close("select share of held-out", round(float((role == "select").sum() / (role != "train").sum()), 4), C.SELECT_SHARE, 0.05)
    rep.info["n_by_role"] = {r: int((role == r).sum()) for r in ("train", "select", "eval")}
    save_npz(run.path("s02_sites", "sites.npz"), chrom=chrom, pos_ins=pos_ins, pos19=pos19,
             evol=evol, role=role.astype("U6"))
    finish(run, "s02_sites", rep)

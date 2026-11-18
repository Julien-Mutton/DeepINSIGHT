"""s08 design: every arm's model inputs, checked row by row against the ORIGINAL files.

For random training sites and benchmark variants, each row's source values are rebuilt with
independent code paths (byte-bisection BED lookup instead of the streaming join, a fresh
liftover, a fresh reference reader + fresh HyenaDNA, the composition function on a fresh hg19
read, re-loaded SAEs) and the row `arms.Data.inputs` gives the trainer must equal them.
Shuffled arms are checked against the source row they were permuted from. A swapped-row control
must fail. Train and benchmark inputs of an arm must share columns and standardisation.
"""
import numpy as np
import pyfaidx
import torch

from arms import ARMS, FSAE, SAE_NAMES, SEEDED, Data, width
from common import C, CheckReport, finish, log
from lctx import seqcomp
from lctx.coords.liftover import Lifter
from lctx.embed.hyena import HyenaEmbedder
from lctx.io.features import RandomAccessBed
from lctx.io.genome import Reference
from lctx.sae import topk

N_ROWS = 300


class Source:
    """Rebuild a row's ingredients from the originals, using only its coordinates."""

    def __init__(self):
        self.rab, self.lifter = RandomAccessBed(C.INPUTS.feature_bed), Lifter(C.INPUTS.chain)
        self.ref38 = Reference(C.INPUTS.hg38)
        self.fa19 = pyfaidx.Fasta(C.INPUTS.hg19, as_raw=True, rebuild=False)
        self.emb = HyenaEmbedder(C.INPUTS.hyena_dir)

    def rows(self, D, which, idx):
        if which == "sites":
            chroms, fpos, pos19, s38 = D.names[idx], D.pos_ins[idx], D.pos19[idx], D.pos38[idx]
        else:
            chroms = D.panel.chrom.to_numpy().astype(str)[idx]
            fpos = pos19 = D.panel.pos19.to_numpy()[idx]
            s38 = D.panel_pos38[idx]
        F = np.stack([self.rab.lookup(c, int(p)) for c, p in zip(chroms, fpos)])
        p38 = np.array([self.lifter.lift(c, p)[0] for c, p in zip(chroms, pos19)])
        seqs = [self.ref38.window(c, int(p), C.CONTEXT) for c, p in zip(chroms, p38)]
        E = np.concatenate([self.emb.embed(seqs[i:i + 32])["mean"] for i in range(0, len(seqs), 32)])
        comp = np.stack([seqcomp.composition(self.fa19[c][int(p) - C.CENTER:int(p) - C.CENTER + C.CONTEXT])
                         for c, p in zip(chroms, pos19)])
        return {"F": F, "E": E, "comp": comp, "p38": p38, "stored_p38": s38}


def expected_extra(arm, seed, which, D, src, idx, perm_src, sae):
    if arm == "repl":
        return None
    if arm == "emb":
        return src["E"]
    if arm in SAE_NAMES:
        return topk.encode(sae[(arm, seed)], src["E"])
    if arm == "comp":
        return src["comp"]
    if arm in ("fsae_codes", "fsae_plus"):
        return topk.encode(sae[(FSAE, seed)], src["F"])
    if arm == "posctl":
        return src["F"][:, [D.posctl_feature]]
    if arm == "shuf_emb":
        return perm_src["E"]
    if arm.startswith("shuf_"):
        return topk.encode(sae[(arm[5:], seed)], perm_src["E"])
    if arm.startswith("noise"):                                  # no source: deterministic regeneration
        return D.extra(arm, seed, which)[idx]
    raise KeyError(arm)


def run(run):
    run.gate("s04_features", "s06_embed", "s06b_composition", "s07_sae")
    torch.set_num_threads(32)
    rep = CheckReport("s08_design")
    D = Data(run)
    rng = np.random.default_rng(1)
    src = Source()
    rep.info["rows"] = {r: int(len(v)) for r, v in D.idx.items()} | {"panel_usable": int(D.panel_usable.sum()),
                                                                      "panel_all": int(len(D.panel))}
    samples = {"sites": np.sort(rng.choice(D.idx["train"], N_ROWS, replace=False)),
               "panel": np.sort(rng.choice(np.where(D.panel_usable)[0], N_ROWS, replace=False))}
    rebuilt, permuted = {}, {}
    for w, idx in samples.items():
        rebuilt[w] = src.rows(D, w, idx)
        rep.eq(f"{w}: re-lifted hg38 position == stored", int((rebuilt[w]["p38"] != rebuilt[w]["stored_p38"]).sum()), 0)
        permuted[w] = src.rows(D, w, D.permutation(w, 0)[idx])       # source rows of the shuffled arms (seed 0)
    sae = {(n, s): topk.load(run.path("s07_sae", f"{n}_s{s}.pt"))[0] for n in SAE_NAMES + [FSAE] for s in C.SEEDS}

    for arm in ARMS:
        seeds = C.SEEDS if (arm in SEEDED and not arm.startswith("shuf_")) else (0,)
        for seed in seeds:
            tr_in, pa_in = D.inputs(arm, seed)
            tag = arm + (f"_s{seed}" if len(seeds) > 1 else "")
            rep.true(f"{tag}: train/benchmark share columns + standardisation",
                     tr_in.nf == pa_in.nf and tr_in.nz == pa_in.nz and np.array_equal(tr_in.mu, pa_in.mu)
                     and np.array_equal(tr_in.sd, pa_in.sd))
            rep.eq(f"{tag}: extra columns", tr_in.nz, width(arm))
            if tr_in.nz:                                  # over ALL training rows (rows are genome-sorted)
                Zs = tr_in.dense(D.idx["train"])[:, tr_in.nf:]
                live = tr_in.sd != 1.0                     # constant columns keep sd 1 by definition
                rep.le(f"{tag}: standardised extra columns on train rows: max |mean|", round(float(np.abs(Zs.mean(0)).max()), 8), 1e-6)
                rep.le(f"{tag}: standardised extra columns on train rows: max |sd - 1|",
                       round(float(np.abs(Zs.std(0)[live] - 1).max()) if live.any() else 0.0, 8), 1e-4)
                del Zs
            for w, inp in (("sites", tr_in), ("panel", pa_in)):
                idx = samples[w]
                F = rebuilt[w]["F"]
                if arm == "posctl":
                    F = F[:, [j for j in range(C.N_FEATURES) if j != D.posctl_feature]]
                if arm == "fsae_codes":
                    F = F[:, :0]
                z = expected_extra(arm, seed, w, D, rebuilt[w], idx, permuted[w], sae)
                R = F.astype(np.float64)
                if z is not None:
                    R = np.hstack([R, (np.asarray(z, np.float64) - inp.mu) / inp.sd])
                M = inp.dense(idx)
                nf = F.shape[1]
                feq = np.all(M[:, :nf] == R[:, :nf], axis=1)
                xeq = np.all(np.abs(M[:, nf:] - R[:, nf:]) <= 1e-2, axis=1) if R.shape[1] > nf else np.ones(len(idx), bool)
                real = float((feq & xeq).mean())
                approx = arm in SAE_NAMES or arm.startswith("shuf_")   # TopK ties; a permuted-from row may be unlifted
                rep.ge(f"{tag} [{w}]: rows equal to re-derivation from originals", round(real, 4), 0.99 if approx else 1.0)
                sw = np.roll(np.arange(len(idx)), 1)
                ctrl = np.all(M[sw, :nf] == R[:, :nf], axis=1)
                if R.shape[1] > nf:
                    ctrl &= np.all(np.abs(M[sw, nf:] - R[:, nf:]) <= 1e-2, axis=1)
                rep.le(f"{tag} [{w}]: CONTROL swapped rows equal", round(float(ctrl.mean()), 4),
                       round(real - 0.25, 4) if R.shape[1] == nf else 0.05, control=True)
        log(f"design checked: {arm}")
    finish(run, "s08_design", rep)

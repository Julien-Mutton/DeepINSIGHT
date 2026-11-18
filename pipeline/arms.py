"""The arms of the experiment and the ONE function that builds their model inputs.

Training sites, held-out sites and benchmark variants of an arm are built here with the same
columns, in the same order; extra columns are standardised with statistics of the TRAINING rows.

  arm            x                                         question it answers
  repl           48 features                               R0: LINSIGHT (baseline)
  emb            48 + 256 HyenaDNA mean embedding          R1: does the language model's reading help?
  sae<d>k<k>     48 + d SAE codes of the embedding         R2: does a sparse summary of it help?
  noise_emb      48 + 256 iid N(0,1)                       cost of adding 256 uninformative columns
  noise<d>       48 + d iid N(0,1)                         same, for an SAE of width d
  shuf_emb       48 + embedding rows permuted              real embedding distribution, no link to the site
  shuf_sae<d>k<k>48 + SAE codes rows permuted              same, for the SAE codes (sparse shape kept)
  comp           48 + 4 composition numbers                does simple composition (GC, CpG, repeats) explain it?
  fsae_codes     SAE codes of the 48 features only         same information as R0, re-expressed
  fsae_plus      48 + those codes                          same information as R0, more flexible
  posctl         47 features + 1 via the extra channel     does the extra-column path work at all?
"""
import numpy as np
import pandas as pd

from common import C, load_npz
from lctx.model.linsight import Inputs

SAE_NAMES = [f"sae{d}k{k}" for d, k in C.SAE_GRID]
FSAE = "fsae32k8"
ARMS = (["repl", "emb"] + SAE_NAMES + ["noise_emb"] + [f"noise{d}" for d, _ in C.SAE_GRID]
        + ["shuf_emb"] + [f"shuf_{n}" for n in SAE_NAMES] + ["comp", "fsae_codes", "fsae_plus", "posctl"])
# arms trained with the paper procedure too (5 seeds); every arm is fit by exact maximum likelihood
PAPER_ARMS = ["repl", "emb"] + SAE_NAMES + ["noise_emb"] + [f"noise{d}" for d, _ in C.SAE_GRID] + ["shuf_emb", "posctl"]
NESTED_IN_R0 = [a for a in ARMS if a not in ("repl", "fsae_codes", "posctl")]   # x = [48 | extra]
SEEDED = set(SAE_NAMES) | {f"shuf_{n}" for n in SAE_NAMES} | {"fsae_codes", "fsae_plus"}  # extra depends on seed


def gamma_cols(arm, inputs):
    """Columns entering gamma' in the exact fit: the 48 features only (new inputs go to rho), except
    posctl (must stay an exact reparametrisation of R0) and fsae_codes (codes replace the features)."""
    return inputs.nx if arm in ("posctl", "fsae_codes") else inputs.nf


def r0_start(arm, inputs, p0, posctl_feature):
    """R0's optimum p0 = [b, w(48), c, v(48)] written in `arm`'s parametrisation: the same function of
    the data, with the new columns' weights at 0 (posctl: the moved, standardised feature re-scaled).
    A fit started here can only end at a lower penalised training objective than R0's own (nested).
    None for fsae_codes (not nested: its codes replace the features)."""
    if arm == "repl":
        return np.asarray(p0, np.float64).copy()
    if arm == "fsae_codes":
        return None
    nf0 = C.N_FEATURES
    rho, gam = p0[:1 + nf0], p0[1 + nf0:]
    if arm == "posctl":
        j = posctl_feature
        keep = [i for i in range(nf0) if i != j]
        mu, sd = float(inputs.mu[0]), float(inputs.sd[0])
        head = lambda h: np.r_[h[0] + h[1 + j] * mu, h[1 + np.array(keep)], h[1 + j] * sd]
        return np.r_[head(rho), head(gam)]
    assert inputs.nf == nf0 and gamma_cols(arm, inputs) == nf0
    return np.r_[rho, np.zeros(inputs.nz), gam]


def lrt_df(arm):
    """Degrees of freedom of the nested likelihood-ratio test against R0 (extra columns in rho only)."""
    return width(arm)


def width(arm):
    if arm == "repl":
        return 0
    if arm in ("emb", "noise_emb", "shuf_emb"):
        return C.D_MODEL
    if arm.startswith("noise"):
        return int(arm[5:])
    if arm.startswith("shuf_sae") or arm.startswith("sae"):
        return int(arm.split("sae")[1].split("k")[0])
    if arm == "comp":
        return 4
    if arm in ("fsae_codes", "fsae_plus"):
        return 32
    if arm == "posctl":
        return 1
    raise KeyError(arm)


class Data:
    """Everything the arms are built from, loaded once (fork-shared by training workers)."""

    def __init__(self, run):
        self.run = run
        S = load_npz(run.path("s02_sites", "sites.npz"))
        self.chrom, self.pos_ins, self.pos19, self.evol, self.role = S["chrom"], S["pos_ins"], S["pos19"], S["evol"], S["role"]
        self.names = np.array(C.AUTOSOMES)[self.chrom]
        SF = load_npz(run.path("s04_features", "sites_features.npz"))
        self.F, self.ffound = SF["F"], SF["found"]
        self.E = np.load(run.path("s06_embed", "sites_mean.npy"), mmap_mode="r")
        self.comp = np.load(run.path("s06b_composition", "sites_comp.npy"), mmap_mode="r")
        lift = load_npz(run.path("s03_liftover", "sites_lift.npz"))
        self.pos38 = lift["pos38"]
        self.usable = (self.ffound & (lift["status"] == "ok") & np.isfinite(np.asarray(self.E[:, 0]))
                       & np.isfinite(np.asarray(self.comp)).all(1))
        self.idx = {r: np.where(self.usable & (self.role == r))[0] for r in ("train", "select", "eval")}

        self.panel = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"))
        PF = load_npz(run.path("s04_features", "panel_features.npz"))
        PL = load_npz(run.path("s03_liftover", "panel_lift.npz"))
        assert (PF["key"] == self.panel.key.to_numpy()).all() and (PL["key"] == self.panel.key.to_numpy()).all()
        self.Fp = PF["F"]
        self.Ep = np.load(run.path("s06_embed", "panel_mean.npy"), mmap_mode="r")
        self.comp_p = np.load(run.path("s06b_composition", "panel_comp.npy"), mmap_mode="r")
        self.panel_pos38 = PL["pos38"]
        self.panel_usable = (PF["found"] & (PL["status"] == "ok") & np.isfinite(np.asarray(self.Ep[:, 0]))
                             & np.isfinite(np.asarray(self.comp_p)).all(1) & ~self.panel.label_conflict.to_numpy())
        self.posctl_feature = int(np.argmax(np.abs(load_npz(run.path("s04_features", "published_weights.npz"))["rho_coef"][1:])))

    # ---- extra columns
    def codes(self, name, seed, which):
        return np.load(self.run.path("s07_sae", f"codes_{name}_s{seed}_{which}.npy"), mmap_mode="r")

    def noise(self, d, seed, which):
        n = len(self.F) if which == "sites" else len(self.Fp)
        base = 1000 if which == "sites" else 2000
        return np.random.default_rng(base + 100 * d + seed).standard_normal((n, d)).astype(np.float32)

    def permutation(self, which, seed):
        """Row permutation for the shuffled controls: usable rows are permuted AMONG usable rows
        (sites and benchmark separately); unusable rows map to themselves. So every shuffled row
        carries a real value of the same distribution, taken from another position."""
        usable = self.usable if which == "sites" else self.panel_usable
        u = np.where(usable)[0]
        perm = np.arange(len(usable))
        perm[u] = u[np.random.default_rng((7000 if which == "sites" else 8000) + seed).permutation(len(u))]
        return perm

    def extra(self, arm, seed, which):
        sites = which == "sites"
        if arm == "repl":
            return None
        if arm == "emb":
            return np.asarray(self.E if sites else self.Ep, np.float32)
        if arm in SAE_NAMES:
            return np.asarray(self.codes(arm, seed, which), np.float32)
        if arm == "noise_emb":
            return self.noise(C.D_MODEL, seed, which)
        if arm.startswith("noise"):
            return self.noise(int(arm[5:]), seed, which)
        if arm == "shuf_emb":
            return np.asarray(self.E if sites else self.Ep, np.float32)[self.permutation(which, seed)]
        if arm.startswith("shuf_"):
            return np.asarray(self.codes(arm[5:], seed, which), np.float32)[self.permutation(which, seed)]
        if arm == "comp":
            return np.asarray(self.comp if sites else self.comp_p, np.float32)
        if arm in ("fsae_codes", "fsae_plus"):
            return np.asarray(self.codes(FSAE, seed, which), np.float32)
        if arm == "posctl":
            return (self.F if sites else self.Fp)[:, [self.posctl_feature]].astype(np.float32)
        raise KeyError(arm)

    def inputs(self, arm, seed):
        """(site Inputs, benchmark Inputs) sharing columns and training-row statistics."""
        F, Fp = self.F, self.Fp
        if arm == "posctl":
            keep = [j for j in range(C.N_FEATURES) if j != self.posctl_feature]
            F, Fp = F[:, keep], Fp[:, keep]
        if arm == "fsae_codes":
            F, Fp = np.zeros((len(F), 0), np.float32), np.zeros((len(Fp), 0), np.float32)
        Z, Zp = self.extra(arm, seed, "sites"), self.extra(arm, seed, "panel")
        if Z is None:
            return Inputs(F), Inputs(Fp)
        if Z.shape[1] != width(arm):
            raise ValueError(f"{arm}: {Z.shape[1]} extra columns, expected {width(arm)}")
        tr = self.idx["train"]
        mu = Z[tr].astype(np.float64).mean(0)
        sd = Z[tr].astype(np.float64).std(0)
        Z = np.where(np.isfinite(Z), Z, 0).astype(np.float32)
        Zp = np.where(np.isfinite(Zp), Zp, 0).astype(np.float32)
        return Inputs(F, Z, mu, sd), Inputs(Fp, Zp, mu, sd)

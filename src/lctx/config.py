"""Every path and constant the pipeline uses. Inputs are original files only (docs/DATA.md)."""
import os
import shutil
import sys
from dataclasses import dataclass, fields

# Repository root (…/DeepInsight). Everything else is relative to it unless overridden by an
# environment variable, so the code runs from any clone:
#   DEEPINSIGHT_DATA     folder with the original inputs        (default <repo>/data, see data/README.md)
#   DEEPINSIGHT_OUT      folder for pipeline outputs            (default <repo>/outputs)
#   DEEPINSIGHT_PY       Python of the main environment         (default: the running interpreter)
#   DEEPINSIGHT_PY_LEGACY Python with pandas 0.13 for the benchmark pickles (env `legacy_pandas`)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.abspath(os.environ.get("DEEPINSIGHT_DATA", os.path.join(ROOT, "data")))
OUT = os.path.abspath(os.environ.get("DEEPINSIGHT_OUT", os.path.join(ROOT, "outputs")))
PY = os.environ.get("DEEPINSIGHT_PY", sys.executable)


def _legacy_python():
    if os.environ.get("DEEPINSIGHT_PY_LEGACY"):
        return os.environ["DEEPINSIGHT_PY_LEGACY"]
    conda = os.environ.get("CONDA_EXE")       # …/miniconda3/bin/conda -> …/miniconda3/envs/legacy_pandas
    if conda:
        cand = os.path.join(os.path.dirname(os.path.dirname(conda)), "envs", "legacy_pandas", "bin", "python")
        if os.path.exists(cand):
            return cand
    return shutil.which("python2") or "python2"


PY_LEGACY = _legacy_python()


@dataclass(frozen=True)
class Inputs:
    """The original files (docs/DATA.md). Layout under DEEPINSIGHT_DATA is fixed (data/README.md)."""
    bench_dir: str = os.path.join(DATA, "benchmark")
    ins: str = os.path.join(DATA, "linsight", "INSIGHT_database.ins")
    feature_bed: str = os.path.join(DATA, "linsight", "genomic_feature.bed")
    linsight_bw: str = os.path.join(DATA, "linsight", "LINSIGHT.bw")
    linsight_src: str = os.path.join(ROOT, "third_party", "linsight_cpp")
    hg19: str = os.path.join(DATA, "genomes", "hg19.fa")
    hg38: str = os.path.join(DATA, "genomes", "hg38.ml.fa")
    chain: str = os.path.join(DATA, "genomes", "hg19ToHg38.over.chain.gz")
    hyena_dir: str = os.path.join(DATA, "hyenadna", "hyenadna-medium-160k-seqlen")

    def missing(self):
        return [f.name for f in fields(self) if not os.path.exists(getattr(self, f.name))]


INPUTS = Inputs()

# upstream identities (docs/DATA.md)
HYENA_WEIGHTS_SHA256 = "02729e5fcbd426a3965a2cca789dae8bd0ed9e24289cac60e5ca9e37525e555a"
CHAIN_MD5 = "35887f73fe5e2231656504d1f6430900"
LINSIGHT_BW_SIZE = 1_395_672_914

# benchmark (Huang et al. 2017, after Ritchie et al. 2014)
POSITIVE = "hgmd_reg"
TIERS = {"unmatched": 0.897, "matched_by_tss": 0.759, "matched_by_region": 0.660}   # paper AUROC
EXPECTED_COUNTS = {"hgmd_reg": 1614, "unmatched": 161400, "matched_by_tss": 16140, "matched_by_region": 5027}

AUTOSOMES = tuple(f"chr{i}" for i in range(1, 23))

# HyenaDNA window: the site sits at index CENTER of a CONTEXT-bp window
CONTEXT = 1001
CENTER = CONTEXT // 2
D_MODEL = 256

# LINSIGHT (L-INSIGHT-fit.cpp / README)
BETA = (0.7567672441, 0.2053143291, 1.0 - 0.7567672441 - 0.2053143291)
RHO_SHIFT = -3.0
ALLELE_FREQ_CUTOFF = 0.15
MIN_THETA_LAMBDA = 1e-5
MINIBATCH, LR_RHO, LR_GAMMA, EPS, EPOCHS = 100, 1e-3, 1e-2, 1e-6, 20
N_FEATURES = 48

# design decisions (docs/SPEC.md)
EXCLUDE_BP = 1000
BLOCK_BP = 1_000_000
HELDOUT_FRAC = float(os.environ.get("DEEPINSIGHT_HELDOUT_FRAC", 0.50))  # held-out genome blocks (0.50: likelihood analysis; the HGMD read of plan v1 used 0.10)
SELECT_SHARE = 0.10          # share of held-out blocks used only for SAE-config selection
SAE_GRID = ((32, 8), (64, 16), (128, 32))
SEEDS = tuple(int(x) for x in os.environ.get("DEEPINSIGHT_SEEDS", "0,1,2,3,4").split(","))  # plan v1 HGMD read: 0,1,2
BOOTSTRAP_REPS, CLUSTER_BP = 2000, 100_000
ALPHA, POWER = 0.05, 0.80
PRIOR_SDS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)   # prior widths for the 48 feature weights: chosen ONCE, by R0, on `select` sites
PRIOR_SDS_EXTRA = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0)  # prior widths for the new columns: chosen per arm and seed on `select`
LATENT_CLAMP = 30.0          # |latent| bound in the exact-likelihood fit (rho underflows beyond it anyway)


def out(run, *parts):
    """<DEEPINSIGHT_OUT>/<run>/<parts...>; run = 'shared' or 'rate_<r>'."""
    p = os.path.join(OUT, run, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p

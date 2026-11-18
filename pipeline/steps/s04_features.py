"""s04 features: the 48 LINSIGHT features for every site (at pos_ins, as LINSIGHT-fit does) and
every benchmark variant (at pos19, where the published LINSIGHT.bw reproduces the paper).

LINSIGHT.bw is, by construction, rho(features) with the published weights. So:
  * identical feature vectors must carry identical LINSIGHT.bw values (functional dependence);
  * u = -log(-log(bw)/3) must be EXACTLY linear in the features (Gompertz link) -> least squares
    recovers the published rho-weights (b, w) with R^2 ~ 1;
  * those weights applied to OUR looked-up features must reproduce bw at our sites and variants.
A control with features taken from other rows must fail each of these.
"""
import numpy as np
import pandas as pd

from common import C, CheckReport, finish, load_npz, log, save_npz
from lctx.io.features import lookup_many
from lctx.io.genome import BigWig


def gompertz_inverse(bw):
    return -np.log(-np.log(np.clip(bw, 1e-12, 1 - 1e-12)) / 3.0)


def recover_weights(F, bw):
    """Least squares of the Gompertz-inverse of LINSIGHT.bw on [1, F]."""
    m = np.isfinite(bw) & (bw > 0) & (bw < 1)
    X = np.hstack([np.ones((m.sum(), 1)), F[m].astype(np.float64)])
    u = gompertz_inverse(bw[m])
    coef, *_ = np.linalg.lstsq(X, u, rcond=None)
    r2 = 1 - ((u - X @ coef) ** 2).sum() / ((u - u.mean()) ** 2).sum()
    return coef, float(r2)


def reproduces(F, bw, coef, tol=2e-3):
    m = np.isfinite(bw)
    pred = np.exp(-3 * np.exp(-(coef[0] + F[m].astype(np.float64) @ coef[1:])))
    return float(np.mean(np.abs(pred - bw[m]) <= tol))


def functional_dependence(F, bw):
    """Fraction of rows whose bw equals the median bw of their identical-feature-vector group."""
    m = np.isfinite(bw)
    df = pd.DataFrame(F[m]).astype(np.int8 if np.all(np.isin(F[m], (0, 1))) else np.float32)
    g = df.groupby(list(df.columns), sort=False).ngroup().to_numpy()
    med = pd.Series(bw[m]).groupby(g).transform("median").to_numpy()
    return float(np.mean(np.abs(bw[m] - med) <= 1e-4))


def run(run):
    run.gate("s01_benchmark", "s02_sites")
    rep = CheckReport("s04_features")
    bwr = BigWig(C.INPUTS.linsight_bw)
    rng = np.random.default_rng(0)

    S = load_npz(run.path("s02_sites", "sites.npz"))
    names = np.array(C.AUTOSOMES)[S["chrom"]]
    Fs, fs = lookup_many(C.INPUTS.feature_bed, names, S["pos_ins"])
    log(f"site features found {fs.mean():.5f}")
    rep.ge("sites: features_found_frac", round(float(fs.mean()), 5), 0.97)
    pan = pd.read_csv(run.path("s01_benchmark", "panel.csv.gz"))
    Fp, fp = lookup_many(C.INPUTS.feature_bed, pan.chrom.to_numpy(), pan.pos19.to_numpy())
    rep.ge("benchmark: features_found_frac", round(float(fp.mean()), 5), 0.97)
    rep.true("all features finite where found", bool(np.isfinite(Fs[fs]).all() and np.isfinite(Fp[fp]).all()))
    rep.info["feature_values"] = sorted(set(np.unique(Fs[fs]).tolist()))[:20]

    # published weights recovered from LINSIGHT.bw on benchmark variants
    bwp = pan.linsight_published.to_numpy()
    coef, r2 = recover_weights(Fp[fp], bwp[fp])
    rep.ge("Gompertz-inverse(LINSIGHT.bw) linear in features: R^2", round(r2, 6), 0.999)
    rep.info["published_rho_weights_recovered"] = coef.tolist()
    rep.ge("benchmark: features reproduce LINSIGHT.bw (|err| <= 2e-3)", round(reproduces(Fp[fp], bwp[fp], coef), 5), 0.99)
    # many rows share common feature vectors, so a control must fall clearly BELOW the real value
    real_p = reproduces(Fp[fp], bwp[fp], coef)
    shuf = rng.permutation(np.where(fp)[0])
    rep.le("benchmark: CONTROL features of other rows reproduce bw", round(reproduces(Fp[shuf], bwp[fp], coef), 4),
           round(real_p - 0.2, 4), control=True)
    rep.ge("benchmark: functional dependence bw <- features", round(functional_dependence(Fp[fp], bwp[fp]), 5), 0.99)

    # our site features against the published track at the same LINSIGHT coordinate
    sub = rng.choice(np.where(fs)[0], min(50000, fs.sum()), replace=False)
    bws = bwr.values_at(names[sub].tolist(), S["pos_ins"][sub].tolist())
    real_s = reproduces(Fs[sub], bws, coef)
    rep.ge("sites: features reproduce LINSIGHT.bw at pos_ins", round(real_s, 5), 0.99)
    rep.le("sites: CONTROL features of other sites", round(reproduces(Fs[rng.permutation(sub)], bws, coef), 4),
           round(real_s - 0.2, 4), control=True)
    bws_far = bwr.values_at(names[sub].tolist(), (S["pos_ins"][sub] + 5000).tolist())
    rep.le("sites: CONTROL bw 5 kb away", round(reproduces(Fs[sub], bws_far, coef), 4), round(real_s - 0.2, 4),
           control=True)

    save_npz(run.path("s04_features", "sites_features.npz"), F=np.nan_to_num(Fs), found=fs)
    save_npz(run.path("s04_features", "panel_features.npz"), key=pan.key.to_numpy().astype("U40"),
             F=np.nan_to_num(Fp), found=fp)
    save_npz(run.path("s04_features", "published_weights.npz"), rho_coef=coef)
    finish(run, "s04_features", rep)

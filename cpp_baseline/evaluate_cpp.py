"""Score the HGMD benchmark with the C++ LINSIGHT fit and compare with the published track.

    python cpp_baseline/evaluate_cpp.py [--params PARAMETER_TXT] [--rate 0.001]

Uses the benchmark panel (s01) and its 48 features (s04) of a pipeline run, so run
`python -m deepinsight run --rate 0.001 --to s04_features` first. The final epoch of parameter.txt
([b_rho, w_rho(48), b_gamma, w_gamma(48)] per row) is applied with LINSIGHT's link
rho = exp(-3 exp(-(b + w.x))), as in L-INSIGHT-pred.cpp. This is a fixed, published model: nothing is
chosen with the labels. Our recorded result: results/cpp_baseline/results.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from lctx import config as C                       # noqa: E402
from lctx.eval.metrics import auroc                # noqa: E402
from lctx.io.benchmark import tier_mask            # noqa: E402


def load_params(path, epoch=-1):
    rows = [np.array(l.split(), float) for l in open(path) if l.strip()]
    p = rows[epoch]
    return p[: len(p) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(C.OUT, "cpp_baseline", "parameter.txt"))
    ap.add_argument("--rate", type=float, default=0.001)
    a = ap.parse_args()
    run = os.path.join(C.OUT, f"rate_{a.rate:g}")
    pan = pd.read_csv(os.path.join(run, "s01_benchmark", "panel.csv.gz"), dtype={"row_id": str})
    with np.load(os.path.join(run, "s04_features", "panel_features.npz"), allow_pickle=False) as d:
        assert (d["key"] == pan["key"].to_numpy()).all(), "panel and feature rows differ"
        F, found = d["F"].astype(np.float64), d["found"]
    w = load_params(a.params)
    cpp = np.exp(C.RHO_SHIFT * np.exp(-(w[0] + F @ w[1:])))
    pub = pan["linsight_published"].to_numpy()
    ok = found & np.isfinite(cpp) & np.isfinite(pub)
    y = pan["label"].to_numpy()
    res = {"params": a.params, "tiers": {}}
    for tier, paper in C.TIERS.items():
        m = tier_mask(pan, tier) & ok
        res["tiers"][tier] = {"n": int(m.sum()), "n_pos": int(y[m].sum()), "paper": paper,
                              "auroc_cpp_refit": auroc(cpp[m], y[m]), "auroc_published_track": auroc(pub[m], y[m])}
        r = res["tiers"][tier]
        print(f"{tier:18s} C++ {r['auroc_cpp_refit']:.4f}  published track {r['auroc_published_track']:.4f}  paper {paper:.3f}")
    out = os.path.join(C.OUT, "cpp_baseline", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()

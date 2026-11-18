"""Shared step machinery: run directory, logging, gates, saving keyed arrays."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from lctx import config as C                                  # noqa: E402
from lctx.checks import CheckReport, require                  # noqa: E402

STEPS = ["s00_inputs", "s01_benchmark", "s02_sites", "s03_liftover", "s04_features",
         "s05_embedder", "s06_embed", "s06b_composition", "s07_sae", "s08_design", "s09_train",
         "s09b_mle", "s09c_groundtruth", "s10_evaluate",
         "s11_report"]


class Run:
    """One pipeline run = one sample rate; everything it makes lives in <DEEPINSIGHT_OUT>/rate_<r>/."""

    def __init__(self, rate):
        self.rate = float(rate)
        self.name = f"rate_{self.rate:g}"
        self.dir = os.path.join(C.OUT, self.name)
        os.makedirs(os.path.join(self.dir, "logs"), exist_ok=True)

    def path(self, step, *parts):
        p = os.path.join(self.dir, step, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def checks_path(self, step):
        return self.path(step, "checks.json")

    def gate(self, *steps):
        for s in steps:
            require(self.checks_path(s))

    def info(self, step):
        return json.load(open(self.checks_path(step)))["info"]


_LOG = None


def set_log(path):
    global _LOG
    _LOG = open(path, "a")


def log(*a):
    msg = time.strftime("[%H:%M:%S] ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOG:
        _LOG.write(msg + "\n")
        _LOG.flush()


def finish(run, step, rep):
    rep.save(run.checks_path(step))
    log(rep.summary())
    if not rep.ok:
        raise SystemExit(f"step {step} FAILED its checks: {[c['name'] for c in rep.failures()]}")
    return rep


def save_npz(path, **arrays):
    np.savez(path, **arrays)
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

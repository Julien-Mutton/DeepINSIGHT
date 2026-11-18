import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def synthetic_evol(rng, n):
    """Realistic INSIGHT records: mostly monomorphic, some low/high frequency."""
    t = rng.choice(3, n, p=[0.9, 0.07, 0.03])
    af = np.array([0.0, 0.05, 0.5], np.float32)[t]
    pmaj = rng.uniform(0.95, 1.0, n).astype(np.float32)
    pmin = np.where(t > 0, rng.uniform(0, 0.05, n), 0).astype(np.float32)
    th = rng.uniform(1e-6, 0.02, n).astype(np.float32)
    lam = rng.uniform(1e-6, 0.01, n).astype(np.float32)
    return np.column_stack([af, pmaj, pmin, th, lam]).astype(np.float32)

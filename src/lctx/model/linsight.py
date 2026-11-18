"""LINSIGHT model and the paper's training procedure (L-INSIGHT-fit.cpp).

    rho     = exp(shift * exp(-(b + w.x)))         shift = -3 (Gompertz)
    gamma'  = sigmoid(c + v.x)
Parameters: [b, w(nx), c, v(nx)], zero-initialised. AdaGrad on mini-batches of `mb` sites with
SUMMED gradients; lr_rho for the first half, lr_gamma for the second. The inner loop is
`_kernel.c`; `fit_epoch_reference` is a plain-numpy transcription used by the tests.
"""
import ctypes
import os
import subprocess

import numpy as np

from ..config import BETA, RHO_SHIFT, MINIBATCH, LR_RHO, LR_GAMMA, EPS

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_kernel.c")
_LIB = os.path.join(_HERE, "_build", "liblinsight.so")
_lib = None


def _kernel():
    global _lib
    if _lib is None:
        if not os.path.exists(_LIB) or os.path.getmtime(_LIB) < os.path.getmtime(_SRC):
            os.makedirs(os.path.dirname(_LIB), exist_ok=True)
            # -O2 without -ffast-math: IEEE semantics, same as the C++ build
            subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-o", _LIB, _SRC, "-lm"], check=True)
        L = ctypes.CDLL(_LIB)
        dp = np.ctypeslib.ndpointer(np.float64, flags="C")
        fp = np.ctypeslib.ndpointer(np.float32, flags="C")
        ip = np.ctypeslib.ndpointer(np.int64, flags="C")
        i64, i32, dbl = ctypes.c_int64, ctypes.c_int, ctypes.c_double
        L.fit_epoch.restype = i64
        L.fit_epoch.argtypes = [i64, ip, fp, i32, fp, i32, dp, dp, dp, dp, dp, dp,
                                dbl, dbl, i64, dbl, dbl, dbl, dp, dp, dp]
        L.score.restype = None
        L.score.argtypes = [i64, ip, fp, i32, fp, i32, dp, dp, dp, dbl, dp, dp, dp]
        _lib = L
    return _lib


def gompertz_rho(latent, shift=RHO_SHIFT):
    return np.exp(shift * np.exp(-np.asarray(latent, np.float64)))


class Inputs:
    """x = [F raw | (Z - mu) / sd].  mu, sd come from TRAINING sites and are shared by every
    matrix built for the same arm (train sites, held-out sites, benchmark)."""

    def __init__(self, F, Z=None, mu=None, sd=None):
        self.F = np.ascontiguousarray(F, np.float32)
        n = len(self.F)
        if Z is None:
            self.Z = np.zeros((n, 0), np.float32)
            self.mu = self.sd = np.zeros(0)
        else:
            self.Z = np.ascontiguousarray(Z, np.float32)
            if len(self.Z) != n:
                raise ValueError("F and Z row counts differ")
            self.mu = np.ascontiguousarray(mu, np.float64)
            sd = np.asarray(sd, np.float64)
            self.sd = np.ascontiguousarray(np.where(sd > 0, sd, 1.0))
        self.nf, self.nz = self.F.shape[1], self.Z.shape[1]
        self.nx = self.nf + self.nz
        if self.nx > 8192:
            raise ValueError("kernel supports at most 8192 inputs")

    def dense(self, idx):
        X = self.F[idx].astype(np.float64)
        if self.nz:
            X = np.hstack([X, (self.Z[idx].astype(np.float64) - self.mu) / self.sd])
        return X


class Trainer:
    def __init__(self, inputs, terms, beta0=BETA[0], shift=RHO_SHIFT, mb=MINIBATCH,
                 lr_rho=LR_RHO, lr_gamma=LR_GAMMA, eps=EPS):
        self.x, self.terms = inputs, terms
        self.beta0, self.shift, self.mb = beta0, shift, mb
        self.lr_rho, self.lr_gamma, self.eps = lr_rho, lr_gamma, eps
        P = 2 * (1 + inputs.nx)
        self.para, self.grad, self.hist = np.zeros(P), np.zeros(P), np.zeros(P)

    def epoch(self, order):
        order = np.ascontiguousarray(order, np.int64)
        A, B, D, F = self.terms
        return _kernel().fit_epoch(len(order), order, self.x.F, self.x.nf, self.x.Z, self.x.nz,
                                   self.x.mu, self.x.sd, A, B, D, F, self.beta0, self.shift,
                                   self.mb, self.lr_rho, self.lr_gamma, self.eps,
                                   self.para, self.grad, self.hist)


def score(inputs, para, idx=None, shift=RHO_SHIFT):
    """(rho, gamma', latent contribution of the extra columns) for rows idx (default all)."""
    idx = np.ascontiguousarray(np.arange(len(inputs.F)) if idx is None else idx, np.int64)
    n = len(idx)
    rho, gp, le = np.empty(n), np.empty(n), np.empty(n)
    _kernel().score(n, idx, inputs.F, inputs.nf, inputs.Z, inputs.nz, inputs.mu, inputs.sd,
                    np.ascontiguousarray(para, np.float64), shift, rho, gp, le)
    return rho, gp, le


def fit_epoch_reference(state, X, terms, order, beta0=BETA[0], shift=RHO_SHIFT, mb=MINIBATCH,
                        lr_rho=LR_RHO, lr_gamma=LR_GAMMA, eps=EPS):
    """Site-by-site numpy transcription of updateGradient/updateParameters (tests only)."""
    A, B, D, F = terms
    para, grad, hist = state["para"], state["grad"], state["hist"]
    H = 1 + X.shape[1]
    lrs = np.where(np.arange(len(para)) < H, lr_rho, lr_gamma)
    counter = 0
    for s in order:
        x = X[s]
        lat_r = para[0] + float(np.dot(para[1:H], x))
        lat_g = para[H] + float(np.dot(para[H + 1:], x))
        rho = np.exp(shift * np.exp(-lat_r))
        gp = 1 / (1 + np.exp(-lat_g))
        dR = A[s] * (B[s] + D[s] * (gp * beta0)) - F[s]
        dG = A[s] * D[s]
        P = dR * rho + F[s]
        dR, dG = -dR / P, -dG / P
        rI = dR * -1 * shift * np.exp(shift * np.exp(-lat_r)) * np.exp(-lat_r)
        gI = dG / beta0 * gp * gp * np.exp(-lat_g)
        grad[0] += rI
        grad[1:H] += rI * x
        grad[H] += gI
        grad[H + 1:] += gI * x
        counter += 1
        if counter == mb:
            hist += grad ** 2
            para -= grad / (np.sqrt(hist) + eps) * lrs
            grad[:] = 0
            counter = 0
    return state


def cpp_order(chroms, pos):
    """The C++ visits chromosome files in std::map (string) order, positions ascending."""
    return np.lexsort((np.asarray(pos), np.asarray(chroms).astype(str)))

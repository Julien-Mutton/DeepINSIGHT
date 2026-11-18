"""TopK sparse autoencoder (Gao et al. 2024, "Scaling and evaluating sparse autoencoders", arXiv:2406.04093).

    x_n   = (x - mu) / scale                       mu, scale from the training rows
    z     = TopK(ReLU(W_enc (x_n - b_dec) + b_enc))
    x_hat = z W_dec + b_dec                         decoder rows kept at unit norm
Loss: MSE on x_n + AuxK (k_aux = 2k, coefficient 1/32) on latents dead for > dead_steps
batches. Encoder initialised as the decoder transpose. Adam.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    d_sae: int = 32
    k: int = 8
    lr: float = 1e-3
    epochs: int = 10
    batch: int = 4096
    seed: int = 0
    aux_coef: float = 1 / 32
    dead_steps: int = 500


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        W = torch.randn(d_sae, d_in)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W.clone())
        self.W_enc = nn.Parameter(W.clone().T.contiguous())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("scale", torch.ones(()))

    def normalise(self, x):
        return (x - self.mu) / self.scale

    def pre(self, xn):
        return torch.relu((xn - self.b_dec) @ self.W_enc + self.b_enc)

    @staticmethod
    def topk(a, k):
        if k >= a.shape[-1]:
            return a
        v, i = a.topk(k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, i, v)

    def encode(self, x):
        return self.topk(self.pre(self.normalise(x)), self.k)

    def forward(self, x):
        xn = self.normalise(x)
        z = self.topk(self.pre(xn), self.k)
        return z @ self.W_dec + self.b_dec, z, xn


def fvu(xn, xhat):
    return float(((xn - xhat) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())


def fit(X, cfg: SAEConfig, log=None):
    torch.manual_seed(cfg.seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    m = TopKSAE(Xt.shape[1], cfg.d_sae, cfg.k)
    m.mu.copy_(Xt.mean(0))
    m.scale.copy_(((Xt - m.mu) ** 2).sum(1).mean().sqrt() / np.sqrt(Xt.shape[1]))
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    since = torch.zeros(cfg.d_sae, dtype=torch.long)
    g = torch.Generator().manual_seed(cfg.seed)
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), cfg.batch):
            xhat, z, xn = m(Xt[perm[i:i + cfg.batch]])
            loss = ((xhat - xn) ** 2).sum(1).mean()
            fired = (z > 0).any(0)
            since[fired] = 0
            since[~fired] += 1
            dead = since > cfg.dead_steps
            if dead.any():
                a = m.pre(xn) * dead.float()
                za = m.topk(a, min(2 * cfg.k, int(dead.sum())))
                loss = loss + cfg.aux_coef * ((za @ m.W_dec - (xn - xhat).detach()) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():                     # keep decoder rows unit norm
                m.W_dec.grad -= (m.W_dec.grad * m.W_dec).sum(1, keepdim=True) * m.W_dec
            opt.step()
            with torch.no_grad():
                m.W_dec /= m.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            tot += loss.item() * len(xn)
        if log:
            log(f"  sae d={cfg.d_sae} k={cfg.k} s{cfg.seed} epoch {ep + 1}: loss {tot / len(Xt):.4f}")
    return m.eval()


@torch.no_grad()
def encode(m, X, batch=1 << 18):
    out = np.empty((len(X), m.d_sae), np.float32)
    for i in range(0, len(X), batch):
        out[i:i + batch] = m.encode(torch.as_tensor(np.asarray(X[i:i + batch], np.float32))).numpy()
    return out


@torch.no_grad()
def metrics(m, X):
    xhat, z, xn = m(torch.as_tensor(np.asarray(X, np.float32)))
    return {"fvu": fvu(xn, xhat), "l0": float((z > 0).sum(1).float().mean()),
            "dead_frac": float(1 - (z > 0).any(0).float().mean())}


def save(m, cfg, path):
    torch.save({"state": m.state_dict(), "cfg": asdict(cfg), "d_in": m.d_in}, path)


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**d["cfg"])
    m = TopKSAE(d["d_in"], cfg.d_sae, cfg.k)
    m.load_state_dict(d["state"])
    return m.eval(), cfg

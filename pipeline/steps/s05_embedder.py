"""s05 embedder: the pretrained HyenaDNA loads completely and behaves like a causal DNA LM
whose index CENTER is the site.

Checks: every backbone tensor loaded; next-base prediction accuracy on real hg38 windows well
above chance (random weights do not get there); deterministic; batch-invariant; causal:
mutating base CENTER leaves states < CENTER unchanged and changes state CENTER; mutating
CENTER+1 leaves state CENTER unchanged (no off-by-one in the window centring).
"""
import numpy as np
import torch

from common import C, CheckReport, finish, load_npz, log
from lctx.embed.hyena import HyenaEmbedder
from lctx.io.genome import Reference


def _mutate(s, i):
    b = "C" if s[i] != "C" else "G"
    return s[:i] + b + s[i + 1:]


def run(run):
    run.gate("s00_inputs", "s03_liftover")
    torch.set_num_threads(16)
    rep = CheckReport("s05_embedder")
    emb = HyenaEmbedder(C.INPUTS.hyena_dir)
    rep.ge("backbone_tensors_loaded", emb.n_loaded, 100)
    ref38 = Reference(C.INPUTS.hg38)
    S = load_npz(run.path("s02_sites", "sites.npz"))
    L = load_npz(run.path("s03_liftover", "sites_lift.npz"))
    names = np.array(C.AUTOSOMES)[S["chrom"]]
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(L["status"] == "ok")[0], 64, replace=False)
    seqs = [ref38.window(names[i], L["pos38"][i], C.CONTEXT) for i in idx]
    acc = emb.next_base_accuracy(seqs)
    rep.ge("next_base_accuracy_on_hg38_windows", round(acc, 4), 0.35, "chance ~0.25-0.30")
    rand = ["".join(rng.choice(list("ACGT"), C.CONTEXT)) for _ in range(8)]
    rep.le("CONTROL next_base_accuracy_on_random_sequence", round(emb.next_base_accuracy(rand), 4), 0.30, control=True)

    h1, h2 = emb.hidden(seqs[:4]), emb.hidden(seqs[:4])
    rep.eq("deterministic (max |diff|)", float((h1 - h2).abs().max()), 0.0)
    single = torch.cat([emb.hidden([s]) for s in seqs[:4]])
    rep.le("batch_invariance (max |diff|)", float((h1 - single).abs().max()), 1e-3)
    rep.eq("hidden_shape", list(h1.shape), [4, C.CONTEXT, C.D_MODEL])

    s = seqs[0]
    base = emb.hidden([s])[0]
    mc = emb.hidden([_mutate(s, C.CENTER)])[0]
    mn = emb.hidden([_mutate(s, C.CENTER + 1)])[0]
    rep.le("causal: states < CENTER unchanged by mutating CENTER", float((mc[:C.CENTER] - base[:C.CENTER]).abs().max()), 1e-4,
           "FFT convolutions add ~1e-6 rounding")
    rep.ge("state CENTER changed by mutating CENTER", float((mc[C.CENTER] - base[C.CENTER]).abs().max()), 0.1)
    rep.le("state CENTER unchanged by mutating CENTER+1", float((mn[C.CENTER] - base[C.CENTER]).abs().max()), 1e-4)
    rep.ge("CONTROL state CENTER+1 changed by mutating CENTER+1", float((mn[C.CENTER + 1] - base[C.CENTER + 1]).abs().max()), 0.1,
           control=True)
    log(f"next-base accuracy {acc:.3f}")
    finish(run, "s05_embedder", rep)

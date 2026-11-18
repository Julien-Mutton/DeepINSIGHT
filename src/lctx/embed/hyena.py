"""HyenaDNA (medium-160k) embeddings of a site's sequence window.

Model code: the official `standalone_hyenadna.py` (third_party/hyena_dna, vendored verbatim).
Weights: the HuggingFace checkpoint; backbone keys are loaded with the official rule
(`huggingface.py::load_weights`: scratch key k <- checkpoint key 'model.' + k). The LM head is
tied to the token embedding in the checkpoint (`model.lm_head.weight`), used only by checks.

Tokens: the official CharacterTokenizer ids A,C,G,T,N = 7..11 (other characters -> [UNK] = 6),
WITHOUT special tokens, as in HyenaDNA's hg38 pretraining loader (add_eos=False).
Representation of a window: mean of the final hidden states over all positions (256-d);
the state at the centre index is returned too.
"""
import json
import os
import sys

import numpy as np
import torch

from ..config import ROOT, CENTER

sys.path.insert(0, os.path.join(ROOT, "third_party", "hyena_dna"))

VOCAB = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
UNK = 6
_LUT = np.full(256, UNK, np.int64)
for ch, i in VOCAB.items():
    _LUT[ord(ch)] = i
    _LUT[ord(ch.lower())] = i


def tokenize(seqs):
    """list of equal-length strings -> LongTensor (B, L)."""
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("all windows must have the same length")
    a = np.frombuffer("".join(seqs).encode("ascii"), np.uint8).reshape(len(seqs), L)
    return torch.from_numpy(_LUT[a])


class HyenaEmbedder:
    def __init__(self, model_dir, device="cpu"):
        from standalone_hyenadna import HyenaDNAModel
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.model = HyenaDNAModel(**cfg, use_head=False)
        ck = torch.load(os.path.join(model_dir, "weights.ckpt"), map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        scratch = self.model.state_dict()
        missing = [k for k in scratch if "backbone" in k and "model." + k not in sd]
        if missing:
            raise KeyError(f"{len(missing)} backbone keys missing from checkpoint, e.g. {missing[:3]}")
        for k in scratch:
            if "backbone" in k:
                if scratch[k].shape != sd["model." + k].shape:
                    raise ValueError(f"shape mismatch for {k}")
                scratch[k] = sd["model." + k]
        self.model.load_state_dict(scratch)
        self.model.eval().to(device)
        self.lm_head = sd["model.lm_head.weight"].float()
        self.n_loaded = sum("backbone" in k for k in scratch)
        self.device = device

    @torch.no_grad()
    def hidden(self, seqs):
        return self.model(tokenize(seqs).to(self.device)).float().cpu()

    @torch.no_grad()
    def embed(self, seqs):
        h = self.hidden(seqs)
        return {"mean": h.mean(1).numpy(), "center": h[:, CENTER].numpy()}

    @torch.no_grad()
    def next_base_accuracy(self, seqs):
        """Fraction of positions where argmax over A,C,G,T of the tied LM head = next base."""
        ids = tokenize(seqs)
        h = self.model(ids.to(self.device)).float().cpu()
        logits = h @ self.lm_head.T
        acgt = torch.tensor([7, 8, 9, 10])
        pred = acgt[logits[:, :-1, acgt].argmax(-1)]
        tgt = ids[:, 1:]
        m = torch.isin(tgt, acgt)
        return float((pred[m] == tgt[m]).float().mean())

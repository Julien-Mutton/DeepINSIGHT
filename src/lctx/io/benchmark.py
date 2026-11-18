"""The GWAVA/LINSIGHT noncoding benchmark, read from the original Python-2 pandas pickles.

The pickles need pandas 0.13 (env `legacy_pandas`); `export_pickles()` runs a tiny script
there that writes only the columns we use (row id, chr, start, end, cls, seq_A..T one-hot)
as tab-separated text. Coordinates are hg19, `start` 0-based (checked against hg19 in s01).
"""
import io
import os
import subprocess

import numpy as np
import pandas as pd

from ..config import PY_LEGACY, POSITIVE, EXPECTED_COUNTS

SOURCES = tuple(EXPECTED_COUNTS)
BASES = ("A", "C", "G", "T")

_EXPORT = r'''
import sys, pandas as pd
df = pd.read_pickle(sys.argv[1])
cols = ["chr", "start", "end", "cls", "seq_A", "seq_C", "seq_G", "seq_T"]
out = df[cols]
out.index.name = "row_id"
out.to_csv(sys.stdout, sep="\t")
'''


def export_pickle(path, py=PY_LEGACY):
    r = subprocess.run([py, "-c", _EXPORT, path], capture_output=True, check=True)
    return pd.read_csv(io.BytesIO(r.stdout), sep="\t", dtype={"row_id": str})


def load_panel(bench_dir, sources=SOURCES):
    frames = []
    for src in sources:
        df = export_pickle(os.path.join(bench_dir, f"{src}.pandas"))
        onehot = df[[f"seq_{b}" for b in BASES]].to_numpy()
        valid = (onehot.sum(1) == 1) & np.isin(onehot, (0, 1)).all(1)
        ref = np.where(valid, np.array(BASES)[onehot.argmax(1)], "N")
        frames.append(pd.DataFrame({
            "key": [f"bench|{src}|{r}" for r in df["row_id"]],
            "source": src, "row_id": df["row_id"].astype(str),
            "chrom": df["chr"].astype(str), "pos19": df["start"].astype(np.int64),
            "end": df["end"].astype(np.int64), "cls": df["cls"].astype(int),
            "label": int(src == POSITIVE), "ref": ref}))
    return pd.concat(frames, ignore_index=True)


def tier_mask(panel, tier):
    """Rows of one comparison: all positives + that tier's negatives."""
    return ((panel["source"] == POSITIVE) | (panel["source"] == tier)).to_numpy()

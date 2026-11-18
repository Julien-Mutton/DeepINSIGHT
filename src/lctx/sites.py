"""Training-site bookkeeping: exclusion around benchmark variants and the block split."""
import numpy as np


def min_distance(chroms, pos, ref_chroms, ref_pos):
    """Distance from each (chrom, pos) to the nearest reference position on the same chromosome
    (inf when the chromosome has none)."""
    chroms, pos = np.asarray(chroms).astype(str), np.asarray(pos, np.int64)
    ref_chroms, ref_pos = np.asarray(ref_chroms).astype(str), np.asarray(ref_pos, np.int64)
    out = np.full(len(pos), np.inf)
    for c in np.unique(chroms):
        m = chroms == c
        r = np.sort(ref_pos[ref_chroms == c])
        if not len(r):
            continue
        p = pos[m]
        j = np.searchsorted(r, p)
        left = np.abs(p - r[np.clip(j - 1, 0, len(r) - 1)])
        right = np.abs(r[np.clip(j, 0, len(r) - 1)] - p)
        out[m] = np.minimum(left, right)
    return out


def block_split(chroms, pos, block_bp, heldout_frac, seed, max_tries=1000, select_share=0.5):
    """Assign whole `block_bp` blocks to held-out with probability `heldout_frac`, redrawn until
    every chromosome has sites on both sides. Held-out blocks are then divided by block
    (seed + 1; a block is 'select' with probability select_share, else 'eval'; redrawn until every
    chromosome has both) into 'select' and 'eval'. Returns (role array of 'train'/'select'/'eval', block ids)."""
    chroms = np.asarray(chroms).astype(str)
    blocks = np.char.add(np.char.add(chroms, ":"), (np.asarray(pos, np.int64) // block_bp).astype(str))
    ub, inv = np.unique(blocks, return_inverse=True)
    ucs = np.unique(chroms)

    def every_chrom_has(mask_a, mask_b):
        return all(mask_a[chroms == c].any() and mask_b[chroms == c].any() for c in ucs)

    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        held = (rng.random(len(ub)) < heldout_frac)[inv]
        if every_chrom_has(held, ~held):
            break
    else:
        raise RuntimeError("could not place every chromosome on both sides")
    rng2 = np.random.default_rng(seed + 1)
    for _ in range(max_tries):
        half = (rng2.random(len(ub)) < select_share)[inv]
        if every_chrom_has(held & half, held & ~half):
            break
    else:
        raise RuntimeError("could not give every chromosome both select and eval blocks")
    role = np.where(~held, "train", np.where(half, "select", "eval"))
    return role, inv

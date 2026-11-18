"""s00 inputs: every original input exists and is the upstream file; unit tests pass."""
import hashlib
import os
import subprocess
import urllib.request

from common import C, CheckReport, finish, log


def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _ucsc_lengths(assembly):
    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{assembly}/bigZips/{assembly}.chrom.sizes"
    with urllib.request.urlopen(url, timeout=60) as r:
        return {l.split()[0]: int(l.split()[1]) for l in r.read().decode().splitlines() if l.strip()}


def run(run):
    rep = CheckReport("s00_inputs")
    I = C.INPUTS
    rep.eq("inputs_missing", I.missing(), [])
    rep.info["inputs"] = {k: {"path": v, "bytes": os.path.getsize(v) if os.path.isfile(v) else None}
                          for k, v in vars(I).items()}
    rep.eq("hyena_weights_sha256_equals_huggingface",
           _hash(os.path.join(I.hyena_dir, "weights.ckpt"), "sha256"), C.HYENA_WEIGHTS_SHA256)
    rep.eq("chain_md5_equals_ucsc", _hash(I.chain, "md5"), C.CHAIN_MD5)
    rep.eq("linsight_bw_size_equals_upstream", os.path.getsize(I.linsight_bw), C.LINSIGHT_BW_SIZE)
    # reference genomes: autosome lengths = UCSC chrom.sizes
    from lctx.io.genome import Reference
    for name, asm, path in (("hg19", "hg19", I.hg19), ("hg38", "hg38", I.hg38)):
        ref = Reference(path)
        try:
            ucsc = _ucsc_lengths(asm)
            bad = [c for c in C.AUTOSOMES if ref.lengths.get(c) != ucsc.get(c)]
            rep.eq(f"{name}_autosome_lengths_equal_ucsc", bad, [])
        except OSError as e:                      # offline: record, do not pass silently
            rep.eq(f"{name}_autosome_lengths_equal_ucsc", f"unreachable: {e}", [])
    r = subprocess.run([C.PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=C.ROOT,
                       capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    rep.eq("unit_tests_exit_code", r.returncode, 0, tail)
    log(tail)
    finish(run, "s00_inputs", rep)

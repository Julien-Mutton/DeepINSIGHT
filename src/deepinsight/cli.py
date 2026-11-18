"""One command for the whole project.

    python -m deepinsight paths                 where inputs/outputs are; which inputs are missing
    python -m deepinsight fetch                 download the public inputs (scripts/fetch_data.sh)
    python -m deepinsight build                 compile LINSIGHT C++ (reference binaries) and the C kernel
    python -m deepinsight test [--data]         unit tests (--data: tests on the real input files)
    python -m deepinsight run --rate 0.001      the canonical pipeline s00 ... s09c, s11
            [--from STEP] [--to STEP] [--only STEP] [--read-benchmark]
    python -m deepinsight steps                 list the steps with one line each
    python -m deepinsight cpp-baseline          authors' C++ LINSIGHT on all sites + HGMD AUROC
    python -m deepinsight paper                 compile the report (needs tectonic)
    python -m deepinsight all --rate 0.001      build, test, run, paper

s10_evaluate opens the HGMD labels under a single-read lock (one read per frozen analysis plan).
`run` therefore skips it unless --read-benchmark is given.
"""
import argparse
import os
import shutil
import subprocess
import sys

from lctx import config as C

REPO = C.ROOT
STEPS = ["s00_inputs", "s01_benchmark", "s02_sites", "s03_liftover", "s04_features", "s05_embedder",
         "s06_embed", "s06b_composition", "s07_sae", "s08_design", "s09_train", "s09b_mle",
         "s09c_groundtruth", "s10_evaluate", "s11_report"]
ABOUT = {
    "s00_inputs": "check the original inputs against upstream (sizes, checksums) and run the unit tests",
    "s01_benchmark": "HGMD positives + 3 negative sets -> one panel; published LINSIGHT AUROCs reproduced",
    "s02_sites": "parse the INSIGHT database, sample sites at --rate, exclude 1 kb around variants, split 1-Mb blocks",
    "s03_liftover": "hg19 -> hg38 for every site and variant (HyenaDNA reads hg38)",
    "s04_features": "join the 48 LINSIGHT annotations; recover the published weights as a check",
    "s05_embedder": "load HyenaDNA and test it (next-base accuracy, determinism, causality)",
    "s06_embed": "256-d mean HyenaDNA embedding of the 1,001-bp window of every site and variant",
    "s06b_composition": "4 composition numbers per window (control input)",
    "s07_sae": "TopK sparse autoencoders on train-site embeddings (3 sizes x 5 seeds) + codes",
    "s08_design": "define every model's input x; re-derive sampled rows from the originals",
    "s09_train": "fit LINSIGHT and every variant with the authors' procedure (AdaGrad, 20 epochs)",
    "s09b_mle": "fit every model by exact penalised ML; held-out likelihood test with block bootstrap",
    "s09c_groundtruth": "power: simulate genomes with a planted sequence effect and refit",
    "s10_evaluate": "THE locked HGMD read (AUROC + paired bootstrap); one read per plan",
    "s11_report": "REPORT.md / REPORT.html of the run",
}


def sh(cmd, **kw):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
    r = subprocess.run(cmd, cwd=kw.pop("cwd", REPO), env=_env(), **kw)
    if r.returncode:
        sys.exit(r.returncode)


def _env():
    e = dict(os.environ)
    e.setdefault("DEEPINSIGHT_PY", sys.executable)
    e["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "pipeline"),
                                       e.get("PYTHONPATH", "")])
    return e


def cmd_paths(a):
    print(f"repo     {REPO}\ndata     {C.DATA}   (DEEPINSIGHT_DATA)\noutputs  {C.OUT}   (DEEPINSIGHT_OUT)")
    print(f"python   {C.PY}\nlegacy   {C.PY_LEGACY}   (DEEPINSIGHT_PY_LEGACY; reads the benchmark pickles)")
    print(f"split    held-out fraction {C.HELDOUT_FRAC}, seeds {C.SEEDS}\n")
    from dataclasses import fields
    for f in fields(C.INPUTS):
        p = getattr(C.INPUTS, f.name)
        print(f"  {'ok     ' if os.path.exists(p) else 'MISSING'}  {f.name:12s} {p}")
    miss = C.INPUTS.missing()
    print("\nall inputs present" if not miss else f"\nmissing: {', '.join(miss)} (see data/README.md)")


def cmd_fetch(a):
    sh(["bash", "scripts/fetch_data.sh"])


def cmd_build(a):
    sh(["make", "-C", "third_party/linsight_cpp"])
    from lctx.model import linsight
    linsight._kernel()
    print("built: third_party/linsight_cpp/LINSIGHT-{prep,fit,score}, src/lctx/model/_build/liblinsight.so")


def cmd_test(a):
    sh([sys.executable, "-m", "pytest"] + (["-m", "data"] if a.data else []))


def _select(a):
    i0 = STEPS.index(a.from_) if a.from_ else 0
    i1 = STEPS.index(a.to) if a.to else len(STEPS) - 1
    sel = [a.only] if a.only else STEPS[i0:i1 + 1]
    if "s10_evaluate" in sel and not a.read_benchmark:
        print("note: skipping s10_evaluate (the locked HGMD read); pass --read-benchmark to include it")
        sel = [s for s in sel if s != "s10_evaluate"]
    return sel


def cmd_run(a):
    if C.INPUTS.missing():
        cmd_paths(a)
        sys.exit("inputs missing; run `python -m deepinsight fetch` and see data/README.md")
    for s in _select(a):
        print(f"\n### {s} (rate {a.rate:g}): {ABOUT[s]}", flush=True)
        sh([sys.executable, "pipeline/run_step.py", s, "--rate", str(a.rate)])
    print(f"\nfinished; outputs in {os.path.join(C.OUT, f'rate_{a.rate:g}')}")


def cmd_steps(a):
    for s in STEPS:
        print(f"{s:18s} pipeline/steps/{s}.py  {ABOUT[s]}")


def cmd_cpp(a):
    sh(["bash", "cpp_baseline/run_linsight_cpp.sh"])
    sh([sys.executable, "cpp_baseline/evaluate_cpp.py", "--rate", str(a.rate)])


def cmd_paper(a):
    tec = shutil.which("tectonic") or sys.exit("tectonic not found (conda install -c conda-forge tectonic)")
    sh([tec, "main.tex"], cwd=os.path.join(REPO, "paper"))


def cmd_all(a):
    cmd_build(a)
    a.data = False
    cmd_test(a)
    cmd_run(a)
    if shutil.which("tectonic"):
        cmd_paper(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deepinsight", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("paths").set_defaults(f=cmd_paths)
    sub.add_parser("fetch").set_defaults(f=cmd_fetch)
    sub.add_parser("build").set_defaults(f=cmd_build)
    sub.add_parser("steps").set_defaults(f=cmd_steps)
    t = sub.add_parser("test")
    t.add_argument("--data", action="store_true", help="tests on the real input files (slow)")
    t.set_defaults(f=cmd_test)
    for name, fn in (("run", cmd_run), ("all", cmd_all)):
        r = sub.add_parser(name)
        r.add_argument("--rate", type=float, default=0.001, help="share of INSIGHT sites (0.001 = 0.1%%)")
        r.add_argument("--from", dest="from_", choices=STEPS)
        r.add_argument("--to", choices=STEPS)
        r.add_argument("--only", choices=STEPS)
        r.add_argument("--read-benchmark", action="store_true", help="include s10_evaluate (locked HGMD read)")
        r.set_defaults(f=fn)
    c = sub.add_parser("cpp-baseline")
    c.add_argument("--rate", type=float, default=0.001, help="run whose s01/s04 panel files to score")
    c.set_defaults(f=cmd_cpp)
    sub.add_parser("paper").set_defaults(f=cmd_paper)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()

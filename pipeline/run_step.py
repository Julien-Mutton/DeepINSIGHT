#!/usr/bin/env python3
"""Run one pipeline step:  python pipeline/run_step.py <step> [--rate 0.001]"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import STEPS, Run, set_log, log   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--rate", type=float, default=0.001)
    a = ap.parse_args()
    run = Run(a.rate)
    set_log(os.path.join(run.dir, "logs", f"{a.step}.log"))
    log(f"=== {a.step}  run {run.name}")
    importlib.import_module(f"steps.{a.step}").run(run)
    log(f"=== {a.step} done")


if __name__ == "__main__":
    main()

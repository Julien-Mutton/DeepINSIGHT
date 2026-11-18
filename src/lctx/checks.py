"""Check reports and gates.

Every block writes `checks.json` = {block, ok, time, checks[], info{}}. A check is
(name, value, expectation, passed). A *control* is a check run on purpose-broken input;
it passes when the broken input is detected, i.e. it demonstrates that the real check
has the power to fail.
"""
import json
import math
import os
import time


def _jsonable(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


class CheckReport:
    def __init__(self, block):
        self.block = block
        self.checks = []
        self.info = {}

    def _add(self, name, value, expect, passed, detail="", control=False):
        self.checks.append({"name": name, "value": _jsonable(value), "expect": expect,
                            "passed": bool(passed), "control": control, "detail": detail})
        return bool(passed)

    def ge(self, name, value, lo, detail="", control=False):
        return self._add(name, value, f">= {lo}", value is not None and value >= lo, detail, control)

    def le(self, name, value, hi, detail="", control=False):
        return self._add(name, value, f"<= {hi}", value is not None and value <= hi, detail, control)

    def eq(self, name, value, target, detail="", control=False):
        return self._add(name, value, f"== {target}", value == target, detail, control)

    def close(self, name, value, target, tol, detail="", control=False):
        return self._add(name, value, f"{target} ± {tol}", abs(value - target) <= tol, detail, control)

    def true(self, name, cond, detail="", control=False):
        return self._add(name, bool(cond), "== True", bool(cond), detail, control)

    def merge(self, other, prefix=""):
        for c in other.checks:
            self.checks.append({**c, "name": prefix + c["name"]})
        self.info.update({prefix + k: v for k, v in other.info.items()})
        return self

    @property
    def ok(self):
        return all(c["passed"] for c in self.checks)

    def failures(self):
        return [c for c in self.checks if not c["passed"]]

    def summary(self):
        n, p = len(self.checks), sum(c["passed"] for c in self.checks)
        lines = [f"[{self.block}] {'PASS' if self.ok else 'FAIL'} ({p}/{n})"]
        for c in self.checks:
            tag = "ok " if c["passed"] else "XX "
            ctl = " (control)" if c["control"] else ""
            lines.append(f"  {tag} {c['name']:<58} {str(c['value']):>14}  {c['expect']}{ctl}")
        return "\n".join(lines)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"block": self.block, "ok": self.ok, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "checks": self.checks, "info": self.info}, f, indent=1, default=_jsonable)
        return path


class GateError(RuntimeError):
    pass


def require(path):
    """Refuse to continue unless the upstream block's checks.json exists and passed."""
    if not os.path.exists(path):
        raise GateError(f"upstream gate missing: {path}")
    with open(path) as f:
        d = json.load(f)
    if not d.get("ok"):
        bad = [c["name"] for c in d["checks"] if not c["passed"]]
        raise GateError(f"upstream gate failed: {path}: {bad[:5]}")
    return d

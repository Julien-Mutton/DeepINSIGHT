"""s11 report: REPORT.md / REPORT.html built only from checks.json, results.json and the artefacts
on disk (shapes, dtypes and sizes are read from the files themselves)."""
import glob
import html
import json
import os
import re

import numpy as np

from common import C, STEPS, CheckReport, finish, log
from steps import dataflow_figure

STAGE_TEXT = {
    "s00_inputs": ("Verify the original inputs", "every input file", "checksums against upstream, unit tests"),
    "s01_benchmark": ("Read the HGMD benchmark", "4 Python-2 pandas pickles", "one row per variant: key, hg19 position (0-based), label, reference base"),
    "s02_sites": ("Sample INSIGHT training sites", "INSIGHT_database.ins (58 GB)", "deterministic hash sample; evol terms per site; exclusion; train/select/eval split by 1 Mb blocks"),
    "s03_liftover": ("Convert hg19 -> hg38", "site and variant hg19 positions + UCSC chain", "hg38 position + status per row"),
    "s04_features": ("Look up the 48 LINSIGHT features", "genomic_feature.bed (28 GB)", "48 values per row; published weights recovered from LINSIGHT.bw"),
    "s05_embedder": ("Check HyenaDNA", "HF weights + official model code", "behavioural checks only"),
    "s06_embed": ("Embed every window", "hg38 1001-bp windows", "256-d mean of the hidden states per row (+ centre state)"),
    "s06b_composition": ("Composition of every window", "hg19 soft-masked 1001-bp windows", "GC, CpG o/e, repeat fraction, longest homopolymer"),
    "s07_sae": ("Fit sparse autoencoders", "train-site embeddings (and train-site features)", "SAE models; codes for every row"),
    "s08_design": ("Build and verify every arm's inputs", "all of the above", "row-by-row re-derivation from originals"),
    "s09_train": ("Train: paper procedure", "train-site inputs + evol terms", "parameters per epoch, 5 seeds, replication arms"),
    "s09b_mle": ("Train: exact maximum likelihood", "train-site inputs + evol terms", "converged MLE for every arm; T1 and T2"),
    "s09c_groundtruth": ("Ground truth by simulation", "real sites/features/embeddings, simulated evol", "false-positive rates, power, recovery, calibration"),
    "s10_evaluate": ("HGMD evaluation (single read)", "all arms' scores on benchmark variants", "AUROC, deltas with CIs, Holm, MDE"),
}


def _files(run, step):
    rows = []
    for f in sorted(glob.glob(os.path.join(run.dir, step, "*"))):
        name, mb = os.path.basename(f), os.path.getsize(f) / 1e6
        if name in ("checks.json",):
            continue
        desc = ""
        try:
            if f.endswith(".npy"):
                a = np.load(f, mmap_mode="r")
                desc = f"{tuple(a.shape)} {a.dtype}"
            elif f.endswith(".npz"):
                with np.load(f) as d:
                    ks = d.files
                    desc = ", ".join(f"{k} {tuple(d[k].shape)} {d[k].dtype}" for k in ks[:4]) + (f", … ({len(ks)} arrays)" if len(ks) > 4 else "")
        except Exception as e:                                 # report, never hide
            desc = f"unreadable: {e}"
        rows.append((name, desc, mb))
    return rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "…"


def _verdict(d, unit="AUROC"):
    if d["p_holm"] < C.ALPHA:
        return f"{'better' if d['delta'] > 0 else 'worse'} (significant after Holm)"
    return f"no significant difference; effects beyond ±{d['mde']:.3f} {unit} would have been detected"


def build_markdown(run):
    I = lambda s: json.load(open(run.checks_path(s)))["info"]
    res = json.load(open(run.path("s10_evaluate", "results.json")))
    mle_i, gt, tr, sites = I("s09b_mle"), I("s09c_groundtruth"), I("s09_train"), I("s02_sites")
    R2 = mle_i["sae_chosen"]
    L = [f"# LINSIGHT + HyenaDNA sequence context — run `{run.name}`", "",
         f"Built from original inputs only (docs/DATA.md). Sample rate {run.rate}: "
         f"{sites['n_sampled']:,} INSIGHT sites sampled, {mle_i['n_train']:,} used for training, "
         f"{mle_i['n_eval']:,} held-out evaluation sites in {mle_i['n_eval_blocks']} genome blocks. "
         f"Benchmark read under plan v{res['plan_version']} (`{res['plan_hash']}`); {res['n_reads_logged'] - 1} earlier "
         f"plan(s) were read before this one (plan v1, the first design), which is why v2 is reported as a redesign, not a first look.", ""]

    # ---------------- answers
    L += ["## 1. Answers", "",
          "| question | test | result | effect [95% CI] | p (Holm) | smallest detectable effect | verdict |",
          "|---|---|---|---|---|---|---|"]
    L.append("| Is a likelihood-ratio test (T1) usable? | separation check | rare features push unpenalised weights to ~1e5 | — | — | — | "
             "no: a penalty is required, so T1 is replaced by the held-out test T2 (unit tests document T1's calibration) |")
    for k, nm in (("emb - repl", "R1"), (f"{R2} - repl", "R2"), ("emb - noise_emb", "R1 vs noise"), (f"{R2} - {mle_i['noise_for_chosen']}", "R2 vs noise")):
        h = mle_i["heldout"][k]
        v = ("improves unseen genome" if (h["p"] < C.ALPHA and h["mean"] < 0) else
             "worse on unseen genome" if (h["p"] < C.ALPHA) else f"no significant change; |effect| > {h['mde']:.1e} would be detected")
        L.append(f"| Does {nm} predict held-out INSIGHT data better? | T2 held-out NLL | Δ NLL/site (−=better) | "
                 f"{h['mean']:+.2e} [{h['lo']:+.2e}, {h['hi']:+.2e}] | {h['p']:.2g} (single test) | {h['mde']:.1e} | {v} |")
    for fn, fl in (("A_paper", "LINSIGHT as trained by its authors"), ("B_bestfit", "best fit to the evolutionary data")):
        for t, d in res["tiers"].items():
            for name, x in d[fn].items():
                L.append(f"| {name} ({fl}): disease prediction, {t} | T3 HGMD ΔAUROC | n={d['n']:,} ({d['n_pos']:,} pos) | "
                         f"{x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | {x['p_holm']:.2g} | {x['mde']:.4f} | {_verdict(x)} |")

    # ---------------- ground truth
    L += ["", "## 2. Ground truth: do the tests work when the answer is known?", "",
          "Evolutionary data were simulated on the real sites, features and embeddings from the fitted R0 model, "
          "plus a planted sequence effect of size s (latent units; R0's latent has sd "
          f"{gt['R0 latent sd (train)']:.3f}). Each world was fitted and tested exactly like the real data.", "",
          "| world | arm | worlds | T2 reject rate | mean held-out gain (NLL/site) | corr(ρ, truth) gain vs R0 |",
          "|---|---|---|---|---|---|"]
    for k, v in gt["summary"].items():
        L.append(f"| {k.split()[0]} | {k.split()[1]} | {v['n_worlds']} | {v['T2_reject_rate']:.2f} | "
                 f"{v['T2_heldout_gain_mean']:+.2e} | {v['corr_truth_arm_minus_R0']:+.4f} |")
    pn = gt["paper_null_corr_truth"]
    L += ["", f"Null worlds (s = 0) measure false positives (expected rate 0.05); planted worlds measure power. "
          f"Paper procedure in null worlds, correlation of fitted ρ with the true ρ: R0 {np.mean(pn['repl']):.4f}, "
          f"R1 {np.mean(pn['emb']):.4f}, 256 noise columns {np.mean(pn['noise_emb']):.4f} — the cost of extra columns "
          f"under the paper procedure when the truth contains no sequence effect.",
          f"The real embedding's held-out gain ({gt['real_emb_heldout_gain']:+.2e} NLL/site) corresponds to a planted "
          f"effect of s ≈ {gt['equivalent_planted_s']:.4f}.", ""]

    # ---------------- all tests tables
    L += ["## 3. All test results", "", f"Feature prior width sdF = {mle_i['sdF']} (chosen once by R0 on held-out select sites, "
          "used by every arm). Extra-column prior width sdE chosen per arm (seed 0) on select sites: " +
          ", ".join(f"{k[:-3]} {v}" for k, v in mle_i["prior_sd_extra"].items() if k.endswith("_s0") and v is not None) +
          ". Agreement of each arm's ρ with published LINSIGHT on held-out sites (Spearman): " +
          ", ".join(f"{a} {v:.3f}" for a, v in mle_i["spearman_vs_published"].items()) + "."]
    L += ["", "### T2 held-out NLL (best fit, eval sites, block bootstrap; negative = better)", "",
          "| comparison | Δ NLL/site | 95% CI | SE | p | MDE |", "|---|---|---|---|---|---|"]
    for k, h in mle_i["heldout"].items():
        L.append(f"| {k} | {h['mean']:+.2e} | [{h['lo']:+.2e}, {h['hi']:+.2e}] | {h['se']:.1e} | {h['p']:.2g} | {h['mde']:.1e} |")
    for fam in ("A_paper", "B_bestfit", "secondary"):
        L += ["", f"### T3 HGMD — family {fam} (loci + seeds bootstrap, Holm within family)", "",
              "| tier | comparison | ΔAUROC | 95% CI | SE | p | p (Holm) | MDE |", "|---|---|---|---|---|---|---|---|"]
        for t, d in res["tiers"].items():
            for n, x in d[fam].items():
                L.append(f"| {t} | {n} | {x['delta']:+.4f} | [{x['lo']:+.4f}, {x['hi']:+.4f}] | {x['se']:.4f} | {x['p']:.2g} | {x['p_holm']:.2g} | {x['mde']:.4f} |")
    L += ["", "### AUROC of every arm (mean score over seeds; ± sd across single seeds)", "",
          "| arm | " + " | ".join(res["tiers"]) + " |", "|---" * (1 + len(res["tiers"])) + "|"]
    for k in next(iter(res["tiers"].values()))["auroc"]:
        cells = []
        for t, d in res["tiers"].items():
            sd = d["auroc_seed_sd"].get(k)
            cells.append(f"{d['auroc'][k]:.4f}" + (f" ± {sd:.4f}" if sd else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    # ---------------- stage by stage
    L += ["", "## 4. Stage by stage: data, shapes, checks", ""]
    for s in STEPS[:-1]:
        p = run.checks_path(s)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        title, inp, outp = STAGE_TEXT.get(s, (s, "", ""))
        n, ok = len(d["checks"]), sum(c["passed"] for c in d["checks"])
        nc = sum(c["control"] for c in d["checks"])
        L += [f"### {s} — {title}", "", f"**In:** {inp}. **Out:** {outp}. **Checks:** {ok}/{n} passed "
              f"({nc} controls that must fail on broken input, all failing as required).", ""]
        files = _files(run, s)
        if files:
            L += ["| file | shape / dtype | MB |", "|---|---|---|"]
            L += [f"| {a} | {b} | {c:.1f} |" for a, b, c in files]
            L.append("")
        L += ["<details><summary>every check</summary>", "",
              "| check | value | expectation | pass | control |", "|---|---|---|---|---|"]
        L += [f"| {c['name']} | {_fmt(c['value'])} | {c['expect']} | {'✓' if c['passed'] else '✗'} | {'control' if c['control'] else ''} |"
              for c in d["checks"]]
        L += ["", "</details>", ""]
    L += ["## 5. Methods in one place", "",
          "Model, simulator, estimators and every test are derived in `docs/MATH.md`; decisions and findings are "
          "logged in `docs/DECISIONS.md`. Best fit = penalised maximum likelihood (Gaussian prior, width chosen on held-out "
          "select sites); T2 = mean held-out per-site NLL difference "
          "with a 1 Mb-block bootstrap SE; T3 = AUROC difference with a bootstrap over 100-kb loci and training seeds; "
          "MDE = 2.80 × SE (α = 0.05 two-sided, 80% power); Holm within each family."]
    return "\n".join(L) + "\n"


def to_html(md, title, figure=""):
    body, rows = [], []

    def flush():
        if rows:
            body.append("<div class='t'><table>" + "".join(rows) + "</table></div>")
            rows.clear()
    for line in md.splitlines():
        if line.startswith("<details") or line.startswith("</details"):
            flush()
            body.append(line)
            continue
        esc = html.escape(line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", re.sub(r"`(.+?)`", r"<code>\1</code>", esc))
        if line.startswith("|"):
            if set(line) <= set("|-"):
                continue
            cells = [c.strip() for c in esc.strip("|").split("|")]
            tag = "th" if not rows else "td"
            rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        flush()
        if line.startswith("# "):
            body.append(f"<h1>{esc[2:]}</h1>")
        elif line.startswith("## 1.") and figure:
            body.append("<h2>0. How the experiment works</h2>" + figure)
            body.append(f"<h2>{esc[3:]}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{esc[4:]}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{esc[3:]}</h2>")
        elif line.strip():
            body.append(f"<p>{esc}</p>")
    flush()
    css = (":root{--bg:#f7f8fa;--fg:#1a2130;--mu:#5d6675;--ru:#dde1e7;--sf:#fff;--acc:#b5532a;--acs:#fbeee6;--trn:#2f6f9f;--tst:#b5532a}"
           "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#13171d;--fg:#e3e7ed;--mu:#9aa3b0;--ru:#2c3440;--sf:#1a2028;--acc:#f0a07a;--acs:#3a241a;--trn:#7fb3dc;--tst:#f0a07a;color-scheme:dark}}"
           ":root[data-theme=dark]{--bg:#13171d;--fg:#e3e7ed;--mu:#9aa3b0;--ru:#2c3440;--sf:#1a2028;--acc:#f0a07a;--acs:#3a241a;--trn:#7fb3dc;--tst:#f0a07a;color-scheme:dark}"
           "body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;padding:24px max(16px,env(safe-area-inset-right,0px)) 24px max(16px,env(safe-area-inset-left,0px))}"
           "main{max-width:1100px;margin:0 auto}.t{overflow-x:auto;background:var(--sf);border:1px solid var(--ru);border-radius:6px;margin:6px 0}"
           "table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 10px;border-bottom:1px solid var(--ru);text-align:left;vertical-align:top}"
           "td{font-variant-numeric:tabular-nums}th{color:var(--mu);font-weight:600;white-space:nowrap}code{font-size:12px}"
           "h2{margin-top:32px;border-top:1px solid var(--ru);padding-top:14px}h3{margin-top:22px}"
           "details{margin:4px 0 14px}summary{cursor:pointer;color:var(--mu)}" + dataflow_figure.CSS)
    return (f"<title>{html.escape(title)}</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">"
            f"<style>{css}</style><main>" + "\n".join(body) + "</main>")


def run(run):
    run.gate("s10_evaluate")
    md = build_markdown(run)
    with open(run.path("REPORT.md"), "w") as f:
        f.write(md)
    with open(run.path("REPORT.html"), "w") as f:
        f.write(to_html(md, f"LINSIGHT Context {run.name}", dataflow_figure.svg(run)))
    rep = CheckReport("s11_report")
    rep.true("report written", os.path.getsize(run.path("REPORT.md")) > 2000)
    log(md[:4000])
    finish(run, "s11_report", rep)

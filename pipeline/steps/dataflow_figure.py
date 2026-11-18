"""The report's walkthrough for a reader who does not know the project (plain language, this run's
numbers). HTML/CSS using the page's theme tokens."""
import json


def _k(x):
    return f"{x / 1e6:.2f} million" if x >= 1e6 else f"{round(x / 1e3):,} thousand"


def svg(run):  # name kept for the caller; returns HTML
    info = lambda s: json.load(open(run.checks_path(s)))["info"]
    S, L, M = info("s02_sites"), info("s03_liftover"), info("s09b_mle")
    res = json.load(open(run.path("s10_evaluate", "results.json")))
    n_all = sum(S["counts_all_sites"].values())
    n_panel = sum(L["benchmark: status_counts"].values())
    R2 = M["sae_chosen"]
    d_sae, k_sae = R2[3:].split("k")
    one_in = round(n_all / S["n_sampled"])
    u = res["tiers"]["unmatched"]["auroc"]
    pr = res["tiers"]["unmatched"]["A_paper"]
    pb = res["tiers"]["unmatched"]["B_bestfit"]
    h1 = M["heldout"]["emb - repl"]

    def card(tag, title, body, cls=""):
        return f'<div class="fl-card {cls}"><span class="fl-tag">{tag}</span><h4>{title}</h4>{body}</div>'

    def step(n, t):
        return f'<div class="fl-step"><span class="fl-no">{n}</span><h3>{t}</h3></div>'

    recipes = [("R0", "A", "LINSIGHT's own 48 facts: the baseline", ""),
               ("R1", "A + B", "adds the language model's reading", ""),
               ("R2", "A + C", "adds the compressed reading", ""),
               ("noise", "A + random numbers", "same number of columns as B or C: what does adding <i>anything</i> cost?", "fl-ctl"),
               ("shuffled", "A + B (or C) of another position", "real values, wrong position: is it the content or the shape?", "fl-ctl"),
               ("composition", "A + GC, CpG, repeats", "is HyenaDNA just measuring simple composition?", "fl-ctl"),
               ("feature-SAE", "A re-expressed by an SAE", "no new information at all: what does extra flexibility do?", "fl-ctl"),
               ("positive", "A, one fact moved to the new slot", "must equal R0 exactly: proves the new slot works", "fl-ctl")]
    rec = "".join(f'<div class="{c}"><b>{a}</b><span>{b}</span><em>{e}</em></div>' for a, b, e, c in recipes)

    return f"""
<div class="fl" role="group" aria-label="How the experiment works, step by step">
  <div class="fl-q"><span class="fl-qk">The question</span>
    LINSIGHT scores how strongly evolution protects each letter of the human genome, from 48 known facts about
    that spot. <b>Does adding what a DNA language model (HyenaDNA) reads in the surrounding DNA make those scores
    better?</b> "Better" is checked three ways: does it carry any signal at all, does it predict unseen genome
    better, and does it find disease-causing changes better.</div>

  {step(1, "Two separate sets of DNA positions")}
  <div class="fl-two">
    {card("TRAINING DATA · used to learn", f"{S['n_sampled']:,} random genome positions",
      f"<p>One in {one_in:,} of the {n_all / 1e9:.2f} billion positions in the INSIGHT database. For each: <b>how much people vary there and how it differs from other apes</b> (protected positions vary less). <b>No disease information.</b></p><ul><li>Positions within 1,000 letters of a test variant are removed, so the test is never seen in training.</li><li>The genome is cut into 1-million-letter blocks: half the blocks train the models ({_k(M['n_train'])} positions), half are held back ({_k(M['n_eval'])} positions) to test on unseen genome.</li></ul>", "fl-tr")}
    {card("TEST DATA · used once, at the end", f"{n_panel:,} known DNA changes",
      "<p>The LINSIGHT paper's benchmark: <b>1,614 disease-causing changes</b> outside genes (HGMD) versus harmless common changes, in three comparisons of increasing difficulty: harmless changes taken anywhere, matched by distance to a gene, or from the same neighbourhoods.</p>", "fl-te")}
  </div>
  <div class="fl-join"><span>every position from both sets goes through exactly the same steps 2 and 3</span></div>

  {step(2, "Describe each position")}
  <div class="fl-three">
    {card("A · what LINSIGHT already uses", "48 known facts", "<p>Conserved across species? Protein-binding site? Enhancer? Near a gene? Each yes/no.</p><div class='fl-ex'><code>0 1 0 0 1 0 … 1</code><small>48 values</small></div>")}
    {card("B · new: the language model's reading", "HyenaDNA summary", "<p>The 1,001 DNA letters centred on the position are read by HyenaDNA; its output is averaged into 256 numbers. (Locations are first converted to the genome version HyenaDNA learned, hg19 → hg38, checked letter by letter.)</p><div class='fl-ex'><code>…GATTACA<mark>C</mark>TTAGGC…</code><small>→ 256 numbers</small></div>")}
    {card("C · new: compressed version of B", "Sparse summary", f"<p>A sparse autoencoder, learned from the training positions only, rewrites the 256 numbers as {d_sae} features with {k_sae} switched on at a time.</p><div class='fl-ex'><code>0 0 2.1 0 0.7 0 …</code><small>{d_sae} values</small></div>")}
  </div>

  {step(3, "Recipes compared — each control answers one question")}
  <div class="fl-recipes">{rec}</div>

  {step(4, "Learn a score for each recipe, two ways (A and B)")}
  <div class="fl-two">
    {card("A · LINSIGHT as its authors train it", "Original training recipe", "<p>Rebuilt from LINSIGHT's C++ code and checked to give identical numbers. It stops early, which keeps it close to published LINSIGHT. Answers: <b>does sequence improve LINSIGHT?</b></p>", "fl-tr")}
    {card("B · best fit to the evolutionary data", "Penalised maximum likelihood", "<p>Finds the weights that make the training positions' data most probable, with a mild pull towards zero (needed: some rare facts would otherwise get infinite weight). The 48 facts get the same pull in every recipe; the new columns get their own, tuned on held-back positions, so a useless addition shrinks back to exactly R0. Proven on simulated data to recover a known answer. Answers: <b>does sequence carry information about selection?</b></p>", "fl-tr")}
  </div>

  {step(5, "Test — each result with its error bars and the smallest effect it could detect")}
  <div class="fl-three">
    {card("Why not a simple significance test of the fit?", "Not valid here", "<p>The classic likelihood-ratio test needs an unpenalised fit, which does not exist on the real data (rare facts get infinite weight). So the evidence comes from unseen data instead (T2, T3), with its false-alarm rate measured in step 6.</p>")}
    {card("T2 · predicts unseen genome?", "Held-out test", "<p>Score how probable the held-back half of the genome is under each recipe. Error bars resample whole genome blocks.</p>")}
    {card("T3 · finds disease changes?", "HGMD AUROC", "<p>Chance a disease change scores above a harmless one (0.5 = coin flip). Error bars resample genome regions and training runs; 18 main comparisons corrected for multiple testing. Labels used once.</p>", "fl-te")}
  </div>

  {step(6, "Proof the tests work: simulated worlds with a known answer")}
  <div class="fl-card fl-wide">
    <p>On the real positions, facts and HyenaDNA readings, the evolutionary data are <b>simulated</b> from LINSIGHT's
    own model, once with no sequence effect and several times with a planted HyenaDNA effect of known size. The
    whole analysis is run on each. That measures how often each test raises a false alarm and how large an
    effect it reliably finds, so a "no effect" result has a stated size limit.</p></div>

  {step(7, "Headline (unmatched comparison; full tables below)")}
  <div class="fl-res">
    <div><span>published LINSIGHT</span><b>{u['published']:.3f}</b></div>
    <div class="fl-r0"><span>R0 (recipe A / B)</span><b>{u['paper:repl']:.3f} / {u['mle:repl']:.3f}</b></div>
    <div><span>R1 + language model (A / B)</span><b>{u['paper:emb']:.3f} / {u['mle:emb']:.3f}</b></div>
    <div><span>R2 + compressed (A / B)</span><b>{u['paper:' + R2]:.3f} / {u['mle:' + R2]:.3f}</b></div>
    <div class="fl-ctl"><span>R1 − R0, recipe A (95% CI)</span><b>{pr['R1 - R0']['delta']:+.3f}</b><span>[{pr['R1 - R0']['lo']:+.3f}, {pr['R1 - R0']['hi']:+.3f}]</span></div>
  </div>
  <p class="fl-take">Held-out genome (T2): R1 changes the per-position fit by {h1['mean']:+.1e}
  (95% CI {h1['lo']:+.1e} to {h1['hi']:+.1e}; an effect of {h1['mde']:.1e} would have been detected).
  Section 1 states every answer with its test, error bars and detectable size.</p>
</div>"""


CSS = (
    ".fl{display:grid;gap:12px;margin:6px 0 4px}"
    ".fl h3{margin:0;font-size:17px}.fl h4{margin:2px 0 6px;font-size:15px}.fl p{margin:0 0 6px;max-width:75ch}"
    ".fl ul{margin:4px 0 0;padding-left:18px;display:grid;gap:4px;font-size:13.5px}"
    ".fl-q{background:var(--sf);border:1px solid var(--ru);border-radius:8px;padding:14px 16px;font-size:15px}"
    ".fl-qk{display:block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mu);margin-bottom:4px}"
    ".fl-step{display:flex;align-items:center;gap:10px;margin-top:10px}"
    ".fl-no{flex:none;width:28px;height:28px;border-radius:50%;display:grid;place-items:center;"
    "background:var(--fg);color:var(--bg);font-weight:700;font-size:14px}"
    ".fl-two{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
    ".fl-three{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
    ".fl-card{background:var(--sf);border:1px solid var(--ru);border-radius:8px;padding:12px 14px;min-width:0;font-size:14px}"
    ".fl-tr{border-top:4px solid var(--trn)}.fl-te{border-top:4px solid var(--tst)}"
    ".fl-tag{display:block;font-size:10.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--mu)}"
    ".fl-tr .fl-tag{color:var(--trn)}.fl-te .fl-tag{color:var(--tst)}"
    ".fl-ex{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;background:var(--bg);border-radius:5px;padding:6px 8px;margin:6px 0}"
    ".fl-ex code{font-size:13px}.fl-ex small{color:var(--mu)}"
    ".fl-ex mark{background:var(--acs);color:var(--acc);font-weight:700;padding:0 1px}"
    ".fl-join{text-align:center;color:var(--mu);font-size:13px;position:relative}"
    ".fl-join span{background:var(--bg);padding:0 10px;position:relative}"
    ".fl-join:before{content:'';position:absolute;left:0;right:0;top:50%;border-top:1px dashed var(--ru)}"
    ".fl-recipes{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}"
    ".fl-recipes div,.fl-res div{background:var(--sf);border:1px solid var(--ru);border-radius:8px;padding:10px;display:grid;gap:3px;min-width:0}"
    ".fl-recipes b{font-size:15px}.fl-recipes span{font-weight:600;font-size:13.5px}.fl-recipes em{font-style:normal;color:var(--mu);font-size:12.5px}"
    ".fl-ctl{border-style:dashed!important}"
    ".fl-res{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}"
    ".fl-res span{color:var(--mu);font-size:12.5px}.fl-res b{font-size:20px;font-variant-numeric:tabular-nums}"
    ".fl-r0{border-color:var(--trn)!important}"
    ".fl-take{font-size:14.5px;margin:2px 0 0}"
    "@media (max-width:760px){.fl-two,.fl-three{grid-template-columns:1fr}.fl-recipes,.fl-res{grid-template-columns:1fr 1fr}}"
)

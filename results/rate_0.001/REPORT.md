# LINSIGHT + HyenaDNA sequence context — run `rate_0.001`

Built from original inputs only (docs/DATA.md); every block gated by its checks (§4). Training sites: 1,212,074 train / 72,029 select / 78,009 eval (sample rate 0.001). HGMD used once, for evaluation (plan `89b069b68c299e39`).

## 1. HGMD benchmark (AUROC)

| tier | n (pos) | paper | published | R0 repl | R1 emb | R2 sae64k16 | noise_emb | noise64 | posctl |
|---|---|---|---|---|---|---|---|---|---|
| unmatched | 161329 (1564) | 0.897 | 0.8977 | 0.8870 | 0.7245 | 0.8240 | 0.8272 | 0.8495 | 0.8844 |
| matched_by_tss | 16910 (1564) | 0.759 | 0.7617 | 0.7484 | 0.6832 | 0.7312 | 0.7187 | 0.7298 | 0.7452 |
| matched_by_region | 6286 (1564) | 0.66 | 0.6645 | 0.6643 | 0.6466 | 0.6651 | 0.6544 | 0.6571 | 0.6627 |

## 2. Differences (paired cluster bootstrap, 95% CI)

| tier | R1 - R0 | R2 - R0 | R1 - noise_emb | R2 - noise64 | R2 - R1 | posctl - R0 | R0 - published |
|---|---|---|---|---|---|---|---|
| unmatched | -0.1625 [-0.1818, -0.1430] ↓ | -0.0629 [-0.0747, -0.0518] ↓ | -0.1027 [-0.1226, -0.0816] ↓ | -0.0254 [-0.0365, -0.0148] ↓ | +0.0995 [+0.0860, +0.1128] ↑ | -0.0026 [-0.0035, -0.0017] ↓ | -0.0107 [-0.0149, -0.0068] ↓ |
| matched_by_tss | -0.0652 [-0.0832, -0.0482] ↓ | -0.0172 [-0.0270, -0.0079] ↓ | -0.0356 [-0.0549, -0.0160] ↓ | +0.0013 [-0.0090, +0.0115] n.s. | +0.0480 [+0.0361, +0.0605] ↑ | -0.0032 [-0.0040, -0.0024] ↓ | -0.0133 [-0.0193, -0.0076] ↓ |
| matched_by_region | -0.0177 [-0.0307, -0.0044] ↓ | +0.0009 [-0.0062, +0.0081] n.s. | -0.0079 [-0.0228, +0.0070] n.s. | +0.0080 [-0.0003, +0.0163] n.s. | +0.0186 [+0.0091, +0.0280] ↑ | -0.0016 [-0.0023, -0.0009] ↓ | -0.0003 [-0.0079, +0.0069] n.s. |

## 3. Held-out INSIGHT likelihood and agreement with published LINSIGHT (no HGMD)

| arm | select NLL/site | eval NLL/site | Spearman vs LINSIGHT.bw | sd of extra-column latent (benchmark) | finite |
|---|---|---|---|---|---|
| repl | 0.0774025 | 0.0785261 | 0.9229 | 0.0000 | True |
| emb | 0.0774059 | 0.0785251 | 0.3802 | 0.2244 | True |
| sae32k8 | 0.0774003 | 0.0785300 | 0.6369 | 0.1238 | True |
| sae64k16 | 0.0773989 | 0.0785305 | 0.5595 | 0.1519 | True |
| sae128k32 | 0.0774151 | 0.0785292 | 0.4779 | 0.3345 | True |
| noise_emb | 0.0773936 | 0.0785251 | 0.6013 | 0.1367 | True |
| noise32 | 0.0774007 | 0.0785261 | 0.7127 | 0.0534 | True |
| noise64 | 0.0774045 | 0.0785261 | 0.6739 | 0.0739 | True |
| noise128 | 0.0774019 | 0.0785205 | 0.6450 | 0.0987 | True |
| posctl | 0.0773767 | 0.0784938 | 0.9238 | 0.0942 | True |

SAE config chosen by held-out select NLL: **sae64k16** (matched noise arm `noise64`).
R0 ρ-weights vs published weights (recovered exactly from LINSIGHT.bw): Pearson 0.860.

## 4. Block checks

| step | result | checks | controls |
|---|---|---|---|
| s00_inputs | PASS | 7/7 | 0 |
| s01_benchmark | PASS | 15/15 | 2 |
| s02_sites | PASS | 21/21 | 2 |
| s03_liftover | PASS | 17/17 | 4 |
| s04_features | PASS | 10/10 | 3 |
| s05_embedder | PASS | 10/10 | 2 |
| s06_embed | PASS | 9/9 | 2 |
| s07_sae | PASS | 36/36 | 0 |
| s08_design | PASS | 82/82 | 32 |
| s09_train | PASS | 13/13 | 0 |
| s10_evaluate | PASS | 3/3 | 0 |

Per-seed AUROCs are in `s10_evaluate/results.json`; every check with value and threshold is in `<step>/checks.json`.

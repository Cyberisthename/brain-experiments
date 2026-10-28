# XOR_DELAY confirmatory replication results

Prereg: `PREREG_xor_rep.md` (locked 2026-10-28 15:47 America/New_York, before eval).
Rows kept: 624. Eval seeds: 200–223 (n=24).
Parent exploratory (seeds 0–7, NOT confirmatory): FRANKEN 0.525 vs ER 0.213 vs SHUF −0.159.

## Ridge α (tuned on 90–92 only, XOR d=5 REALMYELIN)

| Arm | α |
|---|---:|
| `FRANKEN` | 1e-08 |
| `ER` | 1e-06 |
| `SHUF` | 0.01 |
| `DENS` | 0.01 |
| `FLY_CX` | 1e-08 |
| `HUMAN` | 0.0001 |
| `SPEC` | 1e-08 |

## Primary: XOR_DELAY d=5, REALMYELIN — means

| Arm | mean ± sem | n |
|---|---:|---:|
| `FRANKEN` | 0.4787 ± 0.0183 | 24 |
| `ER` | 0.1867 ± 0.0545 | 24 |
| `SHUF` | 0.1306 ± 0.0042 | 24 |
| `DENS` | 0.1490 ± 0.0040 | 24 |
| `FLY_CX` | 0.6335 ± 0.0075 | 24 |
| `HUMAN` | 0.1269 ± 0.0033 | 24 |
| `SPEC` | 0.3411 ± 0.0315 | 24 |

Metric ↑ better (1 − NRMSE).

## Primary hypothesis (Bonferroni ×2, one-sided Wilcoxon)

| Contrast | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf |
|---|---:|---:|---:|---:|---:|
| FRANKEN > ER | +0.2920 | [+0.1833, +0.4007] | 1.135 | 2.563e-06 | 5.126e-06 |
| FRANKEN > SHUF | +0.3480 | [+0.3096, +0.3865] | 3.820 | 5.96e-08 | 1.192e-07 |

**Verdict: SUPPORTED**

## Secondary: XOR delay depth (REALMYELIN, primary arms)

| d | FRANKEN | ER | SHUF | FRANKEN−ER Δ | p_raw (>) |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.976 | 0.900 | 0.274 | +0.077 | 5.96e-08 |
| 3 | 0.975 | 0.898 | 0.303 | +0.077 | 5.96e-08 |
| 5 | 0.479 | 0.187 | 0.131 | +0.292 | 2.563e-06 |
| 8 | 0.154 | -0.517 | 0.097 | +0.671 | 1.788e-07 |
| 10 | -0.244 | -1.443 | 0.073 | +1.200 | 5.96e-08 |

## Secondary: delay mode at d=5 (FRANKEN / ER / SHUF)

| Mode | FRANKEN | ER | SHUF |
|---|---:|---:|---:|
| REALMYELIN | 0.4787±0.0183 | 0.1867±0.0545 | 0.1306±0.0042 |
| FIXED | 0.5348±0.0149 | 0.3077±0.0291 | 0.1382±0.0041 |
| NODELAY | 0.0540±0.0055 | -0.2599±0.0506 | 0.0821±0.0030 |

## Mechanism probe (FRANKEN lesions, seeds 200–207)

| Condition | mean ± sem | n | Δ vs intact |
|---|---:|---:|---:|
| LES_NONE | 0.4557 ± 0.0242 | 8 | +0.0000 |
| LES_FLY | 0.2178 ± 0.0427 | 8 | -0.2379 |
| LES_HUMAN | 0.6689 ± 0.0142 | 8 | +0.2132 |
| LES_BRIDGES | 0.4989 ± 0.0172 | 8 | +0.0432 |

## Honest verdict

**Primary SUPPORTED.** FRANKEN XOR_DELAY d=5 beats ER (p_bonf=5.126e-06) and SHUF (p_bonf=1.192e-07) on fresh seeds 200–223.

Mechanism: fly_CX lesion kills the effect; human lesion helps; bridges optional. FLY_CX alone outperforms the full graft. Do **not** claim mouse/human tissue adds XOR compute.

Limits: reservoir 1−NRMSE on delayed XOR is not biological XOR computation; SPEC is a spectrum-preserving layer-misalignment control, not a full eigen-null model; DENS/FLY_CX/HUMAN are secondary.


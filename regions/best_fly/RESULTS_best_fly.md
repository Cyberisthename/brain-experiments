# Smartest-fly results

Prereg: `PREREGISTRATION_best_fly.md`.

**BEST recipe locked on tune seeds 90–92:** `franken_realmyelin`

Composite: `MC/10 - NARMA + PC_HARD`.

## Tune scores

| candidate | MC | NARMA | PC_HARD | S |
|---|---:|---:|---:|---:|
| fly_cx_fixed | 14.7392 | 0.5038 | 0.8337 | 1.8038 |
| franken_fixed | 16.7709 | 0.5024 | 0.8698 | 2.0445 |
| franken_realmyelin | 17.8876 | 0.4949 | 0.8898 | 2.1836 |
| franken_realmyelin_only | 16.6294 | 0.4968 | 0.8957 | 2.0618 |

## Eval means ± sd (seeds 0–7)

| arm | MC ↑ | NARMA ↓ | PC_HARD ↑ |
|---|---:|---:|---:|
| FLY_CX_PLAIN | 6.4106±0.6803 | 0.6177±0.0246 | 0.7596±0.0699 |
| FLY_CX_FIXED | 15.2904±0.9094 | 0.5145±0.0287 | 0.8023±0.0622 |
| FRANKEN_REALMYELIN | 16.9571±1.3931 | 0.4850±0.0254 | 0.8814±0.0366 |
| BEST | 16.9571±1.3931 | 0.4850±0.0254 | 0.8814±0.0366 |
| ER_BEST | 19.9175±1.7278 | 0.3835±0.0262 | 0.9263±0.0470 |
| SHUF_BEST | 15.6026±2.5106 | 0.3783±0.0270 | 0.9345±0.0319 |
| DENS_BEST | 13.6138±1.6819 | 0.3754±0.0332 | 0.9470±0.0253 |

## Primary: BEST vs controls (Bonferroni ×9, paired Wilcoxon)

- BEST vs ER_BEST MC: Δ=-2.9605, p_raw=0.01562, p_bonf=0.1406, favor=ER_BEST
- BEST vs ER_BEST NARMA: Δ=+0.1014, p_raw=0.007812, p_bonf=0.07031, favor=ER_BEST
- BEST vs ER_BEST PC_HARD: Δ=-0.0450, p_raw=0.007812, p_bonf=0.07031, favor=ER_BEST
- BEST vs SHUF_BEST MC: Δ=+1.3544, p_raw=0.25, p_bonf=1, favor=BEST
- BEST vs SHUF_BEST NARMA: Δ=+0.1067, p_raw=0.007812, p_bonf=0.07031, favor=SHUF_BEST
- BEST vs SHUF_BEST PC_HARD: Δ=-0.0532, p_raw=0.007812, p_bonf=0.07031, favor=SHUF_BEST
- BEST vs DENS_BEST MC: Δ=+3.3432, p_raw=0.007812, p_bonf=0.07031, favor=BEST
- BEST vs DENS_BEST NARMA: Δ=+0.1096, p_raw=0.007812, p_bonf=0.07031, favor=DENS_BEST
- BEST vs DENS_BEST PC_HARD: Δ=-0.0656, p_raw=0.007812, p_bonf=0.07031, favor=DENS_BEST

## Secondary: BEST vs FLY_CX_FIXED (Bonferroni ×3)

- MC: Δ=+1.6666, p_raw=0.07812, p_bonf=0.2344, favor=BEST
- NARMA: Δ=-0.0296, p_raw=0.1094, p_bonf=0.3281, favor=BEST
- PC_HARD: Δ=+0.0791, p_raw=0.01562, p_bonf=0.04688, favor=BEST

## Verdict

NO significant BEST-vs-control difference after Bonferroni. Honest read: random wiring with the same delay tricks matches the stacked real recipe on these tasks. Secondary: BEST also beat FLY_CX_FIXED on 1 task(s).

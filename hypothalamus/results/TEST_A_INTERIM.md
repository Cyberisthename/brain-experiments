# Hypothalamus Test A INTERIM (America/New_York)

Generated 2026-11-05 ~6:45 PM ET after Test A arms finished; Tests B/C still running.
Arms complete: HOME_O_ERR, RAND_MOD, BASE (8 seeds 3400-3407 x 4 tasks).
This is NOT the locked full A/B/C RESULTS.md. Bonferroni shown for A-only (4 contrasts) and for full plan (12).

## Means ± std
### XOR_HARD (higher better)
- HOME_O_ERR: 0.5164 ± 0.0166
- RAND_MOD: 0.5095 ± 0.0098
- BASE: 0.5082 ± 0.0090

### NARMA (lower better)
- HOME_O_ERR: 1.5077 ± 0.6100
- RAND_MOD: 2.2199 ± 1.3730
- BASE: 1.3337 ± 0.3051

### CHAR (lower better)
- HOME_O_ERR: 2.2727 ± 0.0323
- RAND_MOD: 2.2772 ± 0.0354
- BASE: 2.2706 ± 0.0333

### TRANSFER (higher better)
- HOME_O_ERR: 0.7120 ± 0.0394
- RAND_MOD: 0.7054 ± 0.0256
- BASE: 0.7086 ± 0.0472

## Planned A contrasts
- HOME_O_ERR vs RAND_MOD on XOR_HARD: effect=+0.0070 (signed + = HOME better), Wilcoxon p=0.4609, wins 4/8
  A-only α=0.0125: FAIL; full-plan α=0.004167: FAIL; HOME better
- HOME_O_ERR vs BASE on XOR_HARD: effect=+0.0082 (signed + = HOME better), Wilcoxon p=0.3828, wins 5/8
  A-only α=0.0125: FAIL; full-plan α=0.004167: FAIL; HOME better
- HOME_O_ERR vs RAND_MOD on NARMA: effect=+0.7122 (signed + = HOME better), Wilcoxon p=0.1484, wins 6/8
  A-only α=0.0125: FAIL; full-plan α=0.004167: FAIL; HOME better
- HOME_O_ERR vs BASE on NARMA: effect=-0.1740 (signed + = HOME better), Wilcoxon p=0.4609, wins 3/8
  A-only α=0.0125: FAIL; full-plan α=0.004167: FAIL; HOME worse/tied

## Hypothesis A (prereg)
Hypothesis: HOME_O_ERR beats RAND_MOD and BASE on XOR_HARD and/or NARMA.
Falsify if HOME_O_ERR ≤ RAND_MOD, or both ≤ BASE.
- XOR_HARD: HOME vs RAND beats; HOME vs BASE beats (means H=0.5164 R=0.5095 B=0.5082)
- NARMA: HOME vs RAND beats; HOME vs BASE does not beat (means H=1.5077 R=2.2199 B=1.3337)

**Interim call: mixed / exploratory** — see contrasts; confirmatory bar tight at n=8. Wait for full RESULTS.md after B/C.

Do not treat this as the final A/B/C verdict.

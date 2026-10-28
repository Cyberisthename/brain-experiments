# Synthetic myelination results

Prereg: `PREREGISTRATION_myelin.md`.

## MC (8 seeds)

| arm | mean | sd |
|---|---:|---:|
| FIXED | 45.0553 | 0.5408 |
| ADAPT | 43.0282 | 0.4018 |
| NODELAY | 40.4365 | 0.2237 |
| ADAPT-SHUF | 37.7051 | 0.1851 |

### Contrasts (Bonferroni ×2: ADAPT vs FIXED, ADAPT vs ADAPT-SHUF)

- ADAPT vs FIXED: Δ=-2.027, p_raw=0.0001554, p_bonf=0.0003108
- ADAPT vs ADAPT-SHUF: Δ=5.323, p_raw=0.0001554, p_bonf=0.0003108

## Maze

All 32 maze values are NaN (run path bug). Primary score is MC.

## Verdict
FIXED > ADAPT > NODELAY > ADAPT-SHUF on MC. Adaptive loses to fixed but beats shuffled delay map. Maze: missing.

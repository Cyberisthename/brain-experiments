# Myelin mechanism results

Prereg: `PREREG_myelin_mechanism.md` (locked 2026-10-29 08:01 America/New_York, before eval).
Rows kept: 497. Seeds 800–823. Leak=0.2. Graph=HUMAN_LAYERED.

## Plain words

We kept the real human+fly layered graph fixed and only changed **which neuron gets which delay**, using rules inspired by how real H01 myelin lines up with layer, degree, E/I×layer, and feedforward depth. Goal: see which rule gets at least halfway from a global myelin shuffle toward real placement on NARMA, without beating real placement.

## Descriptive myelin correlations (pre-eval)

- Spearman(mf, indeg)=0.15130551125107328
- Spearman(mf, outdeg)=0.00786979477675131
- Spearman(mf, L4-BFS depth)=-0.035823853554405045
- Layer means: {'L2/3': 0.06390267678128864, 'L4': 0.06604363854299808, 'L5': 0.0874513055389019, 'L6': 0.1156967327237342}
- E/I means: {'E': 0.07396522652208368, 'I': 0.07750752261467013}

## NARMA means (↓ better)

| Arm | NARMA | MC | PC_HARD | recovery |
|---|---:|---:|---:|---:|
| `REAL` | 0.395±0.009 | 18.754±0.217 | 0.898±0.008 | — |
| `SHUF` | 0.560±0.016 | 17.869±0.241 | 0.898±0.007 | — |
| `NODELAY` | 0.584±0.011 | 6.464±0.225 | 0.859±0.009 | — |
| `RULE_LAYER` | 0.512±0.017 | 18.155±0.285 | 0.899±0.007 | 0.293 |
| `RULE_DEGREE` | 0.564±0.011 | 17.152±0.217 | 0.889±0.008 | -0.027 |
| `RULE_EI_LAYER` | 0.478±0.020 | 18.290±0.228 | 0.898±0.007 | 0.496 |
| `RULE_FF_DEPTH` | 0.460±0.017 | 18.515±0.244 | 0.884±0.008 | 0.608 |

REAL−SHUF gap (SHUF−REAL on NARMA) = 0.1645

## Primary rules (Bonferroni ×4)

| Rule | mean | vs SHUF Δ | p_raw | p_bonf | recovery | ≥REAL | pass |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| `RULE_LAYER` | 0.5118 | -0.0481 | 0.04291 | 0.1717 | 0.293 | yes | no |
| `RULE_DEGREE` | 0.5644 | 0.0045 | 0.07554 | 0.3021 | -0.027 | yes | no |
| `RULE_EI_LAYER` | 0.4784 | -0.0816 | 0.0007292 | 0.002917 | 0.496 | yes | no |
| `RULE_FF_DEPTH` | 0.4599 | -0.1000 | 1.192e-07 | 4.768e-07 | 0.608 | yes | yes |

**Verdict: SUPPORTED**

Passing rules: RULE_FF_DEPTH.

Reservoir ≠ biology claim about myelin.

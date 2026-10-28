# Trained bridges results

Prereg: `PREREGISTRATION_trained_bridges.md`.
Rows kept: 192. Seeds: [0, 1, 2, 3, 4, 5, 6, 7].

## Mean ± sem by arm × task

| Arm | MC | NARMA | PC_HARD | XOR_DELAY |
|---| ---: | ---: | ---: | ---: |
| `FRANKEN_FIXED` | 19.3658±0.3044 | 0.3792±0.0075 | 0.9064±0.0136 | 0.5253±0.0261 |
| `FRANKEN_ES` | 19.3660±0.3045 | 0.3792±0.0075 | 0.9065±0.0136 | 0.5252±0.0261 |
| `ER_FIXED` | 20.8847±0.2994 | 0.3828±0.0104 | 0.9358±0.0108 | 0.2130±0.0659 |
| `ER_ES` | 20.8842±0.2993 | 0.3827±0.0104 | 0.9359±0.0108 | 0.2135±0.0656 |
| `SHUF_FIXED` | 19.9807±0.3063 | 0.4701±0.0108 | 0.9428±0.0126 | -0.1589±0.1184 |
| `SHUF_ES` | 19.9808±0.3063 | 0.4701±0.0108 | 0.9428±0.0126 | -0.1596±0.1188 |

Direction: MC / PC_HARD / XOR_DELAY ↑ better; NARMA ↓ better.

## Training gains (direction-normalized: ⊕ = ES helped)

| Wiring | MC | NARMA | PC_HARD | XOR_DELAY |
|---| ---: | ---: | ---: | ---: |
| `FRANKEN` | +0.0002±0.0002 | +0.0000±0.0000 | +0.0000±0.0000 | -0.0001±0.0002 |
| `ER` | -0.0005±0.0003 | +0.0000±0.0000 | +0.0000±0.0000 | +0.0004±0.0003 |
| `SHUF` | +0.0002±0.0001 | +0.0000±0.0000 | +0.0000±0.0000 | -0.0006±0.0004 |

## Primary family — held-out tasks (Bonferroni ×4)

| Contrast | Task | mean Δgain | p_raw | p_bonf | sig |
|---|---|---:|---:|---:|---|
| gain_FRANKEN − gain_ER | PC_HARD | -0.0000 | 0.9453 | 1 | no |
| gain_FRANKEN − gain_ER | XOR_DELAY | -0.0006 | 0.4609 | 1 | no |
| gain_FRANKEN − gain_SHUF | PC_HARD | -0.0000 | 0.8438 | 1 | no |
| gain_FRANKEN − gain_SHUF | XOR_DELAY | +0.0005 | 0.3125 | 1 | no |

## Secondary family — train tasks (Bonferroni ×4)

| Contrast | Task | mean Δgain | p_raw | p_bonf | sig |
|---|---|---:|---:|---:|---|
| gain_FRANKEN − gain_ER | MC | +0.0007 | 0.1484 | 0.5938 | no |
| gain_FRANKEN − gain_ER | NARMA | -0.0000 | 0.07812 | 0.3125 | no |
| gain_FRANKEN − gain_SHUF | MC | +0.0000 | 0.9453 | 1 | no |
| gain_FRANKEN − gain_SHUF | NARMA | +0.0000 | 0.8438 | 1 | no |

## Exploratory: FIXED FRANKEN vs FIXED ER / SHUF (raw p)

| Contrast | Task | mean Δ | p_raw |
|---|---|---:|---:|
| FIXED FRANKEN − FIXED ER (⊕ better) | MC | -1.5189 | 0.007812 |
| FIXED FRANKEN − FIXED ER (⊕ better) | NARMA | +0.0035 | 0.6406 |
| FIXED FRANKEN − FIXED ER (⊕ better) | PC_HARD | -0.0294 | 0.007812 |
| FIXED FRANKEN − FIXED ER (⊕ better) | XOR_DELAY | +0.3123 | 0.007812 |
| FIXED FRANKEN − FIXED SHUF (⊕ better) | MC | -0.6149 | 0.03906 |
| FIXED FRANKEN − FIXED SHUF (⊕ better) | NARMA | +0.0909 | 0.007812 |
| FIXED FRANKEN − FIXED SHUF (⊕ better) | PC_HARD | -0.0363 | 0.007812 |
| FIXED FRANKEN − FIXED SHUF (⊕ better) | XOR_DELAY | +0.6843 | 0.007812 |

## ES training (from cached logs on ES rows)

- Bridge edges: 6718 (mean)
- Fitness first→last gen (mean): +0.3037 → +0.3420
- Final ρ (mean): 0.9005

## Honest verdict

**Primary:** no gain-contrast survives Bonferroni on PC_HARD / XOR_DELAY. Training bridges does **not** unlock a real-wiring advantage over ER/SHUF on held-out tasks under this budget (H3 / Level-1 continues).
**Secondary:** no train-task gain-contrast survives Bonferroni.
Raw mean gains ⊕: FRANKEN/MC: +0.0002; FRANKEN/NARMA: +0.0000; FRANKEN/PC_HARD: +0.0000; ER/NARMA: +0.0000; ER/PC_HARD: +0.0000; ER/XOR_DELAY: +0.0004; SHUF/MC: +0.0002; SHUF/NARMA: +0.0000; SHUF/PC_HARD: +0.0000

Limits: antithetic ES G=10 (20 fitness rollouts), short fitness horizon, bridges-only, frozen blocks. Not biological learning.

No GitHub push.

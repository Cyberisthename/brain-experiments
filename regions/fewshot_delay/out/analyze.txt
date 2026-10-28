# Few-shot + matched delays — results

Prereg: `PREREGISTRATION_fewshot_delay.md`.
Eval seeds 0–7. Focus n_train=100.

## Locked ridge α (tune seeds 90–92, MC @ n_train=100)

| arm | α |
|---|---:|
| DENS_DEL | 0.01 |
| DENS_NOD | 0.1 |
| ER_DEL | 0.0001 |
| ER_NOD | 0.01 |
| FLY_CX_DEL | 1e-06 |
| FLY_CX_NOD | 1e-06 |
| FRANKEN_DEL | 1e-06 |
| FRANKEN_NOD | 0.0001 |
| SHUF_DEL | 0.01 |
| SHUF_NOD | 0.1 |

## Learning curve means (MC ↑)

| arm | 25 | 50 | 100 | 200 | 500 | 1000 | 3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLY_CX_NOD | 1.33 | 1.99 | 3.42 | 4.62 | 5.57 | 6.19 | 6.77 |
| FRANKEN_NOD | 1.41 | 2.12 | 3.27 | 4.15 | 4.98 | 5.63 | 6.32 |
| ER_NOD | 1.62 | 2.34 | 3.44 | 4.65 | 5.48 | 6.19 | 6.87 |
| SHUF_NOD | 1.72 | 2.50 | 3.87 | 5.29 | 6.37 | 7.11 | 7.83 |
| DENS_NOD | 1.76 | 2.51 | 3.93 | 5.40 | 6.58 | 7.39 | 8.13 |
| FLY_CX_DEL | 4.99 | 6.89 | 9.35 | 11.46 | 13.44 | 14.62 | 15.94 |
| FRANKEN_DEL | 4.48 | 6.10 | 8.06 | 10.75 | 14.26 | 16.40 | 18.81 |
| ER_DEL | 4.34 | 6.23 | 9.26 | 12.25 | 16.23 | 18.49 | 21.01 |
| SHUF_DEL | 3.32 | 5.44 | 8.16 | 11.13 | 14.19 | 15.96 | 18.08 |
| DENS_DEL | 3.45 | 5.59 | 8.10 | 10.97 | 14.38 | 16.39 | 18.59 |

## Focus table — n_train=100

| arm | MC ↑ | NARMA ↓ | PC_HARD ↑ | PR (mech) |
|---|---:|---:|---:|---:|
| FLY_CX_NOD | 3.415±0.560 | 0.711±0.058 | 0.220±0.460 | 1.0 |
| FRANKEN_NOD | 3.274±0.520 | 0.752±0.044 | 0.058±0.651 | 1.0 |
| ER_NOD | 3.442±0.653 | 0.784±0.049 | 0.736±0.124 | 1.3 |
| SHUF_NOD | 3.866±0.753 | 0.776±0.036 | 0.728±0.118 | 1.5 |
| DENS_NOD | 3.934±0.834 | 0.789±0.052 | 0.770±0.115 | 1.6 |
| FLY_CX_DEL | 9.354±0.590 | 0.616±0.064 | 0.411±0.281 | 1.0 |
| FRANKEN_DEL | 8.062±0.583 | 0.638±0.093 | 0.250±0.778 | 1.0 |
| ER_DEL | 9.261±0.598 | 1.144±0.425 | 0.841±0.083 | 1.4 |
| SHUF_DEL | 8.162±0.617 | 0.782±0.127 | 0.807±0.096 | 1.6 |
| DENS_DEL | 8.101±0.680 | 0.819±0.134 | 0.824±0.094 | 1.8 |

## PRIMARY — FRANKEN_DEL vs delay-matched controls @ n=100 (Bonferroni ×9)

| real | ctrl | task | real | ctrl | Δ | p_raw | p_bonf | sig | favor |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| FRANKEN_DEL | ER_DEL | MC | 8.062 | 9.261 | -1.199 | 0.007812 | 0.07031 | no | ER_DEL |
| FRANKEN_DEL | ER_DEL | NARMA | 0.638 | 1.144 | -0.506 | 0.007812 | 0.07031 | no | FRANKEN_DEL |
| FRANKEN_DEL | ER_DEL | PC_HARD | 0.250 | 0.841 | -0.591 | 0.007812 | 0.07031 | no | ER_DEL |
| FRANKEN_DEL | SHUF_DEL | MC | 8.062 | 8.162 | -0.101 | 0.4609 | 1 | no | SHUF_DEL |
| FRANKEN_DEL | SHUF_DEL | NARMA | 0.638 | 0.782 | -0.145 | 0.007812 | 0.07031 | no | FRANKEN_DEL |
| FRANKEN_DEL | SHUF_DEL | PC_HARD | 0.250 | 0.807 | -0.557 | 0.01562 | 0.1406 | no | SHUF_DEL |
| FRANKEN_DEL | DENS_DEL | MC | 8.062 | 8.101 | -0.039 | 1 | 1 | no | DENS_DEL |
| FRANKEN_DEL | DENS_DEL | NARMA | 0.638 | 0.819 | -0.181 | 0.007812 | 0.07031 | no | FRANKEN_DEL |
| FRANKEN_DEL | DENS_DEL | PC_HARD | 0.250 | 0.824 | -0.574 | 0.007812 | 0.07031 | no | DENS_DEL |

## SECONDARY — FRANKEN_NOD vs NOD controls @ n=100 (replicate) (Bonferroni ×9)

| real | ctrl | task | real | ctrl | Δ | p_raw | p_bonf | sig | favor |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| FRANKEN_NOD | ER_NOD | MC | 3.274 | 3.442 | -0.167 | 0.3828 | 1 | no | ER_NOD |
| FRANKEN_NOD | ER_NOD | NARMA | 0.752 | 0.784 | -0.032 | 0.02344 | 0.2109 | no | FRANKEN_NOD |
| FRANKEN_NOD | ER_NOD | PC_HARD | 0.058 | 0.736 | -0.678 | 0.007812 | 0.07031 | no | ER_NOD |
| FRANKEN_NOD | SHUF_NOD | MC | 3.274 | 3.866 | -0.592 | 0.03906 | 0.3516 | no | SHUF_NOD |
| FRANKEN_NOD | SHUF_NOD | NARMA | 0.752 | 0.776 | -0.024 | 0.03906 | 0.3516 | no | FRANKEN_NOD |
| FRANKEN_NOD | SHUF_NOD | PC_HARD | 0.058 | 0.728 | -0.671 | 0.01562 | 0.1406 | no | SHUF_NOD |
| FRANKEN_NOD | DENS_NOD | MC | 3.274 | 3.934 | -0.660 | 0.02344 | 0.2109 | no | DENS_NOD |
| FRANKEN_NOD | DENS_NOD | NARMA | 0.752 | 0.789 | -0.037 | 0.02344 | 0.2109 | no | FRANKEN_NOD |
| FRANKEN_NOD | DENS_NOD | PC_HARD | 0.058 | 0.770 | -0.713 | 0.007812 | 0.07031 | no | DENS_NOD |

## Delay-survival family — MC @ n=100 (Bonferroni ×3)

Per-seed gap = FRANKEN − ctrl. Test whether gap_DEL − gap_NOD differs from 0 (Wilcoxon).

| ctrl | mean gap_NOD | mean gap_DEL | mean(gap_DEL−gap_NOD) | p_raw | p_bonf | sig | interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| ER | -0.167 | -1.199 | -1.032 | 0.03906 | 0.1172 | no | advantage shrank / flipped under delays |
| SHUF | -0.592 | -0.101 | +0.491 | 0.07812 | 0.2344 | no | advantage grew under delays |
| DENS | -0.660 | -0.039 | +0.621 | 0.1094 | 0.3281 | no | advantage grew under delays |

## Noise family — mean input-noise MC retention σ>0 (Bonferroni ×6)

| real | ctrl | real_ret | ctrl_ret | Δ | p_raw | p_bonf | sig | favor |
|---|---|---:|---:|---:|---:|---:|---|---|
| FRANKEN_DEL | ER_DEL | 0.383 | 0.382 | +0.001 | 0.6406 | 1 | no | FRANKEN_DEL |
| FRANKEN_DEL | SHUF_DEL | 0.383 | 0.401 | -0.017 | 0.01562 | 0.09375 | no | SHUF_DEL |
| FRANKEN_DEL | DENS_DEL | 0.383 | 0.397 | -0.014 | 0.01562 | 0.09375 | no | DENS_DEL |
| FLY_CX_DEL (size-confounded) | ER_DEL | 0.416 | 0.382 | +0.034 | 0.007812 | 0.04688 | YES | FLY_CX_DEL |
| FLY_CX_DEL (size-confounded) | SHUF_DEL | 0.416 | 0.401 | +0.016 | 0.01562 | 0.09375 | no | FLY_CX_DEL |
| FLY_CX_DEL (size-confounded) | DENS_DEL | 0.416 | 0.397 | +0.019 | 0.007812 | 0.04688 | YES | FLY_CX_DEL |

## Mechanism — participation ratio vs MC@100 (exploratory)

| arm | mean PR | MC@100 |
|---|---:|---:|
| FLY_CX_NOD | 1.0 | 3.415 |
| FRANKEN_NOD | 1.0 | 3.274 |
| ER_NOD | 1.3 | 3.442 |
| SHUF_NOD | 1.5 | 3.866 |
| DENS_NOD | 1.6 | 3.934 |
| FLY_CX_DEL | 1.0 | 9.354 |
| FRANKEN_DEL | 1.0 | 8.062 |
| ER_DEL | 1.4 | 9.261 |
| SHUF_DEL | 1.6 | 8.162 |
| DENS_DEL | 1.8 | 8.101 |

Pearson r(PR, MC@100) across arms = **0.140** (H3 predicted negative: lower PR → better few-shot).

## Honest verdict

### Headline (not a breakthrough — a correction)

**With a fair per-arm ridge α (tuned on seeds 90–92 only), the earlier “real wiring wins few-shot” result does not replicate — with or without delays.**

### Locked numbers

- **NOD @ n=100 MC:** FRANKEN 3.27 ≈ ER 3.44; SHUF 3.87; DENS 3.93. Secondary Bonf wins for FRANKEN_NOD: **0/9**.
- **DEL @ n=100 MC:** FRANKEN 8.06; ER 9.26; SHUF 8.16; DENS 8.10. Primary Bonf wins for FRANKEN_DEL: **0/9**. Delays ~2.5× everyone’s few-shot MC.
- **NARMA @ n=100 (DEL):** FRANKEN 0.638 vs ER 1.144 / SHUF 0.782 / DENS 0.819 — raw p=0.0078 favoring FRANKEN on all three, but **p_bonf≈0.07 (not significant)**. Suggestive, not claimable.
- **Noise retention (DEL):** FRANKEN ≈ controls (0.383 vs 0.38–0.40). No Bonf win. FLY_CX_DEL edges ER/DENS on retention but is size-confounded.
- **Delay-survival gaps:** no Bonf-significant change of FRANKEN−ctrl MC gap under delays.
- **Mechanism:** PR≈1–2 for all arms; r(PR, MC@100)=+0.14 (H3 predicted negative — rejected / uninformative; states look near rank-1 after wash).

### Why this contradicts `regions/robust` few-shot

Robust used **α=1e-6 for every arm**. Here locked α for random NOD arms rose to **0.01–0.1** (see alpha table). Fixed tiny α lets high-dimensional random reservoirs overfit at n=100 and look worse; real arms (already low-dimensional) were less hurt. **The few-shot “connectome advantage” was largely a regularization confound**, not a topology miracle.

### Prereg predictions

| Pred | Result |
|---|---|
| P1 (NOD few-shot replicate) | **Rejected** under fair α |
| P2 (DEL few-shot survives) | **Rejected** (MC); NARMA suggestive only |
| P3 (DEL noise retention) | **Rejected** for FRANKEN |
| P4 / H0 | **Supported** — delays dominate; random with same delays matches real at few-shot MC |

**Do not claim a scientific breakthrough.** The useful result is negative-but-informative: delays still help a lot, and prior few-shot wins need fair α to be believed.

## NARMA learning curve means (↓)

| arm | 25 | 50 | 100 | 200 | 500 | 1000 | 3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FLY_CX_NOD | 1.315 | 0.797 | 0.711 | 0.658 | 0.626 | 0.613 | 0.590 |
| FRANKEN_NOD | 0.981 | 0.815 | 0.752 | 0.721 | 0.683 | 0.664 | 0.631 |
| ER_NOD | 1.007 | 0.816 | 0.784 | 0.756 | 0.728 | 0.713 | 0.682 |
| SHUF_NOD | 0.861 | 0.797 | 0.776 | 0.762 | 0.740 | 0.729 | 0.701 |
| DENS_NOD | 0.861 | 0.804 | 0.789 | 0.759 | 0.734 | 0.721 | 0.689 |
| FLY_CX_DEL | 1.132 | 0.718 | 0.616 | 0.558 | 0.528 | 0.513 | 0.492 |
| FRANKEN_DEL | 1.522 | 0.922 | 0.638 | 0.528 | 0.498 | 0.479 | 0.459 |
| ER_DEL | 2.108 | 1.809 | 1.144 | 0.595 | 0.528 | 0.487 | 0.461 |
| SHUF_DEL | 1.093 | 0.877 | 0.782 | 0.638 | 0.572 | 0.541 | 0.509 |
| DENS_DEL | 1.167 | 0.949 | 0.819 | 0.616 | 0.566 | 0.535 | 0.504 |


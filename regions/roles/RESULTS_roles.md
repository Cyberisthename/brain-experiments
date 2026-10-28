# Regions doing their real job — results

Prereg: `PREREGISTRATION_roles.md`. Aggregated from `out/runs.jsonl` (144 jobs).

## MB associative (p1_mean; matched use_lh=1, lh=real)

| mb_mode | p1_mean | vs real p |
|---|---:|---:|
| real | 0.644 ± 0.029 (n=8) | — |
| rand | 0.653 ± 0.047 (n=8) | 0.5611 |
| shuf | 0.637 ± 0.052 (n=8) | 0.8166 |

## LH mode (mb=real, use_lh=1)

| lh_mode | p1_mean | vs real p |
|---|---:|---:|
| real | 0.644 ± 0.029 (n=8) | — |
| rand | 0.569 ± 0.032 (n=8) | 0.0019 |
| shuf | 0.541 ± 0.027 (n=8) | 0.0009 |

LH real vs rand/shuf survives Bonferroni ×6 (p_bonf ≈ 0.011 / 0.0056).

## TuBu heading_err (↓ better)

| mode | mean ± sd |
|---|---:|
| real | 8.925 ± 1.148 (n=8) |
| rand | 9.817 ± 0.990 (n=8) |
| shuf | 8.375 ± 1.536 (n=8) |

real vs rand p=0.141; real vs shuf p=0.442.

## Maze success

| mode | success |
|---|---:|
| real | 0.410 ± 0.176 (n=8) |
| rand | 0.398 ± 0.166 (n=8) |
| randmod | 0.358 ± 0.217 (n=8) |

real vs rand p=1.000; real vs randmod p=0.673.

## Verdict
- MB wiring: **null** vs rand/shuf.
- LH real mode: **win** vs rand/shuf on p1 (survives mild Bonf).
- TuBu / maze: **null**.

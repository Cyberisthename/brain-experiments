# DIAGNOSIS: human RANDOM_LAYERED MC 8.85

## Verdict
**Real, not a fluke, not a size/sign/scaling bug.** Sparse ER add-on has a much fatter spectrum than the real human block → higher MC. Mouse is the opposite (dense), so mouse RANDOM did not win.

## Extra seeds (MC only, leak=0.2, compress_one path)
| seed | HUMAN_LAYERED | RANDOM_LAYERED | R−H |
|------|---------------|----------------|-----|
| 0 | 6.25 | **8.85** | +2.60 |
| 1 | 6.28 | 8.74 | +2.46 |
| 2 | 5.75 | 9.11 | +3.36 |
| 3 | 4.82 | 6.88 | +2.06 |
| mean | 5.78 | **8.40** | +2.62 |

Seed 0’s 8.85 is typical, not a one-off strat.

## Bug checks (clean)
- Same N (2087), same L4/L5 indices, same bridge in/out counts (1968/1400), same signs, global ρ≈0.9
- ER nnz ≈ real (14831 vs 14863); block radii both ≈1.0
- Row-sign match = 1.0 for both (matches how modules were built: W[a,b] *= sign[a])

## What drives it (human REAL vs ER @ ρ=1)
| property | HUMAN real | ER | mouse real | mouse ER |
|----------|------------|-----|------------|----------|
| density | 0.0034 | 0.0034 | **0.028** | 0.027 |
| reciprocity | 0.039 | 0.004 | 0.227 | 0.028 |
| frac neg edges | 0.109 | 0.166 | 0.348 | 0.101 |
| **\|eig\|>0.5** | **11** | **769** | 4 | **1** |
| eff. rank (L4-driven) | 1.51 | 3.07 | 1.10 | 1.86 |
| active frac (std>0.01) | 0.29 | 0.88 | 0.27 | 1.00 |

**Driver:** human cut is very sparse. ER with |N(0,1)| weights, after spectral-radius=1 scaling, keeps a large bulk of mid-size eigenvalues → richer reservoir states → higher memory capacity. Real human wiring is low-rank / low-spectral-bulk after the same scaling (only 11 |λ|>0.5, eff_rank≈1.5), so MC stays ~6.

Mouse module is ~8× denser; after ρ=1 its ER weights shrink and the bulk collapses (|eig|>0.5 = 1), so mouse RANDOM_LAYERED (smoke 5.80) does **not** beat HYBRID_LAYERED (6.23).

## Plain takeaway
8.85 is the sparse-ER spectrum doing the work, not a bug and not a seed trick. Copying “the seed” just re-runs the same ER arm — every seed we tried also beats human by ~2–3 MC. Full 8-seed run will almost certainly keep RANDOM_LAYERED on top for human; that rejects H3 (human hierarchy beating matched ER) on MC unless something else flips.

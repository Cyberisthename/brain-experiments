# Addendum (2026-11-01): NARMA timing scores

The original RESULTS.md showed NARMA as nan because seed 2207 returned nan in every arm.
Dropping seed 2207 (failed in all 7 arms, so the comparison stays paired) gives n=7:

| arm | NARMA (lower better) |
|---|---:|
| HUMAN_SC | 0.5505 |
| HUMAN_SC_MYELIN | 0.7519 |
| INDIVIDUAL | 0.7519 |
| MYELIN_ONLY | 0.8075 |
| WEIGHT_SHUFFLE | 0.9423 |
| DEGREE_PRESERVING_SHUFFLE | 1.0090 |
| ER_RANDOM | 1.0583 |

Paired one-sided Wilcoxon, n=7:
- HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE (preregistered primary): 7/7 wins, p=0.0078 (< Bonferroni 0.01)
- HUMAN_SC vs ER_RANDOM / WEIGHT_SHUFFLE / DEGREE_PRESERVING_SHUFFLE: 7/7 wins each, p=0.0078
- HUMAN_SC_MYELIN vs MYELIN_ONLY: 3/7, p=0.29

Caveat: excluding seed 2207 is a post-hoc deviation from the prereg; seed 2207's NaN should be investigated and rerun.

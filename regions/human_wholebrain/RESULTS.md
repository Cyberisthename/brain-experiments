# human_wholebrain RESULTS

> **NARMA fix (2026-11-01):** seed 2207 previously returned NaN on all arms
> (unstable NARMA-10 target → Inf → NaN loss). Fixed in `make_narma_batch` by
> rejecting/resampling explosive trajectories; only seed 2207 NARMA was rerun.
> See `FIX_NOTES.md`. Prior write-up saved as `RESULTS_v1.md`.


Runs loaded: 224

## Means (metric)

| arm | XOR_HARD ↑ | XOR2BIT ↑ | CHAR ↓ | NARMA ↓ | n/seed |
|---|---:|---:|---:|---:|---:|
| HUMAN_SC | 0.5595 | 0.6215 | 2.0541 | 0.5532 | 8 |
| HUMAN_SC_MYELIN | 0.5252 | 0.6046 | 2.2983 | 0.7556 | 8 |
| DEGREE_PRESERVING_SHUFFLE | 0.5102 | 0.5030 | 2.4475 | 1.0092 | 8 |
| WEIGHT_SHUFFLE | 0.5405 | 0.5225 | 2.3317 | 0.9537 | 8 |
| ER_RANDOM | 0.5078 | 0.5030 | 2.2906 | 1.0553 | 8 |
| MYELIN_ONLY | 0.7433 | 0.7447 | 2.1856 | 0.8132 | 8 |
| INDIVIDUAL | 0.6001 | 0.5843 | 2.2862 | 0.7547 | 8 |

## Primary contrasts (Bonferroni α=0.0100, m=5)

| contrast | task | mean_a | mean_b | pairedΔ(a better+) | p | verdict |
|---|---|---:|---:|---:|---:|---|
| HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE | XOR_HARD | 0.5252 | 0.5102 | 0.0150 | 0.05469 | NOT SUPPORTED |
| HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE | NARMA | 0.7556 | 1.0092 | 0.2536 | 0.003906 | SUPPORTED |
| HUMAN_SC vs ER_RANDOM | XOR_HARD | 0.5595 | 0.5078 | 0.0517 | 0.2305 | NOT SUPPORTED |
| HUMAN_SC vs WEIGHT_SHUFFLE | XOR_HARD | 0.5595 | 0.5405 | 0.0190 | 0.6289 | NOT SUPPORTED |
| HUMAN_SC_MYELIN vs MYELIN_ONLY | XOR_HARD | 0.5252 | 0.7433 | -0.2181 | 1 | NOT SUPPORTED |

**Overall primary:** at least one SUPPORTED

## INDIVIDUAL spread (secondary)

- XOR_HARD: mean=0.6001 std=0.0957 n=8
- XOR2BIT: mean=0.5843 std=0.0848 n=8
- CHAR: mean=2.2862 std=0.0535 n=8
- NARMA: mean=0.7547 std=0.0285 n=8

Data: Griffa et al. 2019 Zenodo 2872624 + Lausanne consensus fiber lengths (local only).

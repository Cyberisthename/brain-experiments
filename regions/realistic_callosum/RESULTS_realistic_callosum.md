# RESULTS — realistic_callosum
Prereg: `PREREG_realistic_callosum.md` (frozen 2026-10-30 09:51:38 EDT).
Left-only readout; delayed hemispheric XOR (d=5). Seeds 2100–2111 (n=12).
Primary family: 6 paired Wilcoxon, Bonferroni α=0.05/6.

## Summary (mean ± sd)

| arm | XOR ↑ | CHAR ↓ | NARMA ↓ | n_bridge (typ) |
|---|---|---|---|---|
| REAL_CALLOSUM | 1.000±0.000 | 1.889±0.017 | 0.484±0.041 | 160 |
| REAL_CALLOSUM_SHUFFLED | 1.000±0.000 | 1.875±0.020 | 0.461±0.038 | 160 |
| INDIVIDUAL_CALLOSUM | 1.000±0.000 | 1.894±0.018 | 0.500±0.029 | 160 |
| CALLOSUM | 1.000±0.000 | 1.866±0.018 | 0.534±0.024 | 160 |
| RAND_BRIDGE | 1.000±0.000 | 1.866±0.016 | 0.493±0.018 | 160 |
| NO_BRIDGE | 0.973±0.015 | 1.928±0.020 | 0.543±0.021 | 0 |
| DENSE_BRIDGE | 0.509±0.007 | 2.806±0.115 | 1.035±0.039 | 12800 |

## Primary tests (Bonferroni ×6)

| comparison | task | dir | mean_a | mean_b | p raw | p adj | pass? |
|---|---|---|---|---|---|---|---|
| REAL_CALLOSUM vs REAL_CALLOSUM_SHUFFLED | XOR | greater | 1.000 | 1.000 | 1 | 1 | no |
| REAL_CALLOSUM vs CALLOSUM | XOR | greater | 1.000 | 1.000 | 1 | 1 | no |
| REAL_CALLOSUM vs RAND_BRIDGE | XOR | greater | 1.000 | 1.000 | 1 | 1 | no |
| REAL_CALLOSUM vs NO_BRIDGE | XOR | greater | 1.000 | 0.973 | 0.0002441 | 0.001465 | YES |
| REAL_CALLOSUM vs REAL_CALLOSUM_SHUFFLED | NARMA | less | 0.484 | 0.461 | 0.9451 | 1 | no |
| REAL_CALLOSUM vs NO_BRIDGE | NARMA | less | 0.484 | 0.543 | 0.0004883 | 0.00293 | YES |

**Primary verdict:** NOT SUPPORTED — REAL beats SHUFFLED on XOR: False; REAL beats NO_BRIDGE on XOR: True. (2/6 family tests significant after Bonferroni).

## Secondary (raw p)

- REAL vs REAL_CALLOSUM_SHUFFLED CHAR (less): 1.889 vs 1.875, raw p=0.9995
- REAL vs CALLOSUM CHAR (less): 1.889 vs 1.866, raw p=1
- REAL vs RAND_BRIDGE CHAR (less): 1.889 vs 1.866, raw p=1
- REAL vs NO_BRIDGE CHAR (less): 1.889 vs 1.928, raw p=0.0002441
- REAL vs DENSE_BRIDGE CHAR (less): 1.889 vs 2.806, raw p=0.0002441

- INDIVIDUAL XOR mean±sd=1.000±0.000; REAL group=1.000±0.000
- NO_BRIDGE XOR (bridge-required check): 0.973±0.015 (expect near chance ~0.5)

## Honesty

Null allowed. Numbers from `out/runs.jsonl` only. Connectome data not committed.

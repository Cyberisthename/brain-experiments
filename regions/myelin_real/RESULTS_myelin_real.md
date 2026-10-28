# Real H01 myelin delay results

Prereg: `PREREGISTRATION_myelin_real.md`.

## HUMAN_LAYERED

| task | NODELAY | REAL_MYELIN | SHUF_MYELIN | ADAPT |
|---|---:|---:|---:|---:|
| MC | 5.4897±1.2458 | 15.7253±1.5580 | 14.6008±1.5677 | 4.6082±0.7222 |
| NARMA | 0.6502±0.0461 | 0.4966±0.0379 | 0.6267±0.0638 | 0.6251±0.0902 |
| PC_HARD | 0.8603±0.0845 | 0.8979±0.0457 | 0.8983±0.0477 | 0.8290±0.0535 |

## FRANKEN

| task | NODELAY | REAL_MYELIN | SHUF_MYELIN | ADAPT |
|---|---:|---:|---:|---:|
| MC | 6.8839±0.8164 | 15.6189±0.9425 | 15.5370±1.0119 | 3.4105±1.1846 |
| NARMA | 0.5802±0.0309 | 0.4871±0.0258 | 0.4880±0.0262 | 0.6050±0.0694 |
| PC_HARD | 0.8799±0.0346 | 0.8847±0.0434 | 0.8835±0.0500 | 0.8675±0.0364 |

## Primary REAL vs SHUF (Bonferroni ×6)

- HUMAN_LAYERED MC: Δ=+1.1246, p_raw=0.1949, p_bonf=1 (REAL>)
- HUMAN_LAYERED NARMA: Δ=-0.1301, p_raw=0.0003108, p_bonf=0.001865 (REAL<)
- HUMAN_LAYERED PC_HARD: Δ=-0.0004, p_raw=0.9591, p_bonf=1 (REAL<)
- FRANKEN MC: Δ=+0.0819, p_raw=0.8785, p_bonf=1 (REAL>)
- FRANKEN NARMA: Δ=-0.0009, p_raw=0.9591, p_bonf=1 (REAL<)
- FRANKEN PC_HARD: Δ=+0.0012, p_raw=0.7984, p_bonf=1 (REAL>)

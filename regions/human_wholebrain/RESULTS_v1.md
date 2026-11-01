# human_wholebrain RESULTS

Runs loaded: 224

## Means (metric)

| arm | XOR_HARD ↑ | XOR2BIT ↑ | CHAR ↓ | NARMA ↓ | n/seed |
|---|---:|---:|---:|---:|---:|
| HUMAN_SC | 0.5595 | 0.6215 | 2.0541 | nan | 8 |
| HUMAN_SC_MYELIN | 0.5252 | 0.6046 | 2.2983 | nan | 8 |
| DEGREE_PRESERVING_SHUFFLE | 0.5102 | 0.5030 | 2.4475 | nan | 8 |
| WEIGHT_SHUFFLE | 0.5405 | 0.5225 | 2.3317 | nan | 8 |
| ER_RANDOM | 0.5078 | 0.5030 | 2.2906 | nan | 8 |
| MYELIN_ONLY | 0.7433 | 0.7447 | 2.1856 | nan | 8 |
| INDIVIDUAL | 0.6001 | 0.5843 | 2.2862 | nan | 8 |

## Primary contrasts (Bonferroni α=0.0100, m=5)

| contrast | task | mean_a | mean_b | pairedΔ(a better+) | p | verdict |
|---|---|---:|---:|---:|---:|---|
| HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE | XOR_HARD | 0.5252 | 0.5102 | 0.0150 | 0.05469 | NOT SUPPORTED |
| HUMAN_SC_MYELIN vs DEGREE_PRESERVING_SHUFFLE | NARMA | nan | nan | nan | nan | NOT SUPPORTED |
| HUMAN_SC vs ER_RANDOM | XOR_HARD | 0.5595 | 0.5078 | 0.0517 | 0.2305 | NOT SUPPORTED |
| HUMAN_SC vs WEIGHT_SHUFFLE | XOR_HARD | 0.5595 | 0.5405 | 0.0190 | 0.6289 | NOT SUPPORTED |
| HUMAN_SC_MYELIN vs MYELIN_ONLY | XOR_HARD | 0.5252 | 0.7433 | -0.2181 | 1 | NOT SUPPORTED |

**Overall primary:** NOT SUPPORTED (no primary contrast passed)

## INDIVIDUAL spread (secondary)

- XOR_HARD: mean=0.6001 std=0.0957 n=8
- XOR2BIT: mean=0.5843 std=0.0848 n=8
- CHAR: mean=2.2862 std=0.0535 n=8
- NARMA: mean=nan std=nan n=8

Data: Griffa et al. 2019 Zenodo 2872624 + Lausanne consensus fiber lengths (local only).

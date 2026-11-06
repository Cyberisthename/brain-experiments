# callosum_scaled RESULTS

Runs loaded: 672
N labels present: [512, 1024]
Primary claim N (largest with TRANSFER paired seeds): **1024**

## Means at N_label=512

| arm | XOR_HARD ↑ | CHAR ↓ | NARMA ↓ | TRANSFER ↑ | SPLIT_BIND ↑ | DELAYED_MATCH ↑ | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| REAL_SC | 0.5292 | 2.1543 | 1.7395 | 0.9410 | 0.5594 | 0.5322 | 8 |
| DEGREE_SHUFFLE | 0.5210 | 2.1658 | 1.6906 | 0.9142 | 0.5596 | 0.5693 | 8 |
| WEIGHT_SHUFFLE | 0.5396 | 2.2000 | 1.7669 | 0.8585 | 0.5889 | 0.5596 | 8 |
| SPARSE_BRIDGE | 0.6842 | 1.9947 | 1.7971 | 0.9766 | 0.7161 | 0.9082 | 8 |
| NO_BRIDGE | 0.5166 | 2.0312 | 1.4600 | 0.9394 | 0.5841 | 0.5547 | 8 |
| DENSE | 0.5097 | 2.4903 | 2.0611 | 0.6069 | 0.5280 | 0.5225 | 8 |
| AGCC | 0.5042 | 2.0581 | 2.2346 | 0.8654 | 0.5268 | 0.5312 | 8 |

## Means at N_label=1024

| arm | XOR_HARD ↑ | CHAR ↓ | NARMA ↓ | TRANSFER ↑ | SPLIT_BIND ↑ | DELAYED_MATCH ↑ | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| REAL_SC | 0.5448 | 1.9303 | 2.0964 | 0.9978 | 0.6276 | 0.6680 | 8 |
| DEGREE_SHUFFLE | 0.5339 | 1.9385 | 2.0013 | 0.9996 | 0.5907 | 0.6279 | 8 |
| WEIGHT_SHUFFLE | 0.5524 | 1.9803 | 2.2517 | 0.9419 | 0.5938 | 0.6221 | 8 |
| SPARSE_BRIDGE | 0.6827 | 1.7842 | 1.6920 | 0.9995 | 0.7192 | 0.9492 | 8 |
| NO_BRIDGE | 0.5479 | 1.7892 | 1.8500 | 0.9988 | 0.6360 | 0.7334 | 8 |
| DENSE | 0.5055 | 2.3068 | 2.8280 | 0.6082 | 0.5390 | 0.5342 | 8 |
| AGCC | 0.5104 | 1.8818 | 4.8287 | 0.9523 | 0.5356 | 0.5674 | 8 |

## Primary contrasts at N=1024 (Bonferroni α=0.0083, m=6)

| contrast | task | mean_a | mean_b | pairedΔ(a better+) | p | verdict |
|---|---|---:|---:|---:|---:|---|
| REAL_SC vs DEGREE_SHUFFLE | TRANSFER | 0.9978 | 0.9996 | -0.0018 | 0.7109 | NOT SUPPORTED |
| REAL_SC vs DEGREE_SHUFFLE | SPLIT_BIND | 0.6276 | 0.5907 | 0.0369 | 0.1562 | NOT SUPPORTED |
| REAL_SC vs DEGREE_SHUFFLE | DELAYED_MATCH | 0.6680 | 0.6279 | 0.0400 | 0.1914 | NOT SUPPORTED |
| REAL_SC vs NO_BRIDGE | TRANSFER | 0.9978 | 0.9988 | -0.0011 | 0.5938 | NOT SUPPORTED |
| REAL_SC vs DEGREE_SHUFFLE | NARMA | 2.0964 | 2.0013 | -0.0951 | 0.5781 | NOT SUPPORTED |
| REAL_SC vs WEIGHT_SHUFFLE | TRANSFER | 0.9978 | 0.9419 | 0.0559 | 0.003906 | SUPPORTED |

**Overall primary:** PARTIAL (some primary contrasts passed; map claim on TRANSFER did not)

Data: Griffa et al. 2019 Zenodo 2872624 + Caminiti/FA delay priors (local only).
Prereg: PREREG_callosum_scaled.md

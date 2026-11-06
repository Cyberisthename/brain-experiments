# QMS-1 RESULTS (20 seeds, 981 held-out test q)
| solver | true evals | wall s | opt-set gap to exact | =exact cfg | =default cfg | test correct | coverage | ans-cov | sel-acc | conf-wrong | boundary acc | multi-hop | F | test O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default_E_auto | 0 | 0.00 | 0.005 | 18/20 | 20/20 | 0.911 | 0.701 | 0.995 | 0.878 | 0.086 | 0.865 | 0.987 | 0.003 | +0.740 |
| exact | 1024 | 0.66 | 0.000 | 20/20 | 18/20 | 0.911 | 0.701 | 0.995 | 0.878 | 0.086 | 0.865 | 0.987 | 0.003 | +0.740 |
| random | 64 | 0.04 | 0.014 | 1/20 | 1/20 | 0.909 | 0.619 | 0.932 | 0.921 | 0.049 | 0.869 | 0.962 | 0.013 | +0.811 |
| greedy | 11 | 0.01 | 0.005 | 18/20 | 20/20 | 0.911 | 0.701 | 0.995 | 0.878 | 0.086 | 0.865 | 0.987 | 0.003 | +0.740 |
| sa | 64 | 0.04 | 0.005 | 18/20 | 20/20 | 0.911 | 0.701 | 0.995 | 0.878 | 0.086 | 0.865 | 0.987 | 0.003 | +0.740 |
| surrogate_classical | 64 | 0.05 | 0.087 | 0/20 | 0/20 | 0.859 | 0.635 | 0.903 | 0.873 | 0.081 | 0.806 | 0.962 | 0.003 | +0.698 |
| surrogate_qaoa_p1 | 64 | 0.28 | 0.116 | 0/20 | 0/20 | 0.826 | 0.597 | 0.845 | 0.869 | 0.078 | 0.777 | 0.924 | 0.003 | +0.669 |
| surrogate_qaoa_p2 | 64 | 1.47 | 0.087 | 0/20 | 0/20 | 0.870 | 0.645 | 0.919 | 0.875 | 0.081 | 0.822 | 0.962 | 0.003 | +0.708 |

T1 exact-optimum vs default conf-wrong: diff +0.0000, p 1.0000, Holm 1.0000 -> FAIL
T2 correct non-inferior 2pp: diff +0.0000, p 0.0005, Holm 0.0010 -> PASS

Surrogate R2 vs true objective (all 1024): median 0.68, min -0.44
QAOA vs classical surrogate argmax config differ: p1 9/20, p2 2/20; max test-correct diff 0.034
QAOA prob on surrogate optimum: p1 mean 0.0146, p2 0.0429 (uniform 0.0010)
Hardware rule (>=25% seeds differ OR >=2pp): MET
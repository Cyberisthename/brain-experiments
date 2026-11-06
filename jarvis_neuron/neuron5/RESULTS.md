# NEURON-5 RESULTS  (n=1045 q, 20 seeds)
| system | correct | coverage | ans-cov | sel-acc | conf-wrong | multi-hop | median ms | storage B | acc after +1000 | F | S(equal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact | 0.480 | 0.674 | 0.623 | 0.581 | 0.282 | 0.527 | 3.70 | 314830 | 0.488 | 0.000 | -0.586 |
| rag | 0.391 | 0.357 | 0.302 | 0.525 | 0.169 | 0.144 | 2.77 | 3637185 | 0.391 | 0.006 | -1.344 |
| A | 0.638 | 0.685 | 0.802 | 0.654 | 0.237 | 0.880 | 2.91 | 346773 | 0.646 | 0.003 | -0.044 |
| E_auto | 0.879 | 0.651 | 0.921 | 0.891 | 0.071 | 0.964 | 3.18 | 351982 | 0.879 | 0.000 | +0.138 |
| P_energy | 0.896 | 0.713 | 0.983 | 0.868 | 0.094 | 0.988 | 2.90 | 351931 | 0.896 | 0.000 | +0.230 |
- H5a conf_wrong P<best: diff +0.0230 p 0.0005 Holm 0.0015 FAIL
- H5b ans-coverage NI 10pp: diff +0.0623 p 0.0005 Holm 0.0015 PASS
- H5c latency P<best (log ms): diff -0.1502 p 0.0001 Holm 0.0004 PASS
- H5d correct NI 2pp: diff +0.0163 p 0.0005 Holm 0.0015 PASS
**Primary supported: True**
P_energy ranks first in 100% of 27 weight combos
SA/exact solver agreement: [274, 279]
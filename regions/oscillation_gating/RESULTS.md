# RESULTS — oscillation_gating (FULL PROTOCOL)

Prereg: `PREREG_oscillation_gating.md` (frozen 2026-10-30 07:34:00 EDT).

Primary (Bonferroni α=0.05/9): COHERENT vs {INCOHERENT, NO_GATE, RAND_MATCH} on XOR↑, CHAR↓, NARMA↓. Seeds 1100–1111 (n=12).

Hparams: `{"lr": 0.005, "leak": 0.4, "epochs": 45, "steps_per_epoch": 20, "batch_xor": 48, "batch_char": 40, "batch_narma": 32, "seq_xor": 36, "seq_char": 48, "seq_narma": 64, "char_train_chars": 30000, "wd": 0.0, "emb_dim": 32}`


## Summary — CTC + myelin arms (mean ± sd)

| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | n_params | nnz |
|---|---|---|---|---|---|
| COHERENT | 0.966±0.013 | 1.793±0.014 | 0.535±0.014 | 1608 | 1127 |
| INCOHERENT | 0.998±0.002 | 1.795±0.015 | 0.518±0.012 | 1608 | 1127 |
| NO_GATE | 1.000±0.000 | 1.773±0.016 | 0.501±0.017 | 1608 | 1127 |
| HUB | 0.971±0.013 | 1.792±0.014 | 0.534±0.023 | 1608 | 1127 |
| RAND_MATCH | 0.589±0.009 | 2.109±0.019 | 0.740±0.014 | 1608 | 1127 |
| MYELIN_ONLY | 1.000±0.000 | 1.783±0.015 | 0.473±0.030 | 1608 | 1127 |
| MYELIN_COHERENT | 0.968±0.009 | 1.791±0.015 | 0.516±0.013 | 1608 | 1127 |

## Summary — callosum arms (split-input XOR; mean ± sd)

| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | nnz | n_bridge |
|---|---|---|---|---|---|
| CALLOSUM | 1.000±0.000 | 1.697±0.016 | 0.490±0.018 | 1633 | 160 |
| RAND_BRIDGE | 1.000±0.000 | 1.696±0.017 | 0.496±0.019 | 1633 | 160 |
| NO_BRIDGE | 1.000±0.000 | 1.706±0.015 | 0.533±0.026 | 1473 | 0 |
| DENSE_BRIDGE | 1.000±0.000 | 1.568±0.018 | 0.454±0.017 | 14273 | 12800 |

## Primary tests (paired Wilcoxon, Bonferroni ×9)

| comparison | task | direction | mean_a | mean_b | p raw | p adj | pass? |
|---|---|---|---|---|---|---|---|
| COHERENT vs INCOHERENT | XOR | greater | 0.966 | 0.998 | 1 | 1 | no |
| COHERENT vs INCOHERENT | CHAR | less | 1.793 | 1.795 | 0.02612 | 0.2351 | no |
| COHERENT vs INCOHERENT | NARMA | less | 0.535 | 0.518 | 0.9995 | 1 | no |
| COHERENT vs NO_GATE | XOR | greater | 0.966 | 1.000 | 1 | 1 | no |
| COHERENT vs NO_GATE | CHAR | less | 1.793 | 1.773 | 1 | 1 | no |
| COHERENT vs NO_GATE | NARMA | less | 0.535 | 0.501 | 1 | 1 | no |
| COHERENT vs RAND_MATCH | XOR | greater | 0.966 | 0.589 | 0.0002441 | 0.002197 | YES |
| COHERENT vs RAND_MATCH | CHAR | less | 1.793 | 2.109 | 0.0002441 | 0.002197 | YES |
| COHERENT vs RAND_MATCH | NARMA | less | 0.535 | 0.740 | 0.0002441 | 0.002197 | YES |

**Primary CTC verdict:** NOT SUPPORTED — COHERENT does not beat INCOHERENT on all three primary endpoints after Bonferroni. (3/9 comparisons significant).


## Secondary: myelin (uncorrected raw p)

- MYELIN_ONLY vs NO_GATE on XOR (greater): 1.000 vs 1.000, raw p=1

- MYELIN_ONLY vs NO_GATE on NARMA (less): 0.473 vs 0.501, raw p=0.0002441

- MYELIN_ONLY vs NO_GATE on CHAR (less): 1.783 vs 1.773, raw p=1

- MYELIN_COHERENT vs MYELIN_ONLY on XOR (greater): 0.968 vs 1.000, raw p=1

- MYELIN_COHERENT vs MYELIN_ONLY on NARMA (less): 0.516 vs 0.473, raw p=1

- MYELIN_COHERENT vs MYELIN_ONLY on CHAR (less): 1.791 vs 1.783, raw p=1

- MYELIN_COHERENT vs COHERENT on XOR (greater): 0.968 vs 0.966, raw p=0.3667

- MYELIN_COHERENT vs COHERENT on NARMA (less): 0.516 vs 0.535, raw p=0.0004883

- MYELIN_COHERENT vs COHERENT on CHAR (less): 1.791 vs 1.793, raw p=0.1506

- MYELIN_COHERENT vs NO_GATE on XOR (greater): 0.968 vs 1.000, raw p=1

- MYELIN_COHERENT vs NO_GATE on NARMA (less): 0.516 vs 0.501, raw p=0.9998

- MYELIN_COHERENT vs NO_GATE on CHAR (less): 1.791 vs 1.773, raw p=1


**Myelin plain answer:** 
MYELIN_ONLY NARMA=0.473 XOR=1.000; MYELIN_COHERENT NARMA=0.516 XOR=0.968; NO_GATE NARMA=0.501 XOR=1.000; COHERENT NARMA=0.535 XOR=0.966. 
Myelin+sync does **not** clearly beat both myelin-alone and sync-alone on XOR/NARMA (raw p≥0.05 on at least one required contrast).


## Secondary: corpus callosum (uncorrected raw p; split-input XOR)

- CALLOSUM vs RAND_BRIDGE on XOR (greater): 1.000 vs 1.000, raw p=1

- CALLOSUM vs RAND_BRIDGE on NARMA (less): 0.490 vs 0.496, raw p=0.1018

- CALLOSUM vs RAND_BRIDGE on CHAR (less): 1.697 vs 1.696, raw p=0.7153

- CALLOSUM vs NO_BRIDGE on XOR (greater): 1.000 vs 1.000, raw p=1

- CALLOSUM vs NO_BRIDGE on NARMA (less): 0.490 vs 0.533, raw p=0.0004883

- CALLOSUM vs NO_BRIDGE on CHAR (less): 1.697 vs 1.706, raw p=0.0002441

- CALLOSUM vs DENSE_BRIDGE on XOR (greater): 1.000 vs 1.000, raw p=1

- CALLOSUM vs DENSE_BRIDGE on NARMA (less): 0.490 vs 0.454, raw p=1

- CALLOSUM vs DENSE_BRIDGE on CHAR (less): 1.697 vs 1.568, raw p=1


**Callosum plain answer:** 
CALLOSUM XOR=1.000; 
RAND_BRIDGE XOR=1.000; 
NO_BRIDGE XOR=1.000; 
DENSE_BRIDGE XOR=1.000; 
CALLOSUM does **not** clearly beat random bridge and no bridge on split XOR.


## Secondary: HUB (descriptive)

- HUB vs COHERENT on XOR (greater): raw p=0.1331

- HUB vs COHERENT on CHAR (less): raw p=0.311

- HUB vs COHERENT on NARMA (less): raw p=0.2847

- HUB vs RAND_MATCH on XOR (greater): raw p=0.0002441

- HUB vs RAND_MATCH on CHAR (less): raw p=0.0002441

- HUB vs RAND_MATCH on NARMA (less): raw p=0.0002441


## Learning curves
See `figs/oscillation_gating.png`.


## Honesty notes

- Full protocol: n=12 seeds (1100–1111), XOR+CHAR+NARMA, N=160, epochs≈45 × 20 steps.

- Fast pilot (2026-10-29) was undertrained (XOR≈chance); baselines here required XOR≥0.70 on tune seeds.

- Hyperparams tuned on seeds 90–91 only (NO_GATE + RAND_MATCH floor + grid).

- Gate mean-normalized so COHERENT vs INCOHERENT differ in timing, not mean gain.

- Myelin delays from real H01 myelin_frac→{1..5}, RULE_FF_DEPTH assignment on inter edges.

- Callosum DENSE_BRIDGE has higher nnz (not edge-matched); others edge-matched to CALLOSUM.

- Null allowed.


## Headline answers (for report-back)

1. **Coherence beats incoherence?** NO (primary CTC).

2. **Myelin+sync beats either alone?** NO / unclear (raw).

3. **Callosum beats random + no bridge?** NO / partial (raw XOR).


## Secondary FLY followup
See `RESULTS_FLY_FOLLOWUP.md` (separate labeled secondary run of FLY EI_SBM vs NO_GATE / MYELIN_ONLY / CALLOSUM; not part of primary prereg).


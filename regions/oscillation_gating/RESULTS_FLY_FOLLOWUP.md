# RESULTS — SECONDARY FLY FOLLOWUP (separate from primary CTC eval)

Labeled secondary run. Same oscillation_gating harness, seeds 1100–1111, epochs/hparams as main. FLY = fly CX / EI_SBM wiring from `bio_wired_net` (FRAC_E≈0.592, EI block dens scaled to match modular nnz), Dale softplus×esign, delay=1, no oscillation gate.

**Not part of the primary preregistered CTC analysis.** Compared fairly against NO_GATE (modular), MYELIN_ONLY, and CALLOSUM from the main eval.

Meta: fly_nnz=1127 modular_nnz=1127 dens_scale=1.255615234375 frac_E=0.59375


## Summary (mean ± sd)

| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | n_params | nnz | notes |
|---|---|---|---|---|---|---|
| FLY | 0.782±0.033 | 2.028±0.014 | 0.705±0.017 | 1608 | 1127 | EI_SBM Dale; delay=1; no gate |
| NO_GATE | 1.000±0.000 | 1.773±0.016 | 0.501±0.017 | 1608 | 1127 | modular; delay=2 inter; no gate |
| MYELIN_ONLY | 1.000±0.000 | 1.783±0.015 | 0.473±0.030 | 1608 | 1127 | modular; H01 RULE_FF_DEPTH delays |
| CALLOSUM | 1.000±0.000 | 1.697±0.016 | 0.490±0.018 | 2274 | 1633 | hemi split + homotopic bridge; split XOR |

## Pairwise vs FLY (paired Wilcoxon, uncorrected raw p)

- FLY vs NO_GATE on XOR (FLY greater): 0.782 vs 1.000, raw p=1, FLY_better=False

- FLY vs NO_GATE on CHAR (FLY less): 2.028 vs 1.773, raw p=1, FLY_better=False

- FLY vs NO_GATE on NARMA (FLY less): 0.705 vs 0.501, raw p=1, FLY_better=False

- FLY vs MYELIN_ONLY on XOR (FLY greater): 0.782 vs 1.000, raw p=1, FLY_better=False

- FLY vs MYELIN_ONLY on CHAR (FLY less): 2.028 vs 1.783, raw p=1, FLY_better=False

- FLY vs MYELIN_ONLY on NARMA (FLY less): 0.705 vs 0.473, raw p=1, FLY_better=False

- FLY vs CALLOSUM on XOR (FLY greater): 0.782 vs 1.000, raw p=1, FLY_better=False

- FLY vs CALLOSUM on CHAR (FLY less): 2.028 vs 1.697, raw p=1, FLY_better=False

- FLY vs CALLOSUM on NARMA (FLY less): 0.705 vs 0.490, raw p=1, FLY_better=False


## Plain comparison

- XOR: FLY=0.782, NO_GATE=1.000, MYELIN_ONLY=1.000, CALLOSUM=1.000

- CHAR: FLY=2.028, NO_GATE=1.773, MYELIN_ONLY=1.783, CALLOSUM=1.697

- NARMA: FLY=0.705, NO_GATE=0.501, MYELIN_ONLY=0.473, CALLOSUM=0.490


Note: CALLOSUM uses split-input XOR (different task formulation) — CHAR/NARMA are the cleaner cross-architecture compares; XOR vs CALLOSUM is descriptive only.


Null allowed. Secondary / exploratory — not Bonferroni-corrected.


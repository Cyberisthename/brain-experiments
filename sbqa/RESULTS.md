# RESULTS — SBQA (Simulated Bio-Quantum Annealing)

**Classical / quantum-inspired only. Not real qubits.**

Prereg: `PREREG_SBQA.md`. Outputs in `/tmp/sbqa_chk`.

## Discrete wiring search (NARMA NRMSE ↓)

| Method | mean ± std (n) |
|--------|----------------|
| sbqa | 0.5908 ± 0.0629 (n=8) |
| sa | 0.5717 ± 0.0533 (n=8) |
| random | 0.5771 ± 0.0508 (n=8) |
| greedy | 0.5814 ± 0.0588 (n=8) |

### Primary tests (Wilcoxon, SBQA better = lower)

- **sbqa_vs_sa**: mean 0.5908 vs 0.5717, wins 3/8, p=0.9453
- **sbqa_vs_random**: mean 0.5908 vs 0.5771, wins 4/8, p=0.875
- **sbqa_vs_greedy**: mean 0.5908 vs 0.5814, wins 3/8, p=0.875

## Continuous weight training (NARMA NRMSE ↓) — negative control

| Method | mean ± std (n) |
|--------|----------------|
| sbqa | 0.4531 ± 0.0128 (n=8) |
| adam | 0.4479 ± 0.0066 (n=8) |
| restarts | 0.5318 ± 0.0179 (n=8) |
| sgld | 0.4563 ± 0.0213 (n=8) |

- **sbqa_vs_adam**: mean 0.4531 vs 0.4479, wins 5/8, p=0.6797

## Ablation

- sbqa: 0.5876 ± 0.0420 (n=8)
- sbqa_metropolis: 0.5993 ± 0.0571 (n=8)
- sbqa_nomask: 0.5844 ± 0.0600 (n=8)

## Verdicts

- **H1 (discrete SBQA beats SA & RANDOM): NOT SUPPORTED** at Bonferroni-corrected α.
- **H2 (Adam ≥ SBQA on continuous): CONSISTENT** — SBQA did not significantly beat Adam (smooth-landscape prior).
- **H3 (ablation):** see means above; no strong claim without full n=8.

Honest note: smoke / partial runs are under-powered; trust full Kaggle n=8.

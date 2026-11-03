# RESULTS — SBQA (Simulated Bio-Quantum Annealing)

**Classical / quantum-inspired only. Not real qubits.**

Prereg: `PREREG_SBQA.md`. Outputs in `/workspace/flybrain/sbqa/out`.

## Discrete wiring search (NARMA NRMSE ↓)

| Method | mean ± std (n) |
|--------|----------------|
| sbqa | 0.9511 ± 0.0000 (n=1) |
| sa | 0.9138 ± 0.0000 (n=1) |
| random | 0.7468 ± 0.0000 (n=1) |
| greedy | 0.9514 ± 0.0000 (n=1) |

### Primary tests (Wilcoxon, SBQA better = lower)

- **sbqa_vs_sa**: mean 0.9511 vs 0.9138, wins 0/1, p=1
- **sbqa_vs_random**: mean 0.9511 vs 0.7468, wins 0/1, p=1
- **sbqa_vs_greedy**: mean 0.9511 vs 0.9514, wins 1/1, p=1

## Continuous weight training (NARMA NRMSE ↓) — negative control

| Method | mean ± std (n) |
|--------|----------------|
| sbqa | 1.0305 ± 0.0000 (n=1) |
| adam | 1.0305 ± 0.0000 (n=1) |
| restarts | 1.0305 ± 0.0000 (n=1) |
| sgld | 0.9862 ± 0.0000 (n=1) |

- **sbqa_vs_adam**: mean 1.0305 vs 1.0305, wins 0/1, p=1

## Ablation

- sbqa: 0.9711 ± 0.0000 (n=1)
- sbqa_metropolis: 0.8787 ± 0.0000 (n=1)
- sbqa_nomask: 1.1576 ± 0.0000 (n=1)

## Verdicts

- **H1 (discrete SBQA beats SA & RANDOM): NOT SUPPORTED** at Bonferroni-corrected α.
- **H2 (Adam ≥ SBQA on continuous): CONSISTENT** — SBQA did not significantly beat Adam (smooth-landscape prior).
- **H3 (ablation):** see means above; no strong claim without full n=8.

Honest note: smoke / partial runs are under-powered; trust full Kaggle n=8.

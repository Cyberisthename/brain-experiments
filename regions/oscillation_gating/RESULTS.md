# RESULTS — oscillation_gating (FAST PILOT)

Prereg: `PREREG_oscillation_gating.md` (frozen 2026-10-29 13:57:30 EDT).

Primary (Bonferroni α=0.05/6): COHERENT vs {INCOHERENT, NO_GATE, RAND_MATCH} on XOR↑ and CHAR↓. Seeds 1100–1105 (n=6 pilot).


## Summary (mean ± sd over seeds)

| arm | XOR acc ↑ | CHAR CE ↓ | n_params | nnz |
|---|---|---|---|---|
| COHERENT | 0.514±0.015 | 2.578±0.029 | 1270 | 885 |
| INCOHERENT | 0.521±0.028 | 2.577±0.029 | 1270 | 885 |
| NO_GATE | 0.526±0.034 | 2.577±0.029 | 1270 | 885 |
| HUB | 0.516±0.013 | 2.578±0.029 | 1270 | 885 |
| RAND_MATCH | 0.495±0.041 | 2.568±0.024 | 1270 | 885 |

## Primary tests (paired Wilcoxon, Bonferroni)

| comparison | task | direction | p raw | p adj | pass? |
|---|---|---|---|---|---|
| COHERENT vs INCOHERENT | XOR | greater | 0.8906 | 1 | no |
| COHERENT vs INCOHERENT | CHAR | less | 1 | 1 | no |
| COHERENT vs NO_GATE | XOR | greater | 0.9219 | 1 | no |
| COHERENT vs NO_GATE | CHAR | less | 1 | 1 | no |
| COHERENT vs RAND_MATCH | XOR | greater | 0.1562 | 0.9375 | no |
| COHERENT vs RAND_MATCH | CHAR | less | 1 | 1 | no |

**Primary verdict:** NOT SUPPORTED (pilot) — COHERENT does not beat INCOHERENT on both primary endpoints after Bonferroni. (0/6 comparisons significant).


## Secondary: HUB (descriptive)

- HUB vs COHERENT on XOR (greater): raw p=0.2188 (uncorrected)

- HUB vs COHERENT on CHAR (less): raw p=0.4219 (uncorrected)

- HUB vs RAND_MATCH on XOR (greater): raw p=0.1562 (uncorrected)

- HUB vs RAND_MATCH on CHAR (less): raw p=1 (uncorrected)


## Learning curves
See `figs/oscillation_gating.png` (mean metric vs epoch ± SEM).


## Honesty notes

- Fast pilot: n=6 seeds (1100–1105), XOR+CHAR only, N=128, epochs≈25.

- Hyperparams tuned on seeds 90–91 only (COHERENT + INCOHERENT).

- Gate mean-normalized so COHERENT vs INCOHERENT differ in timing, not mean gain.

- Null allowed; low power — provisional.


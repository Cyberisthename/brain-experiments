# Results: synthetic hippocampus on the fly CX
Run 2026-10-27 per PREREGISTRATION_hippo.md (literature and design there). Code: hippo.py; raw: out/jobs.jsonl (416 jobs: 288 tuning + 128 test); chosen leak/ρ: out/best.json; analysis: analyze.py.
Deviations: none from the design. Hebbian CA3 plasticity was optional and NOT run. Readout asymmetry as preregistered (CX-alone reads 2,875 neurons; module conditions read 3,275).

## Test results (8 seeds each; mean ± sd. MC and PC ↑ better; NARMA, T0 ↓ better)
| task   | CX              | RAND             | SHUF             | HIPPO          |   p H vs CX |   p H vs RAND |   p H vs SHUF |
|:-------|:----------------|:-----------------|:-----------------|:---------------|------------:|--------------:|--------------:|
| MC     | 13.59 ± 0.55    | 14.57 ± 0.13     | 13.62 ± 0.47     | 13.74 ± 0.32   |    0.328    |      0.000155 |       0.878   |
| NARMA  | 0.4805 ± 0.027  | 0.3619 ± 0.023   | 0.3438 ± 0.09    | 0.3281 ± 0.027 |    0.000155 |      0.0207   |       0.878   |
| PC     | 0.7578 ± 0.034  | 0.71 ± 0.023     | 0.6759 ± 0.028   | 0.7256 ± 0.015 |    0.0311   |      0.171    |       0.00621 |
| T0     | 0.06397 ± 0.018 | 0.04649 ± 0.0084 | 0.04293 ± 0.0098 | 0.047 ± 0.0079 |    0.0281   |      1        |       0.442   |

Worst grid cell (tuning seeds; MC/PC lowest, NARMA/T0 highest):
| task   |     CX |   RAND |   SHUF |   HIPPO |
|:-------|-------:|-------:|-------:|--------:|
| MC     | 6.7014 | 6.9795 | 4.3348 |  6.3657 |
| NARMA  | 0.6669 | 0.5853 | 0.5928 |  0.592  |
| PC     | 0.6075 | 0.5775 | 0.415  |  0.575  |
| T0     | 0.0696 | 0.0639 | 0.0634 |  0.0681 |

Bonferroni (α=0.05/12) survivors: 2 of 12

Bonferroni survivors: HIPPO beats CX-alone on NARMA (0.328 vs 0.481, p=1.6e-4); RAND beats HIPPO on MC (14.57 vs 13.74, p=1.6e-4).

## Plain summary
- Adding ~3,000 extra neurons helped NARMA and T0 a lot, but random modules helped just as much as the hippocampus. Hippocampal structure itself bought nothing measurable on those tasks.
- On memory (MC), the random module was best; the hippocampus was no better than CX alone.
- On pattern completion — the task meant to favour a hippocampus — **CX alone was best** (75.8% vs 72.6% hippocampus, p=0.03, not Bonferroni). Hippocampus did beat its shuffled version (72.6% vs 67.6%, p=0.006, not Bonferroni) and was ≈ random module (71.0%).
- Robustness: hippocampus worst-case was about the same as CX alone (slightly worse on MC/PC, slightly better on NARMA/T0); the shuffled hippocampus was the least robust (MC worst 4.3, PC worst 0.42).

## Predictions vs outcome
- Q1 (HIPPO doesn't beat RAND on MC/NARMA/T0): **held** (RAND actually better on MC).
- Q2 (modules beat CX alone on MC): **partly** — RAND yes, HIPPO and SHUF no.
- Q3 (HIPPO beats CX on PC, ≈ RAND/SHUF): **wrong on the first part** (CX alone better); HIPPO ≈ RAND held; HIPPO > SHUF (uncorrected), so the wiring order matters somewhat vs a scrambled version.
- Q4 (HIPPO robustness no worse than CX): **roughly held** (small differences both ways).

## Caveats
No plasticity: a real hippocampus stores patterns by changing CA3 synapses (Hebbian), so this tests the wiring alone, which is exactly the case theory says shouldn't help much. kWTA DG with tanh rate units is a crude model. Many tuning winners sit at the grid edge (leak 0.9 / ρ 1.1). PC readout is a ridge fit that can do "completion" itself, which likely favours the plain CX. Next step would be Hebbian CA3 storage (exploratory).

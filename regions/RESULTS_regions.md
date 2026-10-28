# Results: growing the fly CX with real regions vs random neurons
Run 2026-10-27 per PREREGISTRATION_regions.md, with the deviations listed below. Runner: run_reduced.py. Raw per-job results: out/jobs.jsonl. Analysis: analyze_reduced.py. Graph sizes: out/steps.json.

## Deviations from the preregistration (decided before S3/S4 results; for compute/RAM, user's machine shared)
- S0–S2: full preregistered plan (3×3 leak/ρ grid, 2 tuning seeds, 8 test seeds, MC/NARMA/T0).
- S3 and S4: **MC and T0 only (NARMA dropped), 1 tuning seed (900), 4 test seeds (0–3).** With n=4 vs 4 the smallest possible Mann–Whitney p is 0.029, so no S3/S4 result can survive Bonferroni.
- S0 "ADD" is identical to REAL by construction (nothing is added), so the S0 REAL-vs-ADD comparison is trivially null.
- Run with 3 worker processes, nice 10, single-threaded BLAS.

## Test results (mean ± sd; MC ↑ better, NARMA NRMSE and T0 MSE ↓ better)
| step                      |     N | task   | REAL                   | ADD                   | SHUF                   |   p_vs_ADD |   p_vs_SHUF |
|:--------------------------|------:|:-------|:-----------------------|:----------------------|:-----------------------|-----------:|------------:|
| S0 CX                     |  2875 | MC     | 13.59 ± 0.55 (n=8)     | 13.59 ± 0.55 (n=8)    | 14.62 ± 0.74 (n=8)     |     1      |    0.00466  |
| S0 CX                     |  2875 | NARMA  | 0.4805 ± 0.027 (n=8)   | 0.4805 ± 0.027 (n=8)  | 0.1344 ± 0.012 (n=8)   |     0.959  |    0.000155 |
| S0 CX                     |  2875 | T0     | 0.06397 ± 0.018 (n=8)  | 0.06397 ± 0.018 (n=8) | 0.02019 ± 0.0031 (n=8) |     0.878  |    0.000155 |
| S1 +mushroom body         |  8479 | MC     | 13.13 ± 0.46 (n=8)     | 13.04 ± 0.43 (n=8)    | 13.12 ± 0.4 (n=8)      |     0.505  |    0.959    |
| S1 +mushroom body         |  8479 | NARMA  | 0.4922 ± 0.02 (n=8)    | 0.4928 ± 0.02 (n=8)   | 0.1943 ± 0.04 (n=8)    |     0.798  |    0.000155 |
| S1 +mushroom body         |  8479 | T0     | 0.0626 ± 0.017 (n=8)   | 0.0632 ± 0.017 (n=8)  | 0.03906 ± 0.011 (n=8)  |     0.959  |    0.00466  |
| S2 +LH/olfactory PN/TuBu  |  9870 | MC     | 13.73 ± 0.28 (n=8)     | 13.33 ± 0.44 (n=8)    | 12.63 ± 1.6 (n=8)      |     0.0281 |    0.0207   |
| S2 +LH/olfactory PN/TuBu  |  9870 | NARMA  | 0.4869 ± 0.025 (n=8)   | 0.4941 ± 0.022 (n=8)  | 0.2889 ± 0.17 (n=8)    |     0.645  |    0.0104   |
| S2 +LH/olfactory PN/TuBu  |  9870 | T0     | 0.06586 ± 0.024 (n=8)  | 0.06414 ± 0.022 (n=8) | 0.04548 ± 0.0099 (n=8) |     0.721  |    0.0207   |
| S3 +rest of central brain | 32383 | MC     | 9.159 ± 0.5 (n=4)      | 13.59 ± 0.37 (n=4)    | 14.79 ± 0.15 (n=4)     |     0.0286 |    0.0286   |
| S3 +rest of central brain | 32383 | T0     | 0.07759 ± 0.021 (n=4)  | 0.06765 ± 0.029 (n=4) | 0.02352 ± 0.0031 (n=4) |     0.343  |    0.0286   |
| S4 +visual projection     | 40421 | MC     | 9.827 ± 0.21 (n=4)     | 13.78 ± 0.12 (n=4)    | 15.22 ± 0.47 (n=4)     |     0.0286 |    0.0286   |
| S4 +visual projection     | 40421 | T0     | 0.07263 ± 0.0094 (n=4) | 0.05875 ± 0.013 (n=4) | 0.01734 ± 0.0034 (n=4) |     0.2    |    0.0286   |

|   step |   ('worst_MC', 'ADD') |   ('worst_MC', 'REAL') |   ('worst_MC', 'SHUF') |   ('std_MC', 'ADD') |   ('std_MC', 'REAL') |   ('std_MC', 'SHUF') |   ('frac_within50', 'ADD') |   ('frac_within50', 'REAL') |   ('frac_within50', 'SHUF') |
|-------:|----------------------:|-----------------------:|-----------------------:|--------------------:|---------------------:|---------------------:|---------------------------:|----------------------------:|----------------------------:|
|      0 |                  6.7  |                   6.7  |                   0.5  |                2.21 |                 2.21 |                 5.79 |                       0.89 |                        0.89 |                        0.44 |
|      1 |                  6.57 |                   6.6  |                   3.43 |                2.13 |                 2.15 |                 3.17 |                       1    |                        1    |                        0.67 |
|      2 |                  6.52 |                   6.61 |                   4.14 |                2.16 |                 2.33 |                 3.4  |                       0.89 |                        0.89 |                        0.78 |
|      3 |                  5.64 |                   5.23 |                   0.66 |                2.65 |                 1.74 |                 5.88 |                       0.67 |                        1    |                        0.44 |
|      4 |                  5.49 |                   5.16 |                   1.62 |                2.43 |                 1.71 |                 4.82 |                       0.67 |                        1    |                        0.56 |

(Second table: robustness from the tuning grid, MC across the 9 leak/ρ cells — worst cell, std, fraction of cells within 50% of the best.)
Significance: 15 of 26 REAL-vs-null comparisons p<0.05 uncorrected; 3 survive Bonferroni (0.05/26), all from S0 REAL vs SHUF (NARMA, T0) and S1 NARMA vs SHUF.

## Plain summary
- Adding the real mushroom body / LH / TuBu (S1–S2, up to 9,870 neurons) changed almost nothing, and real regions did about the same as the same neurons with random wiring (ADD). Only S2 MC shows real slightly ahead of ADD (13.73 vs 13.33, p=0.028, not Bonferroni-significant).
- Adding the rest of the central brain (S3, 32k neurons) or visual projection (S4, 40k) made the real network **worse** at memory (MC ~9.2–9.8 vs 13.6 at S0), while random-wired additions stayed ~13.6–13.8 and fully shuffled graphs ~14.8–15.2. T0: real ≈ ADD (n.s.), both far worse than SHUF.
- Degree-shuffled graphs beat the real fly wiring on NARMA and T0 at every step (e.g. S0 NARMA 0.13 vs 0.48).
- Robustness: the real wiring keeps a much better worst-case MC than SHUF at every step (e.g. S3 5.2 vs 0.7) and the smallest spread; vs ADD it is about the same (ADD keeps the real CX core, which seems to carry the robustness).

## Predictions vs outcome
- P1 (growing helps everyone a bit on MC/NARMA): **mostly wrong.** Real got worse at S3/S4; ADD flat; SHUF roughly flat on MC and got worse on NARMA (0.13→0.29 by S2).
- P2 (REAL doesn't beat ADD/SHUF on tuned performance): **right.** It never beats SHUF; vs ADD it's equal except S2 MC (small real edge) and S3/S4 MC (real clearly worse).
- P3 (REAL more robust worst-case MC at every step): **right vs SHUF**; vs ADD roughly tied (ADD slightly higher worst-case at S3/S4, but REAL has lower spread and more cells within 50%).
- P4 (real regions help more than random additions): **predicted no — held.** Real regions did not help more; at whole-brain scale they hurt memory.

## Caveats
Fixed 2,875-neuron random readout subset at all sizes (so bigger brains get no bigger readout); tiny tuning grid where many winners sit at the edge (ρ=1.1); one rate model with synapse counts as weights; S3/S4 have only 4 seeds and no NARMA.

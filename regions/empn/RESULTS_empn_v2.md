# EMPN v2 results — eligibility / RPE + shared learner fixes

Prereg: `PREREGISTRATION_empn_v2.md` (diagnosis addenda locked before this run).

160 jobs (10 arms × ±TR × 8 seeds).

## Shared fixes (all arms)
1. Block-norm readout: DN ⊕ (z if RANDMOD) ⊕ obs ⊕ bias — fixes flat DN drowning Markov obs.
2. Explore-until-first-food; ε 1→0.1; eval ε=0.05; count bonus; α=0.1; Wq±5.
3. CFS goals gated until ≥5 food memories (risk term kept).
4. +TR: pure three-factor dW=η·e·(R−R̄); τ tuned on RANDMOD+TR seeds 900/901.

## Success and risk (mean over seeds)

| Agent | success | train food | threat loom | steps | short safe | short loom | sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| RANDMOD | 0.565 | 0.200 | 0.102 | 24.9 | 0.551 | 0.102 | 26.9 |
| EMPN-noCFS | 0.500 | 0.188 | 0.132 | 25.2 | 0.612 | 0.132 | 42.4 |
| FLY | 0.462 | 0.163 | 0.124 | 26.3 | 0.549 | 0.124 | 27.9 |
| RANDMOD+TR | 0.123 | 0.125 | 0.256 | 25.8 | 0.368 | 0.256 | 31.5 |
| EMPN+TR | 0.106 | 0.038 | 0.028 | 24.5 | 0.250 | 0.028 | 105.4 |
| EMPN+SCNC+MGM+TR | 0.102 | 0.025 | 0.031 | 30.6 | 0.289 | 0.031 | 100.4 |
| EMPN-noCFS+TR | 0.092 | 0.050 | 0.166 | 23.5 | 0.284 | 0.166 | 88.3 |
| EMPN+SCNC+TR | 0.087 | 0.038 | 0.022 | 30.5 | 0.198 | 0.022 | 99.3 |
| EMPN+SCNC+ALL3+TR | 0.077 | 0.075 | 0.043 | 22.1 | 0.323 | 0.043 | 114.6 |
| EMPN+SCNC+CVB+TR | 0.033 | 0.038 | 0.027 | 44.2 | 0.126 | 0.027 | 104.2 |
| EMPN+SCNC+CTB+TR | 0.031 | 0.050 | 0.014 | 34.6 | 0.145 | 0.014 | 133.8 |
| EMPN+SCNC | 0.027 | 0.013 | 0.034 | 16.4 | 0.102 | 0.034 | 86.1 |
| FLY+TR | 0.025 | 0.038 | 0.154 | 28.3 | 0.224 | 0.154 | 33.2 |
| EMPN+SCNC+CTB | 0.023 | 0.038 | 0.030 | 29.4 | 0.161 | 0.030 | 162.0 |
| EMPN+SCNC+CVB | 0.021 | 0.013 | 0.005 | 43.9 | 0.101 | 0.005 | 85.2 |
| EMPN+SCNC+ALL3 | 0.012 | 0.013 | 0.009 | 46.2 | 0.141 | 0.009 | 164.3 |
| EMPN+SCNC+ALL3RAND+TR | 0.010 | 0.025 | 0.022 | 30.0 | 0.096 | 0.022 | 93.2 |
| EMPN+SCNC+MGM | 0.010 | 0.000 | 0.020 | 47.5 | 0.120 | 0.020 | 86.7 |
| EMPN+SCNC+ALL3RAND | 0.004 | 0.025 | 0.069 | 42.0 | 0.254 | 0.069 | 86.0 |
| EMPN | 0.000 | 0.038 | 0.059 | nan | 0.254 | 0.059 | 45.5 |

## +TR vs no-TR

| Base | noTR | +TR | Δ |
|---|---:|---:|---:|
| EMPN | 0.000 | 0.106 | +0.106 |
| EMPN+SCNC | 0.027 | 0.087 | +0.060 |
| EMPN+SCNC+ALL3 | 0.012 | 0.077 | +0.065 |
| EMPN+SCNC+ALL3RAND | 0.004 | 0.010 | +0.006 |
| EMPN+SCNC+CTB | 0.023 | 0.031 | +0.008 |
| EMPN+SCNC+CVB | 0.021 | 0.033 | +0.013 |
| EMPN+SCNC+MGM | 0.010 | 0.102 | +0.092 |
| EMPN-noCFS | 0.500 | 0.092 | -0.408 |
| FLY | 0.462 | 0.025 | -0.437 |
| RANDMOD | 0.565 | 0.123 | -0.442 |

## Learning curves (train food rate / 20-ep bin)

| ep | FLY | RANDMOD | EMPN | EMPN-noCFS | EMPN+SCNC | FLY+TR | RANDMOD+TR | EMPN+TR | EMPN-noCFS+TR |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 39 | 0.00 | 0.00 | 0.25 | 0.12 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 |
| 59 | 0.25 | 0.25 | 0.00 | 0.12 | 0.00 | 0.00 | 0.25 | 0.00 | 0.00 |
| 79 | 0.00 | 0.12 | 0.00 | 0.12 | 0.00 | 0.12 | 0.00 | 0.00 | 0.00 |
| 99 | 0.12 | 0.25 | 0.00 | 0.38 | 0.00 | 0.00 | 0.25 | 0.00 | 0.00 |
| 119 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.25 | 0.25 | 0.00 | 0.12 |
| 139 | 0.38 | 0.38 | 0.00 | 0.38 | 0.00 | 0.00 | 0.12 | 0.25 | 0.12 |
| 159 | 0.38 | 0.62 | 0.00 | 0.38 | 0.00 | 0.00 | 0.12 | 0.00 | 0.12 |
| 179 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 199 | 0.25 | 0.25 | 0.00 | 0.38 | 0.00 | 0.00 | 0.12 | 0.12 | 0.12 |

## Honest takeaway

Best arm: **RANDMOD** at success 0.565. RANDMOD 0.565 vs EMPN 0.000. EMPN-noCFS 0.500 shows CFS goals still hurt when TBU stores. Shared block-norm readout lifted FLY off the floor; pure eligibility +TR did not systematically beat TD.

See also `DIAGNOSIS_organ_harm.md` and `PREREGISTRATION_empn_v2_organ_fix.md` for the v2b fix rerun.

Raw: `out/runs_v2.jsonl`. Curves: `out/curves_v2.jsonl`.

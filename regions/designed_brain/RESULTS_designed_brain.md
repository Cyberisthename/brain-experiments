# Designed brain (CHIMERA) results

Prereg: `PREREG_designed_brain.md` (locked 2026-10-28 19:37 America/New_York, before eval).
Rows kept: 864. Eval seeds: 500–523 (n=24).

## Design in plain words

A synthetic 4000-neuron directed network whose excitatory/inhibitory labels and four E/I connection-class densities+weight magnitudes are matched to FlyWire CX statistics (stochastic block model; no copied edges). Neurons get cortical-like layers; each neuron's myelin fraction is drawn from the real H01 distribution for that layer, then edge delays follow the same myelin→delay map used in the human patch (more myelin → shorter delay). Weak spectral tempering keeps the bulk of large eigenvalues from filling like an ER graph.

- N=4000, nnz=570086, frac_E=0.59175, spectral frac\|λ\|>0.5 after temper=0.03125, mf_mean=0.0670882630244726, layers={'L2/3': 2800, 'L4': 560, 'L5': 400, 'L6': 240}.

## Ridge α (90–92)

| Arm | XOR | NARMA | MC | PC_HARD |
|---|---:|---:|---:|---:|
| `DESIGN` | 1e-08 | 1e-08 | 1e-08 | 0.1 |
| `DESIGN_EI_ONLY` | 1e-08 | 1e-08 | 1e-08 | 0.1 |
| `DESIGN_MYELIN_ONLY` | 0.0001 | 1e-08 | 1e-06 | 1e-08 |
| `FLY_CX` | 1e-08 | 1e-08 | 1e-08 | 0.01 |
| `HUMAN` | 0.0001 | 1e-08 | 1e-08 | 1e-08 |
| `TEAM_ER` | 0.0001 | 1e-08 | 1e-06 | 0.1 |
| `ER` | 1e-06 | 1e-08 | 1e-06 | 1.0 |
| `DENS` | 0.01 | 1e-06 | 0.0001 | 1.0 |
| `SHUF` | 0.01 | 1e-06 | 0.0001 | 10.0 |

Composite z (tune): μ_xor=0.3348, σ_xor=0.2886, μ_narma=0.3776, σ_narma=0.0520.

## Means

| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ | MC ↑ | PC_HARD ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `DESIGN` | 4000 | 0.2010±0.0211 | 0.4279±0.0066 | -0.7154±0.0743 | 19.8599±0.2014 | 0.8884±0.0063 |
| `DESIGN_EI_ONLY` | 4000 | 0.8401±0.0033 | 0.3884±0.0062 | 0.7708±0.0605 | 17.4821±0.1641 | 0.8782±0.0063 |
| `DESIGN_MYELIN_ONLY` | 4000 | 0.1336±0.0052 | 0.3062±0.0048 | 0.3373±0.0483 | 21.8164±0.2104 | 0.9069±0.0116 |
| `FLY_CX` | 2875 | 0.6561±0.0097 | 0.4466±0.0067 | -0.1068±0.0670 | 17.8480±0.2238 | 0.8525±0.0098 |
| `HUMAN` | 2087 | 0.1212±0.0045 | 0.3961±0.0070 | -0.5480±0.0678 | 18.6725±0.1967 | 0.8590±0.0129 |
| `TEAM_ER` | 4962 | 0.2746±0.0136 | 0.2917±0.0051 | 0.7211±0.0635 | 20.7455±0.2440 | 0.9026±0.0100 |
| `ER` | 4000 | 0.6374±0.0167 | 0.3288±0.0056 | 0.9933±0.0656 | 18.9118±0.2043 | 0.9112±0.0056 |
| `DENS` | 4000 | 0.1602±0.0055 | 0.3769±0.0068 | -0.2960±0.0647 | 19.4200±0.2582 | 0.9473±0.0058 |
| `SHUF` | 4000 | 0.1593±0.0052 | 0.3951±0.0100 | -0.4729±0.0981 | 19.1578±0.3033 | 0.9484±0.0069 |

## Primary hypotheses (Bonferroni ×4)

| Hypothesis | mean Δ | 95% CI | d_z | p_raw | p_bonf | pass |
|---|---:|---:|---:|---:|---:|:---:|
| H1 DESIGN>ER composite | -1.7087 | [-1.8206, -1.5968] | -6.449 | 1 | 1 | ✗ |
| H2 DESIGN>TEAM_ER composite | -1.4365 | [-1.5751, -1.2978] | -4.374 | 1 | 1 | ✗ |
| H3 XOR non-inf vs FLY_CX (m=0.05) | -0.4051 | [-0.4520, -0.3581] | -3.642 | 1 | 1 | ✗ |
| H4 NARMA non-inf vs HUMAN (m=0.03) | -0.0018 | [-0.0089, +0.0053] | -0.107 | 0.6159 | 1 | ✗ |

**Overall verdict: NOT SUPPORTED**

## Ablations / secondary (raw)

- **DESIGN vs DESIGN_EI_ONLY composite**: Δ=-1.4862, p_raw=1, d_z=-7.180
- **DESIGN vs DESIGN_MYELIN_ONLY composite**: Δ=-1.0527, p_raw=1, d_z=-4.812
- **DESIGN vs DENS composite**: Δ=-0.4194, p_raw=1, d_z=-1.967
- **DESIGN vs SHUF composite**: Δ=-0.2425, p_raw=0.9956, d_z=-0.572
- **DESIGN vs FLY_CX XOR**: Δ=-0.4551, p_raw=1, d_z=-4.091
- **DESIGN vs HUMAN NARMA (lower better: HUMAN−DESIGN)**: Δ=-0.0318, p_raw=1, d_z=-1.900

## Honest verdict

**Primary NOT SUPPORTED.** DESIGN comp=-0.7153921302954109, ER=0.9933256433223437, TEAM_ER=0.721061354633424; XOR DESIGN=0.20100000569119816 vs FLY_CX=0.6560725114292981; NARMA DESIGN=0.4278604850120174 vs HUMAN=0.39607197071518857.
 Limits: synthetic SBM ≠ biology; myelin rule is layer-conditional sampling from H01, not axon geometry; DESIGN adjacency frozen across seeds (Win/task RNG vary); TEAM_ER is larger (two blocks).


## Push / artifacts

- **Verdict:** NOT SUPPORTED (all four primary H1–H4 fail Bonferroni)
- **Figure:** `regions/designed_brain/figs/designed_brain.png` (local; PNG not pushed)
- **Runtime:** ~10.7 min wall (tune+eval, WORKERS=2) — done 2026-10-28 19:52 America/New_York (EDT)
- **GitHub SHA (results commit):** `404d10ec73f7a7a4ca007a6da21a92ae1d2b478b` on `Cyberisthename/brain-experiments` main
  (code/prereg also in `9515fc83…`, meta in `d59b0db9…`)

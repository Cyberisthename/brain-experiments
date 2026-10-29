# GATED TEAM results — Design 1+8 (experts + neuromod + timescales)

Prereg: `PREREG_gated_team.md` (locked 2026-10-29 07:46 America/New_York, before eval).
Rows kept: 768. Eval seeds: 700–723 (n=24).
GATED_REAL = FLY_CX ⊕ HUMAN, zero bridges, expert ridges + convex gate + neuromod gain + species leaks.

## Locked hypers (tune seeds 90–92)

| Arm | γ_fly | γ_human | leak_fly | leak_human | leak_single |
|---|---:|---:|---:|---:|---:|
| `GATED_REAL` | 0.5 | 1.0 | 0.5 | 0.2 | 0.2 |
| `GATED_REAL_noLeak` | 0.5 | 1.0 | 0.2 | 0.2 | 0.2 |
| `GATED_ER` | 0.5 | 0.5 | 0.5 | 0.2 | 0.2 |
| `GATED_EI` | 0.5 | 1.0 | 0.5 | 0.2 | 0.2 |
| `TEAM` | 0.0 | 0.0 | 0.35 | 0.35 | 0.35 |
| `FLY_CX` | 0.0 | 0.0 | 0.35 | 0.35 | 0.35 |
| `HUMAN` | 0.0 | 0.0 | 0.35 | 0.35 | 0.35 |
| `ER4962` | 0.0 | 0.0 | 0.35 | 0.35 | 0.35 |

## Ridge α (tuned 90–92, per arm × task)

| Arm | XOR | NARMA | MC | PC_HARD |
|---|---:|---:|---:|---:|
| `GATED_REAL` | 1e-08 | 1e-08 | 1e-08 | 1e-06 |
| `GATED_REAL_noLeak` | 1e-08 | 1e-08 | 1e-06 | 1e-06 |
| `GATED_ER` | 0.0001 | 1e-08 | 0.0001 | 1.0 |
| `GATED_EI` | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| `TEAM` | 1e-08 | 1e-08 | 1e-08 | 0.01 |
| `FLY_CX` | 1e-08 | 1e-08 | 1e-08 | 0.01 |
| `HUMAN` | 1e-08 | 1e-08 | 1e-08 | 1e-06 |
| `ER4962` | 1e-08 | 1e-08 | 1e-08 | 0.1 |

Composite z locked from tune: μ_xor=0.8151, σ_xor=0.2495, μ_narma=0.2978, σ_narma=0.0533.

## Primary tasks + composite (means)

| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ |
|---|---:|---:|---:|---:|
| `GATED_REAL` | 4962 | 0.9645±0.0010 | 0.3338±0.0113 | -0.0376±0.1065 |
| `GATED_REAL_noLeak` | 4962 | 0.7213±0.0107 | 0.3350±0.0115 | -0.5363±0.1093 |
| `GATED_ER` | 4962 | 0.6260±0.3179 | 0.1828±0.0111 | 0.7005±0.6502 |
| `GATED_EI` | 4962 | 0.9610±0.0008 | 0.3336±0.0112 | -0.0432±0.1049 |
| `TEAM` | 4962 | 0.8837±0.0040 | 0.3197±0.0081 | -0.0680±0.0780 |
| `FLY_CX` | 2875 | 0.9294±0.0022 | 0.3741±0.0092 | -0.4865±0.0878 |
| `HUMAN` | 2087 | 0.1372±0.0376 | 0.3257±0.0087 | -1.6199±0.1053 |
| `ER4962` | 4962 | 0.9379±0.0016 | 0.2973±0.0080 | 0.2510±0.0745 |

## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon)

| Hypothesis | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf | pass |
|---|---:|---:|---:|---:|---:|:---:|
| H1 joint specialists (max p of NI) | — | — | — | 0.01243 | 0.0373 | ✓ |
| H2 GATED_REAL>GATED_ER composite | -0.7381 | [-2.0846, +0.6085] | -0.231 | 1 | 1 | ✗ |
| H3 GATED_REAL>TEAM composite | +0.0304 | [-0.0728, +0.1336] | 0.124 | 0.2195 | 0.6584 | ✗ |

### H1 components

- **H1x XOR NI (margin 0.03):** mean(GATED−FLY)+margin=+0.0652, p=5.96e-08, XOR GATED=0.9645352792443646, FLY=0.9293504659841015
- **H1n NARMA NI (margin 0.02):** mean(HUMAN−GATED)+margin=+0.0119, p=0.01243, NARMA GATED=0.333772155731994, HUMAN=0.32570338063066656

**Verdict: NOT SUPPORTED**

## Secondary tasks

| Arm | MC ↑ | PC_HARD ↑ |
|---|---:|---:|
| `GATED_REAL` | 25.255±0.192 | 0.911±0.010 |
| `GATED_REAL_noLeak` | 17.820±0.215 | 0.911±0.010 |
| `GATED_ER` | 25.976±0.572 | 0.888±0.019 |
| `GATED_EI` | 23.744±0.143 | 0.918±0.010 |
| `TEAM` | 24.139±0.201 | 0.844±0.010 |
| `FLY_CX` | 21.002±0.213 | 0.838±0.005 |
| `HUMAN` | 23.955±0.215 | 0.856±0.009 |
| `ER4962` | 22.064±0.119 | 0.876±0.007 |

- **GATED_REAL vs GATED_REAL_noLeak composite**: Δ=+0.4987, p_raw=5.96e-08, d_z=4.702
- **GATED_REAL vs GATED_EI composite**: Δ=+0.0056, p_raw=0.02821, d_z=0.333
- **GATED_REAL vs ER4962 composite**: Δ=-0.2886, p_raw=1, d_z=-0.990
- **GATED_REAL vs FLY_CX composite**: Δ=+0.4489, p_raw=1.192e-07, d_z=1.680
- **GATED_REAL vs HUMAN composite**: Δ=+1.5822, p_raw=5.96e-08, d_z=3.364
- **GATED_REAL vs FLY_CX XOR**: Δ=+0.0352, p_raw=5.96e-08, d_z=3.354
- **GATED_REAL vs HUMAN NARMA (lower better → use HUMAN-GATED)**: Δ=-0.0081, p_raw=0.8962, d_z=-0.331

## Mean convex gate λ (fly weight) on eval

| Arm | XOR | NARMA | MC | PC_HARD |
|---|---:|---:|---:|---:|
| `GATED_REAL` | 1.000 | 0.016 | 0.817 | 0.001 |
| `GATED_REAL_noLeak` | 1.000 | 0.005 | 0.000 | 0.003 |
| `GATED_ER` | 1.000 | 0.996 | 1.000 | 0.094 |
| `GATED_EI` | 1.000 | 0.018 | 0.849 | 0.120 |

## Honest verdict

**Primary NOT SUPPORTED.** GATED_REAL XOR=0.9645352792443646; FLY_CX=0.9293504659841015; GATED_REAL NARMA=0.333772155731994; HUMAN=0.32570338063066656; comp GATED=-0.03762302410416104; GATED_ER=0.7004634342288029; TEAM=-0.06800835447423413.
 Limits: task cue is experimenter-given; convex gate is linear; zero-bridge tests readout+gain specialization not axons; reservoir ≠ biology; CPU shared with other jobs.

## Runtime

- Tune+eval wall: 2026-10-29 07:50:43 EDT → 2026-10-29 08:22:10 EDT (America/New_York); WORKERS=2.
- Jobs: 768 (8 arms × 24 seeds × 4 tasks). Figure: `figs/gated_team.png` (also `figs/gated_team.svg`).


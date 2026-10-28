# TEAM Frankenstein results — division of labor

Prereg: `PREREG_team_franken.md` (locked 2026-10-28 17:56 America/New_York, before eval).
Rows kept: 672. Eval seeds: 400–423 (n=24).
TEAM = FLY_CX ⊕ HUMAN, zero bridges, shared L4 drive, concatenated ridge.

## Ridge α (tuned 90–92, per arm × task)

| Arm | XOR | NARMA | MC | PC_HARD |
|---|---:|---:|---:|---:|
| `TEAM` | 1e-08 | 1e-08 | 1e-08 | 0.1 |
| `FLY_CX` | 1e-08 | 1e-08 | 1e-08 | 0.01 |
| `HUMAN` | 0.0001 | 1e-08 | 1e-08 | 1e-08 |
| `FRANKEN` | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| `TEAM_ER` | 0.0001 | 1e-08 | 0.0001 | 1.0 |
| `TEAM_EI` | 1e-08 | 1e-08 | 1e-08 | 0.01 |
| `FLY_CX_BIG` | 0.01 | 1e-08 | 1e-06 | 0.1 |

Composite z locked from tune: μ_xor=0.3652, σ_xor=0.2257, μ_narma=0.3707, σ_narma=0.0466.

## Primary tasks + composite (means)

| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ |
|---|---:|---:|---:|---:|
| `TEAM` | 4962 | 0.4643±0.0164 | 0.3793±0.0064 | 0.1282±0.0750 |
| `FLY_CX` | 2875 | 0.6361±0.0113 | 0.4535±0.0069 | -0.2881±0.0792 |
| `HUMAN` | 2087 | 0.1226±0.0039 | 0.3937±0.0066 | -0.7836±0.0732 |
| `FRANKEN` | 6833 | 0.5038±0.0161 | 0.3690±0.0064 | 0.3257±0.0786 |
| `TEAM_ER` | 4962 | 0.2584±0.0148 | 0.3047±0.0082 | 0.4720±0.0918 |
| `TEAM_EI` | 4962 | 0.5693±0.0124 | 0.3782±0.0064 | 0.3721±0.0757 |
| `FLY_CX_BIG` | 4962 | 0.1252±0.0046 | 0.3714±0.0122 | -0.5385±0.1317 |

## Primary hypotheses (Bonferroni ×4, one-sided Wilcoxon)

| Hypothesis | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf | pass |
|---|---:|---:|---:|---:|---:|:---:|
| H1a TEAM>FLY_CX composite | +0.4163 | [+0.3186, +0.5140] | 1.799 | 1.192e-07 | 4.768e-07 | ✓ |
| H1b XOR non-inf (margin 0.03) | -0.1418 | [-0.1821, -0.1015] | -1.486 | 1 | 1 | ✗ |
| H2 TEAM>TEAM_ER composite | -0.3438 | [-0.5313, -0.1563] | -0.774 | 0.9997 | 1 | ✗ |
| H3 TEAM>FLY_CX_BIG composite | +0.6666 | [+0.4331, +0.9001] | 1.206 | 1.132e-06 | 4.53e-06 | ✓ |

**Verdict: NOT SUPPORTED**

## Secondary tasks

| Arm | MC ↑ | PC_HARD ↑ |
|---|---:|---:|
| `TEAM` | 19.039±0.173 | 0.876±0.007 |
| `FLY_CX` | 17.828±0.208 | 0.860±0.005 |
| `HUMAN` | 18.563±0.175 | 0.886±0.009 |
| `FRANKEN` | 19.189±0.190 | 0.874±0.013 |
| `TEAM_ER` | 19.893±0.276 | 0.930±0.007 |
| `TEAM_EI` | 18.500±0.167 | 0.876±0.008 |
| `FLY_CX_BIG` | 16.714±0.298 | 0.933±0.007 |

## Secondary contrasts (raw)

- **TEAM vs HUMAN composite**: Δ=+0.9118, p_raw=5.96e-08, d_z=3.986
- **TEAM vs FRANKEN composite**: Δ=-0.1976, p_raw=0.9991, d_z=-0.767
- **TEAM vs TEAM_EI composite**: Δ=-0.2439, p_raw=1, d_z=-1.276
- **TEAM vs HUMAN NARMA (lower better → reverse)**: Δ=+0.0144, p_raw=0.0001391, d_z=0.908
- **TEAM vs FLY_CX XOR**: Δ=-0.1718, p_raw=1, d_z=-1.801

## Honest verdict

**Primary NOT SUPPORTED.** TEAM comp=0.12815010018212378; FLY_CX=-0.288148371993919; TEAM_ER=0.47195083216778344; FLY_CX_BIG=-0.5384613668637179. XOR TEAM=0.46430405650824147 vs FLY_CX=0.6361435757187517.
 Limits: zero-bridge TEAM is a readout-selection test; FRANKEN is larger (adds mouse); NARMA ↓ / XOR ↑ composite depends on tune-locked z; reservoir ≠ biology.

### What failed (honest)

- **XOR cost:** grafting HUMAN (even with zero bridges) drops XOR from FLY_CX 0.64 → TEAM 0.46 (Δ≈−0.17), so non-inferiority margin 0.03 fails hard — ridge does *not* fully preserve the CX XOR specialist when states are concatenated.
- **TEAM_ER wins composite:** size-matched ER+shuffled-myelin beats real TEAM on composite (driven by NARMA 0.30 vs TEAM 0.38); real wiring is not required for the NARMA half under this z-score.
- **TEAM_EI > TEAM** on composite (EI-shuffled CX + real human) — consistent with xor_cx where EI_SHUF ≥ CX on XOR.
- **H1a/H3 do pass:** TEAM composite beats FLY_CX alone and FLY_CX_BIG (real human myelin/add-on helps vs plain CX / dens addon).

## Runtime

- Tune+eval wall: ~15 min (America/New_York 17:58–18:13, 2026-10-28); WORKERS=2, nice, fc2=True.
- Jobs: 336 primary + 336 secondary = 672 (7 arms × 24 seeds × 4 tasks).
- Figure: `figs/team_franken.png`.


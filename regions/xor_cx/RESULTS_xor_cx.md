# XOR_CX results — FLY_CX alone vs own matched controls

Prereg: `PREREG_xor_cx.md` (locked 2026-10-28 16:23 America/New_York, before eval).
Rows kept: 528. Eval seeds: 300–323 (n=24).
Parent: xor_rep mechanism → fly_CX carries FRANKEN XOR; this tests CX vs CX-matched random.

## Ridge α (tuned 90–92, XOR d=5 FIXED)

| Arm | α |
|---|---:|
| `CX` | 1e-08 |
| `ER` | 1e-08 |
| `SHUF` | 0.0001 |
| `DENS` | 0.01 |
| `EI_SHUF` | 1e-08 |
| `SPEC` | 0.01 |

### Spectral audit (tune seeds, top-64 frac \|λ\|>0.5)

| Arm | mean frac>0.5 |
|---|---:|
| `CX` | 0.172 |
| `ER` | 1.000 |
| `SHUF` | 1.000 |
| `DENS` | 1.000 |
| `EI_SHUF` | 0.016 |
| `SPEC` | 1.000 |

## Primary: XOR_DELAY d=5 FIXED — means

| Arm | mean ± sem | n |
|---|---:|---:|
| `CX` | 0.6606 ± 0.0066 | 24 |
| `ER` | 0.1139 ± 0.1907 | 24 |
| `SHUF` | 0.0259 ± 0.0385 | 24 |
| `DENS` | 0.1554 ± 0.0040 | 24 |
| `EI_SHUF` | 0.7025 ± 0.0060 | 24 |
| `SPEC` | 0.1563 ± 0.0041 | 24 |

Metric ↑ better (1 − NRMSE).

## Primary hypothesis (Bonferroni ×3, one-sided Wilcoxon)

| Contrast | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf |
|---|---:|---:|---:|---:|---:|
| CX > ER | +0.5467 | [+0.1495, +0.9439] | 0.581 | 6.557e-06 | 1.967e-05 |
| CX > SHUF | +0.6348 | [+0.5554, +0.7141] | 3.378 | 5.96e-08 | 1.788e-07 |
| CX > DENS | +0.5052 | [+0.4909, +0.5196] | 14.838 | 5.96e-08 | 1.788e-07 |

**Verdict: SUPPORTED**

## Secondary: XOR delay depth

| d | CX | ER | SHUF | DENS | CX−ER Δ | p_raw (>) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.986 | 0.884 | 0.586 | 0.378 | +0.101 | 5.96e-08 |
| 3 | 0.986 | 0.884 | 0.555 | 0.361 | +0.103 | 5.96e-08 |
| 5 | 0.661 | 0.114 | 0.026 | 0.155 | +0.547 | 6.557e-06 |
| 8 | 0.369 | -1.107 | 0.006 | 0.110 | +1.475 | 1.192e-07 |
| 10 | 0.143 | -6.473 | 0.004 | 0.084 | +6.617 | 5.96e-08 |

## Secondary: EI_SHUF and SPEC at d=5

- **EI_SHUF** mean=0.7025; CX−EI_SHUF Δ=-0.0419, p_raw=1, d_z=-0.988
- **SPEC** mean=0.1563; CX−SPEC Δ=+0.5044, p_raw=5.96e-08, d_z=14.423

## Honest verdict

**Primary SUPPORTED.** Real FlyWire CX beats ER, SHUF, and DENS on XOR_DELAY d=5 (all p_bonf < 0.05) under fresh seeds 300–323 and per-arm α.
 SPEC temper **failed to match** CX spectral bulk (frac\|λ\|>0.5 stayed 1.0 vs CX 0.172 after 25 iters) and scored like DENS (~0.16); cannot conclude from SPEC whether spectral sparsity alone explains the CX edge. EI_SHUF (frac≈0.016) slightly *exceeded* CX (0.70 vs 0.66) → preserving E/I edge classes + rewiring does not destroy the advantage (and may even help); exact CX wiring within E/I classes is not uniquely required.

Limits: reservoir 1−NRMSE ≠ biological XOR; EI_SHUF preserves E/I edge classes not biology; SPEC temper is a heuristic spectral-bulk match, not an exact eigen-null model; FIXED delays (no myelin on fly).


## Runtime

- Tune+eval wall: ~14 min (America/New_York 16:26–16:40, 2026-10-28); WORKERS=2, nice, fc2=True.
- Jobs: 144 primary (d=5, 6 arms×24) + 384 secondary (d∈{2,3,8,10}, 4 arms×24) = 528.
- Figure: `figs/xor_cx.png`.

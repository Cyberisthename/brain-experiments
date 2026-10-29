# Preregistration: ei_cross_species — does fly CX E/I XOR advantage generalize?

**Locked:** 2026-10-29 08:11 America/New_York (EDT), BEFORE any ei_cross_species eval scores.  
**Dir:** `/workspace/flybrain/regions/ei_cross_species/`  
**Parent motivation (prior studies, not this eval):** `xor_cx` / `designed_brain` — FlyWire CX XOR_DELAY d=5 advantage is largely carried by E/I class structure (EI_SHUF ≈ CX; DESIGN_EI_ONLY exploratory ≈0.84). Ask: does REAL or EI-preserving structure beat size-matched ER on XOR across species modules that have E/I labels?

## Species / modules (E/I required)

| ID | Module | E/I source | Include? |
|---|---|---|---|
| `MOUSE` | MICrONS / hybrid `mouse_module.npz` (N≈1871) | `sign` from cell type in module | YES |
| `HUMAN` | H01 patch `human_module.npz` (N≈2087) | `sign` from cell type / HF | YES |
| `FINCH` | zebra finch Area X j0251 `areax_module.npz` (N≈7576) | `sign` from cell type (MSN/INT…) | YES |
| `ELEGAN` | Witvliet W5 / Cook COOK_H | **E/I unavailable** (`ei_status=unknown`; all weights unsigned +1) | **SKIP** — documented in SOURCES |
| `DESIGN_EI` | confirmatory: fly-CX E/I SBM N=4000 (`DESIGN_EI_ONLY` recipe) | synthetic E/I from CX stats | YES (confirmatory H, not in species Bonferroni) |

## Task / recipe (locked)

- Task: **XOR_DELAY d=5**, metric **1−NRMSE** (↑ better). Bit-aligned with `xor_cx` / `trained_bridges`.  
- Delays: FIXED Uniform{1..5} per edge (no myelin map). Leak=0.2.  
- Wash/tr/te = 200/1500/500. Ridge α tuned per (species, arm) on seeds **90–92**.  
- Eval seeds **900–923** (n=24).  
- Drive: L4 indices when layer labels exist; else first `max(32, N//20)` neurons (finch/elegans-style fallback).  
- WORKERS from env (default **4** for Kaggle CPU).

## Arms (per species module)

| Arm | Definition |
|---|---|
| `REAL` | real adjacency, scaled ρ=0.9 |
| `ER` | size+nnz-matched ER; out-sign from real `sign[pre]` |
| `SHUF` | degree-preserving endpoint shuffle (frankenstein `degree_shuffle`) |
| `EI_SHUF` | rewire within (sign_pre, sign_post) classes; keep magnitudes (`ei_class_shuffle`) |
| `SBM_EI` | stochastic block model from empirical E/I→E/I densities + |w| pools of REAL (no copied edges); spectral temper soft |

`DESIGN_EI` confirmatory uses arms `DESIGN_EI_ONLY` vs `ER4000` only (same N=4000, nnz targeted to fly CX dens×N² scale as designed_brain).

## Primary hypotheses

**Per included species** S ∈ {MOUSE, HUMAN, FINCH}:  
H_S: mean XOR(REAL) > mean XOR(ER) **OR** mean XOR(EI_SHUF) > mean XOR(ER),  
one-sided Wilcoxon on paired seeds (higher better),  
p_bonf = min(1, n_species · min(p_REAL_vs_ER, p_EI_SHUF_vs_ER)).  

Species passes if p_bonf < 0.05.  
**Study verdict SUPPORTED** if ≥1 species passes; else **NOT SUPPORTED**. Report each species honestly.

## Confirmatory (not in species Bonferroni)

**H_design:** DESIGN_EI_ONLY > ER4000 on XOR d=5, seeds 900–923, one-sided Wilcoxon, α=0.05 (single test). Relates to exploratory designed_brain mean ≈0.84.

## Ops

- Prereg FIRST; no peeking.  
- Kaggle CPU notebook preferred for eval (4 cores); this box may only smoke.  
- Push code+prereg (+results when available). Reservoir ≠ biology claim.

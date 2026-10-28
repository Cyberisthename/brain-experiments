# Preregistration: Fly CX angular path integration (compass)

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any eval scores are written  
**Dir:** `/workspace/flybrain/regions/compass/`

## Motivation

On generic reservoir tasks (MC, NARMA, PC_HARD), random wiring matches or beats real wiring once FIXED delays and per-arm ridge α are matched (`regions/best_fly`, `regions/fewshot_delay`). Real circuits may still win on the job they evolved for.

The fly central complex (CX) implements a ring-attractor heading compass: EPG bump encodes heading; PEN / PFN carry angular-velocity / optic-flow updates into the ellipsoid body / protocerebral bridge. This experiment asks whether **real CX wiring** keeps heading under angular-velocity drive better than ER / degree-shuffle / dens-matched controls, with and without FIXED delays.

## Arms (eval seeds 0–7)

| Arm | Wiring | Delays | ω injection | Landmark injection |
|---|---|---|---|---|
| `FLY_CX_NOD` | real fly CX | all d=1 | real PEN indices (n=42) | real EPG indices (n=51) when cue on |
| `FLY_CX_DEL` | real fly CX | FIXED Uniform{1..dmax} per edge | real PEN | real EPG |
| `ER_NOD` / `ER_DEL` | ER same N+nnz, signed by random E/I match | NOD / FIXED | **random** set size n_PEN | **random** set size n_EPG |
| `SHUF_NOD` / `SHUF_DEL` | degree-preserving shuffle of real W | NOD / FIXED | random set size n_PEN | random set size n_EPG |
| `DENS_NOD` / `DENS_DEL` | dens-matched (same N+nnz, \|w\| bootstrap) | NOD / FIXED | random set size n_PEN | random set size n_EPG |

PEN / EPG indices come from FlyWire-style `ann.tsv` cell_type labels joined on `fly_cx_module.npz` root_ids (`data/fly_cx_types.npz`). Control injection sets are seed-tied (`seed + 4242`) and independent of the real PEN/EPG sets.

## Task

**Input:** 1-D angular velocity ω(t) (smooth OU-filtered noise) always injected into the ω set. Optional intermittent **visual landmark cues**: when the cue is ON, also inject (cos θ, sin θ) into the landmark set (simulates a brief heading reset / visual landmark).

**Target:** current heading as (cos θ, sin θ), θ(t) = ∫ ω.

**Readout:** ridge (bias column) from full reservoir state → (cos, sin). α tuned **per arm** on seeds **90–92 only**.

### Metrics (all ↓ better; circular RMSE in radians)

1. **HEADING** — train+test with intermittent landmarks (blocks: 50 ON / 150 OFF). Mean circular error on test window.
2. **DRIFT** — train with landmarks; **test with landmarks OFF** (path integration alone). Mean circular error on test.
3. **GEN** — same as HEADING schedule, but test ω amplitude = **2×** training scale (unseen speed). Mean circular error on test.

Circular error at t: `arccos(clip(ĉ·c + ŝ·s, -1, 1))` after L2-normalizing the predicted (ĉ,ŝ) (zero vector → error π).

### Protocol constants

- Leak = 0.2; W scaled to ρ = 0.9 (fast `radius_cached`).  
- `dmax = 5` FIXED delays (same recipe as `regions/myelin` / `regions/best_fly`).  
- wash / train / test = **200 / 1200 / 400**.  
- Training ω scale σ_ω = 0.15 rad/step (OU τ=20). GEN test uses 2σ_ω.  
- Landmark gain = 1.0 into EPG/random landmark set; ω gain = 1.0 into PEN/random ω set.  
- α grid: `{1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0}` — pick argmin mean HEADING RMSE on tune seeds. Locked to `out/alpha_choice.json` before eval.  
- Fast kernels: `compress/fast_kernels.py` + bucketed delay CSR (same as best_fly / fewshot_delay).  
- No Hebbian, no maze, no frankenstein graft.

## Stats

- Per-seed paired Wilcoxon signed-rank (two-sided; all-zero → p=1).  
- **Primary family (Bonferroni ×9):** `FLY_CX_DEL` vs `{ER_DEL, SHUF_DEL, DENS_DEL}` × `{HEADING, DRIFT, GEN}`.  
  Direction: lower error favors FLY. Report Δ = FLY − ctrl (negative favors FLY).  
- **Secondary family (Bonferroni ×9):** `FLY_CX_NOD` vs `{ER_NOD, SHUF_NOD, DENS_NOD}` × same three metrics.  
- Exploratory: DEL vs NOD within FLY_CX; raw p only.  
- Verdict rule: claim “real CX wins its native task” only if ≥1 primary contrast survives Bonferroni **and** mean FLY_CX_DEL is better (lower) than all three delay-matched controls on that metric. Otherwise honest null / mixed.

## Predictions (locked)

- **P1:** Real CX + FIXED delays beats delay-matched random controls on DRIFT (path integration without landmarks) — the native job.  
- **P2:** HEADING (with landmarks) is easier for everyone; real wiring advantage shrinks.  
- **P3:** GEN (faster turns) stresses integrators; real ring structure may help or collapse — report either way.  
- **P4 (null we respect):** If FLY_CX_DEL ≈ controls after correction, delays/readout dominate and CX wiring is not special under this proxy.

## Honesty / ops

- No GitHub push.  
- `nice -n 19`, WORKERS≤2.  
- Report negatives. Do not call biology a win from FLY > NOD-only comparisons.

# Preregistration: Fish1 zebrafish module (± delays) + FRANKEN graft

**Status:** LOCKED design, **data access BLOCKED** (see `SOURCE_fish1.json`, `RESULTS_zebrafish.md`).  
**Date:** 2026-10-28. Written before any Fish1 eval scores (none exist).

## Gate

If Fish1 edges cannot be obtained without a login / CAVE token → **STOP**. Do not invent a synthetic zebrafish graph. Document the blocker and leave this prereg as the intended design for when a token is available.

## Intended module (when unlocked)

- Source: Fish1 CAVE `fish1_v250915`, tables `somas` + `synapses_axde_label`.
- Cut: connected region ~2,000–7,000 neurons (prefer hindbrain integrator or tectum), E/I from molecular tags when present.
- Weight: synapse count (or size if available) × presynaptic sign (+1 E / −1 I; `na` → drop or neutral per build log).
- Artifact: `data/fish_module.npz` + `SOURCE_fish1.json` update (`module_built: true`).

## Arms (eval seeds 0–7; tune α on 90–92 only)

| Arm | Wiring | Delays |
|---|---|---|
| FISH_NOD / FISH_DEL | real Fish1 module | all d=1 / FIXED Uniform{1..5} |
| ER_NOD / ER_DEL | ER same N+nnz, E/I match | NOD / DEL |
| SHUF_NOD / SHUF_DEL | degree-preserving shuffle | NOD / DEL |
| DENS_NOD / DENS_DEL | dens-matched | NOD / DEL |
| FRANKEN_FISH_NOD/DEL | FRANKEN blocks + Fish graft | NOD / DEL |
| FRANKEN_RAND_NOD/DEL | FRANKEN + same-size random graft | NOD / DEL |

Leak=0.2, ρ=0.9, fast kernels, per-arm ridge α tuned on seeds 90–92 (MC criterion), never reuse tune seeds in eval.

## Tasks / stats

- Tasks: MC ↑, NARMA ↓, PC_HARD ↑ (cue 0.35, noise 1.5).
- Primary family (Bonferroni): FISH_DEL vs {ER,SHUF,DENS}_DEL × {MC,NARMA,PC_HARD}.
- Graft family: FRANKEN_FISH_DEL vs FRANKEN_RAND_DEL × same tasks.
- If Fish1 myelin annotations exist: REAL vs SHUF myelin delays as in `regions/myelin_real` (separate Bonferroni family).

## Predictions (locked)

- **P1 (skeptical):** once N/nnz/delays/α matched, FISH ≈ controls on generic tasks (Level-1 scaling).
- **P2:** worth retesting only if a fish-native task (e.g. escape / optic-flow) is added later.

## Honesty

Report block if auth fails. No GitHub push. No tokens in files.

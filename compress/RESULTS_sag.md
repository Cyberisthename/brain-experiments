# Sparse Adapter Gateway (SAG) results

Prereg: `PREREGISTRATION_sag.md` (locked before this run). 7 arms × 2 methods (CONSOL, MAGP) × 8 seeds × 3 tasks = **336 jobs**. All arms share grad clip 1.0 and weight clip ±2. NARMA seed 7 still overflowed to 1e9 on every trained arm (1/8 seeds); clean means exclude that seed.

## Parameter counts and wall time (mean over seeds)

| Arm | params | n_tgt CONSOL / MAGP | MC sim (s) | MC train (s) | T0 sim (s) |
|---|---:|---:|---:|---:|---:|
| NONE | 0 | 15 / 39 | 0.12 | 0 | 0.26–0.97 |
| LORA15 | 1,438 | 15 / 39 | 0.44 | 2.7 | 0.16 |
| **SAG** | **1,500** | 15 / 39 | **0.62** | **10.0** | 0.27 |
| SAG_RAND | 1,500 | 15 / 39 | 0.61 | 10.1 | 0.27 |
| SAG_DENSE | 1,500 | 15 / 39 | 0.57 | 9.3 | 0.19 |
| ADAPT | 11,504 | 15 / 39 | 0.56 | 3.2 | 0.18 |
| LORA | 11,504 | 15 / 39 | 0.44 | 2.9 | 0.16 |

SAG is **not faster** than ADAPT at eval (0.62 vs 0.56 s) and is ~3× slower to train (straight-through top-K). Sparse matmul helps vs a naive dense residual, but hard top-K + ST overhead eats the gain.

## Task means (clean = exclude 1e9 blow-ups)

### CONSOL (f=0.25, ~15 E-PG clusters)

| Arm | MC ↑ | NARMA ↓ (clean) | T0 ↓ |
|---|---:|---:|---:|
| NONE | 11.88 | 0.485 | 0.0432 |
| LORA15 | 12.63 | 0.475 | 0.0278 |
| SAG | **10.89** | 0.501 | 0.0418 |
| SAG_RAND | 10.02 | 0.989 | 0.0433 |
| SAG_DENSE | 14.33 | 0.474 | 0.0132 |
| ADAPT | 18.07 | 0.475 | **0.0002** |
| LORA | **18.57** | 0.474 | 0.0008 |

### MAGP (f=0.25, ~39 E-PG cells kept)

| Arm | MC ↑ | NARMA ↓ (clean) | T0 ↓ |
|---|---:|---:|---:|
| NONE | 12.17 | 0.495 | 0.0406 |
| LORA15 | 13.08 | 0.475 | 0.0192 |
| SAG | **11.20** | 0.510 | 0.0453 |
| SAG_RAND | 9.49 | 0.682 | 0.0574 |
| SAG_DENSE | **17.39** | 0.455 | 0.0077 |
| ADAPT | 17.83 | 0.469 | **0.0002** |
| LORA | **18.50** | 0.467 | 0.0012 |

## Prereg questions (Bonferroni over 36 SAG-vs-* tests; 18 survive)

| Q | Answer |
|---|---|
| **G1: Does SAG keep MC/T0 gains at ~1/8 params?** | **No.** SAG MC is *worse* than NONE (CONSOL 10.89 vs 11.88, MAGP 11.20 vs 12.17; both p_bonf≈0.011). On T0, SAG ≈ NONE (no gain). ADAPT/LORA (11.5k) still dominate. LORA15 (~1.4k, same budget) beats SAG on MC and T0. |
| **G2: Is SAG faster than ADAPT?** | **No** at eval (0.62 vs 0.56 s MC); **much slower** to train (~10 vs ~3 s) because of ST top-K. |
| **G3: Do heading targets matter?** | **Yes for the soft gateway.** SAG_DENSE MAGP (39 E-PG) MC 17.39 ≫ CONSOL (15 clusters) 14.33 — more real heading cells help when all gateways fire. Under **hard** top-K, both SAG and SAG_RAND sit near/below NONE; SAG vs SAG_RAND is n.s. (learned U targets do not help once only 10/500 units are active). |
| **G4: SAG vs SAG_DENSE (top-K ablate)** | Hard top-K **hurts**: SAG_DENSE ≫ SAG on MC and T0 (all p_bonf&lt;0.05). Sparsity of *activation*, not of params, is the bottleneck. |
| **G5: NARMA** | With clips, clean NARMA is flat across arms (~0.47–0.51). Seed-7 still hits 1e9 on every trained arm; NONE never blows. Adapter training remains fragile on NARMA. |

## Honest takeaway
A 500-unit hard top-K=10 gateway into E-PG/P-EG **does not** preserve the dense-adapter MC/T0 gains at 1.5k params. The same param budget as a *dense* residual into those cells (SAG_DENSE) recovers most of the MAGP MC gain (17.4 vs ADAPT 17.8) and a chunk of T0 — so the E-PG sink is useful, but forcing 2% activity kills the benefit. SAG is also not a free lunch on wall time. Prefer LoRA/ADAPT when accuracy matters; prefer NONE when speed matters; SAG as specified loses on both axes here.

Raw: `out/sag.jsonl`. Tables: `out/sag_analysis.md`. Smoke (seed 0) logged in the prereg before this run.

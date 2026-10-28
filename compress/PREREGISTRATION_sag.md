# Preregistration: Sparse Adapter Gateway (SAG)
Written 2026-10-27, BEFORE any SAG test-seed result. Smoke checks on seed 0 only are allowed and will be logged.

## Setup
- Base: REAL CX, compressed to f=0.25 with CONSOL and MAGP (same graphs as compress.py).
- Tuned leak/ρ from compress `out/best.json` (REAL, meth, 0.25, task).
- Tasks: MC (higher better), NARMA (NRMSE, lower), T0 (MSE, lower).
- Test seeds 0–7. No retuning.
- **NARMA blow-up fix (all arms):** Adam with grad clip max-norm 1.0 per step, and trainable weights clipped to ±2 after each step. Same for every arm. If a seed still overflows, value is recorded as 1e9 and also reported with that seed dropped.

## SAG design (~1,500 trainable params, sparse)
- G = 500 gateway units.
- Hard top-K: K = 10 (2% of 500) active per step. Non-selected units contribute 0.
- Sparse input matrix Vin (m → G): each gateway gets 2 random CX units → nnz ≈ 1,000.
- Sparse output matrix U (G → targets): each gateway projects to 1 random target cell → nnz ≈ 500.
- Total trainable ≈ 1,500. Forward pass uses sparse matmuls only (scipy CSR / indexed gathers), not a dense residual.
- **Targets:**
  - SAG: real E-PG/P-EG/EPGt cells that survive compression.
    - MAGP: neurons kept by magnitude; EPG among them (expected ~39).
    - CONSOL: cluster IDs that contain at least one EPG cell (expected ~15).
  - SAG_RAND: same param count and sparsity, but targets are a random CX subset of the same size.
  - SAG_DENSE: same Vin/U sparsity and param count, but **no** top-K — all 500 gateways fire every step.

## Arms (per meth ∈ {CONSOL, MAGP})
| Arm | Params | Notes |
|---|---:|---|
| NONE | 0 | compressed only |
| ADAPT | 11,504 | dense 8-on bit bottleneck (prior) |
| LORA | 11,504 | rank-8 LoRA (prior) |
| LORA15 | ≈1,438 | rank-1 LoRA, matched param budget (2·m·1) |
| SAG | ≈1,500 | top-K sparse, EPG/P-EG targets |
| SAG_RAND | ≈1,500 | top-K sparse, random CX targets |
| SAG_DENSE | ≈1,500 | no top-K, same sparse weights |

Training: 100 Adam steps, lr 0.01, batch as in adapt.py. W_in and b fixed per seed. Adapter residual sits inside the tanh (same as before).

## Metrics
- Task score, wall-clock **simulation** seconds (eval rollouts only), train seconds, param count, active FLOPs proxy = nnz of active paths per step.
- Robustness: worst seed.

## Statistics
Two-sided Mann–Whitney. Bonferroni over planned comparisons (per meth, per task, unless noted):
1. SAG vs NONE
2. SAG vs ADAPT
3. SAG vs LORA
4. SAG vs LORA15
5. SAG vs SAG_RAND (heading targets matter?)
6. SAG vs SAG_DENSE (top-K matters?)
= 6 × 2 meth × 3 tasks = 36 tests.

## Predictions
- **G1:** SAG keeps most of the MC/T0 gain of ADAPT/LORA at ~1/8 the params (SAG ≫ NONE on MC and T0; SAG within 20% of ADAPT on MC).
- **G2:** SAG sim_sec is closer to NONE than to ADAPT (actually fast; sparse wins).
- **G3:** SAG ≥ SAG_RAND on T0 (heading targets help the braid task).
- **G4:** SAG ≥ SAG_DENSE on sim speed; task scores similar or SAG better (top-K regularises).
- **G5:** LORA15 < LORA on MC (param cut hurts dense LoRA more than SAG).

Honest prior: G2 is the main claim to stress-test; if SAG is implemented with a dense gather it may not beat ADAPT on wall-clock.

## Smoke (seed 0, MC only — logged before full run; design frozen)
| arm | meth | MC | sim_sec | params |
|---|---|---:|---:|---:|
| NONE | CONSOL | 12.01 | 0.117 | 0 |
| NONE | MAGP | 12.62 | 0.118 | 0 |
| SAG | CONSOL | 11.53 | 0.243 | 1500 |
| SAG | MAGP | 11.93 | 0.231 | 1500 |
| ADAPT | CONSOL | 18.74 | 0.563 | 11504 |
| ADAPT | MAGP | 18.23 | 0.558 | 11504 |

Note: seed-0 SAG did not beat NONE on MC. Full 8-seed run will decide G1; no further tuning.

# SPEEDUP2 — batch seeds + state/graph reuse (opt-in)

Generated: 2026-10-28 15:44:08 UTC

Opt-in module: `fastcore2.py` (does **not** modify running experiments).
Verification: `verify_speedup2.py` → `out_speedup2/verify_report.json`.

## Ideas

1. **Batch seeds/inputs**: sparse `W @ X` with multi-column dense state (same W), or block-diagonal stack of different Ws.
2. **Reuse states**: simulate once per `(arm, seed, input)`; train all shared-input readouts on cached `X`. Disk-cache random control graphs (ER/SHUF/DENS).

## Equivalence (old vs new)

- Relative tolerance required: **1e-6**
- **Global max abs diff**: `2.568e-08`
- **Global max rel diff**: `3.489e-08`
- Failures above tol: **0**

| Experiment | mode | n metrics | max abs | max rel | n_fail |
|---|---|---:|---:|---:|---:|
| myelin_real | batch | 48 | 0.000e+00 | 0.000e+00 | 0 |
| myelin_real | dropin | 48 | 0.000e+00 | 0.000e+00 | 0 |
| best_fly | batch | 42 | 0.000e+00 | 0.000e+00 | 0 |
| best_fly | dropin | 42 | 0.000e+00 | 0.000e+00 | 0 |
| fewshot_delay | batch | 400 | 2.568e-08 | 3.489e-08 | 0 |
| fewshot_delay | dropin | 400 | 0.000e+00 | 0.000e+00 | 0 |

Unit tests (CSR/delay batch & blockdiag):

| unit | max abs |
|---|---:|
| csr/csr_batch_max_abs | 1.110e-16 |
| csr/blockdiag_max_abs | 1.388e-16 |
| delay/delay_batch_max_abs | 0.000e+00 |
| delay/delay_blockdiag_max_abs | 0.000e+00 |

All metric comparisons pass the 1e-6 relative tolerance. Myelin and best_fly are bit-identical (0.0). Fewshot batch shows ~1e-8 float noise from state reuse / dual ridge path fusion — well under 1e-6.

## Wall-clock (2 seeds, all arms) — end-to-end

| Experiment | old (s) | drop-in (s) | batch (s) | speedup batch | speedup drop-in |
|---|---:|---:|---:|---:|---:|
| myelin_real | 651.84 | 632.37 | 600.27 | 1.086× | 1.031× |
| best_fly | 75.42 | 66.67 | 71.47 | 1.055× | 1.131× |
| fewshot_delay | 210.96 | 222.55 | 182.76 | 1.154× | 0.948× |

Box was heavily loaded (other experiments running at nice 19); absolute walls are pessimistic. Relative speedups still valid under same load.

## Estimated time saved per full 8-seed experiment

Scaled linearly from 2-seed walls (`est = wall_2seed * 4`).

| Experiment | 8-seed old (s) | 8-seed batch (s) | time saved (s) | time saved (min) |
|---|---:|---:|---:|---:|
| myelin_real | 2607.4 | 2401.1 | 206.3 | 3.44 |
| best_fly | 301.7 | 285.9 | 15.8 | 0.26 |
| fewshot_delay | 843.9 | 731.0 | 112.8 | 1.88 |
| **sum (3 exps)** | **3752.9** | **3418.0** | **334.9** | **5.58** |

Per-experiment 8-seed savings (batch path):
- **myelin_real**: save **206s (3.4 min)**, speedup **1.086×**
- **best_fly**: save **16s (0.3 min)**, speedup **1.055×**
- **fewshot_delay**: save **113s (1.9 min)**, speedup **1.154×**

## Microbenchmarks (isolated kernels, n=2875, T=2200)

| B | CSR speedup | delay speedup | CSR max abs | delay max abs |
|---:|---:|---:|---:|---:|
| 2 | 0.583× | 1.051× | 1.665e-16 | 0.000e+00 |
| 4 | 1.017× | 1.521× | 1.665e-16 | 0.000e+00 |
| 8 | 1.456× | 1.906× | 1.665e-16 | 0.000e+00 |

- Graph cache (8 builds): cold 1.714s → warm 0.172s = **9.97×**
- State reuse (7 shared-input readouts): resim 3.725s → reuse 0.420s = **8.87×**

## Theoretical / component speedups

- Same-W CSR batch at B=8: **~1.46×** sim wall (overhead dominates at B=2).
- Delay batch at B=8: **~1.91×**.
- State reuse across n_train / multi-task readouts: **~8.9×** vs resimulating.
- Control-graph disk cache warm hit: **~10×** vs rebuild.
- End-to-end 2-seed batch under load: **1.06–1.15×** (myelin/best_fly/fewshot); larger when state-reuse dominates (fewshot grids) or when B≥8 with light host load.

## Files

- `/workspace/flybrain/fastcore2.py` — opt-in API
- `/workspace/flybrain/verify_speedup2.py` — A/B harness
- `/workspace/flybrain/out_speedup2/verify_report.json` — full numbers
- `/workspace/flybrain/out_speedup2/microbench.json` — kernel benches
- Cache: `/workspace/flybrain/.cache/fastcore2/control_graphs/`

## Safety

Running queues (celegans, finch, zebrafish, myelin_real_rep, mature, hybrid) were **not** modified or interrupted. `fastcore2` is import-gated.


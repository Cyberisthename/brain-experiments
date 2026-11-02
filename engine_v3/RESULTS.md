# Engine v3 results

**Time zone:** America/New_York. **Built:** 2026-11-02.

## Validation (features off)

| Check | Result |
|---|---|
| Unit tests (`pytest engine_v3/tests`) | **9/9 passed** |
| Callosum forward vs v2 sparse OscRNN | **PASS** — max abs err **0.0** |
| Wholebrain dense→sparse port forward | **PASS** — max abs err **~1.5e-7** |
| Synced train (identical init + batches) | **PASS** — loss/out diff **0.0** |
| Mini metric XOR_HARD (short hp, 2 seeds) | within ~0.014 (tol diagnostic) |
| Mini metric NARMA (short hp) | **not comparable** — different train seed offset than v2 `seed+17*n_label`; dynamics proven by synced train |

**Verdict: validation PASS** for migrating science onto v3 with flags off.

## Speed

See `BENCH.md`. Forward speedup vs v2 callosum OscRNN ≈ **1.02–1.06×** at N=512/1024 under load (ring buffer only). Synthetic N=2048 forward ~51–61 ms (B=16,T=48). **Below** the 1.5× end-to-end gate — Phase-2 batching/compile/GPU still required before flipping default region scripts.

## Feature ablation

Prereg: `PREREG_v3_features.md` (written before runs).  
Job: `run_ablation.py` — 6 features × 4 tasks × 8 seeds = **192** jobs, resume-safe jsonl.  
**Status at first write:** running under `nice -n 19`, 1 worker, threads=1 (callosum PID 362279 untouched).  
First row: BASE XOR_HARD seed=3300 metric≈0.519 (~91 s/job) → ETA ~5 h for full grid.

Re-run `python3 engine_v3/analyze_ablation.py` when `out/ablation_runs.jsonl` completes; paste Wilcoxon table here.

### Ablation verdicts

*(pending full grid — do not interpret partial means as confirmatory)*

## Anything broken

- End-to-end speedup gate (≥1.5×) **not met** yet (expected; Phase-1 ring only).
- Standalone metric-parity helper used a different seed than callosum `train_task` for NARMA; not a dynamics bug.
- `torch.compile` not enabled by default (optional flag).

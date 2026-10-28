# Plasticity boost results

Prereg: `PREREGISTRATION_plasticity.md`.

## Status: NO DATA (run bug)

All 60 tune jobs returned `success=nan`. Cause: `empn_v2.job` expects a 4-tuple `(arm, seed, use_tr, tau)` but plasticity passed a 3-tuple → ValueError. `picks.json` = `{}`; full test jobs = 0.

Not a biological null. Needs patch to v2b 4-tuple API, then nice-rerun after hybrid frees CPU.

Raw: `out/runs.jsonl` (errors only).

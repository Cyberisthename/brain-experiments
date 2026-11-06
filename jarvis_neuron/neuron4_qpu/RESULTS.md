# NEURON-4 QPU follow-up — Pilot results (NEURON-3 E)

**When:** 2026-11-05 America/New_York  
**Pilot tag:** `NEURON-3-E` (dev only; swap to NEURON-4 when ready)  
**Instance:** `problems/instance_neuron_3_e_n16_s20261105.json` (Amendment 2)  
**sha256:** `2b38b6f4f6b3c039322c42bdb70846f9b4bfb9f054d46fa816886adac342b273`  
**No IBM job. Token not used.**

## What was tested

QPU/QAOA **optimizes keep/cut** on 16 frozen boundary edges. It does **not** run boundary learning.

| Type | count |
| ---- | ----- |
| contradicts | 5 |
| supported | 3 |
| supersedes | 2 |
| proposed | 2 |
| uncertain_source | 2 |
| unrelated_similar | 2 |

Quadratic RZZ terms: **3** (same-group supported↔contradicts).

## Solver table (higher objective_score better)

| Solver | Objective | Boundary acc | Confident-wrong | Coverage | Storage B | Q | notes |
| ------ | --------- | ------------ | --------------- | -------- | --------- | - | ----- |
| Brute force (exact) | **16.0** | **1.00** | 0.030 | 0.569 | 29259 | −16.0 | 0.26 s, 2^16 evals |
| Simulated annealing | **16.0** | **1.00** | 0.030 | 0.569 | 29259 | −16.0 | matched budget |
| Greedy / local | **16.0** | **1.00** | 0.030 | 0.569 | 29259 | −16.0 | 66 evals |
| Random (1000) | 12.95 | 0.812 | 0.030 | 0.569 | 29301 | −12.95 | below optimum |
| Ideal QAOA p=1 | **16.0** | **1.00** | 0.030 | 0.569 | 29259 | −16.0 | depth≈6 / decomp≈12 |
| Ideal QAOA p=2 | **16.0** | **1.00** | 0.030 | 0.569 | 29259 | −16.0 | depth≈11 / decomp≈23 |
| Baseline keep-all | 5.7 | — | — | — | — | — | |
| Baseline cut-all | 0.0 | — | — | — | — | — | |
| Gold mask | 16.0 | 1.00 | — | — | — | — | = exact optimum |

Bitstring at optimum: `1100111100001111` (= gold keep/cut).

Confident-wrong / coverage use the pilot proxy (E_boundary reference outcomes + mask safety adjustments); see instance `eval_detail.note`.

## Hardware plan (not submitted)

| | p=1 | p=2 |
| - | --- | --- |
| Qubits | 16 | 16 |
| Shots | 4096 | 4096 |
| RZZ / layer | 3 | 3 |
| Est. QPU s (pessimistic) | **8 s** | **10 s** |
| Cap | 40 s | 40 s |
| Leave ≥90 s of 345? | yes (335–337 s left) | yes |
| `passes_gate` | true | true |

## Is hardware worth testing?

**Gate criteria (PREREG): PASS** — ideal QAOA matches brute-force objective and confident-wrong; cost ≤40 s with reserve.

**Recommendation: do not spend QPU time on this pilot.** Classical brute force already has the exact optimum in <1 s. A hardware run would only measure noise/decoherence on a solved instance, not improve the mask. **No quantum advantage claim** (and none is possible at this size against exact classical).

Revisit after NEURON-4 exports a harder boundary slice (more variables or denser frustration) where QAOA and matched classical disagree on the five metrics.

## NEURON-4 swap

```
.venv/bin/python generate_problem.py --source neuron4 --edges PATH/to/export.json --n 16
.venv/bin/python solvers_classical.py problems/instance_*.json
.venv/bin/python qaoa_ideal.py problems/instance_*.json
```

Export schema: `{ "edges": [[a,b,type],...], "facts": [...], "questions": [...] }`.

## Files

- `PREREG.md` (+ Amendments 1–2)
- `generate_problem.py`, `solvers_classical.py`, `qaoa_ideal.py`, `run_all.py`
- `problems/instance_neuron_3_e_n16_s20261105.json`, `qubo_*.npz`, `meta_*.json`
- `results/classical_*.json`, `results/qaoa_*.json`
- `results/archive_pre_amend2/` — superseded linear-only pilot

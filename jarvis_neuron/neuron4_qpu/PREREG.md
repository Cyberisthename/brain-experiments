# NEURON-4 QPU follow-up — Boundary configuration optimization (PREREG)

**Written:** 2026-11-05 (America/New_York), **before** any solver or simulator results.
**Status:** PREPARE ONLY. No IBM QPU job. No token use. Ideal simulator (local Aer/statevector) only until the user approves a hardware plan.
**Pilot data:** NEURON-3 `E_boundary` outputs (dev pilot). Swap to NEURON-4 outputs when that run finishes. Do not modify NEURON-1/2/3/4 run trees.

---

## What this experiment is

The QPU (or QAOA simulator) **optimizes keep/cut decisions on a frozen set of boundary edges** already produced by the classical memory system. It does **not** run the boundary-learning / relationship-discovery algorithm.

| Role | Who does it |
| ---- | ----------- |
| Learn / type edges from text | Classical memory (NEURON-3 E pilot; later NEURON-4) |
| Choose which candidate edges to KEEP vs CUT | Combinatorial optimizer (classical + ideal QAOA; hardware only if approved) |

## What this experiment is not

- Not “running NEURON-4 on a QPU.”
- Not a claim of quantum advantage. At n≈10–20 binaries, classical brute force is exact and cheap; QAOA is a methods check.
- Not an unrestricted classical search vs a tiny QPU search. All solvers see the **same** frozen QUBO of the **same** size and a **matched** evaluation budget where applicable.

---

## Scientific question

Given a frozen candidate edge set \(C=\{e_1,\ldots,e_n\}\), \(n\in[10,20]\), does any optimizer (brute force, simulated annealing, greedy/local, random, ideal QAOA p=1..2, later hardware) find a keep/cut mask that improves the five tracked metrics vs a trivial baseline (keep-all / cut-all / E-default), under the **frozen** objective below?

**Primary decision for “is hardware worth testing”:** only if ideal QAOA is competitive with matched classical on objective **and** does not worsen confident-wrong vs brute-force optimum on the pilot — and the estimated QPU time is ≤40 s while leaving ≥90 s of the 345 s balance. Even then, hardware is a noise/calibration check, not an advantage claim.

---

## Variables

Binary \(x_i\in\{0,1\}\): **1 = KEEP** edge \(e_i\) in retrieval / probe aggregation; **0 = CUT**.

Structural edges (`in_project`, `has_fact`, `same_row`) are **never** decision variables (always kept). Candidates are fact–fact boundary-typed edges from E’s vocabulary:

`supported`, `supersedes`, `contradicts`, `unrelated_similar`, `uncertain_source`, `proposed`.

---

## Frozen objective (QUBO / Ising) — locked before results

Minimize Ising energy \(H(z)=\sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j\) with \(z_i=2x_i-1\in\{-1,+1\}\).

Equivalent QUBO in \(x\): minimize
\[
Q(x)=\sum_i a_i x_i + \sum_{i<j} b_{ij} x_i x_j
\]
with coefficients frozen as follows (weights fixed here; not tuned after seeing solver outputs):

### Linear \(a_i\) (cost of KEEP)

| Edge type | \(a_i\) | Rationale (E design) |
| --------- | ------ | -------------------- |
| `unrelated_similar` | \(+3.0\) | Must not spread activation; prefer CUT |
| `uncertain_source` | \(+1.5\) | Alone should not support a confident answer |
| `proposed` | \(+0.5\) | Weak; prefer not to rely on |
| `supported` | \(-2.0\) | Prefer KEEP |
| `supersedes` | \(-2.0\) | Prefer KEEP (outdated→current) |
| `contradicts` | \(-1.0\) | Prefer KEEP so abstention can fire |
| other / unknown | \(0.0\) | Neutral |

Storage term (always on): \(a_i \mathrel{+}= \lambda_S\) with \(\lambda_S = 0.05\) (small preference for sparsity).

### Quadratic \(b_{ij}\) (both KEEP)

| Condition on pair \((e_i,e_j)\) | \(b_{ij}\) | Rationale |
| ------------------------------- | --------- | --------- |
| Same `(entity,attr)` group, different values, both `supported` or both `proposed` | \(+2.5\) | Dual KEEP without contradict/supersede link → risky confident conflict |
| One `supersedes` and one `supported` in same group (consistent chain) | \(-1.0\) | Complementary boundary evidence |
| Both `unrelated_similar` sharing an endpoint | \(+1.0\) | Extra distractors |
| One `contradicts` and one `supported` on same conflicting values | \(-0.5\) | Contradiction signal + support → abstention path |
| Otherwise | \(0.0\) | |

No other terms. **No data-dependent refit** of \(a_i,b_{ij}\) after prereg. Instance files store the numeric \(a,b\) (or \(h,J\)) explicitly.

**Objective score** reported in tables = \(-Q(x)\) (higher is better) so “optimization quality” reads naturally beside the other metrics.

---

## Metrics (every solver, every instance)

| Metric | Definition |
| ------ | ---------- |
| Objective score | \(-Q(x)\) on the frozen QUBO |
| Boundary accuracy | Fraction of decision vars matching the **gold keep/cut mask** (defined below) |
| Confident-wrong rate | On the frozen mini-eval question set, fraction of answered questions that are confident-wrong under E’s read rules with edges masked by \(x\) |
| Coverage | Fraction of mini-eval questions that produce a non-abstain answer |
| Storage cost | Serialized size (bytes) of kept candidate edges (+ fixed structural edges always counted once per instance) |

### Gold keep/cut mask (for boundary accuracy only; **not** fed to solvers)

- Gold KEEP: types in `{supported, supersedes, contradicts}`
- Gold CUT: types in `{unrelated_similar, uncertain_source, proposed}`
- Ties / unknown: gold CUT

Solvers never receive the gold mask. It is used only for the boundary-accuracy metric.

### Mini-eval (confident-wrong + coverage)

A frozen set of ≤80 questions drawn from NEURON-3 seed-0 test items (or the analogous NEURON-4 slice later), stored in the instance file. Retrieval uses E’s boundary rules with candidate edges filtered by \(x\); structural edges always on. Scoring: +confident-wrong / coverage as in NEURON-3 utility definitions (abstain ≠ wrong).

---

## Instance construction (generator)

1. Load boundary-aware memory outputs (pilot: rebuild E edges via NEURON-3 code **read-only**; production: load NEURON-4 export).
2. Filter to fact–fact typed edges; sample **n=16** candidates (seed `INSTANCE_SEED=20261105`) stratified by type when possible (at least one of each available type).
3. Build \(a_i,b_{ij}\) from the frozen rules above.
4. Attach gold mask, mini-eval questions, storage baselines.
5. Write exact instance to `problems/instance_*.json` (+ numpy-friendly `qubo_*.npz`).

**NEURON-4 swap:** generator accepts `--source neuron4 --edges PATH` with schema `{edges: [[a,b,type],...], facts: [...], questions: [...]}`. Same QUBO rules.

Pilot tag: every pilot artifact is marked `"pilot": "NEURON-3-E"`.

---

## Solvers (matched size; same Q)

| Solver | Budget |
| ------ | ------ |
| Brute force | Exact: enumerate \(2^n\) (n≤20) |
| Simulated annealing | `SA_STEPS = 20 * 2^n / 32` energy evals (cap 200_000), 5 restarts, temp schedule locked |
| Greedy / local search | Start from keep-all and cut-all; single-bit flips until local min; 2 starts |
| Random | `R_SAMPLES = min(1000, 2^n)` uniform bitstrings, report best by \(Q\) |
| Ideal QAOA | Qiskit Aer statevector (or density if needed); p=1 and p=2; COBYLA on γ,β; `maxiter=100`; measure distribution → best sampled bitstring among `shots_eval=4096` from the optimized state (statevector: sample from probabilities) |

All classical stochastic methods use seed `SOLVER_SEED=42` (+ restart index).

---

## Ideal QAOA details (local only)

- Cost Hamiltonian = Ising form of frozen \(Q\).
- Mixer = transverse field \(\sum X_i\).
- No IBM runtime, no fake backend noise model required for the prepare phase (noise is for hardware report later).
- Record: best \(Q\), bitstring, all five metrics, wall time, parameter history summary.

---

## Hardware plan gate (do not submit)

If (and only if) the user later approves, a hardware job must satisfy:

| Constraint | Limit |
| ---------- | ----- |
| Qubits | \(n\) (one per variable), plus 0 ancilla |
| QAOA depth | p≤2 |
| Shots | ≤4096 |
| Estimated QPU seconds | **≤40 s** |
| Reserve | Leave **≥90 s** of the **345 s** balance |

Estimate formula (logged before submit):  
`est_s ≈ overhead_s + shots * (circuit_duration_dt * dt_s) * n_circuits`  
with `n_circuits = 1` for a single optimized parameter set (params from ideal sim), plus optional 1 calibration/verify circuit. Prefer `ibm_brisbane` / least-busy backend with ≥n qubits. **No job in this prepare phase.**

---

## Reporting rules

- Side-by-side table: brute / SA / greedy / random / QAOA-p1 / QAOA-p2 (+ hardware later).
- Never call a QPU win “quantum advantage.”
- Nulls and classical wins reported honestly.
- Do not print, log, or commit `IBM_QUANTUM_TOKEN`.

---

## Accept / reject (pilot)

- **Preparation success:** instance files exist; brute force optimum known; all solvers + ideal QAOA ran; hardware estimate ≤40 s with ≥90 s reserve **or** explicitly “not worth hardware” with reason.
- **Hardware worth testing?** Yes only if ideal QAOA’s objective is within 5% of brute-force best **and** confident-wrong ≤ brute-force’s + 1 question, **and** cost estimate passes the gate. Otherwise: **not worth spending QPU time.**

---

## Paths

- Code + prereg: `/workspace/jarvis/neuron/neuron4_qpu/`
- Problems: `problems/`
- Results: `results/`
- Do not write into `neuron3/`, `neuron4/`, or `frozen/`.

## Amendment log

None at prereg time.


## Amendment 1 (2026-11-05 ET) — candidate sampling

Instance construction prefers multi-edge `(entity, attr, project)` groups so quadratic
`b_ij` terms are present. Frozen QUBO *weights* unchanged. Replaces the first pilot
instance file (n_rzz was 0 under pure type-stratified sampling). Same seed `INSTANCE_SEED`.

## Amendment 2 (2026-11-05 ET) — quadratic extras + type-balanced sampling

Locked **before** regenerating the interacting pilot instance / re-running solvers.

Additional quadratic \(b_{ij}\) when both KEEP and same `(entity,attr,project)`:
| Pair types | \(b_{ij}\) |
| ---------- | --------- |
| `contradicts` + `unrelated_similar` | \(+1.5\) |
| `supported` + `unrelated_similar` | \(+2.0\) |
| `uncertain_source` + `supported` | \(+1.0\) |

Candidate sampling: ≥2 edges per available type when possible, then fill from mixed-type groups, then random fill. `proposed` edges (entity-node→fact) are eligible. Prior linear-only instance superseded; hash recorded in results.

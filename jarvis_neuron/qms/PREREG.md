# QUANTUM MEMORY SEARCH (QMS-1) — classical-first preregistration
Written 2026-11-05 ~3:35 PM ET before any QMS-1 run. No IBM, no token, no QPU. Ideal circuit simulation only (local numpy statevector).
See AUDIT_neuron4_qpu_pilot.md for why the earlier pilot is not used.

## Real optimization instance (from NEURON-4 E-auto, not oracle)
10 binary switches of the E-auto read/boundary policy, applied on top of NEURON-4 E-auto (raw-text facts, classifier-predicted uncertainty, E graph retrieval). Defaults in [ ] = NEURON-4 E-auto behaviour.
x0 drop superseded (older, certain, different value in same group) [1]
x1 abstain on unresolved contradiction (else majority value) [1]
x2 abstain when only uncertain-predicted facts support [1]
x3 project scope (else cross-project twins allowed) [1]
x4,x5 uncertainty threshold offset: thr + {(0,0):0, (1,0):-0.15, (0,1):+0.15, (1,1):+0.3} [0,0]
x6 'latest' intent picks newest date only (else all live facts) [0]
x7 require >=2 agreeing certain facts to answer [0]
x8 treat facts from files named PREREG* as uncertain (filename cue) [0]
x9 fallback to exact-search candidates if graph retrieval finds none [0]
Retrieval is computed once per question and cached; every config's answers are re-read and re-scored from those candidates (actual replay, not constants).

## Data
NEURON-4 corpus 1, whole-project split per seed. OPTIMIZATION set = up to 10/kind questions from TRAIN projects. EVALUATION = held-out test-project questions (NEURON-4 recipe), never seen by any optimizer. Independent seeds 200-219.

## Objective (locked)
O(x) = correct - 2*confident_wrong on the optimization set (NEURON utility). The previously locked S = A - L - M - F (equal weights) is also reported; L (latency) and M (storage) do not depend on x here, so they are reported but cannot drive the choice; F is measured once for chosen configs (+1000 synthetic memories).

## Solvers (same instance, equal budget of B=64 true objective evaluations, except exact)
exact enumeration (1024 evals, reference); random (64); greedy single-bit from default (<=64); simulated annealing (64);
QUBO-surrogate: fit quadratic model by ridge on 64 random evaluations, then (a) exact classical minimisation of the surrogate, (b) ideal QAOA p=1,2 statevector sim of the surrogate (grid+Nelder-Mead on angles), best of 64 samples re-scored with true objective (counts within the 64? No: QAOA uses the 64 fit evals + 1 final eval; classical surrogate same). Wall-clock reported per solver.

## Metrics on held-out test, per solver's chosen config
correct, coverage, answerable coverage, selective acc, confident-wrong, boundary accuracy (= correct on conflict/uncertain/latest/distractor kinds), multi-hop, storage, median latency, forgetting F, S.
Also: optimization-set objective gap to exact optimum; generalisation gap (train O vs test O).

## Tests (Holm over 2): T1 exact-optimum config vs E-auto default on test confident-wrong (cluster sign-flip); T2 test correct non-inferior 2 pp.
## Hardware decision rule (locked)
Hardware has scientific value only if, on the ideal simulator, QAOA's surrogate optimum differs from classical surrogate minimisation on >=25% of seeds OR its test outcomes differ by >=2 pp. Otherwise: no hardware experiment (classical exact is trivial at n=10).

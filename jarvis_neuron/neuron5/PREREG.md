# NEURON-5 — Physics-Inspired Memory Engine (preregistration)
Written 2026-11-05 ~3:20 PM ET while NEURON-4 seeds were still finishing; before any NEURON-5 run and before NEURON-4 analysis was read. No QPU, no external models.

## Engine (system "P_energy")
- Store: append-only adapter log. Each memory chunk = one adapter segment; new memories are appended as new segments; nothing is overwritten or deleted.
- Facts: auto-extracted from raw memory text exactly as NEURON-4 (no gold flags). Uncertainty probability p_unc from the NEURON-4 classifier recipe trained on TRAIN projects only.
- Retrieval: NEURON-4 graph retrieval (A edges, same byte cap) to get candidate facts for (project, entity, attr).
- Energy over binary states s_v (one per candidate value v):
  E(s) = -sum_v (h_v - b) s_v + J * sum_{v<v'} s_v s_v'
  h_v = sum over in-project facts f with value v of (1 - p_unc_f) * lam^(newer_disagreeing_f)   [lam=0.25 down-weights a fact that is superseded by a later-dated different value]
  cross-project same entity/attr facts contribute 0 (boundary); J = 10 (hard mutual exclusion).
- Solver: exact enumeration of all states (candidates are <= 12 per query); a simulated-annealing solver is run as a check on a 5% sample and must agree.
- Answer = value in the minimum-energy state; abstain if ground state is empty or energy gap (2nd-lowest minus lowest) < tau.
  b and tau are chosen on TRAIN-project dev questions only (grid b in {0.25,0.5,0.75}, tau in {0.1,0.25,0.5,0.75}; maximise utility = correct - 2*confident_wrong). Confidence = gap.
- Multi-hop: same two-step bridge as NEURON-3/4 systems.

## Baselines
best NEURON-4 auto system (whichever of E_auto / D_auto had lower corpus-1 confident-wrong in the frozen NEURON-4 analysis), A, exact, RAG — re-run, unchanged code, on the same splits.

## Data / splits / seeds
Corpus 1 (NEURON-3/4 corpus), whole-project hold-out as NEURON-4. INDEPENDENT seeds 100-119 (not used in NEURON-4), 20 seeds.
Corpus 2 omitted: its RAG baseline OOM-killed repeatedly on this 16 GB box during NEURON-4. Stated in advance as a limitation.

## Metrics
recall (overall correct), coverage, answerable coverage, selective accuracy, confident-wrong, abstain, multi-hop, storage (native bytes), retrieval latency (median ms/query, wall clock, same process, sequential),
forgetting after +1,000 synthetic memories (NEURON-3 synth_interference): F = max(0, acc_before - acc_after) + corruption, corruption = share of questions answered correctly before and confidently wrong after.

## Multi-objective score (locked)
S = w1*A - w2*L - w3*M - w4*F, A = overall correct; L = latency / max latency among systems (same seed); M = storage / max storage among systems; F as above.
Primary weights: w1=w2=w3=w4=1. Sensitivity: every combination of w2,w3,w4 in {0.5,1,2} with w1=1 (27 combos); report share of combos where P_energy ranks first.

## Tests (pooled 20 seeds, clustered by (seed, fact-group), Holm over all four)
H5a: P_energy confident-wrong < best-N4 (sign-flip, 10k).  H5b: answerable coverage non-inferior to best-N4, margin 10 pp.
H5c: median latency lower than best-N4 (paired per-question sign-flip on log latency).  H5d: overall correct non-inferior, margin 2 pp.
PRIMARY supported iff (H5a and H5b) or (H5c and H5d and storage <= best-N4).

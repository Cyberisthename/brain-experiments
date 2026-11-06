# NEURON-QK1 Quantum Boundary Classifier — PREREG DRAFT (design frozen v1; NOT run; NOT approved for QPU)
2026-11-05 ~4:00 PM ET. Separate from NEURON-6. Required input: NEURON-6 independently human-reviewed, adjudicated items (BLOCKED until they exist).
No QPU time, no token loading, no hardware jobs, no Kaggle, no JARVIS integration. No advantage claims. A user-mentioned "2026 benchmark with 29 comparisons" is UNVERIFIED and is not cited.

## Task
Classify each evidence pair/item boundary into {supports, contradicts, supersedes, different_context, weak_evidence, insufficient} (labels = N6 adjudicated human relations). Then plug each classifier into E's read rule and RERUN actual retrieval/answering.

## Features (8; all computed at inference from evidence text + provenance only; qk1/features.py)
f1 source reliability — ONLY from reviewer-independent observable provenance proxies (hedge cues, PREREG/draft markers, explicit erratum); human reliability ratings are labels/targets, never features. Missing -> 0 + missing-indicator.
f2 contradiction strength — relative numeric disagreement |a-b|/max(|a|,|b|) for same entity/attribute; 0 if non-numeric/absent.
f3 time difference — |date_a-date_b| in days from text dates (log1p); missing -> indicator.
f4 relationship depth — shortest path length between the two facts in a graph built ONLY from train-split documents (+ the item's own evidence at inference).
f5 context similarity — TF-IDF cosine of surrounding context (vocab fit on train only).
f6 evidence completeness — share of required slots (entity, attribute, value, unit, date, source) present.
f7 independent supporting-source count — count of distinct origin_ids agreeing; copies deduped by text hash / origin_id (copies count once).
f8 source conflict — number of distinct origin_ids disagreeing.
Missingness indicators appended for f1, f3, f4 (stated extra inputs, used by all classifiers identically).

## Leakage controls
Pipeline receives an ItemView without final/annotations fields (assert); gold labels only joined in evaluator. Split by origin/project group BEFORE scaling, feature selection, kernel bandwidth, graph construction. Assertions in qk1/pipeline.py.

## Splits / tuning
Outer: locked TEST = N6 TEST groups (fresh projects/origins). Inner: grouped K=5 CV on train for tuning. Matched tuning budget: each model gets 40 hyperparameter configurations (random search, same seed protocol). Repeats: 10 data-split seeds (grouped) x 3 optimisation seeds; seeds are NOT treated as new independent documents.

## Models (same 8+3 features for all kernel models)
L0 logistic regression (sanity) · K1 tuned RBF SVM · K2 random-Fourier-feature linear classifier · G1 graph classifier (label propagation over train-only fact graph + features; extra structural input disclosed) ·
Q1 ideal (noiseless statevector) shallow quantum fidelity kernel: angle encoding RY(x_i) per qubit + one ring of ZZ(x_i x_j) + RY layer, k(x,x') = |<phi(x)|phi(x')>|^2, then classical SVM; bandwidth/scale tuned in the same budget.
Controls: oracle (human relations) and corrupted relations (rate r in {0.1,0.3,0.5}, seeds 7001-7005), paired on same items.

## Metrics
Classification: macro-F1, per-class recall, calibration (ECE), grouped bootstrap CI. Retrieval (rerun): confident-wrong, answerable coverage, selective accuracy, correct abstention, false/missed conflict, AURC.

## Decision gate (locked; fixes the QMS-1 flaw — differing choices are NOT enough)
Q1 is "worth a hardware experiment" only if ALL: (a) on TEST retrieval, Q1 reduces confident-wrong vs best tuned classical (K1/K2/G1, chosen on train CV) by >=2 pp absolute, Holm-adjusted p<0.05 with grouped (project) sign-flip, (b) answerable coverage within 3 pp (non-inferior), (c) direction reproduces in >=8/10 data-split seeds, (d) the effect survives a realistic noise model (shot noise at planned shots + depolarizing 1e-3/1e-2 per 2q gate) in simulation, (e) cost estimate passes the budget below. Otherwise: no hardware. Even if passed: hardware result is a noise check, not an advantage claim, and a quantum-kernel result is reported separately from any JARVIS improvement.

## QPU cost model (estimate only; qk1/cost.py; no guarantee)
Kernel needs n_train(n_train-1)/2 train pairs + n_test*n_train test pairs circuits; each with S shots, R repeats, plus calibration circuits. est_s = jobs*overhead + circuits*S*R*t_shot. Reserve >= 90 s of balance must remain; any run requires explicit user approval.

# NEURON-4 — Autonomous Relationship Discovery (preregistration)
Written 2026-11-05 ~1:15 PM ET, before any NEURON-4 system was run. No QPU, no external/pretrained models.
Frozen inputs: see HASHES_FROZEN.txt (NEURON-3 run script copied verbatim to n3lib.py, sha256 2a392b92…; must match).

## Question
NEURON-3's boundary-aware E got its edge labels (uncertain / prereg flags, groupings) from the same fact metadata
the grader used. Can E (and D) reach low confident-wrong rates when they must infer relation types from raw text?

## Data & splits
- Corpus 1: NEURON-3 corpus (128 RESULTS/PREREG/audit .md files, 18 projects). Gold = NEURON-3 evaluator rules, computed by the evaluator only.
- Split per seed by WHOLE PROJECT (hash(seed, project) < 0.5 -> test). Test questions only from held-out projects; dev
  questions + classifier training labels only from train projects. 20 seeds (0-19).
- Corpus 2 (cross-source, never used for training/tuning): other markdown on the box with numeric tables
  (brain_qa, bookish-umbrella, chatbot*, pr164, jarvis_model, Jarvis-2v, v, aee), deduped by content hash, gold by the same evaluator rules.

## Systems
| name | facts it reads | relation labels |
|---|---|---|
| exact | auto-extracted from memory text | none (flat reader) |
| rag | auto-extracted | none |
| A | auto-extracted | NEURON-3 normal edges built from text-derived keys/values/dates |
| D-auto | auto-extracted | weights from *predicted* uncertainty |
| E-auto | auto-extracted | supports/contradicts/supersedes/unrelated-similar/uncertain-source inferred from text: value disagreement, header dates, project scope, and a learned uncertainty classifier |
| E-oracle | evaluator gold facts | NEURON-3 E labels (upper bound, leaky by design) |
Uncertainty classifier: numpy logistic regression on hashed char/word n-grams of the cell text, heading and file name; trained per seed on train-project facts only; threshold picked on train only (max F1, 3-fold within train).

## Leakage controls
- Systems receive memory views {id, text, project parsed from header} and query views {text, intent, project, entity, attr, sattr, sval}; gold keys (gold, kind, cluster, uncertain, prereg, gold_fact, src) are asserted absent.
- Auto facts are re-extracted from memory text; uncertain/prereg fields are deleted on extraction and asserted absent.
- A Python audit hook logs every file open during system build/answer phases; asserted that none is a gold/results file. Log saved.
- Test question text asserted not to occur verbatim in the corpus.

## Scoring (identical to NEURON-3 `score`)
correct = (answer matches gold value) or (abstained and gold==ABSTAIN); confident-wrong = gave any answer and it is not correct (incl. answering an ABSTAIN question); abstain = returned no answer.
Coverage = answered/all; answerable-coverage = answered/(gold != ABSTAIN); selective acc = correct answers/answered;
appropriate abstention = abstain rate on gold==ABSTAIN; multi-hop = correct on multi_hop_3; storage = native bytes (+ classifier bytes);
AURC = mean over k of error rate among top-k answered questions ranked by confidence (lower better).
Equal budgets: retrieval byte cap = max dev mean bytes read across systems (NEURON-3 recipe).

## Prereg tests (corpus 1 test projects, pooled 20 seeds, clustered by fact-group)
H1a: E-auto confident-wrong < A  (cluster sign-flip, 10k perms, two-sided p)
H1b: E-auto confident-wrong < exact
H1c: E-auto answerable-coverage non-inferior to A, margin 10 pp (p = bootstrap share of diff <= -0.10, 2000 resamples)
H1d: same vs exact
Holm correction over H1a-d, alpha 0.05. PRIMARY HYPOTHESIS SUPPORTED iff all four pass after Holm.
Secondary (descriptive, no claims): D-auto same tests; E-auto vs E-oracle gap; corpus-2 results; classifier F1 on test projects.

## Known limitation stated in advance
The evaluator's "uncertain" gold is itself a text rule (≈, ~, est, pending, PREREG file...). A learned classifier can rediscover it,
so corpus-1 success shows the boundaries are *learnable from text without the grader's labels*, not that they match human judgement. Corpus 2 tests transfer.

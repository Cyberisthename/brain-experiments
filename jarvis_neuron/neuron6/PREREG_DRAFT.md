# NEURON-6 Boundary Reliability — PREREGISTRATION DRAFT (design frozen; benchmark NOT locked)
Status 2026-11-05 ~3:50 PM ET: DESIGN FROZEN v1. No items exist yet. No human review has happened. Benchmark lock (LOCK.md + hashes)
happens only after independent review/adjudication and BEFORE any system is run on test items. No QPU, no Kaggle, no JARVIS changes.
Claims allowed: performance on human-judged boundary cases. Not allowed: "E-auto understands meaning".

## Control
NEURON-4 E_auto unchanged (CONTROL_FREEZE_MANIFEST.md; hashes verified). Any later "targeted change" is a new system id; control stays runnable.

## Categories (user's seven) and operational definitions
| id | category | item contains | correct behaviour |
|---|---|---|---|
| C1 | direct contradiction | >=2 independent sources, same entity/attribute/context/time, different values | flag conflict / abstain (or answer only if one source is adjudicated reliable) |
| C2 | time change | values valid at different documented times | answer value valid at the asked time; "current" = latest documented |
| C3 | different context | apparently conflicting values, different conditions (setup, units, subset) | answer for the asked context; NOT a conflict |
| C4 | weak evidence | single claim, hedged or from source the reviewers rate low-reliability | abstain or answer with low confidence (frozen convention below) |
| C5 | missing evidence | answer absent from memory; near-miss lexical distractors present | abstain |
| C6 | correction | explicit later erratum/correction of earlier claim | answer corrected value |
| C7 | multi-hop | answer needs >=3 linked facts across >=2 documents | correct final value |
Each category must include lexical counterexamples (same words, different meaning) and context/time mismatches.

## Items and provenance (no invented sources)
- Every evidence passage must come from a real, citable document (local research files with path+hash, or external public documents with URL+retrieval date+archived copy). Truth date recorded when the source states one; else "unknown".
- Source reliability is a reviewer judgement with stated reason, never assigned by the builder or a system. Corroboration counts only independent sources: copies/derivatives of the same origin (same text hash, quoted from one origin, same project duplicated across folders) count as ONE source.
- Answer label in {answerable(value), ambiguous/conflict, insufficient}. New random seeds do NOT create new independent documents; independence is at document-origin level.
- Splits: DEV (for building/debugging, may be inspected) and TEST (locked, sealed hashes; no system or builder tuning on it). Split by document origin/project, not by question.

## Sample size (rationale)
- Per category ~100 TEST items -> any rate estimated within about +/-10 pp (95% CI half-width at p=0.5 is 9.8 pp).
- Primary paired contrasts (E-auto vs each of 5 comparators, confident-wrong, McNemar), Holm over 5 -> alpha ~0.01: to detect a 10 pp difference with ~20% discordant pairs at 80% power needs about 240 paired items; 700 TEST items covers this overall and gives category-level descriptives (category tests are secondary).
- TEST = 700 (7 x 100), DEV = 140 (7 x 20). If reviewers can supply fewer, the reduced power is stated before lock; no post-hoc top-up from DEV.

## Review protocol (requires REAL independent people; currently BLOCKED)
- Roles kept separate where feasible: builders (draft candidates), boundary annotators (2 per item, independent), adjudicator (resolves disagreements), answer graders (blind to system identity; never annotated the same item). The builder (this bot) may not act as annotator, adjudicator or grader.
- Agreement: Cohen's kappa per category on label + boundary type; items with unresolved disagreement after adjudication are dropped and counted. Target kappa >= 0.6; below that the category is reported as unreliable.
- Tracking: annotation_log.jsonl with item_id, role, reviewer pseudonym, timestamp, label, reason, version. Review packet: review_packet/.

## Systems (equal evidence/indexing budget: same memory store, same retrieval byte cap = max dev mean, same top-k)
S1 E-auto unchanged · S2 E with human-supplied boundaries (adjudicated relation labels replace inferred ones; read rule unchanged) ·
S3 E with corrupted boundaries: relation labels of a fixed fraction r in {0.1,0.3,0.5} replaced by a uniformly drawn other type, corruption seeds 6001-6005, logged masks · S4 exact search · S5 RAG (NEURON-3 recipe) · S6 plain KG (A edges, no relation types).

## Confidence convention (frozen)
"Answered" = system returns a value. Confidence = system's native score; for risk-coverage, ties broken by item hash. No recalibration after seeing TEST.

## Metrics (TEST; per category and overall; with denominators always printed)
selective accuracy = correct/answered · answerable coverage = answered/answerable · confident-wrong = wrong answers/all items ·
correct abstention = abstained/(conflict+insufficient items) · false-conflict = conflict-abstain/(answerable C2,C3,C6 items) ·
missed-conflict = answered/(C1 items) · retention after updates = accuracy on unchanged items after appending C6 corrections and 1000 unrelated memories (vs before) · storage bytes · median/95th latency ms. Equal-coverage comparison (accuracy at matched coverage via confidence thresholds) and risk-coverage curves + AURC.
Answer grading: numeric normalisation per NEURON-3 `norm_num`; non-numeric answers graded by blind human graders with written rubric.

## Primary hypotheses (on TEST, Holm)
H1 E-auto confident-wrong < exact, RAG, plain KG, at answerable coverage within 10 pp. H2 S2 (human boundaries) - S1 gap reported (headroom). H3 corrupted boundaries degrade monotonically with r (manipulation check).

## Order (user-specified)
1 freeze control (done) -> 2 human-validated benchmark (BLOCKED: reviewers) -> 3 failure analysis -> 4 ONE targeted change to the component behind the largest failure category -> 5 fresh evaluation on new projects/questions/seeds -> 6 shadow-mode JARVIS branch only after passing.

# NEURON-7 Hypothesis Space Reasoning — FAST preregistration

Status: LOCKED BEFORE IMPLEMENTATION OUTCOMES. Small exploratory motif experiment. Not a biological-brain claim, not independent human validation, not a production promotion.

Locked at 2026-11-06 08:05:09 EST (America/New_York).

## Central question
Can a Hypothesis Space Reasoner (HSR) derive more correct conclusions from the SAME available local knowledge without increasing confident errors, versus retrieval and association baselines?

Try to FALSIFY. Do not assume HSR works.

## Scope and invariants
- Write all artifacts under `/workspace/jarvis/neuron/neuron7_compositional/` only.
- Use only existing local `uploaded_facts.jsonl` and `jarvis_notes.jsonl` (and optional copies of extracted facts from `/workspace/jarvis/neuron/reason/` as raw text sources). No Institutional Books, no invented scientific gold claims, no fine-tuning, no QPU, no Kaggle, no git push, no uploads.
- Do NOT modify `/workspace/jarvis/neuron/neuron4/`. Hash before and after.
- Do NOT change JARVIS default chat (CAD hybrid stays default; `--reason` / `--shadow` stay opt-in).
- Do NOT reuse the association-test primary `questions.json` as the scored set.
- Do NOT use the historical 30Q chat development items as primary scored items.
- CPU only, `nice 19`, `OMP/MKL/OpenBLAS` threads=1.

## Architecture under test
Hypothesis Space Reasoning maintains and tests competing hypotheses rather than relying only on direct retrieval or association traversal.

Mechanisms (all required, kept cheap):
1. Counterfactual testing: for each hypothesis, record what evidence would invalidate it.
2. Dependency tracking: record which evidence supports each derived conclusion; drop the conclusion if a dependency fails.
3. Competing hypotheses: generate multiple explanations; rank by supporting evidence, contradictions, and missing information.
4. Recursive decomposition: split multi-hop / compositional questions into subproblems and combine results.

Exploratory extra arm (one revision pass only, not retry-until-convincing):
5. Self-revising HSR: if the selected hypothesis hits a contradiction or failed dependency, drop it once, re-rank remaining hypotheses, and answer from the revised top hypothesis (or abstain). Counts as extra help only if it improves accuracy or confident-wrong vs unrevised HSR.

## Arms
1. `bag` — retrieval + span extract only (no hypothesis space; cannot invent derived values).
2. `assoc` — association graph adapted from `/workspace/jarvis/neuron/reason/system.py` (plain graph readout; no neuron4 edits).
3. `n4_boundary` — same association graph with E-boundary reliability readout from the reason experiment (weak/conflict abstain). Full frozen E_auto / n3lib is NOT run on these free-text compositional items; this arm is the disclosed N4-style reliability baseline, not a claim that full NEURON-4 ran.
4. `hsr` — Hypothesis Space Reasoner (proposal).
5. `hsr_shuffle` — control that should break fake machinery: randomly pair premises / shuffle hypothesis dependencies and ignore contradictions (seeded). If HSR's win disappears here, the machinery is not doing real dependency work.
6. `hsr_revise` — exploratory one-pass self-revision on top of HSR (not required for the primary win).

All arms share the same card corpus, the same extracted atomic premises inventory, and the same candidate retrieval budget except for the preregistered intervention.

## Dataset (FAST)
- Target ~25–40 NEW questions built only from existing local cards.
- Unseen relative to the association-test primary question file: new templates; answers for compositional items are not a single stored source span when the gold is a derived comparison/count/difference/conjunction claim.
- Include slices:
  - `compose` — cross-document composition (premise on card A + premise on card B ⇒ conclusion C not written as one fact)
  - `multihop` — needs an intermediate sub-answer then a second lookup/combine
  - `conflict` — contradictory evidence; gold ABSTAIN
  - `missing` — required premise absent; gold ABSTAIN
  - `uncertain` — only weak/approx evidence; gold ABSTAIN
  - `distractor` — similar-looking wrong pairing should not yield a confident wrong join
  - `direct` — a few single-span lookups to check we did not wreck ordinary facts
- Gold must be licensed by existing source text (comparisons, conjunctions, counts, or explicit abstain). No fake scientific claims invented as gold.
- Freeze cards, questions, settings, and system hashes in `frozen_manifest.json` before first scoring.

## Seeds
- Deterministic arms (`bag`, `assoc`, `n4_boundary`, `hsr`, `hsr_revise`): run once; do not fake replication by repeating identical scores.
- Randomized control `hsr_shuffle`: seeds 0..7.
- Report shuffle as mean ± over seeds; compare HSR to shuffle mean and to deterministic baselines.

## Metrics
Report per arm and per slice:
- accuracy (correct includes correct abstentions on unanswerable items)
- coverage (answered/all)
- answerable coverage
- confident-wrong among answered (undefined if zero answers; do not report as 0)
- abstention quality on conflict/missing/uncertain: correct abstention rate and false abstention on answerable items in those slices
- latency ms/question
- memory overhead: peak RSS during scoring and/or size of hypothesis/dependency structures vs bag

## Locked win rule (primary)
HSR is SUPPORTED only if ALL hold:
1. On the combined `compose`+`multihop` slice, HSR accuracy is strictly greater than BOTH `bag` and `assoc`.
2. HSR confident-wrong on that slice is not higher than either of those two controls (undefined CWR does not count as improvement by itself unless accuracy also rises with non-worse abstention quality).
3. On conflict+missing+uncertain, HSR abstention quality (correct abstentions / those items) is not worse than `n4_boundary`.
4. On `direct`, HSR accuracy is within 10 percentage points of `bag` and confident-wrong is not higher than `bag`.
5. HSR accuracy on `compose`+`multihop` is strictly greater than `hsr_shuffle` mean (falsification control).

Otherwise: NOT SUPPORTED. Call FALSIFIED only if HSR clearly increases confident-wrong vs bag/assoc on compose+multihop, or loses badly on that slice while the shuffle control matches HSR (machinery inert).

`hsr_revise` is exploratory: report whether one revision pass improves accuracy or CWR vs `hsr` without turning into retry-until-pretty (max one revision).

## Deliverables
PREREG.md, RESEARCH.md (existing design note), cards/facts/questions, system.py, run_experiment.py, frozen_manifest.json, summary.json, RESULTS.md, neuron4 hash before/after, exact rerun command, limitations.

No production promotion based on this fast test.

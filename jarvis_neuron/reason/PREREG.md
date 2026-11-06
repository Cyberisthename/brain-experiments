# Association reasoning preregistration

Status: LOCKED BEFORE IMPLEMENTATION AND OUTCOMES. This is a small exploratory motif experiment, not a biological-brain claim or evidence of general understanding.

## Scope and invariants
Follow RESEARCH.md. Use only existing local uploaded_facts, jarvis_notes, and/or frozen NEURON memories; no Institutional Books, invented scientific gold claims, fine-tuning, QPU, uploads, pushes, or live/default answer changes. Write experimental artifacts only under neuron/reason. Import frozen modules read-only with bytecode writing disabled. Hash every neuron4 file before and after. CAD hybrid stays default; --reason and --shadow stay opt-in.

## Hypotheses and systems
H1: E-boundary association reasoning beats no-link bag/retrieval extraction on association/hop questions, while preserving answerable coverage and direct facts.
H2: E-boundary real associations beat degree-matched endpoint-shuffled associations. A tie or shuffle win does not support the real-link-pattern claim.
H3 (secondary): real typed links beat label-shuffled links.
Required arms: no-link bag-of-facts, plain association graph, graph plus E boundaries, degree-matched endpoint shuffle with the same boundary policy/readout, and label shuffle with the same endpoints/readout. Any existing sorter is an extra baseline, not a substitute for no-links. All arms share extracted facts, candidate budget, retrieval inputs, and readout except the preregistered intervention. Hop solvers must not bypass the graph through an unrestricted entity-table join. Verify shuffled degree sequences/link count exactly; if supplied B fails this check, use degree-preserving edge swaps and disclose the change before scoring.

## Dataset and leakage controls
Build gold and questions from existing source-backed text only; attach source paths/card IDs and supporting spans. Include separate direct, association/hop, conflict, uncertain/weak, and similar-unrelated slices where available. Hop pairs must have an explicit same-card or clearly linked-note relationship; mere common tokens are insufficient. Conflict/uncertain items require abstention unless source text explicitly resolves them. Gold labels, answer spans annotated for grading, question types, and source reliability judgments from graders may never be extraction or graph features. Extraction uses raw source text and document structure only. Freeze extracted corpus, question file, scorer, and arm settings with SHA256 before first scoring. No outcome-informed tuning; bugs require a versioned amendment, fresh run, and retention of invalidated results. Previously tuned 30-question chat development items are excluded from primary results; any reuse is reported separately as development leakage. Automatically constructed items and shared historical notes are not independent human validation.

## Seeds and resource policy
Use eight fixed seeds 0..7 for each randomized arm. Keep deterministic controls fixed and do not misrepresent repeated deterministic scores as independent replication. Use nice 19, OMP/MKL/OpenBLAS threads=1 and CPU only. If resources prevent eight seeds, state why; all results remain exploratory. No external services or default chat integrations.

## Metrics and inference
Report means per arm separately for direct, association/hop, conflict, uncertain, and similar-unrelated, plus combined conflict/uncertain. Accuracy includes correct abstentions on genuinely unanswerable items. Coverage is answered/all; answerable coverage is answered/answerable. Confident-wrong is wrong/answered (undefined, not zero, when no answers); selective accuracy is correct/answered. Report counts, seed-level scores, abstentions, and question-level predictions, not only pooled metrics. Freeze answer normalization before scoring.

Primary endpoint is association/hop accuracy difference (E minus comparator). For H1 and H2 use one-sided paired randomization tests on source-card/connected-source clusters: average each question over the eight seeds, aggregate differences within source cluster, then flip cluster signs (enumerate all assignments if <=20 clusters, otherwise 100000 Monte Carlo assignments with seed 20261106 and plus-one correction). This avoids treating seed repeats or sibling questions as independent samples. Holm correction across H1 and H2 at familywise alpha .05. Report raw and adjusted p-values, effect sizes and number of independent source clusters. H3 is secondary exploratory: same cluster test with its unadjusted p-value clearly marked, no confirmatory declaration. If clusters or nonzero differences are too few to meet threshold, say evidence is insufficient; no substitute seed-only test.

## Locked win rule
An overall supported result requires BOTH H1 and H2 positive accuracy differences with Holm-adjusted p<=.05, no increase in association/hop confident-wrong relative to either control, answerable association/hop coverage >=80% and no more than 5 percentage points below either control, and direct accuracy/answerable coverage no more than 5 points below bag with no increase in direct confident-wrong. These thresholds are engineering safeguards, not powered noninferiority proofs. Lower confident-wrong without significant primary accuracy improvement is useful descriptive evidence but not the preregistered win. Ties, mixed results, and limited power are reported plainly as not supported; falsified is reserved for a clear replicated reverse effect rather than a nonsignificant null. Report H3 as exploratory supported/not supported/reversed with its evidence.

## Deliverables
PREREG.md, frozen source/question manifests, runner, question-level results, summary.json, RESULTS.md, and before/after neuron4 hashes. Include exact rerun command, all deviations, limitations, and explicit default-answer/frozen-file invariants. No production promotion based on this small experiment.

Locked at 2026-11-06 07:52:26 EST.

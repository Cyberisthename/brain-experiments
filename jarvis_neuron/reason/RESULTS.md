# Association reasoning experiment
Finished 2026-11-06 07:58:50 EST (America/New_York).

**Overall: NOT SUPPORTED under the locked win rule.**

## Hypotheses
| Hypothesis | Comparator | Hop accuracy difference | Raw one-sided p | Holm p | Verdict |
|---|---|---:|---:|---:|---|
| H1 | bag | +0.000 | 1.000000 | 1.000 | not supported |
| H2 | degree_shuffle | +0.819 | 0.007812 | 0.016 | supported |
| H3 (secondary/exploratory) | label_shuffle | +0.000 | 1.000000 | undefined | not supported |

Primary paired sign flips use seven research-topic source clusters, not 248 independent hop trials. Questions are averaged over seeds before source-cluster aggregation. H1/H2 use Holm correction; H3 is unadjusted exploratory. No nonsignificant null is called falsification.

## Slice means across eight seeds
| Slice | Arm | Questions/seed | Accuracy | Coverage | Answerable coverage | Wrong / answered | Selective accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| direct | bag | 52 | 0.962 | 0.962 | 0.962 | 0.000 | 1.000 |
| direct | plain | 52 | 0.962 | 0.962 | 0.962 | 0.000 | 1.000 |
| direct | E | 52 | 0.962 | 0.962 | 0.962 | 0.000 | 1.000 |
| direct | degree_shuffle | 52 | 0.962 | 0.962 | 0.962 | 0.000 | 1.000 |
| direct | label_shuffle | 52 | 0.962 | 0.962 | 0.962 | 0.000 | 1.000 |
| hop | bag | 31 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| hop | plain | 31 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| hop | E | 31 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| hop | degree_shuffle | 31 | 0.181 | 0.181 | 0.181 | 0.000 | 1.000 |
| hop | label_shuffle | 31 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conflict | bag | 1 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| conflict | plain | 1 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| conflict | E | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| conflict | degree_shuffle | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| conflict | label_shuffle | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| uncertain | bag | 1 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| uncertain | plain | 1 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| uncertain | E | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| uncertain | degree_shuffle | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| uncertain | label_shuffle | 1 | 1.000 | 0.000 | undefined | undefined | undefined |
| similar_unrelated | bag | 7 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| similar_unrelated | plain | 7 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| similar_unrelated | E | 7 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| similar_unrelated | degree_shuffle | 7 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| similar_unrelated | label_shuffle | 7 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conflict_uncertain | bag | 2 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| conflict_uncertain | plain | 2 | 0.000 | 1.000 | undefined | 1.000 | 0.000 |
| conflict_uncertain | E | 2 | 1.000 | 0.000 | undefined | undefined | undefined |
| conflict_uncertain | degree_shuffle | 2 | 1.000 | 0.000 | undefined | undefined | undefined |
| conflict_uncertain | label_shuffle | 2 | 1.000 | 0.000 | undefined | undefined | undefined |
| all | bag | 92 | 0.957 | 0.978 | 0.978 | 0.022 | 0.978 |
| all | plain | 92 | 0.957 | 0.978 | 0.978 | 0.022 | 0.978 |
| all | E | 92 | 0.978 | 0.957 | 0.978 | 0.000 | 1.000 |
| all | degree_shuffle | 92 | 0.702 | 0.681 | 0.696 | 0.000 | 1.000 |
| all | label_shuffle | 92 | 0.978 | 0.957 | 0.978 | 0.000 | 1.000 |

Counts, abstentions, denominator-defined seed counts, per-seed ratios, source clusters, and predictions are in summary.json and runs/*/per_question.jsonl. Undefined error with zero answers is not reported as zero.

## Methods and audit
- Raw-text numerical clause-prefix extractor plus literal author/capital patterns; no model training and no existing answer/question/fact_statement fields as system input. Questions supply literal slot/project cues, not a learned natural-language parser.
- All five arms see the same extracted facts and lexical top-12 source cards. Graphs spread for three hops only within that candidate set. The bag baseline may extract a second span from an already retrieved seed card: it is deliberately not forced to abstain on every hop. Graph readout has no unrestricted global entity-table join.
- Degree control uniformly permutes all undirected fact-edge stubs. Every seed preserves the exact node-labelled degree sequence, edge count, and label multiset; multiedges and self-loops are retained. Label control preserves endpoints and label counts. See graph_audit.json.
- E boundaries block unrelated_similar links, downweight uncertain_source links, reject raw-text weak support and unresolved multiple values. Plain graph/bag use ordinary weighted extraction. No dated supersession examples were available in the selected source cards.
- Same-card associations are explicit source structure; cross-topic lexical similarity is a lure, not evidence. Shared-topic notes are clustered together conservatively. Frozen n3lib/run_neuron4 were inspected read-only but not imported; a local prose adapter avoids their unrestricted two-retrieval hop shortcut.

## Guardrails
```json
{
  "hop_answerable_coverage_at_least_80pct": true,
  "direct_accuracy_within_5pp_of_bag": true,
  "direct_answerable_coverage_within_5pp_of_bag": true,
  "direct_confident_wrong_not_increased": true,
  "hop_coverage_within_5pp_of_bag": true,
  "hop_confident_wrong_not_above_bag": true,
  "hop_coverage_within_5pp_of_degree_shuffle": true,
  "hop_confident_wrong_not_above_degree_shuffle": true
}
```

## Limitations
- Small, automatically templated benchmark. Gold numeric/string answers are literal source spans; conflict/uncertainty labels are text rules, not independent human judgments. The extraction and question construction share a representation, so this tests that representation rather than discovering general understanding.
- All primary hop items use two fields co-located in eight existing cards, seven source-topic clusters. This is a limited association test, not a difficult cross-document deduction test. Plain source-card extraction can already solve it; do not claim graph-specific reasoning from high absolute accuracy alone.
- Existing jarvis_notes were previously used for sorter development. Fresh templates/slots do not make those sources independent held-out data. The historical 30 chat questions and conservatively matched author topics were excluded from primary evaluation; their facts remain retrieval distractors.
- One conflict probe and one weak probe were already synthetic examples in jarvis_notes. They are explicitly local probes, not real scientific claims, and two items cannot validate robust abstention. No new fake scientific claims were invented.
- Similar-unrelated slice contains seven superficially similar metric-slot questions, not independently authored adversarial lures. Literal query-slot matching makes entity disambiguation easier than free chat.
- Source cards summarize earlier experiments; the underlying scientific claims were not independently re-audited here. No Institutional Books corpus was loaded. WordNet raw cards remain candidate distractors but are not used as definition gold questions.
- Eight seeds randomize control graphs; deterministic bag/plain/E repeats are not independent replications. Very few source clusters constrain power and generalization. Nominal confidence is uncalibrated; every wrong non-abstaining answer counts as confident-wrong.
- A motif abstraction only: no claim about biological brains, consciousness, or training the 319M model to understand. No production promotion.

## Safety
```json
{
  "neuron4_unchanged": true,
  "neuron4_files": 69,
  "default_files_unchanged": true,
  "no_training": true,
  "no_qpu": true,
  "no_upload": true,
  "no_git_push": true,
  "no_institutional_books": true
}
```
CAD hybrid default and opt-in --reason/--shadow paths were untouched. All 69 neuron4 files match the original before-run SHA256 manifest. No fine-tune, QPU, Kaggle upload, git push, or external service call.

## Preregistration / deviations
- Original PREREG.md hash remains 288e3eff908982f05bd59951f3041ef997cb7690ce61c87837b74e2888428e53. Corpus, question set, arm settings, system, tests and scorer were SHA256-frozen before any system outcomes. No outcome-informed changes.
- Local execution resumed after the initial unrun status; that status was archived under status_before_run/. This is an execution availability change, not a hypothesis/metric/seed/win-rule amendment.

## Rerun
```bash
cd /workspace/jarvis/neuron/reason
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_experiment.py
```
Uses frozen snapshots; verifies hashes and creates a new run directory. Do not rerun prepare.py/make_questions.py to regenerate the locked data.

Files: PREREG.md, IMPLEMENTATION.md, frozen_manifest.json, source_manifest.json, dataset_audit.json, cards.json, facts.json, questions.json, settings.json, system.py, prepare.py, make_questions.py, test_system.py, run_experiment.py, summary.json, neuron4_hashes_before/after.json, runs/*/.

## Post-run interpretation audit (no rerun or tuning)

- E answered 50/52 direct questions correctly and abstained on the two Homer questions (The Iliad / The Odyssey); the same two retrieval misses occurred in every arm. They were not fixed after scoring.
- E and the bag baseline each got 31/31 association questions. The degree-shuffled control averaged 5.625/31 correct across seeds (range 1/31 to 10/31). This establishes a dependence on intact source-card wiring for this particular graph implementation, not an advantage over ordinary local-card extraction.
- E's overall 90/92 versus bag's 88/92 comes entirely from correctly abstaining on the two pre-existing conflict/weak probes. All three boundary-aware arms also passed those probes; correct typed labels were not necessary for that raw-value/weak-marker safeguard.
- Seven topic clusters were fixed before outcomes. Their independence is not externally established: NEURON-4, NEURON-5 and QMS-1 describe related memory research and may share underlying benchmark provenance. H2's adjusted p=0.015625 is conditional on the preregistered topic-cluster construction, not a robust independent-source replication. No alternate clustering was selected after seeing results.
- Primary outcome remains NOT SUPPORTED: adding graph spreading did not improve on the no-link baseline, and real labels tied label-shuffled labels. High absolute accuracy must not be described as making JARVIS generally smarter.

Original source hashes and all 14 frozen artifact hashes were independently checked after the run; none changed. All 3,680 prediction rows cover 92 unique questions, five arms and eight seeds (not 3,680 independent examples).

## Preserved finding (2026-11-06 ~8:01 AM ET)

User interpretation, recorded after the locked results and not used to retune them:

The association graph did not beat ordinary extraction on hop items (31/31 vs 31/31). The only overall gain for E-boundaries (90/92 vs 88/92) was two correct abstentions on conflict/uncertain probes. That suggests **boundaries are more valuable as a reliability mechanism than as a reasoning accelerator.** Preserve this finding.

Next research direction (user): investigate **compositional reasoning** — derive genuinely new, supported conclusions from existing knowledge — not another association-graph test.

Numbering: NEURON-5 already exists (physics-inspired energy memory, completed). This next study should be logged as a new experiment (suggested name: **NEURON-7 Compositional Reasoning**), not as a rewrite of NEURON-5. NEURON-6 remains the independent human-reviewed boundary-reliability benchmark.

See `/workspace/jarvis/neuron/neuron7_compositional/RESEARCH.md`.

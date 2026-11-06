# Training-free evidence reason/sort evaluation

Finished: 2026-11-06 07:40:50 ET (America/New_York)

**Default unchanged: CAD hybrid.** Use `--reason` or `--mode reason`; REPL `/reason`.

No checkpoint loaded. No generation rescored. No gold metadata enters the reasoner.

| Set / method | n | Accuracy | Coverage | Wrong among answered |
|---|---:|---:|---:|---:|
| full_eval_30q / reason | 30 | 1.000 | 1.000 | 0.000 |
| full_eval_30q / live extractive + WordNet | 30 | 0.733 | 1.000 | 0.267 |
| full_eval_30q / published E-auto shadow | 30 | 0.733 | 0.800 | 0.083 |
| full_eval_30q / live retrieval (gold anywhere, not answer accuracy) | 30 | 1.000 | n/a | n/a |
| jarvis_notes / reason | 15 | 0.733 | 0.600 | 0.000 |
| jarvis_notes / live extractive + WordNet | 15 | 0.000 | 1.000 | 1.000 |
| jarvis_notes / published E-auto shadow | 15 | 1.000 | 0.867 | 0.000 |
| jarvis_notes / live retrieval (gold anywhere, not answer accuracy) | 13 | 0.846 | n/a | n/a |

Historical jarvis_notes retrieval: .923 on 13 answerable questions (global, not current routed retrieval). The prior .923 hybrid figure used retrieval-or-span matching and is not actual answer accuracy.

Historical 30Q: hybrid **.900**, routed retrieval **.967**, gen **.300**, CAD **.433**. These are published, not rerun.

## Scoring

Gold substring in answer ONLY (not support or trace). ABSTAIN requires explicit abstention. Any nonabstain wrong answer counts as confident-wrong; not calibrated confidence.

## Per-question outputs (answer only)

| Set | ID | Gold | Answer | Decision | Correct |
|---|---|---|---|---|---|
| full_eval_30q | france | Paris | Paris | answer | True |
| full_eval_30q | dracula | Bram Stoker | Bram Stoker | answer | True |
| full_eval_30q | frankenstein | Mary Shelley | Mary Shelley | answer | True |
| full_eval_30q | wwi | 1914 | 1914 | answer | True |
| full_eval_30q | hamlet | Shakespeare | William Shakespeare | answer | True |
| full_eval_30q | gibbon | Edward Gibbon | Edward Gibbon | answer | True |
| full_eval_30q | herodotus | Herodotus | Herodotus | answer | True |
| full_eval_30q | thucydides | Thucydides | Thucydides | answer | True |
| full_eval_30q | paine | Thomas Paine | Thomas Paine | answer | True |
| full_eval_30q | jefferson | Thomas Jefferson | Thomas Jefferson | answer | True |
| full_eval_30q | zorblax | Quendara | Quendara | answer | True |
| full_eval_30q | voss | Dr. Mirella Voss | Dr. Mirella Voss | answer | True |
| full_eval_30q | wren | 2147 | 2147 | answer | True |
| full_eval_30q | kestrel | three-phase lattice keys | The Kestrel Protocol uses three-phase lattice keys | answer | True |
| full_eval_30q | amberfall | Queen Lysara | Queen Lysara | answer | True |
| full_eval_30q | xylith | 149 | 149 | answer | True |
| full_eval_30q | okorie | Hana Okorie | Hana Okorie | answer | True |
| full_eval_30q | nexora | seven | Planet Nexora-4 has seven moons | answer | True |
| full_eval_30q | helixwind | Priya Dunn | Priya Dunn | answer | True |
| full_eval_30q | marlowe | Eli Navarro | Eli Navarro | answer | True |
| full_eval_30q | glaucite | blue-green | Glaucite is blue-green in color | answer | True |
| full_eval_30q | northmere | Lux in limine | Lux in limine | answer | True |
| full_eval_30q | atlantis | Poseidonis | Poseidonis | answer | True |
| full_eval_30q | gleeb | Nara Quill | Nara Quill | answer | True |
| full_eval_30q | def_neuron | nerve | [DICT WordNet] neuron / 1. (noun) a cell that is specialized to conduct nerve impulses  syn: nerve cell / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| full_eval_30q | def_synapse | junction | [DICT WordNet] synapse / 1. (noun) the junction between two neurons (axon-to-dendrite) or between a neuron and a muscle  e.g. 'nerve impulses cross a synapse through the action of neurotransmitters' / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| full_eval_30q | def_myelin | fatty | [DICT WordNet] myelin / 1. (noun) a white fatty substance that forms a medullary sheath around the axis cylinder of some nerve fibers  syn: medulla, myeline / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| full_eval_30q | def_photosynthesis | synthesis | [DICT WordNet] photosynthesis / 1. (noun) synthesis of compounds with the aid of radiant energy (especially in plants) / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| full_eval_30q | def_algorithm | rule | [DICT WordNet] algorithm / 1. (noun) a precise rule (or set of rules) specifying how to solve some problem  syn: algorithmic program, algorithmic rule / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| full_eval_30q | def_entropy | uncertainty | [DICT WordNet] entropy / 1. (noun) (communication theory) a numerical measure of the uncertainty of an outcome  e.g. 'the signal contained thousands of bits of information'  syn: information, selective information / 2. (noun) (thermodynamics) a thermodynamic quantity representing the amount of energy in a system that is no longer available for doing mechanical work  e.g. 'entropy increases as matter and energy in the universe degrade to an ultimate state of inert uniformity'  syn: S, randomness / [SOURCE] Princeton WordNet 3.0 via NLTK | dictionary | True |
| jarvis_notes | jn_n4_cwr | 0.064 | 0.064 | answer | True |
| jarvis_notes | jn_n4_vs_exact | yes | yes | answer | True |
| jarvis_notes | jn_n4_labels | no | no | answer | True |
| jarvis_notes | jn_n5 | no | no | answer | True |
| jarvis_notes | jn_qms | no | [abstain] | weak | False |
| jarvis_notes | jn_tutor | NOT SUPPORTED | [abstain] | weak | False |
| jarvis_notes | jn_chat_hybrid | 0.900 | 0.900 | answer | True |
| jarvis_notes | jn_chat_gen | 0.300 | 0.300 | answer | True |
| jarvis_notes | jn_chat_cad | 0.433 | 0.433 | answer | True |
| jarvis_notes | jn_chat_ret | 0.967 | [abstain] | weak | False |
| jarvis_notes | jn_hypo | no | [abstain] | weak | False |
| jarvis_notes | jn_callosum | no | no | answer | True |
| jarvis_notes | jn_stp | STP | STP | answer | True |
| jarvis_notes | jn_probe_kappa_a | ABSTAIN | [abstain] | conflict | True |
| jarvis_notes | jn_probe_lambda | ABSTAIN | [abstain] | uncertain | True |

## Example evidence traces

### Who wrote Common Sense?
Answer: Thomas Paine

- Used who_history_153 [uploads, query overlap 100%]: Thomas Paine wrote Common Sense.
- Grouped 1 related statement(s); no parsed same-slot conflict found.
- Skipped who_literature_10: weak/off-topic query overlap (33%); Homer wrote The Odyssey.
- Skipped other relevant statements: different slot or lower relevance.

### What is shadow probe metric Kappa?
Answer: [abstain]

- Values disagree for the same subject/attribute: 0.91 (jn_probe_kappa_b); 0.42 (jn_probe_kappa_a).
- Abstained; source rank/repetition cannot resolve disagreement.
- Skipped jn_probe_kappa_b: weak/off-topic query overlap (25%); Conflicts with probe A.
- Skipped jn_probe_kappa_b: weak/off-topic query overlap (25%); Local shadow-eval conflict probe B.

### What is shadow probe metric Lambda?
Answer: [abstain]

- Only uncertain top support: jn_probe_lambda: Shadow probe metric Lambda is ~0.55 (approx, unverified).
- Abstained rather than turn an estimate into a verified value.
- Skipped jn_probe_lambda: weak/off-topic query overlap (25%); Local shadow-eval uncertain probe.
- Skipped jn_probe_kappa_b: weak/off-topic query overlap (25%); Conflicts with probe A.

## Honest limitations / comparison caveats

- Small existing development sets, not independent validation; no significance claim. Iterative regression fixes were made after inspecting these sets; intermediate runs retained as reason_eval_development_v0/v1/v2.json.
- Default stays CAD hybrid despite meeting the published 30Q number: notes coverage/routing still fails, and no independent holdout validates the rule changes.
- Historical .900 hybrid matching included supporting sentences (and global mode generation); current reason score is stricter answer-only. Historical numbers are context, not a controlled head-to-head.
- WordNet remains enabled; historical hybrid .900 did not use all automatic WordNet answers. Fresh extractive+WordNet baseline is shown to avoid attributing dictionary improvements to reasoning.
- Prior shadow_eval jarvis_notes hybrid_acc is an optimistic retrieval-or-span surrogate; do not call it answer accuracy. This run reports actual span-only baseline.
- Same synthetic notes supplied question/golds; no independent source validation.
- Only supplied top-3 hits checked. Conflicts outside retrieval, paraphrased slots, negation scope, multi-hop reasoning, and numeric comparisons may be missed.
- NOT SUPPORTED is a reported negative verdict, not automatically an uncertain number; exploratory/PARTIAL statements retain caveats in supporting evidence.
- This sorts/checks retrieved text, not general intelligence, consciousness, or learning.

## Tests

................
----------------------------------------------------------------------
Ran 16 tests in 0.004s

OK


## Safety

{
  "no_qpu": true,
  "no_training": true,
  "no_model_load": true,
  "no_kaggle_upload": true,
  "no_institutional_books_indexed": true,
  "neuron4_all_files_unchanged": true,
  "neuron4_files_hashed": 69
}

## Run

```bash
cd /workspace/jarvis/chat_rag && nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 chat.py --reason --once "What is shadow probe metric Kappa?"
```

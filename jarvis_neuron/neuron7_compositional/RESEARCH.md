# NEURON-7 — Compositional Reasoning (design only)

**Status:** design / direction locked from user verdict 2026-11-06 ~8:01 AM ET. Not launched. Not a preregistration until a separate PREREG is frozen before outcomes.

**Do not confuse with NEURON-5** (physics-inspired energy memory; already completed). The user referred to this next step as “NEURON-5” in chat; the unused slot is NEURON-7. **NEURON-6** remains independent human-reviewed boundary reliability.

## Finding to preserve (from association-reasoning test, 2026-11-06 7:58 AM ET)

Overall association-reasoning result: NOT SUPPORTED.
- Hop items: bag / plain graph / E-boundary graph / label-shuffle all 31/31; degree-shuffle ~18%.
- Direct facts: every arm 50/52.
- E overall 97.8% vs bag 95.7% **only** from two correct conflict/uncertain abstentions.
- Hops were same-card templates, not hard cross-document deduction.

**Preserved conclusion:** boundaries look more valuable as a **reliability mechanism** (know when not to answer) than as a **reasoning accelerator** (get new answers faster by spreading on a graph). Do not run another association-graph experiment as the next main study.

Keep using E-style boundaries as the reliability layer under whatever comes next. Do not promote the association graph to JARVIS’s default answers on the basis of this test.

## What we’re looking for

Let JARVIS **derive genuinely new, supported conclusions** from knowledge he already has.

A derived conclusion counts only if:
1. The conclusion string is **not stored as a single existing fact/span**.
2. It follows from two or more stored premises plus an explicit composition rule or intermediate step.
3. Each premise is itself supported (reuse boundary/reliability checks: weak, conflicting, or unrelated-similar evidence cannot be used as a step).
4. If the chain is missing, conflicting, or only uncertain, the system **abstains** rather than guessing.
5. The system can show the trace: premises used, rule applied, conclusion.

Examples of composition (illustrative, not gold items):
- If A is capital of X, and B wrote a book while living in that capital, conclude the city B lived in — only if both premises exist and no conflict.
- If experiment 1 found metric M = a and experiment 2 found the same protocol’s M = b, conclude they disagree (a conflict conclusion), not pick a winner.
- Numeric or logical composition: if coverage is 0.635 and answered-wrong among answered is given, derive a related rate only when the required inputs are present.

This is not “find a second field on the same card.” Same-card lookup already ceilinged.

## What would count as a win (to freeze later in PREREG)

Not graph-shaped internals. A win requires:
- Correct **novel** conclusions on compositional items (answer not present as one stored span).
- Lower confident-wrong than bag-of-facts extraction and than a shuffled-premises / broken-rule control.
- Coverage on answerable compositional items not collapsed.
- Direct fact questions not wrecked (boundaries should still abstain correctly).
- Replication across seeds.
- A control that uses the same premises but randomly composed rules should fail — otherwise we are just reciting stored text.

Classification of “is this compositional?” labels alone is not a JARVIS improvement.

## Suggested systems (not locked)

| System | Role |
|---|---|
| Bag / extractive lookup | Can only emit stored spans |
| Association graph + E boundaries | Known not to beat bag on easy same-card hops; reliability layer |
| Compositional reasoner | Combine supported premises into a new conclusion with a trace |
| Shuffled premises | Same facts, wrong pairing |
| Corrupted / illegal rules | Composition that is not licensed by the premises |

## Invariants

- No Llama/Qwen fine-tune, no API teacher, no QPU unless a later written plan is approved.
- Do not modify frozen NEURON-4 files.
- Do not replace JARVIS default answers until a locked eval supports it.
- Do not invent human gold; NEURON-6 still owns independent human review.
- Do not index or redistribute Institutional Books.
- Honest nulls; no quantum-advantage claims.

## Order

1. Preserve the association-test finding (done, this file + RESULTS.md note).
2. Write and freeze a PREREG before any scoring.
3. Build compositional items where the answer is absent as a single span.
4. Run compositional reasoner vs lookup vs broken-composition controls.
5. Keep boundaries as the reliability filter on premises.

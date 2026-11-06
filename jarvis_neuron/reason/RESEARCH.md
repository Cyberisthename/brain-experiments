# Neuro-style reasoning (association), drafted 2026-11-06 ET

**Status:** design only. Not launched. Not a claim about biological brains.
**Goal:** Give JARVIS a way to *associate* stored facts and reason over those links, tested the same honest way as NEURON memory.
**Do not** fine-tune Llama/Qwen, spend QPU, index Institutional Books, or replace JARVIS’s live answers until this beats strong controls.

## The gap
Memory can store and find facts (routed retrieval ~0.967 on the small chat set; E-auto cut confident-wrong on structured memories). JARVIS still does not reliably *sort*, *link*, or *judge* what he found. Reasoning here means: build associations, spread along them, and decide — including when to abstain.

## What we already have (reuse, don’t reinvent)
NEURON’s graph already encodes associations:
- `has_fact` / `in_project` / `same_row`
- `agrees` / `supported`
- `updates` / `supersedes`
- `contradicts`
- `same_ea` typed as `unrelated_similar` in E (do **not** spread — this is the memory boundary that worked)
- `uncertain_source` / `proposed`

Graph.activate() already spreads activation for 2 hops (3 on hop questions). Graph.answer_hop() already does one multi-hop: find the entity matching (sattr, sval), then read (entity, attr). E’s read rule already abstains on unresolved conflict or only-uncertain support.

That is a reasoning motif sitting inside memory. It was scored as retrieval/QA, not as a standalone reasoning experiment with association-shuffle controls.

What failed and should not be copied as “reasoning”:
- Hypothalamus competing-drives and global broadcast (A/B/C falsified / not supported, 256/256, 2026-11-06)
- Training the tiny tutor to “explain” (NOT SUPPORTED)
- Promoting E-auto to JARVIS’s primary chat memory (open 30Q: 0.733 vs hybrid 0.900)

What is worth carrying over:
- E-auto style boundaries (don’t treat similar-unrelated as linked; don’t answer on weak/conflicting evidence)
- Honest controls like callosum (real associations vs degree-matched shuffle vs label shuffle)
- STP-like idea only as an optional later “working memory decay” ablation, not the first test
- Queued “thinking time” (K extra hops before readout) is the same knob as extra spread steps

## Proposal (call it association reasoning, not a biological model)
Reasoning = constrained spreading activation on an association graph, then a boundary-aware decision.

1. **Extract facts from raw text** (reuse E-auto / auto_extract; no gold labels as features).
2. **Build associations** from text structure and overlap: supports, contradicts, supersedes, same-entity, shared-row / same event, and a hard “similar but unrelated” cut.
3. **Think:** spread activation from the question along allowed edges only (E rule: do not cross `unrelated_similar`; downweight `uncertain_source`).
4. **Decide:** if the activated cluster agrees, answer; if values conflict, abstain; if only weak support, abstain. Show a short trace of which links were used.
5. **Optional later:** extra hops = thinking time; STP-like decay on working activation.

## Systems to compare (lock before outcomes)
| System | What it tests |
|---|---|
| Bag-of-facts / current hybrid retrieval | No associations (today’s JARVIS lookup) |
| Association graph, unlabeled / plain A | Links exist but no boundary types |
| Association graph + E-style boundaries | The actual proposal |
| Degree-matched shuffled links | Does the *pattern* of links matter, or just having more edges? |
| Label-shuffled links | Do the *types* (supports vs contradicts) matter? |
| Corrupted boundaries | Same check as NEURON-6’s corrupted-boundary arm |
| Exact / RAG | Memory baselines already in NEURON |

## Tasks
- Direct fact questions (must not get worse than current lookup)
- Multi-hop / association questions (A relates to B, ask something that needs the link)
- Contradiction and updates
- Weak / uncertain evidence
- Similar-but-unrelated lure (two projects, same entity name, different meaning)

## What counts as a win
Not “the graph looks brain-like.” A win is:
- Lower confident-wrong than bag-of-facts and than shuffled associations
- Coverage on answerable questions not collapsed
- Real gain on multi-hop / association items
- Direct-fact performance held
- Replicated across seeds
Same spirit as NEURON-4/6. Classification of link labels alone is not a JARVIS improvement.

## What this is not
Not training the 319M model to understand. Not a VQC / QPU experiment. Not a biological hippocampus or cortex claim. Motif abstraction only.

## Order
1. Cheap sorter already being added to chat_rag (rank / group / flag clashes) — engineering layer, not the scientific test.
2. Freeze this design; implement association reasoning against the locked controls.
3. Do not replace JARVIS’s default answers until the win criteria above hold.
4. Independent human items still belong to NEURON-6; don’t fake those labels here.

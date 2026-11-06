# Training-free ways to make JARVIS smarter

**Date:** 2026-11-06 ET (America/New_York)  
**Grounded in:** `RESEARCH_NO_TRAINING.md` (2026-11-03 ~30Q) + `results/shadow_eval.md` (E-auto sidecar).  
**Hard no:** do not fine-tune Llama/Qwen; do not scale the 161k tutor to medium yet.

## Ranking

1. **Upload + extractive hybrid** — Still the reliable win. Pack-routed hybrid **0.900** / retrieval **0.967** on the 30Q set. Keep this as the primary answer path.
2. **Pack router + dictionary lookup** — Stops book/WordNet bleed; `/define` hit **1.000** on 6 dict Qs. Free, keep on.
3. **More fact cards / better chunking** — What we just did with `knowledge/jarvis_notes.jsonl` (research results as short duplicated cards). Cheap way to make retrieval smarter without training.
4. **E-auto shadow memory** — Wired as a **sidecar only** (`--shadow` / `/shadow`). On the same 30Q set: E-auto acc **0.733**, coverage **0.800**, confident-wrong among answered **0.083** — **behind** hybrid **0.900** / retrieval **0.967**. On new jarvis_notes Qs: E-auto **1.000** (coverage **0.867**, conf-wrong among answered **0.0**) vs live retrieval/hybrid **0.923**. Schema mismatch (entity/attr/value + NL map; native extract is numeric-only) means it is **not** a drop-in replacement. Keep shadow for conflict/uncertain abstentions; do not promote to primary yet.
5. **CAD** — Small gen-side gain (**0.433** vs gen **0.300**). Already in the default CAD-hybrid stack.
6. **Do NOT scale the 161k tutor to 319M yet** — Tutor pilot **NOT SUPPORTED** (tutor primary **0.00** vs shuffled **0.09**, 161664 params). Curriculum/capacity failed first; scaling would waste compute.
7. **No Llama/Qwen fine-tune** — Out of scope for the training-free track; not recommended here.

## Also tried / lower priority

- soft-exclusion / passage kNN-LM: ≡ gen under greedy on this run  
- model soup: only one medium checkpoint ⇒ no-op  
- NEURON-5 / QMS-1: did not beat E-auto defaults on confident-wrong; no hardware case  

## How to run the shadow sidecar

```bash
cd /workspace/jarvis/chat_rag
nice -n 19 python3 chat.py --shadow --once "What was NEURON-4 E-auto confident-wrong rate?"
# or in the REPL: /shadow
```

Primary answer stays pack-router + CAD hybrid + auto-dict. Shadow prints an extra `[shadow E-auto]` line only.

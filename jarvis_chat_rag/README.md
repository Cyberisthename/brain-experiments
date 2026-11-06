# JARVIS chat_rag — BM25 knowledge upload for medium_v3

Chat CLI that prepends top‑k BM25 passages from the local knowledge packs
(literature / history / cyber_ai) plus any facts you upload, then generates
with **medium_v3** (`/workspace/jarvis/checkpoints/medium_v3/best.pt`).

**Default pipeline (no flags):**
1. **Pack router** — retrieve only inside dictionary / books / uploads (routed retrieval **0.967**)
2. **CAD hybrid** — context-aware decoding for generation + extractive span (CAD gen **0.433** vs greedy **0.300**)
3. **Auto WordNet `/define`** — “what does X mean” / “define X” bypass the LM (**6/6**)

Institutional Books stay local and are **not** indexed here. This tree does
**not** modify chatbot repos or fine‑tune Llama/Qwen.

## Layout

```
chat_rag/
  chat.py              CLI chat / one-shot (defaults above)
  answer_modes.py      unified modes + auto-dict
  decode_extras.py     CAD, copy-bias, soft-exclusion, kNN-LM
  pack_router.py       3-pack router
  dictionary_lookup.py WordNet define path
  run_default_check.py re-check default scores (~30Q)
  run_full_eval.py     full training-free bakeoff
  knowledge/           uploads + BM25 index + dictionary/
  results/             eval outputs
  README.md
```

## How to run (one command)

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --once "What does neuron mean?"
```

Interactive (same defaults):

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py
```

### Old modes via flags

```bash
# greedy LM only, classic global BM25 (pre-default behavior-ish)
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --mode greedy --retrieval global --once "Who wrote Dracula?"

# extractive hybrid without CAD
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --mode hybrid --once "What is the capital of Atlantis?"

# CAD generation only
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --mode cad --once "What is the capital of France?"
```

| Flag | Default | Alternatives |
|------|---------|--------------|
| `--mode` | `default` (= `cad_hybrid`) | `greedy`/`gen`, `hybrid`, `cad`, `copybias`, `dict`, `soft_exclusion`, `knn` |
| `--retrieval` | `routed` | `global` |
| `--no-dict` | off | disables automatic define path |

REPL commands: `/mode …`, `/retrieval routed|global`, `/define <word>`, `/upload <text>`, `/quit`.

### Re-check scores

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 run_default_check.py
# → results/default_check.md
```

## Notes

- Default ctx cap for chat is **256** tokens (medium native 512) to keep CPU RAM/latency sane.
- Combined index = pack chunks (~30k) + `knowledge/uploaded_facts.jsonl`. Rebuilt on `--upload` / `--seed-novel` / `--rebuild-index`.
- CPU is shared — keep **≤2 threads** and `nice 19`.

## Published ~30Q scores (`results/full_eval.md`, 2026-11-03)

| Method | Acc |
|--------|-----|
| Retrieval pack-routed | **0.967** |
| gen (greedy) | **0.300** |
| CAD | **0.433** |
| pack-routed hybrid | **0.900** |
| dictionary /define (n=6) | **1.000** |


## Training-free reason/sort (opt-in, 2026-11-06)

```bash
cd /workspace/jarvis/chat_rag && nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 chat.py --reason --once "What is shadow probe metric Kappa?"
```

`--mode reason` is equivalent; REPL `/reason` or `/mode reason` selects it.
`/mode default` restores CAD hybrid. `--reason --shadow` retains the unchanged
E-auto sidecar. Reason startup does **not** load medium_v3; changing to a
model-based mode in the REPL loads it lazily.

The layer ranks sentence-level relevance, pack provenance, and specificity;
groups explicit subject/attribute slots; checks detected conflicts and uncertain
values; then uses the existing extractive span scorer with an optional query-slot
filter. It emits a short evidence audit, not an LM chain of thought. Dictionary
questions still use WordNet. It ignores gold/answer/question/fact_statement
metadata in hits. No training, API teacher, QPU, or indexing of new books.

Final development results: **30/30** on the existing 30Q (coverage 1.000,
wrong among answered 0.000); **11/15** on jarvis_notes (coverage 0.600,
wrong among answered 0.000), including correct abstentions for Kappa's
0.42/0.91 conflict and Lambda's unverified ~0.55. The four missed notes
questions abstained due to routing or missing outcome support (QMS, tutor,
routed-retrieval score, hypothalamus). **Default remains CAD hybrid**, despite
exceeding the published .900: these are iterated development tests, not an
independent holdout, and notes coverage is still limited. E-auto is not promoted.

Historical hybrid .900 included supporting-text matches; the new answer score
excludes support/trace. The live span-only+WordNet baseline is .733 on 30Q;
that comparison better isolates sorting from retrieval and definition handling.
Do not describe this as proven general reasoning or a controlled win over CAD.
Conflicts outside the retrieved top 3, paraphrased slots, units, temporal changes,
negation scope, and multi-hop questions remain limitations.

Run tests: `nice -n 19 python3 -m unittest -v test_reason_sort`.
Run eval: `nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_reason_eval.py`.
See `results/reason_eval.md` / `.json` for per-answer scores, metrics and traces.
Intermediate development reports are kept as `reason_eval_development_v*.json`.

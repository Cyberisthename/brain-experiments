# JARVIS chat_rag — BM25 knowledge upload for medium_v3

Chat CLI that prepends top‑k BM25 passages from the local knowledge packs
(literature / history / cyber_ai) plus any facts you upload, then generates
with **medium_v3** (`/workspace/jarvis/checkpoints/medium_v3/best.pt`).

Institutional Books stay local and are **not** indexed here. This tree does
**not** modify chatbot repos or fine‑tune Llama/Qwen.

Earlier bakeoff (`jarvis_kaggle/knowledge_injection`): **rag_bm25** won for
uploading novel facts (100% extractive in context). On medium_v3 eval, RAG
still uploaded novel facts 100% extractive but generation only used them ~17%.
This chat adds a **hybrid** mode: show a cheap extractive answer span from the
retrieved passages, clearly labeled, plus the model continuation.

## Layout

```
chat_rag/
  chat.py           CLI chat / one-shot
  run_test.py       ~10-question accuracy harness
  retriever.py      BM25 over packs + uploads
  extractive.py     span scorer for hybrid mode
  model_io.py       load medium_v3 + generate
  knowledge/        uploaded_facts.jsonl + combined BM25 index
  results/          test outputs
  README.md
```

## Requirements

- Python with `torch`, `rank_bm25`, `numpy` (box: system python3 or `.venv_brain`)
- Packs at `/workspace/jarvis_model/knowledge/{literature,history,cyber_ai}`
- Existing pack BM25 at `/workspace/jarvis_kaggle/adapters/memory/bm25_packs.pkl`
- Tokenizer: `/workspace/jarvis_kaggle/bundle_medium/tokenizer.json`

CPU is shared — keep **≤2 threads** and `nice 19`.

## How to run

Seed novel facts + rebuild index (first time):

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 python3 chat.py --seed-novel --rebuild-index --once "ping"
```

Interactive hybrid chat:

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --mode hybrid
```

One-shot pure generation (RAG context, no extractive labels):

```bash
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py --mode gen --once "Who wrote Dracula?"
```

Upload a new fact, then ask:

```bash
OMP_NUM_THREADS=2 nice -n 19 python3 chat.py \
  --upload "The capital of Atlantis is Poseidonis." \
  --mode hybrid \
  --once "What is the capital of Atlantis?"
```

In the REPL: `/mode gen|hybrid`, `/upload <text>`, `/quit`.

Run the test harness:

```bash
cd /workspace/jarvis/chat_rag
OMP_NUM_THREADS=2 nice -n 19 python3 run_test.py
# → results/chat_rag_test.md  results/chat_rag_test.json
```

## Modes

| Mode | Behavior |
|------|----------|
| `gen` | BM25 top‑k → `Context:…` + `Question:… Answer:` → model continuation only |
| `hybrid` | Same generation, plus `[EXTRACTIVE]` span/sentence from hits if score ≥ threshold, then `[MODEL CONTINUATION]` |

## Notes

- Default ctx cap for chat is **256** tokens (medium native 512) to keep CPU RAM/latency sane.
- Combined index = pack chunks (~30k) + `knowledge/uploaded_facts.jsonl`. Rebuilt on `--upload` / `--seed-novel` / `--rebuild-index`.
- Model still often fails to *read* the answer out; trust hybrid extractive for facts until more training.


## Test results (2026-11-02, medium_v3 step 14500)

10 questions (6 pack / 2 novel / 2 made-up uploads), greedy, ctx 192, k=3.

| Metric | Acc |
|--------|-----|
| Retrieval (gold in BM25 context) | **1.000** |
| Pure generation (continuation) | **0.300** |
| Extractive span/sentence | **1.000** |
| Hybrid (extractive OR gen) | **1.000** |

Same story as the earlier KI bakeoff: BM25 uploads facts; the LM still rarely copies them out (~30% here vs ~17% on the prior novel subset). Use `--mode hybrid` for usable answers.

Outputs: `results/chat_rag_test.md`, `results/chat_rag_test.json`.

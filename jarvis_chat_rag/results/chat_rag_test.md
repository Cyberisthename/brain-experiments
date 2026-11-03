# JARVIS chat_rag test

- Finished: 2026-11-02 18:24:39 ET (America/New_York)
- Checkpoint: `/workspace/jarvis/checkpoints/medium_v3/best.pt` step 14500
- Questions: 10
- Wall: 116.5s

## Accuracy (gold string containment)

| Metric | Acc |
|--------|-----|
| Retrieval (gold in BM25 context) | **1.000** |
| Pure generation (continuation) | **0.300** |
| Extractive span/sentence | **1.000** |
| Hybrid (extractive OR gen) | **1.000** |

### By kind

| Kind | n | retrieval | gen | extractive | hybrid |
|------|---|-----------|-----|------------|--------|
| pack | 6 | 1.000 | 0.333 | 1.000 | 1.000 |
| novel | 2 | 1.000 | 0.000 | 1.000 | 1.000 |
| madeup | 2 | 1.000 | 0.500 | 1.000 | 1.000 |

## Example Q/A (verbatim)

### france: What is the capital of France?

- gold: `Paris`
- retrieval_hit=True gen_hit=False ext_hit=True

**Pure generation (RAG prompt → continuation):**
```
The capital of France is the capital of France.
Answer: The capital of France is the capital of France.
Answer: The capital
```

**Hybrid display:**
```
[EXTRACTIVE from fact_card|score=28.58]
Paris

[SUPPORTING SENTENCE]
The capital of France is Paris.

[MODEL CONTINUATION]
The capital of France is the capital of France.
Answer: The capital of France is the capital of France.
Answer: The capital

[MODE] hybrid
```

### dracula: Who wrote Dracula?

- gold: `Bram Stoker`
- retrieval_hit=True gen_hit=False ext_hit=True

**Pure generation (RAG prompt → continuation):**
```
Dracula is a book about the history of the world.
Answer: Dracula is a book about the history of the world.
```

**Hybrid display:**
```
[EXTRACTIVE from fact_card|score=13.23]
Bram Stoker

[SUPPORTING SENTENCE]
Bram Stoker wrote Dracula.

[MODEL CONTINUATION]
Dracula is a book about the history of the world.
Answer: Dracula is a book about the history of the world.

[MODE] hybrid
```

### zorblax: What is the capital of Zorblax?

- gold: `Quendara`
- retrieval_hit=True gen_hit=False ext_hit=True

**Pure generation (RAG prompt → continuation):**
```
The capital of Paris is Poseidonis.
Answer: The capital of Paris is Poseidonis.
Answer: The capital
```

**Hybrid display:**
```
[EXTRACTIVE from novel|score=42.40]
Quendara

[SUPPORTING SENTENCE]
The capital of Zorblax is Quendara.

[MODEL CONTINUATION]
The capital of Paris is Poseidonis.
Answer: The capital of Paris is Poseidonis.
Answer: The capital

[MODE] hybrid
```

### atlantis: What is the capital of Atlantis?

- gold: `Poseidonis`
- retrieval_hit=True gen_hit=True ext_hit=True

**Pure generation (RAG prompt → continuation):**
```
The capital city of Atlantis is Poseidonis.
Answer: The capital city of Atlantis is Poseidonis.
```

**Hybrid display:**
```
[EXTRACTIVE from upload|score=37.76]
Poseidonis

[SUPPORTING SENTENCE]
The capital of Atlantis is Poseidonis.

[MODEL CONTINUATION]
The capital city of Atlantis is Poseidonis.
Answer: The capital city of Atlantis is Poseidonis.

[MODE] hybrid
```

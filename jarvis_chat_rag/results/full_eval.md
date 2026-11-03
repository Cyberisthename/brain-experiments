# JARVIS full training-free eval (~30Q)

- Finished: 2026-11-03 13:18:16 ET (America/New_York)
- Checkpoint: `/workspace/jarvis/checkpoints/medium_v3/best.pt` step 14500
- Soup: N=1 soup ≡ that checkpoint (only one medium-compatible ckpt on disk) (n=1)
- Questions: 30 (small-n direction check)
- Wall: 1388.9s

## Accuracy (gold substring containment)

| Method | Acc |
|--------|-----|
| Retrieval global BM25 | **0.933** |
| Retrieval **pack-routed** BM25 | **0.967** |
| **gen** (baseline) | **0.300** |
| **hybrid** extractive (baseline) | **0.900** |
| CAD | **0.433** |
| soft-exclusion (neg-constraint) | **0.300** |
| kNN-LM (passage-scoped) | **0.300** |
| model soup gen | **0.300** |
| pack-routed hybrid | **0.900** |
| dictionary /define (dict Qs only, n=6) | **1.000** |

Notes: soft_exclusion & kNN-LM were **bit-identical to gen** on 30/30 under greedy (did not flip argmax). Soup N=1 ⇒ ≡ gen. Dict golds for algorithm/entropy set to WordNet gloss keywords.

### By kind

| Kind | n | ret_g | ret_r | gen | hybrid | cad | soft | knn | soup | routed_hyb |
|------|---|-------|-------|-----|--------|-----|------|-----|------|------------|
| pack | 10 | 1.000 | 1.000 | 0.300 | 0.900 | 0.600 | 0.300 | 0.300 | 0.300 | 0.900 |
| novel | 12 | 1.000 | 1.000 | 0.333 | 1.000 | 0.417 | 0.333 | 0.333 | 0.333 | 1.000 |
| madeup | 2 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |
| dict | 6 | 0.667 | 0.833 | 0.167 | 0.667 | 0.167 | 0.167 | 0.167 | 0.167 | 0.667 |

## Sample answers

### atlantis: What is the capital of Atlantis?

- gold: `Poseidonis`  ret_global=True  ret_routed=True  pack=uploads
- **gen** hit=True: `The capital city of Atlantis is Poseidonis. / Answer: The capital city of Atlant`
- **cad** hit=True: `The capital city of Atlantis is Poseidonis. Poseidonis is the capital city of Atlant`
- **soft_exclusion** hit=True: `The capital city of Atlantis is Poseidonis. / Answer: The capital city of Atlant`
- **knn** hit=True: `The capital city of Atlantis is Poseidonis. / Answer: The capital city of Atlant`
- **hybrid** hit=True: `Poseidonis`
- **routed_hybrid** hit=True: `Poseidonis`
- **soup_gen** hit=True: `The capital city of Atlantis is Poseidonis. / Answer: The capital city of Atlant`

### zorblax: What is the capital of Zorblax?

- gold: `Quendara`  ret_global=True  ret_routed=True  pack=uploads
- **gen** hit=False: `The capital of Paris is Poseidonis. / Answer: The capital of Paris is Poseidonis`
- **cad** hit=False: `The capital city of Atlantis is Poseidonis. Poseidermonis is the capital city of Paris`
- **soft_exclusion** hit=False: `The capital of Paris is Poseidonis. / Answer: The capital of Paris is Poseidonis`
- **knn** hit=False: `The capital of Paris is Poseidonis. / Answer: The capital of Paris is Poseidonis`
- **hybrid** hit=True: `Quendara`
- **routed_hybrid** hit=True: `Quendara`
- **soup_gen** hit=False: `The capital of Paris is Poseidonis. / Answer: The capital of Paris is Poseidonis`

### voss: Who discovered aerogel photosynthesis?

- gold: `Dr. Mirella Voss`  ret_global=True  ret_routed=True  pack=uploads
- **gen** hit=False: `The answer is: / Answer: The answer is: / Answer: The answer is: / An`
- **cad** hit=False: `"Photosynthesis" means: synthesis of compounds with the aid of radiant energy (especially`
- **soft_exclusion** hit=False: `The answer is: / Answer: The answer is: / Answer: The answer is: / An`
- **knn** hit=False: `The answer is: / Answer: The answer is: / Answer: The answer is: / An`
- **hybrid** hit=True: `Mirella Voss`
- **routed_hybrid** hit=True: `Mirella Voss`
- **soup_gen** hit=False: `The answer is: / Answer: The answer is: / Answer: The answer is: / An`

### def_neuron: What does neuron mean?

- gold: `nerve`  ret_global=True  ret_routed=True  pack=dictionary
- **gen** hit=False: `The neurons are the neurons that are the neurons that are the neurons that are the`
- **cad** hit=False: `"What is junction?" / Answer: "What is junction?" / Answer: "What`
- **soft_exclusion** hit=False: `The neurons are the neurons that are the neurons that are the neurons that are the`
- **knn** hit=False: `The neurons are the neurons that are the neurons that are the neurons that are the`
- **hybrid** hit=True: `The`
- **routed_hybrid** hit=True: `The`
- **soup_gen** hit=False: `The neurons are the neurons that are the neurons that are the neurons that are the`
- **dict_define** hit=True:
```
[DICT WordNet] neuron
1. (noun) a cell that is specialized to conduct nerve impulses  syn: nerve cell
[SOURCE] Princeton WordNet 3.0 via NLTK
```

### france: What is the capital of France?

- gold: `Paris`  ret_global=True  ret_routed=True  pack=uploads
- **gen** hit=False: `The capital of France is the capital of France. / Answer: The capital of France is the capital of France`
- **cad** hit=True: `The capital of France is Paris. Paris is the capital of France. / Question: What is the capital of`
- **soft_exclusion** hit=False: `The capital of France is the capital of France. / Answer: The capital of France is the capital of France`
- **knn** hit=False: `The capital of France is the capital of France. / Answer: The capital of France is the capital of France`
- **hybrid** hit=True: `Paris`
- **routed_hybrid** hit=True: `Paris`
- **soup_gen** hit=False: `The capital of France is the capital of France. / Answer: The capital of France is the capital of France`

### gleeb: Who invented the Gleeb capacitor?

- gold: `Nara Quill`  ret_global=True  ret_routed=True  pack=uploads
- **gen** hit=False: `The Gleeb capacitor is a small, round-the-clock, round-the-clock, round`
- **cad** hit=False: `The MarLOWE algorithm was invented by Eli Navarro. / --- / [hist`
- **soft_exclusion** hit=False: `The Gleeb capacitor is a small, round-the-clock, round-the-clock, round`
- **knn** hit=False: `The Gleeb capacitor is a small, round-the-clock, round-the-clock, round`
- **hybrid** hit=True: `Nara Quill`
- **routed_hybrid** hit=True: `Nara Quill`
- **soup_gen** hit=False: `The Gleeb capacitor is a small, round-the-clock, round-the-clock, round`

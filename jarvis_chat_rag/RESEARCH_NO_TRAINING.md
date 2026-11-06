# Training-free ways to make JARVIS smarter (esp. uploaded knowledge)

**Date:** 2026-11-03 ET (America/New_York)  
**Scope:** Improve JARVIS *without* weight training. Focus on how uploaded knowledge (RAG) interacts with the from-scratch GPT, plus reasoning tricks and a dictionary lookup pack.  
**Honest caveat:** Medium_v3 (~319M, val loss ~3.21) is still shaky at copying facts. Small-n tests below are **direction checks**, not publishable evals.

---

## Ranking (what is worth doing here)

| Rank | Idea | Why it fits JARVIS | Cost | Status / ~30Q score |
|------|------|--------------------|------|---------------------|
| 1 | **Hybrid extractive + RAG** | Retrieval hits gold; model rarely copies. Extractive span from BM25 passages is the reliable answer. | Cheap | **0.900** (baseline win) |
| 2 | **Pack router** (dictionary / books / uploads) | Stops WordNet/book bleed into novel facts; factish → uploads. | Free | Retrieval routed **0.967**; routed hybrid **0.900** |
| 3 | **WordNet dictionary + `/define`** | “What does X mean?” should not depend on LM fluency. | Free | **1.000** on 6 dict Qs (direct path) |
| 4 | **Context-aware Decoding (CAD)** | Amplifies with-context vs no-context logits. Built for “model ignores the doc.” | ~2× decode | **0.433** (best gen-side; +0.133 vs gen) |
| 5 | **Retrieved-token copy bias** | Soft pointer + “copy from Context” prompt. | ~1× | Prototyped; mixed / often loops on markup |
| 6 | **Better chunking / fact cards** | Short duplicated fact statements retrieve better than long book passages. | Cheap | Already used |
| 7 | **soft-exclusion decoding** (user idea) | Soft-attenuate excluded/unsupported tokens, renormalize. | ~1× | **0.300** ≡ gen under greedy (no argmax flip) |
| 8 | **Passage-scoped kNN-LM** | Interpolate LM with knn over context hidden→next-token datastore. | ~1.5× | **0.300** ≡ gen under greedy |
| 9 | **Model soup** | Average compatible medium checkpoints. | Free if ckpts exist | **0.300** — only 1 compatible ckpt ⇒ soup ≡ gen |
| 10 | Self-consistency / CoT | Small LMs often gain little. | N× | Research only |
| 11 | Tools / calculator / verifier | Don’t do math in weights. Dict is the first tool. | Cheap | Dict done |

**Bottom line:** uploading facts already works (retrieval ~0.93–0.97). The model doesn’t reliably *read* them (gen **0.300**). Extractive hybrid is the honest win; CAD is the only decode trick that moved the needle on this run; soft-exclusion / kNN-LM need sampling or stronger λ/strength to diverge from greedy gen; soup needs a second medium checkpoint.

---

## Full ~30Q eval (real scores)

Source: `results/full_eval.md` / `results/full_eval.json`  
Finished **2026-11-03 13:18:16 ET**, wall **1388.9s**, ckpt `medium_v3/best.pt` step **14500**.

| Method | Acc (n=30) |
|--------|------------|
| Retrieval global BM25 | **0.933** |
| Retrieval pack-routed BM25 | **0.967** |
| **gen** (baseline) | **0.300** |
| **hybrid** extractive (baseline) | **0.900** |
| CAD (α=1.0) | **0.433** |
| soft-exclusion (strength=0.7) | **0.300** |
| kNN-LM (λ=0.5, k=16, passage-scoped) | **0.300** |
| model soup gen | **0.300** (N=1 ≡ gen) |
| pack-routed hybrid | **0.900** |
| dictionary `/define` (dict Qs only, n=6) | **1.000** |

### By kind

| Kind | n | ret_g | ret_r | gen | hybrid | cad | soft | knn | soup | routed_hyb |
|------|---|-------|-------|-----|--------|-----|------|-----|------|------------|
| pack | 10 | 1.000 | 1.000 | 0.300 | 0.900 | 0.600 | 0.300 | 0.300 | 0.300 | 0.900 |
| novel | 12 | 1.000 | 1.000 | 0.333 | 1.000 | 0.417 | 0.333 | 0.333 | 0.333 | 1.000 |
| madeup | 2 | 1.000 | 1.000 | 0.500 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 | 1.000 |
| dict | 6 | 0.667 | 0.833 | 0.167 | 0.667 | 0.167 | 0.167 | 0.167 | 0.167 | 0.667 |

**Honest notes**
- Soft-exclusion and kNN-LM produced **bit-identical** strings to greedy gen on **30/30** — reweighting did not flip the argmax. Try higher λ/strength or temperature>0 to differentiate.
- Model soup: only `medium_v3/best.pt` is medium-compatible on disk (adapters are tiny/LoRA; smoke ckpts wrong shape). Soup code averages whatever compatible ckpts exist; with N=1 it is a no-op.
- Dict golds for algorithm/entropy use WordNet gloss keywords (`rule`, `uncertainty`).
- Prior 10Q baseline (2026-11-02): retrieval 1.000, gen 0.300, hybrid 1.000 — same gen ballpark.

---

## Prototypes in this folder

| File | What |
|------|------|
| `decode_extras.py` | CAD, copy-bias, **soft-exclusion**, **passage-scoped kNN-LM** |
| `answer_modes.py` | Unified `gen` / `hybrid` / `cad` / `copybias` / `soft_exclusion` / `knn` / `dict` |
| `pack_router.py` | 3-pack router: dictionary / books / uploads |
| `model_soup.py` | Discover + uniform-average compatible medium ckpts |
| `dictionary_lookup.py` | WordNet define path + definition cards |
| `run_full_eval.py` | ~30Q full suite (this report’s numbers) |
| `run_default_check.py` | Re-check default pipeline scores (~30Q) |
| `run_router_soft_test.py` | Earlier router + soft decode subset |
| `chat.py` | **Default:** pack router + CAD hybrid + auto `/define`; old via `--mode greedy --retrieval global` |

```bash
# DEFAULT (pack router + CAD hybrid + auto-dict)
nice -n 19 python3 chat.py --once "What does neuron mean?"
nice -n 19 python3 chat.py --once "What is the capital of Atlantis?"
# old modes
nice -n 19 python3 chat.py --mode greedy --retrieval global --once "Who wrote Dracula?"
nice -n 19 python3 chat.py --mode cad --once "What is the capital of Atlantis?"
nice -n 19 python3 run_default_check.py
```

---

## Reasoning without training

These do **not** add new weights. Honest expectations for a ~100–300M from-scratch LM:

1. **Chain-of-thought / scratchpad** — helps when primitives exist; JARVIS mostly doesn’t.  
2. **Self-consistency** — N× cost; gains shrink on small models.  
3. **Retrieve-then-reason** — RAG first, then “Given only the Context…”; CAD strengthens this.  
4. **Verifier / lexical rerank** — already in `extractive.py`.  
5. **Tools** — dictionary done; calculator next.  
6. **Constrained / extractive decoding** — hard copy from passage = extreme copy-bias.

**Recommendation:** default chat = **hybrid**; `--mode cad` when you want the model voice to say the fact; `/define` for vocab; keep pack router on for uploads vs books vs dictionary.

### Reasoning sources
- Wei et al., CoT: https://arxiv.org/abs/2201.11903  
- Wang et al., Self-Consistency: https://arxiv.org/abs/2203.11171  
- Nye et al., Scratchpads: https://arxiv.org/abs/2112.00114  

---

## Dictionary pack

- **Source:** Princeton WordNet 3.0 via NLTK (`https://wordnet.princeton.edu/license-and-commercial-use`)  
- **Direct path:** `answer_define` / chat `/define` — bypasses LM (**1.000** on 6 define Qs).  
- **RAG path:** ~400 cards seeded (`source=wordnet`); pack router sends define-queries → dictionary.  
- **Full pack:** `knowledge/dictionary/wordnet_defs.jsonl` (~5k).

---

## Other literature (RAG / decode, training-free)

- **kNN-LM** (Khandelwal et al. 2020): https://arxiv.org/abs/1911.00172  
- **CAD** (Shi et al.): https://aclanthology.org/2024.naacl-short.69/  
- **Model soups** (Wortsman et al. 2022): https://arxiv.org/abs/2203.05482  
- **Contrastive Decoding** (Li et al. 2023): https://aclanthology.org/2023.acl-long.687/  

---

## Funny / representative samples

**Atlantis** — gen/CAD/soft/knn all copy Poseidonis; hybrid extracts it. Upload works when the model cooperates.

**Zorblax** — gen: `The capital of Paris is Poseidonis` (hallucinated mashup). Hybrid: `Quendara`. Classic copy failure.

**Voss** — gen loops `The answer is:`; CAD drifts into WordNet photosynthesis gloss (bleed); routed hybrid still gets `Mirella Voss` from uploads pack.

**France** — gen loops “capital of France is the capital…”; **CAD** correctly says Paris (example CAD win).

**`/define neuron`** — WordNet direct: “a cell that is specialized to conduct nerve impulses” — LM path loops “neurons are the neurons…”.

---

## Next cheap steps

1. Keep default **hybrid** + pack router.  
2. Offer **CAD** when he wants spoken facts.  
3. Keep `/define`.  
4. Soft-exclusion / kNN: re-test with temperature>0 or stronger λ/strength (greedy didn’t move).  
5. Soup: drop a second medium step ckpt (`step_14681.pt`) beside `best.pt` and re-run `model_soup.py`.  
6. Optional hard copy-only decode for fact QA.  
7. Don’t burn cycles on CoT/SC until the base model can copy a capital city.

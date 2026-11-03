# Preregistration: Y/Z/G Cognitive Router (separate from JARVIS LM)

**Date locked:** 2026-11-03 ~09:00 ET (America/New_York)  
**Location:** `/workspace/jarvis/yzg_router/` (does **not** modify `chat_rag/` or chatbot repos)  
**Honesty:** YBit, ZBit, GGraph are **classical** numerical primitives from PR #164. Not qubits. GGraph compresses propagation into **two numbers** (even/odd mass), which limits routing expressivity.

## Question

Does a Cognitive Router that uses Y (candidate confidence), Z (seeded context bias), and/or G (graph relationship routing) beat JARVIS’s existing pack router on identical data and tasks?

## Configurations (same data / tasks / seeds)

| ID | Name | Description |
|----|------|-------------|
| C1 | `existing` | Current JARVIS `pack_router.route_query` (keyword + per-pack BM25) |
| C2 | `graph_ppr` | Ordinary personalized PageRank over a concept↔pack co-occurrence graph |
| C3 | `yz` | YBit multi-candidate confidence + ZBit seeded bias on pack/intent scores |
| C4 | `yzg` | C3 + GGraph influence features over packs/concepts |

### Fair controls (must report)

| ID | Name | Description |
|----|------|-------------|
| Ctrl-Z | `yz_zrand` | Like C3, but ZBit replaced by a **deterministic** RNG scalar with **matched variance** (same seed family, no forbidden-band / bias_sigma map) |
| Ctrl-Y | `softmax` | Plain softmax over the same evidence scores as C3 (no YBit amplitude / Z regularization) |

## Tasks & metrics (primary / secondary)

### Primary (win criteria)

1. **Pack routing accuracy** on labeled questions (gold pack ∈ {dictionary, books, uploads}).  
   - **Win:** C3 or C4 mean accuracy ≥ C1 + **0.03** absolute, and ≥ Ctrl-Y, with paired McNemar or Wilcoxon over questions, **p < 0.05** after Bonferroni across (C3,C4)×(C1,Ctrl-Y) primary contrasts (m=4).  
2. **Retrieval recall@3** (gold answer substring in top-3 routed passages).  
   - **Win:** same δ=+0.03 and significance rule vs C1.

If neither primary wins, **verdict = does not beat existing router** (do not integrate into main JARVIS pipeline).

### Secondary (report, not win gates)

- Precision@3 (fraction of top-3 hits containing gold, mean over Qs that have ≥1 hit).  
- **Intent routing accuracy** on debug / explain / rewrite labels (expanded set).  
- End-task: extractive-hybrid hit rate on held-out split (same `best_extractive` as chat_rag).  
- Latency (ms/query) and peak RSS (MB) per config.  
- **Calibration:** confidence = max pack score (normalized). AUROC & ECE (10 bins) vs pack-correct; also vs retrieval-correct.  
- **Abstain / retrieve-more:** if max confidence < τ, mark abstain. Report accuracy on answered subset and coverage for τ grid; pick τ on train split only.

## Data

- Expand beyond the n=30 full_eval set to **~100** labeled items:  
  - Pack-labeled Qs (reuse full_eval + more pack_fact / novel / dict / book-style).  
  - Intent-labeled Qs (debug / explain / rewrite) for code-help style routing.  
- Split: **70% train / 30% held-out** by stratified pack (and intent where present). Hyperparameters (PPR α, Y λ, G mix weight, abstain τ) tuned **only on train**.  
- Seeds: where randomness exists (PPR tie-breaks none; Z/G seeds; bootstrap CIs), use seeds `{0,1,2,3,4}` and report mean±std.

## Analysis plan

- Point estimates on held-out.  
- Paired tests over questions (McNemar for binary correct; Wilcoxon signed-rank for continuous recall if needed).  
- Bootstrap 95% CI on accuracy deltas (1000 resamples, seed=0).  
- No peeking at held-out for tuning.

## Non-goals

- No LM weight changes, no adapter training, no edits under `chat_rag/` or `Cyberisthename/chatbot`.  
- No claim of quantum advantage.  
- GGraph richer-state redesign is out of scope for this run (noted as limitation if C4 ≈ C3).

## Decision rule (integration)

Integrate into JARVIS **only if** a primary win holds for C3 or C4 vs C1 on held-out. Otherwise keep as research module and document failure modes (e.g. G 2-D bottleneck, Z ≈ random control).

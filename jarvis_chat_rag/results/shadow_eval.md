# Shadow E-auto eval (JARVIS chat_rag)

**Finished:** 2026-11-06 12:32:53 ET (America/New_York)  
**Wall:** 5.93s  
**Mode:** sidecar only — default pack-router + CAD hybrid answer path unchanged.

## Schema / limitation

E-auto Graph expects structured {project, entity, attr, intent} queries and entity/attr/value memories. Native N4 auto_extract only keeps NUMERIC markdown table cells. This adapter (1) renders each fact card as a 2-column markdown table plus the raw sentence, (2) runs N4.auto_extract for numeric cells, (3) adds a string-cell extractor over the SAME tables so open chat values work, (4) maps NL questions to entity/attr by token overlap — not fluent free-form QA. Uncertain flags come from UNC regex markers only (no gold-trained classifier).

Memories=485 facts=79 (numeric values≈12, string values≈67).

## Published chat_rag baselines (full_eval.json, 2026-11-03, n=30)

| Method | Acc |
|---|---:|
| Retrieval pack-routed | 0.967 |
| Hybrid extractive | 0.900 |
| Pack-routed hybrid | 0.900 |
| CAD | 0.433 |
| Generation | 0.300 |

## Shadow E-auto on the same 30Q

| Metric | Value |
|---|---:|
| n | 30 |
| E-auto accuracy (correct/n) | 0.7333 |
| E-auto coverage (answered/n) | 0.8 |
| E-auto abstain count | 6 |
| E-auto selective accuracy | 0.9167 |
| E-auto confident-wrong among answered | 0.0833 |
| Recorded retrieval_routed acc (same items) | 0.9667 |
| Recorded hybrid acc (same items) | 0.9 |

**30Q verdict vs hybrid 0.900:** HURT / behind hybrid on this chat set (schema mismatch expected)

## jarvis_notes questions (new cards)

| Metric | Value |
|---|---:|
| n | 15 |
| E-auto accuracy | 1.0 |
| E-auto coverage | 0.8667 |
| E-auto abstain | 2 |
| E-auto selective accuracy | 1.0 |
| E-auto confident-wrong among answered | 0.0 |
| Live retrieval acc (gold in BM25 hits) | 0.9231 |
| Live extractive-hybrid acc (no LM) | 0.9231 |

### Per-question (jarvis_notes)

| id | gold | E-auto | ok? | ret | hyb |
|---|---|---|---|---|---|
| jn_n4_cwr | 0.064 | 0.064 | Y | True | True |
| jn_n4_vs_exact | yes | yes | Y | False | False |
| jn_n4_labels | no | no | Y | True | True |
| jn_n5 | no | no | Y | True | True |
| jn_qms | no | no | Y | True | True |
| jn_tutor | NOT SUPPORTED | NOT SUPPORTED | Y | True | True |
| jn_chat_hybrid | 0.900 | 0.9 | Y | True | True |
| jn_chat_gen | 0.300 | 0.3 | Y | True | True |
| jn_chat_cad | 0.433 | 0.433 | Y | True | True |
| jn_chat_ret | 0.967 | 0.967 | Y | True | True |
| jn_hypo | no | no | Y | True | True |
| jn_callosum | no | no | Y | True | True |
| jn_stp | STP | STP | Y | True | True |
| jn_probe_kappa_a | ABSTAIN | ABSTAIN | Y | None | None |
| jn_probe_lambda | ABSTAIN | ABSTAIN | Y | None | None |

## Integrity

- run_neuron4.py sha256 `9994ec5714617b1db9e966b84381a905e17b3652783c1c23ca077b3ff9d1c1e3` matches launch hash: **True**
- n3lib.py sha256 `2a392b927bdb4c50b1bb8e81d4ff26f1a9eb908a59c215ba20514f7ff41a0aa7` matches HASHES_FROZEN: **True**
- neuron4/ files were not modified.
- Default primary answers were not replaced (shadow sidecar only).
- No QPU, no Kaggle upload, no fine-tune, Institutional Books not indexed.


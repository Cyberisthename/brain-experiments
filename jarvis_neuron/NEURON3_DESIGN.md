# NEURON-3 experimental design (DESIGN ONLY — do not run)

**Status:** Design written after NEURON-2 audit reproduction **succeeded**.
**Do not launch** until the user explicitly approves a run.
**No QPU. No external API models.**

## Provenance gate
- NEURON-1 / NEURON-2 frozen at `frozen/neuron1_v1` and `frozen/neuron2_v1` (read-only).
- Audit: `neuron2_audit/AUDIT.md` — accuracy fields reproduced exactly; false-recall lead survives as a point estimate (conn 16.4% vs exact 23.6%; question-level 95% CI on difference excludes 0; seed-level permutation p≈0.13 with n_seeds=5).
- NEURON-3 must not modify frozen baselines or NEURON-2 originals.

## Scientific question
**Why do connections help?** Specifically: do they reduce *confident wrong answers* by encoding real relationship structure (and boundaries), or by a similarity / label artifact?

## Architectures (five versions)

| Version | Connection design | What it isolates |
| ------- | ----------------- | ---------------- |
| **A** | Normal connections (NEURON-2 `3_neuron_conn` recipe) | Full proposed mechanism |
| **B** | Randomized connections (same degree sequence / same edge count, endpoints shuffled) | Topology vs mere density of links |
| **C** | Incorrect relationship labels (real edges kept, relation types / endpoint semantics scrambled) | Correct wiring vs correct labels |
| **D** | Confidence-weighted connections (edge weight ∝ source confidence / agreement / recency) | Binary graph vs graded strength |
| **E** | Boundary-aware typed edges | Distinguishing proposed / disproved / supported / supersedes / unrelated-similar / uncertain-source |

### Version E — typed edge vocabulary
Every edge must carry exactly one type:

| Type | Meaning |
| ---- | ------- |
| `proposed` | Claim introduced, not yet settled |
| `disproved` | Later evidence rejects an earlier claim |
| `supported` | Later evidence backs a claim |
| `supersedes` | Newer value replaces an older one for the same entity/attr (outdated→current) |
| `unrelated_similar` | Surface similarity without a factual link (hard negative) |
| `uncertain_source` | Link asserted but provenance weak / conflicting sources |

Retrieval / abstention rules for E:
- Prefer `supported` / `supersedes` over `proposed`.
- Never answer from a `disproved` target as if it were current.
- If only `uncertain_source` edges support an answer, **abstain** (or answer with low confidence that counts as abstain under the utility score).
- `unrelated_similar` neighbors are excluded from answer aggregation (boundary).

## Relation-discrimination probe (required)
Separate from QA, score whether the memory graph can classify held-out pairs:

1. Related (true associative link)
2. Contradictory (disproved / mutual exclusion)
3. Similar-but-unrelated (distractor twin)
4. Outdated (superseded value)
5. Uncertain source

**Metric:** macro-F1 + calibration of confidence; report confusion matrix.
This tells us whether NEURON learns **relationships** or only **similarities**.

## Baselines (carry forward, equal budgets)
1. Exact search + zstd (corrected native storage accounting)
2. RAG (TF-IDF + SVD) — local only
3. NEURON A–E as above

Storage pad-to-cap and retrieve-byte caps as in NEURON-2. **Native bytes = uncapped stored size** (do not subtract pad).

## Metrics
- Correct answers
- Confident wrong answers (false recall) — primary
- Appropriate abstentions
- Multi-hop (≥3) recall
- Storage overhead (native bytes, bytes/correct fact)
- Relation-discrimination F1 (versions A–E)
- Forgetting / interference after +1000 memories (optional replication of NEURON-2 H4)

Utility score unchanged: +1 correct, 0 abstain (when gold isn't ABSTAIN), −2 confident wrong; ABSTAIN gold scoring as in NEURON-2.

## Data & no-leakage split
- Prefer real conversation logs if available; else expand the research-markdown corpus with **held-out files** never used to hand-curate facts.
- **Question construction split:**
  - Fact / edge construction may use corpus train files only.
  - Test questions and discrimination pairs come from held-out files + paraphrases generated with a seed frozen before evaluation.
  - Assert no test question string appears in the memory blob (same leakage check as NEURON-2).
- Multiple random seeds (≥20) for QA; report macro-over-seeds and micro-over-questions.

## Sample size / power (7 pp false-recall difference)
Target: detect a **7 percentage-point** false-recall difference (e.g. 23.6% → 16.6%) at two-sided α=0.05.

Using NEURON-2 rates p₁=0.236, p₂=0.166:

| Design | Power | Approx. n (questions per compared pair) |
| ------ | ----- | -------------------------------------- |
| Unpaired two-proportion | 80% | **~511** |
| Unpaired two-proportion | 90% | **~684** |
| Paired (upper bound using p₁+p₂−2p₁p₂) | 80% | **~518** |

**Recommendation:** ≥**600 test questions** total across seeds (e.g. 20 seeds × ≥30 test questions), with a preregistered paired permutation test on question-level false-recall indicators for A vs exact and A vs B/C.
NEURON-2 had n=55 test questions (11/seed × 5); underpowered for confirmatory claims (seed-level p≈0.13).

Observed NEURON-2 McNemar discordant count: 4 / 55 — use this to refine paired power in the prereg before launch.

## Ablation readouts (what would convince us)
- **A ≫ B** on false recall → structure matters, not just edge count.
- **A ≫ C** → labels/semantics matter, not just topology.
- **D ≥ A** → weights help; **D ≈ A** → binary edges enough.
- **E ≫ A** on false recall *and* on relation-discrimination F1 → boundaries are the mechanism.
- If A≈B≈C on QA but E wins discrimination only, connections help representation more than answer aggregation.

## Deliverables before any launch
1. `PREREG.md` with locked seeds, n targets, scoring, and accept/reject rules for each version.
2. Code hash recorded at prereg time.
3. Separate output dir `neuron3/` (never write into frozen/).
4. Per-question logs from day one (lesson from NEURON-2 audit).

## Explicit non-goals
- No IBM QPU time.
- No claim of biological neurons.
- No silent patching of results; corrections go to `*_corrected.json` with provenance.

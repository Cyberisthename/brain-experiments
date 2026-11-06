# JARVIS Tutor Curriculum — Preregistration (pilot)

**Date:** 2026-11-05 (America/New_York)
**Owner:** Lanner / Cyberisthename
**Code root:** `/workspace/jarvis/tutor`
**Hard constraints:** JARVIS is from-scratch (`MINE NO API`). Teacher is programmatic only (rule-based generators/graders in this repo). No Llama/Qwen. No external LLM teacher. No Kaggle launch/stop in this pilot.

## Motivation

The 319M medium_v3 model (eval loss ~3.21) still produces incoherent chat. The idea is to teach it like a student: start in a tiny native symbolic language the model can master, with a teacher that **explains** answers, grades them, and only then bridges to English.

## Hypothesis (one primary)

**H1:** With the **same model**, **same total training-token budget**, and **same example pool**, a **tutor curriculum** (ordered stages + worked explanations + mastery-gated progression with retries of failed items) beats:

1. **No-explanations** — same curriculum order and mastery gating, but targets contain only the answer (explanations stripped).
2. **Shuffled** — same full examples *with* explanations, but random order each epoch and **no** mastery gating / retries.

…on **held-out tasks** and **held-out lengths**, measured by exact-match answer accuracy.

Primary contrast for H1: `tutor` vs mean of the two controls is not enough — we report all three arms. Confirmatory claim requires `tutor` > `no_expl` **and** `tutor` > `shuffled` on the primary metric (Holm-adjusted paired across seeds).

## Model (pilot)

- Architecture: `JarvisTransformer` (plain attention) from `/workspace/jarvis_kaggle/model/transformer.py` — Lanner's own code.
- Size: `d_model=128, n_layers=4, n_heads=4, d_ff=512, max_seq_len=80` (~1M params with the tutor vocab). Chosen for CPU pilot speed; not the 319M.
- Init: fresh random each seed. No pretrained weights.

## Curriculum stages

### Stage 0 — native symbolic language
Tasks over bit-strings: `COPY`, `REV`, `COUNT`, `PARITY`, `ADD`.
Format: `<bos> TASK IN: … <sep> EXPLAIN: … <sep> ANSWER: … <eos>`
Teacher explanations are deterministic templates (step lists), not an API model.

### Stage 1 — bridge
Map bits ↔ English number words; templated `MAP` / `MEANS` sentences.

### Stage 2 — simple English
Templated QA + short definitions from a **built-in static dictionary** in this package (not WordNet live calls during training).

## Arms (matched)

| Arm | Order | Explanations in target | Mastery gate + fail retries |
|-----|-------|------------------------|-----------------------------|
| `tutor` | curriculum | yes | yes |
| `no_expl` | curriculum | no | yes |
| `shuffled` | random each epoch | yes | no |

**Token budget:** each arm trains until `TOKEN_BUDGET` non-pad target tokens have been used for the loss (default 200_000). Same optimizer, LR, batch size.

## Data splits (frozen before training)

- Train lengths: `{2,3,4,5,6}`
- **Held-out lengths:** `{8,10}` (same tasks as train)
- Train tasks: `COPY, REV, PARITY, MAP, DEFINE, QA` (+ Stage0 COUNT for curriculum but…)
- **Held-out tasks:** `COUNT`, `ADD` at train lengths (never in tutor progression targets for training — wait, COUNT is stage0 curriculum)

**Correction (frozen rule):**
- Training task set: `COPY, REV, PARITY, MAP, MEANS, DEFINE, QA`
- Held-out tasks: `COUNT`, `ADD` (never appear in any training example for any arm)
- Held-out lengths: length ∈ `{8,10}` for `COPY, REV, PARITY` only
- In-distribution probe: length ∈ train set, tasks ∈ train set

Seed for data generation: `DATA_SEED = 0` (shared across arms). Model/train seeds: `{0,1,2,3,4}` (≥5).

## Mastery gating (tutor + no_expl only)

After every `PROBE_EVERY` steps, exact-match accuracy on a 32-item probe for the **current curriculum slice**. If accuracy ≥ `MASTERY = 0.75`, advance to the next slice; else re-queue the failed probe items into the next window (retries). Shuffled arm ignores this and streams randomly.

## Metrics (pre-registered)

1. **Primary:** exact-match answer accuracy on **held-out tasks** (macro-average of COUNT and ADD).
2. **Secondary:** exact-match on **held-out lengths**.
3. **Secondary:** exact-match on in-distribution probe.
4. **Process:** final curriculum stage reached (tutor/no_expl only); train loss last 100 steps.

## Stats

- 5 seeds × 3 arms.
- For each metric: paired bootstrap over seeds (tutor − control), 5000 resamples; two-sided.
- Holm correction across the two primary contrasts (`tutor−no_expl`, `tutor−shuffled`) on the primary metric only.
- α = 0.05.

## Decision rule

- **Support H1** if both Holm-adjusted p < 0.05 and tutor mean > both controls on primary metric.
- **Partial** if tutor beats exactly one control after correction.
- **Not supported** otherwise.
- Nulls reported honestly. Pilot does **not** auto-scale to 319M; RESULTS.md says whether scaling is warranted.

## Out of scope this pilot

- Kaggle / 319M training
- External APIs / Llama / Qwen
- Stopping or launching any Kaggle session
- Claiming chat competence from Stage-0 wins alone

## Freeze

Analysis script: `analyze_pilot.py` (single). No peeking at held-out to tune curriculum after first seed.

## Amendment A1 — Verifier / self-training arm (STaR-style)

Added 2026-11-05 after design start, **before** any pilot seed completed. Frozen with this text.

### Extra arm: `star`

Same model, same `TOKEN_BUDGET`, same example **prompts** as the shared pool.

Procedure each cycle (until token budget exhausted):
1. Sample up to `STAR_ATTEMPTS` (default 4) completions from the current model for a prompt (temperature 0.8), asking for `EXPLAIN: … ANSWER: …` format.
2. Programmatic checker (`grade_exact` on the answer span) keeps only completions whose **answer** matches gold.
3. Train next on the kept (reasoning+answer) traces. If none correct for an item, fall back to the gold full example once (so the arm never starves), and mark the item for **error-focused resampling** (higher sample weight next cycle).
4. No curriculum mastery gate; prompt order follows the curriculum slice schedule for fair prompt exposure, but progression is not mastery-gated.

### Optional follow-up (run only if primary 4-arm pilot finishes with time): `student_teacher`

- Teacher model = best `tutor` seed checkpoint (or `star` if tutor fails).
- Generates practice completions for a **fresh** small student; two filters: (a) checker-filtered (b) unfiltered.
- Same token budget for the student. Report whether filtered > unfiltered on held-out tasks.
- Not required for H1 decision.

### Updated arms table

| Arm | Order | Explanations | Mastery / retries | Self-sampled verified traces |
|-----|-------|--------------|-------------------|------------------------------|
| `tutor` | curriculum | gold teacher | yes | no |
| `no_expl` | curriculum | no | yes | no |
| `shuffled` | random | gold teacher | no | no |
| `star` | curriculum prompts | model-sampled, checker-filtered (gold fallback) | error-focused resample | yes |

### Updated hypothesis

**H1** (unchanged primary): `tutor` beats `no_expl` and `shuffled` on held-out tasks.

**H2** (secondary, A1): `star` beats `shuffled` on held-out tasks at matched tokens (self-training with a checker helps vs plain supervised-on-gold-shuffled).

**H3** (secondary, A1): `tutor` beats `star` on held-out tasks (gold programmatic explanations beat noisy self-sampled ones at this scale).

Holm correction for H1's two contrasts only. H2/H3 reported with raw paired bootstrap p; labeled exploratory.


## Amendment A2 — Pilot scale (CPU contention)

Box CPU is shared with other long-running local jobs. Pilot uses a smaller arch
(`d_model=64, n_layers=3, n_heads=4, d_ff=256, max_seq_len=64`), `TOKEN_BUDGET=60000`,
`BATCH_SIZE=8`, and STaR samples on only a fraction of batch items per step.
**Training-token budget remains matched across arms.** Wall-clock is not matched
(star spends extra time on sampling). Student-as-teacher is opt-in via
`RUN_STUDENT_TEACHER=1` after the main 4-arm pilot.

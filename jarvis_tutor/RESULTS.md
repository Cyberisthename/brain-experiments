# JARVIS Tutor Pilot — RESULTS

**Decision (H1):** NOT SUPPORTED H1
**H2 (star>shuffled, exploratory):** not supported
**H3 (tutor>star, exploratory):** not supported
**Worth scaling to 319M now?** False — Do not scale to 319M yet; fix curriculum/capacity on the pilot first or accept null.

## Arm means (primary = held-out tasks COUNT/ADD macro exact-match)

| Arm | primary mean±std | held-out lengths | in-dist probe | final slice |
|-----|------------------|------------------|---------------|-------------|
| tutor | 0.000±0.000 | 0.000 | 0.033 | 0.0 |
| no_expl | 0.040±0.065 | 0.000 | 0.146 | 1.0 |
| shuffled | 0.092±0.071 | 0.000 | 0.021 | — |
| star | 0.010±0.022 | 0.000 | 0.013 | 1.0 |

## Contrasts

- **tutor−no_expl** (H1): Δ=-0.040, p=0.1596, p_Holm=0.1596, 95% CI [-0.100,0.000]
- **tutor−shuffled** (H1): Δ=-0.092, p=0.0004, p_Holm=0.0008, 95% CI [-0.145,-0.043]
- **star−shuffled** (exploratory): Δ=-0.082, p=0.0148, 95% CI [-0.143,-0.022]
- **tutor−star** (exploratory): Δ=-0.010, p=0.6620, 95% CI [-0.030,0.000]

## Plain-words verdict

We trained a tiny 161k-param JARVIS from scratch with a rule-based teacher. Tutor primary=0.00, no_expl=0.04, shuffled=0.09, star=0.01. H1 decision: NOT SUPPORTED H1. Scaling call: Do not scale to 319M yet; fix curriculum/capacity on the pilot first or accept null.

See PREREG.md for frozen design. All quantum/LLM-teacher claims: none — programmatic teacher only.

## Post-hoc diagnostics (not used for H1 decision)

Frozen decision above stands. These notes explain the null so we know what to change **before** any re-prereg:

1. **Tutor never left slice 0** (mean final_slice=0.0). Mastery 0.75 on the first COPY probe was too hard for this tiny net + budget, so the tutor arm burned its 60k tokens almost entirely on short COPY with long EXPLAIN strings.
2. **Token-matched explanations dilute answer supervision.** Same TOKEN_BUDGET means the tutor sees fewer answer tokens than `no_expl` / gets fewer unique items than `shuffled`.
3. **Absolute accuracy is near floor** on held-out COUNT/ADD (best arm shuffled ~0.09). The pilot model is **161,664 params**, not ~1M; capacity + budget were too small for compositional generalization.
4. **STaR keep-rate ≈ 0.4%.** Almost all star traces fell back to gold; self-sampling did not yet produce useful verified reasoning at this scale (H2/H3 not supported).
5. **In-distribution:** `no_expl` led (≈0.15) vs tutor ≈0.03 — stripping explanations helped the small model fit answers under the budget.

### What would need a new prereg (not done here)
- Larger pilot (`small` ~5M) or 5–10× token budget
- Mastery threshold schedule (start 0.5, rise) or step-based slice advance as a backup
- Answer-token budget matching (count only ANSWER: span tokens) instead of all supervised tokens
- Only then consider Kaggle — start from base/small JARVIS, not a cold 319M tutor run

Student-as-teacher (`RUN_STUDENT_TEACHER=1`) was **not** run: optional under A1, and the parent arms are at floor.

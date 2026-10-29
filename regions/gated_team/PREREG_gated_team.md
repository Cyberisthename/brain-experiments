# Preregistration: GATED TEAM — Design 1+8 (experts + neuromod + timescales)

**Locked:** 2026-10-29 07:46 America/New_York (EDT), BEFORE any `gated_team` eval scores.  
**Dir:** `/workspace/flybrain/regions/gated_team/`  
**Source:** `regions/integration_research/RESEARCH.md` Designs 1 + 8.  
**Parent facts:** TEAM concat ridge diluted CX XOR (0.64→0.46); TEAM_ER beat TEAM on composite (`team_franken` NOT SUPPORTED). Hypothesis: **per-module expert ridges + task gate + neuromod gain + species leaks** preserve specialists.

## Question

Does **GATED_REAL** (FLY_CX ⊕ HUMAN_LAYERED, zero bridges) with (a) expert ridges + convex gate, (b) task-cue neuromod gain, (c) per-module leaks (fly fast / human slow) jointly keep XOR non-inferior to FLY_CX alone and NARMA non-inferior to HUMAN alone, and beat fair ER / old TEAM on composite?

## Recipe (frozen)

- **Topology:** block-diagonal `[FLY_CX | HUMAN]`. Bridge σ = 0. Shared L4 drive (union). Global ρ=0.9 after block scale. Wash/tr/te=200/1500/500. DMAX=5.
- **Delays:** REALMYELIN on human-range pre-synapses (same map as `team_franken`); FIXED Uniform{1..5} elsewhere. GATED_ER / ER4962: shuffled myelin where human-range applies.
- **(a) Experts + gate:** Fit ridge on fly states and human states separately (train window only). Convex gate `ŷ = λ p_fly + (1−λ) p_human` with λ ∈ [0,1] fit by least squares on **train only** (closed form, clipped). Alone arms (FLY_CX, HUMAN, ER4962) and TEAM use a single ridge on all states (TEAM = old concat).
- **(b) Neuromod gain:** task cue `(m_fly, m_human)` multiplies each module’s **input and recurrent** gain: `g = 1 + γ · m`. Cue schedule (locked): XOR → (1, 0); NARMA → (0, 1); MC/PC_HARD → (0.5, 0.5). γ_fly, γ_human chosen on tune seeds **90–92 only** from `{0, 0.5, 1.0, 2.0}`.
- **(c) Per-module leak:** fly leak (fast) and human leak (slow) from grids on tune only. Fly grid `{0.25, 0.35, 0.5}`; human `{0.08, 0.12, 0.2}`; alone/TEAM/ER4962 use single leak from `{0.12, 0.2, 0.35}` (tune). GATED_REAL_noLeak forces both modules to leak=0.2 (ablate c).
- Ridge α per arm × task on seeds 90–92; grid `{1e-8,1e-6,1e-4,1e-2,0.1,1,10}`. Lock `out/alpha_choice.json` (alphas, γ, leaks, composite z) before eval.
- Eval seeds: **700–723** (24 NEW). Never peek.
- WORKERS=2, nice. `fastcore2` opt-in if importable. CPU shared with macaque/plasticity jobs.

## Arms

| Arm | Definition |
|---|---|
| `GATED_REAL` | real FLY_CX + real HUMAN; (a)+(b)+(c); real myelin |
| `GATED_REAL_noLeak` | same as GATED_REAL but leak_fly=leak_human=0.2 (no c) |
| `GATED_ER` | ER-matched fly + ER-matched human sizes/densities; same (a)+(b)+(c); **shuffled** myelin |
| `GATED_EI` | E/I-class shuffle CX + real HUMAN; (a)+(b)+(c); real myelin |
| `TEAM` | old concat ridge, leak=tuned single, **no** experts/gate/neuromod (team_franken TEAM) |
| `FLY_CX` | real FLY_CX alone |
| `HUMAN` | real HUMAN alone (REALMYELIN) |
| `ER4962` | single dens/ER size-matched graph N=4962 (no modules); FIXED delays |

Sizes: FLY_CX N=2875; HUMAN N=2087; gated TEAM-like / TEAM / ER4962 N=4962.

## Tasks

- **Primary metrics:** XOR_DELAY d=5 (1−NRMSE, ↑); NARMA-10 (NRMSE, ↓).
- **Secondary:** MC (sum r², ↑); PC_HARD (cue=0.35, noise=1.5, ↑).
- **Composite** (tune-locked z, same formula as `team_franken`):  
  `z_xor = (XOR − μ_xor)/σ_xor`; `z_narma = −(NARMA − μ_narma)/σ_narma`; `composite = 0.5(z_xor + z_narma)`.

## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon, n=24 paired)

1. **H1 (joint specialists):**  
   - **H1x:** XOR non-inferiority margin 0.03: median(`GATED_REAL_XOR − FLY_CX_XOR + 0.03`) > 0.  
   - **H1n:** NARMA non-inferiority margin 0.02: median(`HUMAN_NARMA − GATED_REAL_NARMA + 0.02`) > 0.  
   H1 passes only if **both** hold. Report `p_H1_raw = max(p_H1x, p_H1n)`.
2. **H2:** `mean(GATED_REAL composite) > mean(GATED_ER composite)`.
3. **H3:** `mean(GATED_REAL composite) > mean(TEAM composite)`.

**SUPPORTED** iff all three have `p_bonf = min(1, 3·p_raw) < 0.05` **and** the mean / non-inf inequalities hold.  
Else **NOT SUPPORTED**.

## Secondary (report; not in primary Bonferroni)

- Per-task means all arms; GATED_REAL vs GATED_REAL_noLeak (leak ablate); GATED_REAL vs GATED_EI; GATED_REAL vs ER4962 / FLY_CX / HUMAN on composite.
- Locked γ, leaks, λ distribution on eval (mean λ per task).
- Effect sizes d_z; N noted (GATED > FLY_CX alone).

## Honesty

- Reservoir metrics ≠ biological computation / neuromodulator claims.
- Task cue is experimenter-given (not inferred); MoE gate is linear convex.
- Zero-bridge gated TEAM tests readout + gain specialization, not long-range axons.
- Push code+prereg+results (no big data/secrets). Report negatives.

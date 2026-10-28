# Preregistration: Mega-Frankenstein — stitch more sections + bridge strength

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any mega_franken eval scores are written  
**Dir:** `/workspace/flybrain/regions/mega_franken/`  
**No GitHub push.**

## Motivation

Prior frankenstein grafts (mouse | human | fly_CX) at bridge σ=0.05 behave like
**parallel weakly coupled reservoirs** (`ANALYSIS_cross_species.md` §3). Adding
size helps; real wiring rarely beats size-matched random on MC / NARMA / PC_HARD.

This experiment asks two questions locked before data:

1. Does adding more real sections (finch Area X, *C. elegans*, octopus-style
   **synthetic** VL) improve the stack **beyond** a same-size / same-density
   random replacement of those added blocks?
2. Does **stronger bridging** (σ ∈ {0.05, 0.2, 0.5}) or **realistic port-only
   bridges** change that picture (integration vs parallel reservoirs)?

## Base recipe (locked)

- Blocks always present: **mouse** + **human** + **fly_CX** from
  `regions/frankenstein/data/` (same modules as best_fly).
- Delays: **FIXED** Uniform{1..dmax} on non-human-pre edges; **real H01 myelin →
  delay** on human-pre edges (`dmax=5`, same map as `best_fly` /
  `hybrid_human/data/human_myelin.npz`). Recipe label: `franken_realmyelin`.
- Block ρ=1 before bridges; global ρ=0.9 after stacking.
- Leak=0.2. Hub = fly_CX (no mouse↔human direct bridges; added blocks also
  bridge only via the fly hub unless noted).
- Tasks: **MC** ↑, **NARMA** ↓ NRMSE, **PC_HARD** ↑ (cue_frac=0.35, noise=1.5).
- Eval seeds **0–7**. Ridge α **per arm** tuned on seeds **90, 91, 92** only
  (MC criterion; one rollout, α grid
  `{1e-8,1e-6,1e-4,1e-2,1e-1,1.0,10.0}`; never peek at 0–7).
- Speedup: `fastcore2` opt-in if imported; numerics must match single-traj path.

## Added blocks

| Block | Source | N | Notes |
|---|---|---:|---|
| `finch` | j0251 Area X `regions/finch/data/areax_module.npz` | ≈2000 | Stratified subsample of real neurons; induced subgraph. Real edges only. |
| `worm` | Cook 2019 adult hermaphrodite `COOK_H.npz` | 300 | Full worm; layers mapped: sensory→L4, command (AVA/AVB/AVD/PVC/…)→L5, else L2/3. |
| `octo` | `octo_vl_synthetic.npz` | 2000 | **SYNTHETIC** VL rules — flagged in every row / RESULTS. Not a real connectome. |

## Arms (stack × added-block control)

| Arm | Added blocks | Control of **added** blocks only |
|---|---|---|
| `FRANKEN` | none | baseline (no ER/SHUF variant) |
| `FRANKEN+FINCH` | finch | REAL |
| `FRANKEN+FINCH_ER` | finch | ER same-N / nnz / sign-matched |
| `FRANKEN+FINCH_SHUF` | finch | degree-preserving shuffle |
| `FRANKEN+WORM` | worm | REAL |
| `FRANKEN+WORM_ER` | worm | ER |
| `FRANKEN+WORM_SHUF` | worm | SHUF |
| `MEGA` | finch + worm + octo | all REAL (octo synthetic-but-structured) |
| `MEGA_ER` | finch + worm + octo | each added → ER |
| `MEGA_SHUF` | finch + worm + octo | each added → SHUF |

Mouse / human / fly_CX stay **real** in every arm (controls isolate the *added*
sections, not the whole graft).

## Factors

1. **Bridge strength** σ ∈ {**0.05**, **0.2**, **0.5**} (weight scale of
   Gaussian bridge edges; same edge *counts* as frankenstein builder:
   `n_out = min(3000, max(100, |out_ports|*4))`, same for `n_in`).
2. **Bridge mode**
   - `UNIF`: current-style random bridges between peripheral L4/L5-proxy sets
     and fly hub targets (L4∪L5 proxy).
   - `REALISTIC`: bridges **only** via output-type / projection neurons of the
     peripheral block → fly targets, and fly → input-type neurons:
     - mouse/human/octo: L5 → fly, fly → L4
     - finch: HVC (and LMAN if present) → fly; fly → MSN (or non-projection bulk)
     - worm: command interneurons → fly; fly → sensory

Arm key encoding: `{STACK}__s{sigma}__{UNIF|REAL}` e.g. `MEGA__s0.2__UNIF`.

## Hypotheses (locked)

- **H1 (null we respect):** At σ=0.05, real-added ≈ ER/SHUF-added on all three
  tasks after correction (size/density dominate; replicate Level-1).
- **H2:** Raising σ to 0.2 or 0.5 **changes** the real−random gap (either
  helps real integrate, or hurts everyone via over-mixing). Direction not
  claimed a priori; two-sided tests.
- **H3:** `REALISTIC` bridges differ from `UNIF` at the same σ on ≥1 primary
  contrast (exploratory / secondary family).
- **H4 (MEGA):** MEGA_REAL beats MEGA_ER / MEGA_SHUF on ≥1 task at some σ
  after Bonferroni — only path to a Level-2 “composition” claim here.

## Stats

- Per-seed paired Wilcoxon signed-rank (two-sided; all-zero → p=1).
- **Primary family (Bonferroni ×54):** for bridge=`UNIF` only, for each
  σ ∈ {0.05,0.2,0.5}, for each stack ∈ {FINCH, WORM, MEGA}, contrasts
  REAL vs ER and REAL vs SHUF × {MC, NARMA, PC_HARD}.
  (6 stack-contrasts × 3 σ × 3 tasks = 54.)
- **Secondary (Bonferroni ×54):** same family at bridge=`REALISTIC`.
- **Exploratory (raw p only):** REALISTIC vs UNIF within REAL arms; MEGA_REAL
  vs FRANKEN (size-confounded); mean tables; σ main effect plots.
- Direction: report Δ = REAL − ctrl; MC/PC_HARD higher better; NARMA lower
  better (favor REAL when Δ_MC/PC>0 or Δ_NARMA<0).

## Smoke

1 seed, σ=0.05 UNIF, FRANKEN + FRANKEN+FINCH + MEGA, MC only — before queueing
full grid.

## Honesty

- Octopus block is synthetic; never claim cephalopod biology.
- Do not call a win from MEGA > FRANKEN alone (size confound).
- Report negatives. No GitHub push from this experiment.

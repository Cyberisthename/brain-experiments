# Preregistration: Trained bridges — can FRANKEN learn to route?

**Date:** 2026-10-28 (America/New_York)  
**Status:** preregistered BEFORE any trained_bridges eval scores are written  
**Dir:** `/workspace/flybrain/regions/trained_bridges/`  
**No GitHub push.**  
**Do not touch** the running `mega_franken` job.

## Motivation

Real brains are segmented and **learn how to route** between regions. In our
FRANKEN reservoirs the inter-block **bridges are fixed random** (σ=0.05), and
prior analysis (`ANALYSIS_cross_species.md`) shows the graft behaves like
weakly coupled parallel reservoirs. This experiment asks whether **training
only the bridge weights** (keeping each block’s internal wiring frozen) lets
**real** blocks beat size-/density-matched **random** blocks.

## Base recipe (locked)

- Stack: **FRANKEN** = mouse | human | fly_CX from `frankenstein/data/`.
- Delays: **FIXED** Uniform{1..dmax} on non-human-pre edges; **real H01 myelin
  → delay** on human-pre edges (`dmax=5`, `human_myelin.npz`). Same map as
  `best_fly` / `franken_realmyelin`.
- Block ρ=1 before bridges; global ρ=0.9 after stacking (builder identical to
  `frankenstein.graft_blocks`, bridge σ=0.05).
- Leak=0.2. Drive = union of L4 ports.
- Speedup: import `fastcore2` (opt-in); numerics match single-traj path.
- Ridge α **per arm** tuned on seeds **90, 91, 92** only (MC criterion; α grid
  `{1e-8,1e-6,1e-4,1e-2,1e-1,1.0,10.0}`; never peek at eval seeds 0–7).
- Eval seeds **0–7**.

## Wiring arms (block controls)

| Wiring | Blocks |
|---|---|
| `FRANKEN` | all three blocks **real** |
| `ER` | each block → ER matched N / nnz / sign (same bridges recipe) |
| `SHUF` | each block → degree-preserving shuffle |

## Bridge condition

| Tag | Bridges |
|---|---|
| `FIXED` | random Gaussian bridges σ=0.05, **never updated** |
| `ES` | same init, then **antithetic evolution strategy** on bridge weights only |

Arm key: `{WIRING}_{FIXED|ES}` e.g. `FRANKEN_ES`, `ER_FIXED`.

## Bridge-training method (locked): antithetic ES

**Why ES (not FORCE/RLS):** FORCE/RLS on recurrent delayed bridges is easy to
destabilize; ES has a hard identical budget and stays stable with spectral
control. (One method; NP/Hebbian left as a follow-up.)

**Parameters (identical for FRANKEN / ER / SHUF):**

- Parameter vector θ = weights of **bridge edges only** (edges whose pre and
  post neurons lie in different blocks). Block-diagonal entries are **frozen
  bit-exact** for the whole run.
- Generations **G = 10**; each generation evaluates antithetic pair
  θ±σε → **20 fitness rollouts** per (wiring, seed).
- Perturbation scale σ_es = 0.05 · rms(θ_init) (floor 1e-3).
- Learning rate α_es = 0.5 / (pop_sigma); update
  `θ ← θ + α_es · 0.5 · (f₊ − f₋) · ε` (standard antithetic ES).
- After every write of θ: **restore frozen block diagonals**, then if
  ρ(W)>0.9 multiply **bridge entries only** by `0.9/ρ(W)` (one-shot). Report
  final ρ.
- Fitness rollout (short, train-only): wash=100, tr=800, te=300;
  `fitness = MC/20 − NARMA` (MC↑ NARMA↓). Same RNG scheme per generation
  index across wirings (fair).
- **Train tasks:** MC + NARMA only (via the fitness above).
- **Held-out / generalization tasks (never in fitness):** `PC_HARD`,
  `XOR_DELAY` (2-bit delayed XOR, delay d=5; metric = 1 − NRMSE on ±1 targets).
- FIXED arms: **zero** bridge updates (budget 0); ES arms: exactly 20 fitness
  rollouts. Budget equality is across the three ES wirings.

## Eval tasks

| Task | Direction | In bridge fitness? |
|---|---|---|
| MC | ↑ | yes |
| NARMA | ↓ NRMSE | yes |
| PC_HARD | ↑ (cue_frac=0.35, noise=1.5) | **no** |
| XOR_DELAY | ↑ (1 − NRMSE), d=5 | **no** |

Full eval uses wash/tr/te = 200/1500/500 (MC, NARMA, XOR); PC_HARD as in
`best_fly` (T=1200, wash=200, tr=800).

## Hypotheses (locked)

- **H1 (primary):** The **training gain**
  `gain = trained − fixed` (for NARMA: `gain = fixed − trained` so ⊕ helps)
  is **larger for FRANKEN than for ER and than for SHUF** on held-out
  `PC_HARD` and `XOR_DELAY` after correction.
- **H2 (secondary):** Same gain contrast on train tasks MC / NARMA.
- **H3 (null we respect):** Gains are similar across wirings → learning to
  route does not unlock a real-wiring advantage (Level-1 continues).

## Stats

- Per-seed paired Wilcoxon signed-rank (two-sided; all-zero → p=1).
- **Primary family (Bonferroni ×4):** gain contrasts
  `(gain_FRANKEN − gain_ER)` and `(gain_FRANKEN − gain_SHUF)` ×
  `{PC_HARD, XOR_DELAY}`.
- **Secondary (Bonferroni ×4):** same contrasts × `{MC, NARMA}`.
- Exploratory (raw p): absolute means; ES training curves; final ρ; FIXED
  FRANKEN vs FIXED ER (size-matched baseline replicate).

## Smoke

1 seed, FRANKEN_FIXED + FRANKEN_ES (G=2), MC fitness only — before full run.

## Honesty

- Do not claim biological “learning to route” beyond: gradient-free
  optimization of random bridge weights with frozen blocks.
- Do not call a win from ES > FIXED alone on FRANKEN without beating the
  matched ER/SHUF **gain**.
- Report negatives. No GitHub push from this experiment.

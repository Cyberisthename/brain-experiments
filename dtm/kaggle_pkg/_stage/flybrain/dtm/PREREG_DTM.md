# PREREG: Dynamic Trophic Morphogenesis (DTM / Neuro-Vines)

**Frozen at:** 2026-11-03 ~08:07 America/New_York (UTC-5), **before** held-out eval seeds are scored.  
**Path:** `flybrain/dtm/` (sibling of `sbqa/`; reuses `sbqa` modules read-only; does **not** alter `sbqa/kaggle_pkg/sbqa_kaggle.zip`).  
**Note:** Classical biology-inspired sparse grow/prune. Not quantum. Not backprop through vines.

## Question

Does **DTM (Neuro-Vines)** — local vine sprouting inside a Griffa connectome mask, nutrient updates from forward-only reward-modulated Hebbian signals, famine prune + regrow under a fixed edge budget — beat SBQA, classical SA, random search, greedy hill-climb, and SET/RigL-style magnitude-prune+regrow on the **discrete wiring** task, and match/beat Adam on continuous weight training under equal evaluation budget? Does wall-clock / peak memory **decrease or stay flat** as vines prune (user claim)?

## Data

- Griffa et al. 2019 scale-68 group SC: `data/griffa70_scale68_group_minimal.npz` (Zenodo 2872624, **CC BY 4.0**). Private Kaggle only. See `data/DATA_LICENSE.md`.
- Mask = bio2art expansion (`sbqa.mask.make_mask`).

## Network

- Rate reservoir (from `sbqa.reservoir.MaskedReservoir`): leak α=0.4, ρ=0.9.
- Full: **N_LOCAL=4 → N=272**. Smoke: N_LOCAL=2 → N=136.
- Edge budget `E_budget = round(0.35 × nnz(M))` (same as SBQA discrete).
- Readout: **ridge only** for discrete wiring tests (wiring is the object of study). Continuous arm may train W with Adam baselines; DTM continuous updates vine weights via Hebbian nutrients, readout still ridge or linear.

## DTM algorithm (frozen)

1. **Seed:** start with `E_seed = max(8, E_budget // 5)` random mask-legal edges; nutrient = 1.0 each; store as sparse COO `(post, pre, nutrient, weight)`.
2. **Sprout (each step):** from a random active node, sample up to `k_sprout=4` unused mask-legal neighbors (1-hop in mask adjacency); add vine with nutrient=`n0=0.5`, weight~N(0,0.08) if under soft cap `1.2×E_budget`.
3. **Forward pass:** run reservoir with current sparse W; ridge readout; task score S (NRMSE ↓ or 1−acc ↓ unified energy).
4. **Reward:** `r = baseline − S` (positive if improved). Update `baseline ← 0.9·baseline + 0.1·S`.
5. **Nutrient Hebbian:** for each active edge, accumulate `|x_pre · x_post|` over wash-free steps (mean abs product); `nutrient += η·hebb·r − famine` with `η=0.5`, `famine=0.02` per step (always subtract famine).
6. **Prune:** drop edges with `nutrient ≤ 0`; free their slots (true sparse list shrink).
7. **Budget enforce:** if `|E| > E_budget`, prune lowest-nutrient edges until budget; if `|E| < E_budget` after prune, sprout more (SET-like regrow).
8. **Rescale** spectral radius to ρ after topology change.

## Experiment A — Discrete wiring (primary)

Same task streams / seeds family as SBQA discrete for comparability.  
**Methods (equal reservoir-eval budget B):**

| Method | Notes |
|--------|-------|
| **DTM** | Neuro-Vines as above |
| **SBQA** | Import / reimplement thin wrapper calling `sbqa` discrete SBQA |
| **SA** | Metropolis annealing |
| **RANDOM** | Keep-best random feasible wirings |
| **GREEDY** | Hill-climb + restart |
| **SET** | Magnitude prune lowest-|w| fraction + random regrow in mask (Moussa et al. / SET) |
| **RIGL** | Magnitude prune + regrow proportional to |grad| proxy (= |hebb| from last forward) (Evci et al. RigL-style) |

Budget B full: **80** evals/method/seed. Smoke: 12.  
Seeds held-out: **5100–5107** (n=8). Smoke: 91.  
Primary task: NARMA-10 NRMSE ↓. Secondary: XOR_HARD (d=8) energy = 1−acc.

## Experiment B — Continuous (control)

Train / adapt weights under mask with equal step budget.  
Methods: **DTM**, **Adam**, **SBQA** (Adam+WKB), **SET** (sparse Adam with SET drops), **SGLD**.  
Full: epochs=40, steps/epoch=8, batch=16, seq=32. Smoke: epochs=4, steps=3, batch=4, seq=24.  
Seeds: **5200–5207**. Smoke: 91.

## Efficiency metrics (every method)

- `seconds` wall-clock per seed/method  
- `peak_rss_mb` via `resource.getrusage` (Linux) sampled each eval  
- For DTM only: `nnz_trace` (active edges after each step) — claim: nnz and/or time-per-eval trends down

## Stats

- Primary metric: held-out task energy (lower better).  
- Paired Wilcoxon across 8 seeds.  
- Bonferroni primary family (α_family=0.05, m=5):  
  1. DTM < SA on discrete NARMA  
  2. DTM < RANDOM  
  3. DTM < GREEDY  
  4. DTM < SET  
  5. DTM < SBQA  
- Success: DTM beats **RANDOM and SET** (tests 2 & 4). SBQA comparison exploratory within family.  
- Efficiency: report whether DTM median seconds ≤ SET and whether nnz_final < nnz_init (descriptive; not a pass/fail gate).

## Hypotheses

- **H1:** DTM beats RANDOM and SET on discrete NARMA.  
- **H2:** DTM ≥ GREEDY / SA (no strong claim vs SBQA).  
- **H3 (efficiency):** DTM peak RSS and/or mean seconds ≤ dense-mask Adam; nnz decreases from soft-cap toward ≤ E_budget.  
- **H4 (continuous negative-leaning):** Adam ≤ DTM energy not required; expect Adam ≤ DTM on smooth weights.

## Outputs

- `out/discrete_runs.jsonl`, `out/continuous_runs.jsonl`, `out/summary.json`, `RESULTS.md`  
- Kaggle: separate `kaggle_pkg/dtm_kaggle.zip` (does not replace SBQA zip)

## What this is NOT

Backprop through vines, real trophic biology, or a claim of brain-complete learning. Forward-only local grow/prune + ridge readout.

## Add-on: Membrane Phase-Sliding (MPS)

Frozen before held-out DTM/MPS eval. Classical geometric warp + phase wave — not literal membrane physics.

### Mechanism
1. **Virtual coords** `z_i ∈ R^3` per node. Init: Griffa region centroid broadcast to `n_local` units (LH negative-x / RH positive-x heuristic from `hemis` if present; y/z from spectral embedding of SC), else Laplacian spectral embedding of the mask graph (3 dims).
2. **Distance-limited sprout:** candidate vines only if `‖z_post − z_pre‖ ≤ R` (R = median pairwise dist among mask edges × 1.25) **or** mask 1-hop (union).
3. **Warp (scent bridging):** if two clusters (parcels) have high scent `corr(activity_i, activity_j)` or reward-gradient proxy above τ but mean inter-coord distance > R, temporarily pull parcel centroids together (`z ← (1−λ)z + λ·mid`) for `T_warp` steps so vines can bridge; then relax coords with spring toward init (`z ← (1−μ)z + μ·z0`). Bridged edges **persist** after relax.
4. **Phase wave:** `φ(z,t) = sin(ω t + κ·z_x)`; sprout rate & nutrient uptake × `(1 + 9·(φ)_+)` (peak up to **10×**); prune/famine × `(1 + 9·(−φ)_+)`.

### Arms (discrete NARMA primary; same B/seeds as DTM)

| Arm | Warp | Wave |
|-----|------|------|
| **DTM** | no | no |
| **DTM_MPS_WARP** | yes | no |
| **DTM_WAVE** | no | yes |
| **DTM_BOTH** | yes | yes |
| **LONG_RANDOM** | control: same # long-range edges as warp bridges, placed uniformly at random in mask (no coord warp) |

Primary family extended (Bonferroni m=7, α_family=0.05): prior 5 DTM tests +  
6. DTM_BOTH < LONG_RANDOM on discrete NARMA  
7. DTM_BOTH < DTM on discrete NARMA  

Success add-on: DTM_BOTH beats LONG_RANDOM (test 6). Test 7 exploratory (does MPS help plain DTM?).


## Add-on: Spatial Kinetic Folding (SKF)

Frozen before held-out SKF eval. Classical particle dynamics on virtual coords — not literal folding physics.

### Mechanism
1. **State:** each node has position `z_i ∈ R^4` and velocity `v_i ∈ R^4`. Init: pad MPS/Griffa 3D coords with a spectral 4th dim (or N(0,0.1)).
2. **Affinity force:** pairwise attraction `F_ij = a_ij (z_j − z_i)` where `a_ij` is a loss-derived affinity (parcel/node activity correlation × positive reward proxy). Repulsion soft-core at very short range. Integrate with damping/friction: `v ← (1−γ)v + Δt F`, `z ← z + Δt v`.
3. **Edge formation:** undirected candidate becomes a vine if `‖z_i−z_j‖ ≤ R_skf` **and** mask-legal (or soft allow long-range if already bridged). Keep DTM nutrient/famine on vines.
4. **Phase splitting (clouds):** each node is **K** Gaussian particles `{z_{i,k}, v_{i,k}}` with weights; particles move independently under the same forces. Periodic **phase freeze** every `T_freeze` steps: keep the particle with best local score proxy (highest recent nutrient uptake / lowest contribution to energy), snap node to that particle, prune other particles (low-velocity & low-score branches die).
5. **Friction pruning:** edges whose score-contribution variance (rolling) exceeds τ get **cooled** — velocity coupling frozen / nutrient updates stopped (edge weight locked).

### Arms (discrete NARMA; same B / seeds as DTM family)

| Arm | Clouds K | Notes |
|-----|----------|-------|
| **SKF** | K=4 | full clouds + freeze + friction prune |
| **SKF_K1** | K=1 | control: plain momentum, no splitting |
| **PSO** | — | particle-swarm / evolution-strategies baseline: swarm of `P` full wirings (same E_budget), velocity in binary edge-space, equal eval budget |

Primary family extended again (Bonferroni **m=9**, α_family=0.05): prior tests +  
8. SKF < PSO on discrete NARMA  
9. SKF < SKF_K1 on discrete NARMA  

Success add-on: SKF beats PSO (test 8). Test 9 exploratory (do clouds help?).


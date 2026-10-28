# Preregistration: CHIMERA_DESIGN — synthetic designed reservoir

**Locked:** 2026-10-28 19:37 America/New_York (EDT), BEFORE any `designed_brain` eval scores.  
**Dir:** `/workspace/flybrain/regions/designed_brain/`  
**Motivation (prior results, not eval peeking):**
- `xor_cx`: FLY_CX XOR win is carried by **E/I makeup** (EI_SHUF 0.70 ≥ CX 0.66); real edge list not uniquely required.
- `myelin_real_rep`: **real H01 myelin placement** beats shuffle on NARMA in human patch (0.393 vs 0.557).
- Spectral bulk of real CX is sparse (frac\|λ\|>0.5 ≈ 0.17–0.28); ER/DENS ≈ 1.0.
- `team_franken`: naive concat TEAM failed XOR non-inferiority; TEAM_ER won composite — so a **single** designed net that owns both E/I + myelin rules is the bet.

**Question:** Can a **fresh synthetic** graph (no copied edges) built from CX E/I block statistics + a human-analogue myelin delay rule beat size-matched random controls and stay competitive with FLY_CX on XOR and HUMAN on NARMA?

## Design rules (locked; fit from data only — never eval seeds 500–523)

Extracted once into `out/extracted_stats.json` from `fly_cx_module.npz` + `human_myelin.npz` / human layers (tune-agnostic connectome files).

1. **Size:** N=4000 (within 3000–5000).  
2. **E/I labels:** frac_E = CX frac_E (~0.592); rest I.  
3. **Wiring:** Stochastic block model on 4 directed classes (E→E, E→I, I→E, I→I) using CX **block densities** and **weight magnitude** bootstrap (sample \|w\| from the matching CX class). Out-sign rule: weight sign = sign(pre). No self-loops. **No edges copied from CX.**  
4. **Layers:** assign L2/3, L4, L5, L6 with proportions inspired by human depth mix (for myelin gradient); L4 used as drive ports.  
5. **Myelin→delay:** sample myelin_frac per neuron from the **human empirical distribution for that layer** (with global median fill); delay on edge = `1 + round((1−clip(mf_pre,0,1))·(DMAX−1))`, DMAX=5 (identical functional form as REALMYELIN).  
6. **Spectral:** after ρ=1 scale, if top-64 frac(\|λ\|>0.5) > 0.40, temper: shrink smallest-half \|w\| ×0.7 up to 8 iters, rescaling ρ=1 (weak sparsity prior). Then global ρ=0.9.  
7. Design RNG seed for the **graph template** is fixed at **7** (rules + one draw); eval seeds 500–523 only randomize task IO / Win / (for controls) rewires — the DESIGN adjacency is frozen across eval seeds (delays follow mf; Win/task RNG vary). Ablation graphs freeze similarly.

## Arms

| Arm | Definition |
|---|---|
| `DESIGN` | full CHIMERA (E/I SBM + myelin delay rule + spectral temper) |
| `DESIGN_EI_ONLY` | same E/I SBM graph; **FIXED** Uniform{1..5} delays (no myelin rule) |
| `DESIGN_MYELIN_ONLY` | ER matched N/nnz/sign of DESIGN; **myelin delay rule** by assigned layers |
| `FLY_CX` | real fly CX (FIXED delays) |
| `HUMAN` | real human module (REALMYELIN) |
| `TEAM_ER` | team_franken TEAM_ER rebuild (two ER blocks, shuffled myelin on human-sized) |
| `ER` | ER match of DESIGN (N/nnz/sign); FIXED delays |
| `DENS` | dens_match of DESIGN; FIXED delays |
| `SHUF` | degree-preserving shuffle of DESIGN; FIXED delays |

## Tasks / composite

- XOR_DELAY d=5 (1−NRMSE ↑), NARMA-10 (NRMSE ↓), MC ↑, PC_HARD ↑ — numerics bit-aligned with team_franken / trained_bridges.  
- Leak=0.2; wash/tr/te=200/1500/500; PC_HARD cue=0.35 noise=1.5.  
- Ridge α **per arm × task** on seeds **90–92** only.  
- Composite: z-mean of XOR and −NARMA with μ/σ locked from **tune** pool (all arms), same as team_franken.  
- Eval seeds: **500–523** (n=24).

## Primary hypotheses (Bonferroni ×4, one-sided Wilcoxon)

1. **H1:** DESIGN composite > ER composite  
2. **H2:** DESIGN composite > TEAM_ER composite  
3. **H3:** DESIGN XOR non-inferior to FLY_CX (margin **0.05**): Wilcoxon on (DESIGN_XOR − FLY_CX_XOR + 0.05) greater  
4. **H4:** DESIGN NARMA non-inferior to HUMAN (margin **0.03**, lower-is-better): Wilcoxon on (HUMAN_NARMA − DESIGN_NARMA + 0.03) greater  

**SUPPORTED** iff all four p_bonf=min(1,4·p_raw)<0.05 and mean inequalities hold (H3: mean DESIGN_XOR ≥ mean FLY_CX_XOR − 0.05; H4: mean DESIGN_NARMA ≤ mean HUMAN_NARMA + 0.03).  
Else **NOT SUPPORTED**. Report per-hypothesis pass/fail honestly.

## Secondary

- DESIGN vs DESIGN_EI_ONLY / DESIGN_MYELIN_ONLY (ablations).  
- DESIGN vs FLY_CX / HUMAN / DENS / SHUF on each task.  
- Spectral audit frac\|λ\|>0.5 on DESIGN vs ER.

## Ops

WORKERS≤2, nice. Push code+prereg+results to `Cyberisthename/brain-experiments` (no big data/secrets). Reservoir ≠ biology claim.

# Preregistration: TEAM Frankenstein — division of labor (FLY_CX ⊕ HUMAN)

**Locked:** 2026-10-28 17:56 America/New_York (EDT), BEFORE any `team_franken` eval scores.  
**Dir:** `/workspace/flybrain/regions/team_franken/`  
**Parent facts:** FLY_CX alone best on XOR_DELAY (~0.66; E/I makeup carries it, `xor_cx` SUPPORTED). HUMAN_LAYERED with real H01 myelin best on NARMA (REAL 0.393 vs SHUF 0.557, `myelin_real_rep` REPLICATED).  
**Question:** Does a **TEAM** of real FLY_CX + real HUMAN (real myelin) with **zero bridges** (shared drive, concatenated ridge readout) beat size-matched / alone baselines on a **composite** of XOR and NARMA — i.e. division of labor without mashing?

## Recipe (frozen)

- **TEAM topology:** block-diagonal `[FLY_CX | HUMAN]`. **Bridge σ = 0** (no inter-block edges). Shared input drives **union of L4** ports in both blocks. Readout = ridge on **all** concatenated states (ridge picks specialists).
- Block scale ρ=1 then global ρ=0.9. Leak=0.2. Wash/tr/te=200/1500/500.
- **Delays:** REALMYELIN on human-range pre-synapses (same map as `xor_rep` / `trained_bridges`); FIXED Uniform{1..5} elsewhere (fly / ER / non-human). DMAX=5.
- Tasks (bit-exact with trained_bridges / xor_rep / myelin_real_rep numerics where applicable):
  - **Primary:** XOR_DELAY d=5 (metric 1−NRMSE, ↑ better); NARMA-10 (NRMSE, ↓ better).
  - **Secondary:** MC (sum r², ↑); PC_HARD (cue=0.35, noise=1.5, ↑).
- Ridge α **per arm × per task** on seeds **90, 91, 92** only (XOR/MC/PC: max mean; NARMA: min mean). Grid `{1e-8,1e-6,1e-4,1e-2,0.1,1,10}`. Lock `out/alpha_choice.json` + composite z-params before eval.
- Eval seeds: **400–423** (24 NEW). Never peek.
- `fastcore2` opt-in if importable. WORKERS≤2, nice.

## Arms

| Arm | Definition |
|---|---|
| `TEAM` | real FLY_CX + real HUMAN; zero bridges; real myelin on human |
| `FLY_CX` | real FLY_CX alone (FIXED delays) |
| `HUMAN` | real HUMAN alone (REALMYELIN) |
| `FRANKEN` | old mashed mouse+human+fly_cx, bridges σ=0.05 (`F.build_arm("FRANKEN")`), REALMYELIN |
| `TEAM_ER` | ER-matched fly-sized + ER-matched human-sized blocks; zero bridges; **shuffled** myelin delays on human-sized block |
| `TEAM_EI` | E/I-class shuffle of CX + real HUMAN; zero bridges; real myelin on human (mechanism) |
| `FLY_CX_BIG` | real FLY_CX + dens_match ER block of size/nnz of HUMAN; zero bridges; FIXED delays (size fairness) |

Sizes: FLY_CX N=2875; HUMAN N=2087; TEAM/TEAM_ER/TEAM_EI/FLY_CX_BIG N=4962; FRANKEN N≈6833.

## Composite score (locked)

- From **tune** seeds only, pool XOR and NARMA values across all arms → lock `μ_xor, σ_xor, μ_narma, σ_narma` in `out/alpha_choice.json`.
- Per eval seed/arm:  
  `z_xor = (XOR − μ_xor) / σ_xor`  
  `z_narma = −(NARMA − μ_narma) / σ_narma`  (− so higher better)  
  `composite = 0.5 * (z_xor + z_narma)`
- If σ < 1e-12, treat that z as 0.

## Primary hypotheses (Bonferroni ×4, one-sided Wilcoxon, n=24 paired)

1. **H1a:** `mean(TEAM composite) > mean(FLY_CX composite)`
2. **H1b (XOR non-inferiority, margin 0.03):** `TEAM_XOR − FLY_CX_XOR + 0.03` has median > 0 (Wilcoxon greater on that signed difference)
3. **H2:** `mean(TEAM composite) > mean(TEAM_ER composite)`
4. **H3:** `mean(TEAM composite) > mean(FLY_CX_BIG composite)`

**SUPPORTED** iff all four have p_bonf = min(1, 4·p_raw) < 0.05 **and** the corresponding mean inequalities hold (for H1b: mean(TEAM_XOR) ≥ mean(FLY_CX_XOR) − 0.03).  
Else **NOT SUPPORTED**.

H1 requires both H1a and H1b; failure of either fails the claim that TEAM improves the composite without sacrificing XOR.

## Secondary (report; not in primary Bonferroni)

- Per-task means: XOR, NARMA, MC, PC_HARD for all arms.
- TEAM vs HUMAN on composite / NARMA; TEAM vs FRANKEN on composite.
- TEAM_EI vs TEAM on composite (does E/I CX shuffle kill TEAM?).
- Effect sizes d_z; sizes N noted honestly (TEAM > FLY_CX; FLY_CX_BIG / TEAM_ER match TEAM size).

## Honesty

- Reservoir metrics ≠ biological computation claims.
- Zero-bridge TEAM is a readout-selection / specialization test, not a claim about anatomical long-range axons.
- FRANKEN is larger (adds mouse); not size-matched to TEAM.
- Push code+prereg+results to `Cyberisthename/brain-experiments` (no big data/secrets). Report negatives.

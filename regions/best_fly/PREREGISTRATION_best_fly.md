# Preregistration: Smartest fly — stack what helped

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any best_fly eval scores are written  
**Dir:** `/workspace/flybrain/regions/best_fly/`

## Motivation

Prior arms found:
- Synthetic **FIXED** delays on fly CX → MC ≈ 45 (`regions/myelin`; FIXED > ADAPT > NODELAY on that protocol).
- Real H01 myelin map on human-pre edges → large MC boost vs NODELAY; **REAL > SHUF on NARMA for HUMAN_LAYERED** (p_bonf ≈ 0.002; `regions/myelin_real`).
- Frankenstein hybrid (mouse+human+fly CX) showed PC_HARD gains vs fly alone in fast passes (`regions/frankenstein`).
- Hebbian / Oja warmup did **not** help → **excluded**.
- Maturation (`regions/mature`) still running; partial `full` deltas show **MC drops** with age on both FLY_CX and FRANKEN → **maturation excluded** from BEST until a finished analysis shows real wiring gains *more* than random.

Goal: one stacked recipe vs fair random graphs that get the **same** tricks.

## Arms (eval seeds 0–7)

| Arm | Wiring | Delays |
|---|---|---|
| `FLY_CX_PLAIN` | real fly CX alone | all d=1 |
| `FLY_CX_FIXED` | real fly CX alone | FIXED: per-edge Uniform{1..dmax} (frozen; seed-tied) |
| `FRANKEN_REALMYELIN` | real mouse+human+fly CX graft | human-pre: real H01 myelin→delay; other-pre: FIXED Uniform{1..dmax} |
| `BEST` | recipe locked on tune seeds 90–92 only | see below |
| `ER_BEST` | ER same-size / nnz-matched per block as BEST’s architecture | **identical delay rule** as BEST |
| `SHUF_BEST` | degree-preserving shuffle per block | identical delay rule as BEST |
| `DENS_BEST` | dens-matched (same N+nnz, \|w\| bootstrap) per block | identical delay rule as BEST |

### BEST recipe candidates (tune seeds **90, 91, 92** ONLY)

Pick the candidate with highest composite score on the mean over tune seeds:

`S = mean(MC)/10 − mean(NARMA) + mean(PC_HARD)`

(MC scaled by 1/10 so it does not dominate NARMA≈0.5 and PC≈0.8.)

Candidates:

1. `fly_cx_fixed` — FLY_CX + FIXED delays  
2. `franken_fixed` — FRANKEN + FIXED on all edges  
3. `franken_realmyelin` — FRANKEN + real human myelin + FIXED elsewhere (same as `FRANKEN_REALMYELIN`)  
4. `franken_realmyelin_only` — FRANKEN + real human myelin; non-human-pre d=1  

Locked choice written to `out/best_choice.json` **before** any seed-0–7 BEST/control rows. If ties, prefer the simpler architecture (fly over franken; fewer myelin rules).

If BEST picks a fly-sized recipe, ER/SHUF/DENS controls are **fly-CX-sized** random graphs with FIXED delays. If BEST picks a franken recipe, controls are frankenstein-sized with that delay rule (for `franken_realmyelin*`, controls still get the real myelin map on the human index range of the grafted graph — myelin values follow the human neuron indices; for shuffled wiring the human block still occupies the same index range).

## Protocol constants

- Leak = 0.2; block ρ=1, bridge σ=0.05, global ρ=0.9 (frankenstein builders).  
- `dmax = 5`. Myelin map: `hybrid_human/data/human_myelin.npz`; `d = 1 + round((1−clip(mf,0,1))*(dmax−1))`; NaN mf → median fill.  
- Tasks: **MC** ↑ (wash/tr/te/D = 200/1500/500/100), **NARMA** ↓ NRMSE, **PC_HARD** ↑ (cue_frac=0.35, noise=1.5).  
- Drive = L4 / union of L4s (frankenstein `info["drive"]`).  
- Fast kernels: `compress/fast_kernels.py` + bucketed delay CSR (same as `myelin_real`).  
- No HEBB, no maturation, no maze, no easy PC.

## Stats

- Per-seed paired contrasts (Wilcoxon signed-rank, two-sided; if all-zero diffs use p=1).  
- **Primary family (Bonferroni ×9):** BEST vs {ER_BEST, SHUF_BEST, DENS_BEST} × {MC, NARMA, PC_HARD}.  
- **Secondary family (Bonferroni ×3):** BEST vs FLY_CX_FIXED × {MC, NARMA, PC_HARD}.  
- Exploratory: all other pairwise arm means; report raw p only.  
- Direction: MC/PC_HARD higher better; NARMA lower better (Δ reported as BEST−ctrl; for NARMA, negative Δ favors BEST).

## Predictions (locked)

- **P1**: Delays raise MC vs FLY_CX_PLAIN (replicate FIXED / myelin_real boost).  
- **P2 (primary):** BEST beats its random controls on ≥1 task after Bonferroni — i.e. **real wiring still matters once tricks are shared**.  
- **P3 (secondary):** BEST ≥ FLY_CX_FIXED on the composite (graft+real myelin adds something beyond fly+FIXED alone).  
- **P4 (null we respect):** If BEST ≈ controls after correction, honest verdict = random wiring with the same delay tricks matches the “smartest” stack.

## Honesty / ops

- No GitHub push.  
- `nice -n 19`, WORKERS≤2 (share box with mouse hybrid + mature).  
- Report negatives. Do not call biology a win from BEST > FLY_CX_PLAIN alone.

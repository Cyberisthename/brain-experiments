# Preregistration: Thinking-time test (CFS forward projection × K)
Written BEFORE any thinking-time result. Queued to run only after EMPN v2 finishes.

## Setup
- Maze: same fixed env as EMPN v2 (`regions/empn`, shared learner fixes from v2: block-norm DN⊕obs readout, explore-until-food, count bonus, Wq clip).
- Bridge: **1,500-param dense E-PG adapter** (SAG_DENSE-style: G=500 gateways, no top-K, Vin 2-in/gateway + U 1-out → E-PG/P-EG only). Trained offline on REAL CX at f=0.25 MAGP (same recipe as `compress/sag.py` SAG_DENSE), then frozen into the maze agent as CA1→CX / plan→CX drive.
- Also run the same K sweep on reservoir **MC** and **T0** (compress tasks), with the adapter in the residual path.

## Arms
For each K ∈ {0, 1, 2, 4, 8, 16}:
1. **THINK**: K autonomous CFS forward-projection steps before committing action / before readout flush to DNs.
2. **RANDTGT**: same K, but adapter U targets are random CX cells (matched count), not E-PG/P-EG.
3. **SETTLE**: K extra reservoir settling steps (leak dynamics only) with **no** CFS projection.

Maze agents: FLY baseline + EMPN-noCFS + EMPN (v2 learner). Compress: NONE vs adapter arms above on MC/T0.

## Seeds / stats
8 seeds (0–7). Primary maze metrics: success, threat-under-loom, wall-clock sec vs K. Compress: MC↑, T0↓, sim_sec vs K.
Bonferroni over the family of THINK-vs-SETTLE and THINK-vs-RANDTGT contrasts at each K where we claim a win (prereg primary: K=4 and K=8).

## Predictions
- T1: THINK success rises with K up to a peak (≤8), then flattens or drops (overthink).
- T2: THINK threat-under-loom falls with K vs K=0.
- T3: THINK > SETTLE and THINK > RANDTGT on success at the peak K (heading targets + CFS both needed).
- T4: Wall time grows ~linear in K; MC/T0 do not gain from K (planning is maze-specific).

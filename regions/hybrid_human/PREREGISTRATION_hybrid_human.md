# Preregistration: Fly–human hybrid (H01 cortex graft)
Written **BEFORE** any hybrid_human result (smoke or full). Queued after mouse hybrid.

## Hard caveat (read this first)
This is **not** as reliable as the fly–mouse MICrONS graft.

| | Mouse hybrid | Human hybrid (this) |
|---|---|---|
| Source | MICrONS minnie65 v1412 proofread | Google/Harvard H01 (Shapson-Coe et al. Science 2024) |
| Proofread neurons in module | **1871 / 1871** | **8 / 2087** (public H01 proofread set ≈104 cells total) |
| Edges | Condensed proofread synapses | Auto c3 synapses (20210729 final set), soma-linked |
| Cell types / layers | AIBS metamodel + proofread | ML cell types + soma layer tags from `somas.csv` |

So: feasible for a controlled reservoir test; **not** grounds for “world’s first proofread human–fly transplant” claims. Report auto-segmentation / thin-slab limits in RESULTS.

Data locked in `data/SOURCE.json` and `data/human_module.npz`.
N=2087, nnz=14863, E=1736 / I=351.
Layers: L2/3=995, L4=492, L5=350, L6=250.

## Graft (legacy HUMAN_HYBRID / ER / SHUF)
- Human cortex module; E/I from cell type (INTERNEURON → negative outgoing).
- Legacy out-bridge: L5∪L2/3 → fly E-PG/FB; Win prefers L4.
- Scale human block to ρ=1, bridges σ=0.05, then whole matrix to global ρ=0.9.

## Layered graft
- **In:** fly CX / task drive → **human L4 only**
- **Out:** **human L5 only** → fly E-PG / FB
- RANDOM_LAYERED uses the **same** L4/L5 index sets and same bridge counts/σ; only internal wiring is ER matched to human nnz + pre signs.

## Arms
1. FLY
2. HUMAN_HYBRID — fly + real H01 module (legacy graft)
3. ER — same-size ER, matched density / E/I signs on sampled edges
4. SHUF — degree-preserving shuffle of human edges (signs kept)
5. HUMAN_LAYERED — real H01; in→L4; out←L5
6. RANDOM_LAYERED — ER add-on; identical L4/L5 indices + bridge layout
7. HUMAN_LAYERED+HEBB — HUMAN_LAYERED + Oja on **internal human↔human** edges only
8. RANDOM_LAYERED+HEBB — same Oja η on random add-on internals only

## Hebb (identical for both +HEBB arms)
Oja on existing internal add-on edges: `dw_ij += η · y_i · (x_j − y_i · w_ij)`, η=1e-4.
Wash+train only; bridges and fly weights frozen. Random and human get the **same** η.

## Tasks / stats
MC (lags 1–100), T0, NARMA-10, PC. **8 seeds.**
Bonferroni on prereg contrasts below. No maze in this prereg (compress only).

## Predictions (locked before smoke / full run)
- H1: HUMAN_HYBRID ≥ ER on MC (structure helps).
- H2: HUMAN_HYBRID ≥ SHUF on MC.
- H3: HUMAN_LAYERED beats RANDOM_LAYERED on MC **only if** L4-in/L5-out hierarchy carries usable structure beyond matched size/density; else indistinguishable after Bonferroni.
- H4: HUMAN_LAYERED+HEBB vs RANDOM_LAYERED+HEBB — equal Hebb, so a human win still attributes to wiring.
- H5: Expect weaker / noisier margins than mouse because almost none of the module is proofread.

## Contrasts (Bonferroni)
MC:
1. HUMAN_HYBRID vs ER, vs SHUF, vs FLY (×3)
2. HUMAN_LAYERED vs RANDOM_LAYERED, vs FLY (×2)
3. HUMAN_LAYERED+HEBB vs RANDOM_LAYERED+HEBB (×1)
Secondary: same arm pairs on T0 / NARMA / PC (exploratory; not primary α spend).

# Preregistration: Fly–mouse hybrid (MICrONS cortex graft)
Written BEFORE any hybrid result. Queued after myelination.
**Layered / Hebb arms added 2026-10-27 BEFORE any layered run (predictions locked here).**

## Data (must be real — no fakes)
Source: public MICrONS minnie65 proofread subset + synapses via CAVE (`minnie65_public` / caveclient) or published static exports (BossDB / microns-explorer).
Record in RESULTS: datastack version, materialization, n neurons, n synapses, download date.
If credentials block public access and no static export works, **stop and report** — do not synthesize a pretend connectome.

Actual source locked in `data/SOURCE.json`: Zenodo 10.5281/zenodo.16744240 (v1412), N=1871, E=1680 / I=191.
Layer counts in module: L2/3=557, **L4=542**, L5=301, L6=392, unk=79. L4 is present → no L2/3 fallback needed for input gateway.

## Graft (legacy HYBRID / ER / SHUF)
- Build cortex module from proofread mouse neurons; E/I signs from cell types (inhibitory types → negative outgoing weights).
- Obs → mouse L4-like; mouse L5/L2-3 → fly E-PG/FB (legacy arm mixes L5+L2/3 out).
- Scale mouse block to ρ=1, then whole matrix to global ρ=0.9.

## Layered graft (new)
- **Input gateway:** fly CX / task drive → **mouse L4 only** (σ=0.05 bridges + Win mask).
- **Output:** **mouse L5 only** → fly E-PG / FB (σ=0.05). Not L2/3.
- Mouse block scaled to ρ=1, then global ρ=0.9.
- If L4 were missing: fall back to L2/3 and note in RESULTS (not needed here).

## Arms
1. FLY — fly CX/circuit alone
2. HYBRID — fly + real mouse module (legacy graft: L5∪L2/3 out)
3. ER — same-size Erdős–Rényi module, matched density / E/I fraction
4. SHUF — degree-preserving shuffle of mouse edges (signs preserved)
5. **HYBRID_LAYERED** — real mouse; in→L4 only; out←L5 only
6. **RANDOM_LAYERED** — ER add-on with **identical** L4/L5 index sets and bridge counts/scaling (fair hierarchy control)
7. **HYBRID_LAYERED+HEBB** — same as HYBRID_LAYERED + small Oja Hebbian on **internal mouse↔mouse synapses only**
8. **RANDOM_LAYERED+HEBB** — same Hebb rule/rate on the random add-on’s internal synapses only

## Hebb rule (identical for both +HEBB arms)
- Online Oja on existing internal add-on edges only: `dw_ij += η · y_i · (x_j − y_i · w_ij)` with η=1e-4.
- Applied during wash+train (compress) or train episodes (maze); frozen at eval/test.
- Bridges and fly weights are never plastic. Random and mouse get the **same** η.

## Tasks
MC (lags 1–100), T0, NARMA-10, PC (heading ring), EMPN-v2-style maze.
8 seeds. Bonferroni on prereg contrasts below.

## Predictions (locked before layered smoke / full run)
- H1: HYBRID ≥ ER on MC (legacy; real cortex structure helps).
- H2: HYBRID ≥ SHUF on MC (degree sequence alone insufficient).
- H3: Maze — exploratory; hybrid may or may not beat FLY/ER.
- **H4 (layered):** HYBRID_LAYERED beats RANDOM_LAYERED on MC **only if** mammalian layer hierarchy (L4 in / L5 out) carries usable structure beyond matched size/density. If hierarchy is irrelevant under this reservoir setup, means will be statistically indistinguishable after Bonferroni.
- **H5 (Hebb):** HYBRID_LAYERED+HEBB vs RANDOM_LAYERED+HEBB — Hebb is applied equally, so a win for mouse+Hebb over random+Hebb still attributes to wiring (not to the learning rule). Expect Hebb to help both slightly or neither; do **not** claim plasticity proves hierarchy unless the mouse margin over random grows under Hebb.
- **H6:** Primary layered contrast for “news”: HYBRID_LAYERED vs RANDOM_LAYERED on MC (Bonferroni with HYBRID_LAYERED vs FLY). Secondary: same under +HEBB.

## Contrasts (Bonferroni)
Compress MC:
1. HYBRID vs ER, HYBRID vs SHUF, HYBRID vs FLY (×3, legacy)
2. HYBRID_LAYERED vs RANDOM_LAYERED, HYBRID_LAYERED vs FLY (×2)
3. HYBRID_LAYERED+HEBB vs RANDOM_LAYERED+HEBB (×1)
Maze success: same arm pairs.

## Density-matched control (added 2026-10-27, before full run)

Lesson from human smoke: sparse `|N(0,1)|` ER after ρ=1 gets a fat mid-spectrum and can beat real wiring on MC; dense ER collapses.

- `RANDOM_LAYERED` — nnz-matched ER with `|N(0,1)|` magnitudes (existing)
- `DENS_MATCHED_LAYERED` — same N + same nnz, but `|w|` **bootstrapped from real** `|W.data|` then signed (row convention). Isolates random topology from the randn-magnitude spectrum artifact.
- `DENS_MATCHED_LAYERED+HEBB` — equal Oja η on dens-matched internals

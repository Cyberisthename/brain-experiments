# Preregistration: Frankenstein multi-species graft

**Date:** 2026-10-27  
**Status:** preregistered before smoke / full run  
**Octopus:** **not included** — no public neuron-level optic-lobe (or other cephalopod) edge list without login (see `regions/hybrid_octo/data/SOURCE.json`).

## Modules (each scaled to block ρ=1 before bridges)

| Block | Source | N (approx) | dens |
|---|---|---|---|
| fly CX | EMPN circuit CX subgraph | 2875 | ~0.036 |
| mouse | MICrONS v1412 proofread module | 1871 | ~0.028 |
| human H01 | HF condensed edges, layer-stratified | 2087 | ~0.0034 |

Schema: `W_*`, `root_id`, `sign`, `layer`, `cell_type` (same as hybrid modules).

Fly CX L4/L5: index split (first 400 → L4-like in, next 300 → L5-like out) — CX has no cortical layers; flagged `cx_layer_proxy=true`.

## Graft

- Block-diagonal `[mouse | human | fly_CX]`, each ρ=1.
- Bridges σ=0.05: for mouse and human, L4 ← fly_CX targets, L5 → fly_CX targets (EPG/FB-like = fly L5 proxy ∪ random CX).
- No mouse↔human bridges (hub = fly CX).
- Global spectral radius ρ=0.9 after stacking.

## Arms

1. **FLY_CX** — fly CX alone (scaled)
2. **FRANKEN** — real mouse + real human + real fly CX
3. **ER** — same-size nnz-matched `|N(0,1)|` ER **per block** (known sparse-fat-spectrum bias)
4. **SHUF** — degree-shuffle per block
5. **DENS_MATCHED** — per block: same N+nnz, `|w|` bootstrapped from that block’s real `|W.data|` (density + magnitude matched; isolates topology)
6. **BEST_OF** — per block independently choose `{real, dens_matched}` by mean MC on **tuning seeds {90,91,92}** (never test seeds 0–7); freeze choices; run test seeds with frozen mix
7. **FRANKEN+HEBB** / **DENS_MATCHED+HEBB** — equal Oja η=1e-4 on mouse+human blocks only during wash+train (fly CX frozen)

## Tasks / seeds / stats

- Tasks: MC, T0 (NARMA/PC if time; maze out of scope)
- Test seeds: 0–7; Bonferroni on primary contrasts ×2 (MC, T0): FRANKEN vs ER, vs SHUF, vs DENS_MATCHED, vs FLY_CX, vs BEST_OF
- Leak 0.2; drive = union of mouse L4 + human L4 + fly L4-proxy

## Hypotheses

- **H1:** FRANKEN > FLY_CX on MC (add-ons help).
- **H2:** FRANKEN beats ER / SHUF / DENS_MATCHED on MC **only if** real multi-species wiring carries structure beyond density+weight-matched random. (Human-alone lesson: sparse `|randn|` ER can fake-win — DENS_MATCHED is the fair control.)
- **H3:** BEST_OF ≥ FRANKEN and ≥ DENS_MATCHED (oracle per-block pick helps; if BEST_OF ≈ all-dens, real wiring is not buying MC).

## Smoke

1–3 seeds, MC (+ T0 if cheap), before queueing full 8-seed.

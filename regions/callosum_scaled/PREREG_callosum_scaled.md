# PREREG: callosum_scaled

**Frozen at:** 2026-11-02 (America/New_York), **before** any held-out eval seed is scored.  
**Path:** `regions/callosum_scaled/`  
**Follows:** `CALLOSUM_RESEARCH_2.md` §4; harness patterns from `realistic_callosum` + `human_wholebrain` (incl. NaN-safe NARMA from `human_wholebrain/FIX_NOTES.md`).  
**Data:** local only under `connectomes/` — **never push** `.mat` / edgelists / microstructure volumes.

## Question

At larger reservoir size, with **real fiber-length / myelin delays** and **tasks that require L↔R transfer**, does a **real human structural connectome prior** beat a **degree-preserving shuffle that keeps the same delay multiset**?

## Data (local)

- Griffa et al. 2019 scale-68 group SC + FC + Lausanne consensus fiber lengths:  
  `connectomes/human/griffa70/griffa70_scale68_whole.npz` (Zenodo 2872624, CC BY).
- Delay priors: fiber-length quantile bins + Caminiti (2013) sector velocity modulation + optional callosal FA gain from `connectomes/human/microstructure/delay_prior_suggestions.json`.
- Vohryzek 54 CTRL (Zenodo 3758534) is on disk as HDF5; **not required** for this run — Griffa FL already matches the Vohryzek CTRL FL inter/intra ≈1.8× pattern (research note 2). Using Griffa keeps the prior identical to `human_wholebrain`.

## Network

- Lausanne **68** parcels (34 LH + 34 RH).
- bio2art-style expansion: `N_LOCAL` units per parcel; actual `N = 68 × N_LOCAL`.
- Target sizes and exact `N_LOCAL` / `N`:

| label | N_LOCAL | N (actual) |
|------:|--------:|-----------:|
| 512   | 8       | **544**    |
| 1024  | 15      | **1020**   |
| 2048  | 30      | **2040**   |

- Within-parcel: all-to-all instant (`mask_intra`).
- Cross-parcel: delayed `mask_inter` with softplus×sign (OscRNN family).
- **Default readout:** left-hemisphere units only (except TRANSFER, which reads **LH motor parcels only**).
- OscRNN: leak, Adam, BPTT — same family as `human_wholebrain`.

## Arms

| Arm | Wiring | Delays |
|-----|--------|--------|
| **REAL_SC** | Consensus Griffa SC topology + group weights; FC signs (Dale 80/20 if NaN) | Fiber-length bins × Caminiti sector velocity × callosal FA prior |
| **DEGREE_SHUFFLE** | Maslov–Sneppen rewire of region graph (degree-preserving); weights/signs/lengths **reassigned from the same pools** so the **delay multiset matches REAL_SC** | Same delay rule on reassigned lengths |
| **WEIGHT_SHUFFLE** | Topology fixed; permute nonzero weights/signs/lengths among existing edges | Same delay rule |
| **SPARSE_BRIDGE** | Two-hemi modular ER (matched within-hemi dens) + random sparse callosal bridge matched to real callosal edge **count** | Length-like schedule (short intra / long callosal) |
| **NO_BRIDGE** | Within-hemi only (callosal edges removed from REAL topo) | Intra delays only |
| **DENSE** | Within-hemi as REAL density; **full bipartite** callosal bridge | Fixed mid delay on callosal |
| **AGCC** *(exploratory, not primary)* | NO_BRIDGE + boosted within-hemi density (~1.5× intra edges; Owen-inspired compensation) | Intra delays |

Primary “does the **map** matter?” contrast: **REAL_SC vs DEGREE_SHUFFLE** (matched delays) on the transfer task at the **largest completed N**.

## Tasks

| Task | Metric | Better | Bridge load |
|------|--------|--------|-------------|
| XOR_HARD | accuracy | ↑ | Hemispheric split XOR, d=12, left readout |
| CHAR | CE on Alice | ↓ | Left-driven LM, left readout |
| NARMA | NRMSE | ↓ | Left-driven; **NaN-safe** resampling of unstable targets (`FIX_NOTES.md`) |
| **TRANSFER** | accuracy | ↑ | **Poffenberger-like:** inputs → **RH visual** parcels only; readout → **LH motor** parcels only; delayed bit identity (d = max(8, DMAX+2)). Must cross callosum. |
| **SPLIT_BIND** | accuracy | ↑ | Bimanual-like: independent L and R bit streams; target = XOR(L,R) at lag d=8; left-only readout (Kang posterior-CC motivation). |
| **DELAYED_MATCH** | accuracy | ↑ | Cue bit → RH; after lag `D_MATCH=12` probe → LH; target = match(cue, probe); left readout. |

## Seeds / hparams

- Held-out eval seeds: **2300–2307** (n=8). Never used for tuning.
- Smoke / wiring-check only on seed **91** (not counted).
- Frozen hparams (from realistic_callosum / human_wholebrain), written to `out/hparams.json` before eval:

```
lr=5e-3, leak=0.4, epochs=30, steps_per_epoch=12,
batch_xor=32, batch_char=32, batch_narma=24, batch_xfer=32,
seq_xor=40, seq_char=40, seq_narma=56, seq_xfer=40,
char_train_chars=30000, wd=0.0, emb_dim=32, eval_every=3, DMAX=5
```

## Stats (declared up front)

- Paired Wilcoxon signed-rank across seeds, per (task, N).
- **Primary family** (Bonferroni over m tests below; α_family=0.05):
  1. REAL_SC > DEGREE_SHUFFLE on **TRANSFER** at **largest N** in the completed set
  2. REAL_SC > DEGREE_SHUFFLE on **SPLIT_BIND** at that same N
  3. REAL_SC > DEGREE_SHUFFLE on **DELAYED_MATCH** at that same N
  4. REAL_SC > NO_BRIDGE on **TRANSFER** at that same N
  5. REAL_SC > DEGREE_SHUFFLE on **NARMA** at that same N (continuity with human_wholebrain timing win)
  6. REAL_SC > WEIGHT_SHUFFLE on **TRANSFER** at that same N

  → m=6, α=0.05/6 ≈ 0.00833.

- **Primary claim (SUPPORTED):** contrast (1) passes after Bonferroni at the largest completed N.  
- **PARTIAL:** delays/sparse-bridge win but topography does not (REAL ≈ DEGREE_SHUFFLE; REAL or SPARSE ≫ NO_BRIDGE).  
- **NOT SUPPORTED:** otherwise — report honestly; do not quietly redefine.

- Secondary (descriptive / raw p): means tables at every N; XOR_HARD / CHAR; SPARSE_BRIDGE vs REAL; DENSE; AGCC exploratory.

## Runtime budget & staged N (CPU box, ~24 h)

Benchmark (dense OscRNN, 45 ep × 20 steps): N≈512 ~2.7–4 min/job; N≈1024 ~4× that; N≈2048 ≫.

**Stage plan (locked before eval; amend below if smoke forces it):**

1. **Stage A (must finish):** all 6 primary arms (+ AGCC exploratory) × 6 tasks × seeds 2300–2307 at **N_label=512** (N=544).  
   Jobs ≈ 7 × 6 × 8 = 336 → target wall ≲ 12 h @ 2 workers.
2. **Stage B:** same matrix at **N_label=1024** (N=1020) **only if** projected remaining wall after Stage A ≤ 12 h; else skip and amend.
3. **Stage C (N_label=2048):** **deferred** by default. Run **TRANSFER only × 6 primary arms × 8 seeds** only if ≥8 h remain after A+B.

Primary claim uses the **largest N for which all primary arms × TRANSFER × 8 seeds finished**.

Workers ≤ 2; `nohup`; resume-safe via `out/runs.jsonl` keyed by `(arm, task, seed, n_label)`.

## Honesty

Null allowed. Commit code + PREREG + RESULTS only. Connectome binaries stay gitignored.


## Amendments

### 2026-11-02 (pre-eval) — DEGREE_SHUFFLE class-wise rewire
Clarified DEGREE_SHUFFLE: Maslov–Sneppen is run **separately** on intra-LH, intra-RH, and callosal edge sets (then pools reassigned within class). Full-graph MS was checked at wiring-time and inflated callosal nnz ~3× by destroying hemi modularity, which would confound the topography contrast. Class-wise MS matches research note 2 (“shuffle of inter + intra”) and keeps bridge count matched to REAL_SC. **No eval seeds scored yet.**

### 2026-11-02 (pre-eval) — sparse unit expansion + sparse OscRNN
Full n_local×n_local block expansion at N_label=512 (N=544) made nnz≈86k and
dense/softplus matmuls ~10+ min/job (Stage A ≫ 24 h). **Before any eval seed:**
(1) OscRNN uses sparse edge lists (softplus only on existing edges);
(2) each region edge expands to **n_local** random directed unit edges (bio2art-style
sparse), not a full block; within-parcel stays all-to-all. Topology/weights/delays
at the **region** level are unchanged; only the unit lift is sparsified. Timing
probe must confirm Stage A ≲ 24 h before `--eval`.

### 2026-11-02 (pre-eval) — train budget cut for 24 h wall
Sparse expansion still left ~5–10 min/job at 45×20 (Python BPTT loop). Cut
frozen train budget to epochs=30, steps_per_epoch=12, smaller batches/seq, and
eval_every=3 **before any held-out seed**. Same across all arms. Primary stats
unchanged. Stage B/C still conditional on remaining wall.

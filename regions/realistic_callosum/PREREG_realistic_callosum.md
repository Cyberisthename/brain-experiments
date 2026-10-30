# PREREGISTRATION — realistic_callosum

**Frozen at:** 2026-10-30 09:51:38 EDT (America/New_York)  
**Repo path:** `regions/realistic_callosum/`  
**Status:** preregistered **before** any evaluation scores on held-out seeds 2100–2111.  
**Follows:** `regions/oscillation_gating/` (old CALLOSUM tied RAND_BRIDGE; XOR at ceiling because readout saw both hemispheres).  
**Data notes:** `connectomes/human/griffa70/NOTES_callosum_data.md` (Griffa70 + Rosen–Halgren + Lausanne + OASIS).

## Question

Does a **real human callosum wiring prior** (mostly heterotopic, strong minority homotopic, FC-signed, myelin-style delays) beat (a) the old 100% homotopic `CALLOSUM`, (b) a weight/count-matched shuffle of that real prior, and (c) random / no / dense bridges — on a bridge-required hemispheric XOR, char LM, and NARMA-10?

## Why the old callosum test was uninformative

1. Bridge pattern was **100% homotopic**; real SC is **~85–99% heterotopic** by edge count (Griffa70 scale68 majority ≈15.7% homo; Rosen–Halgren ≈0.6% homo).
2. Split-input XOR still hit **1.0 accuracy with `NO_BRIDGE`** because the linear readout saw **both** hemispheres — the bridge was not required.

## Fix (task)

- **Left-only readout:** `readout` linear layer only sees units in the LEFT half (`N/2`). Right-hemisphere info must cross the bridge.
- **XOR:** 2-channel hemispheric split (left←ch0, right←ch1); target = ch0 XOR ch1; **delayed** with `d=5` (mask from t≥d) so memory + bridging both matter. Metric = accuracy ↑.
- CHAR / NARMA unchanged except left-only readout + left-driven input (right half no direct drive), same as callosum CHAR/NARMA before.

## Model

Same OscRNN family as `oscillation_gating`:
- \(N=160\), two hemispheres of 80; intra-hemi dens≈0.12; no oscillation gate on these arms.
- Trainable masked weights; Adam + BPTT; workers ≤ 2.
- Bridge delays in `{1..5}`; intra delay 0.

## Arms (edge-matched on bridge count except DENSE / NO)

Bridge budget for matched arms: **`2 · N_HEMI = 160` directed** edges (same as old CALLOSUM).

| Arm | Bridge |
|-----|--------|
| `REAL_CALLOSUM` | Pattern from Griffa70 scale-68 **majority** interhemispheric SC (≥35/70 subjects). Upsample 34×34 parcels → 80×80 units by parcel→unit map. Sample 160 directed edges with probability ∝ SC weight (preserve ~15% homotopic mass). **Sign** from group FC on that parcel pair (+1 if FC≥0 else −1; expected ~97% +). **Delay:** hetero / weaker → larger delay (isotonic on hetero-flag + 1/weight). |
| `REAL_CALLOSUM_SHUFFLED` | Same edge **weights, signs, delay multiset, and count**; permute right-side targets (destroy spatial layout / homo identity). |
| `INDIVIDUAL_CALLOSUM` | Same construction but from **one of the 70** subjects’ SC/FC (subject = `(seed_eval − 2100) % 70`); report mean and sd across subjects/seeds. |
| `CALLOSUM` | Old: reciprocal **homotopic only** (left[i]↔right[i]). Fixed delay D=2. Unsigned (+). |
| `RAND_BRIDGE` | 160 random cross-hemi directed edges. D=2. Unsigned (+). |
| `NO_BRIDGE` | Zero cross edges. |
| `DENSE_BRIDGE` | All-to-all cross-hemi (higher nnz; matched only on N + train budget). D=2. Unsigned (+). |

Graph / parcel map seed: **7**. Intra-hemi masks shared across arms (seed 7+909).

## Hypotheses (primary family)

Paired Wilcoxon over seeds 2100–2111 (n=12). Family of **6** tests, Bonferroni α=0.05/6:

On **XOR accuracy ↑**:
1. `REAL_CALLOSUM` > `REAL_CALLOSUM_SHUFFLED`
2. `REAL_CALLOSUM` > `CALLOSUM`
3. `REAL_CALLOSUM` > `RAND_BRIDGE`
4. `REAL_CALLOSUM` > `NO_BRIDGE`

On **NARMA NRMSE ↓**:
5. `REAL_CALLOSUM` < `REAL_CALLOSUM_SHUFFLED`
6. `REAL_CALLOSUM` < `NO_BRIDGE`

**Supported** if REAL beats SHUFFLED on XOR (H1) **and** beats NO_BRIDGE on XOR (H4) after correction.  
H2/H3/H5/H6 are secondary within the family (still Bonferroni-corrected as listed).

## Secondary (descriptive, raw p)

- CHAR CE: REAL vs SHUFFLED / CALLOSUM / RAND / NO.
- INDIVIDUAL_CALLOSUM mean±sd vs REAL group (does person-level wiring help or just add variance?).
- DENSE_BRIDGE vs REAL (capacity upper bound; not edge-matched).
- Bridge required check: `NO_BRIDGE` XOR must be clearly below REAL (expect ~chance if left-only readout works).

## Hyperparameters

Frozen from oscillation_gating `out/hparams.json` (already tuned on unrelated seeds 90–91):  
`lr=0.005, leak=0.4, epochs=45, steps_per_epoch=20, batch_xor=48, batch_char=40, batch_narma=32, seq_xor=36, seq_char=48, seq_narma=64, emb_dim=32`.  
No re-tuning on 2100–2111.

## Evaluation

- Seeds **2100–2111** (n=12), one run per (arm, seed, task).
- Tasks: XOR, CHAR, NARMA.
- ≤2 CPU workers; resume via `out/runs.jsonl`.

## Analysis

`analyze_realistic_callosum.py` → `RESULTS_realistic_callosum.md` + `out/verdict.json`.

## Honesty

Null allowed. Data stay in `connectomes/` (gitignored). Commit code + RESULTS only.

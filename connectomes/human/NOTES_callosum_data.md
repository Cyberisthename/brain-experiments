# Callosum connectome notes (local data — do not push)

Written 2026-10-30 (America/New_York). Plain-English summary of what we downloaded and what we learned for the realistic callosum RNN test.

## Datasets on disk (`/workspace/flybrain/connectomes/human/`)

### 1. Griffa, Alemán-Gómez & Hagmann 2019 — 70 young adults (THIS FILE’S FOLDER)
- **Zenodo:** https://zenodo.org/records/2872624 — DOI `10.5281/zenodo.2872624`
- **License:** CC BY 4.0 (credit the authors; keep data local, never push the `.mat`)
- **File:** `Individual_Connectomes.mat` (669 MB; md5 `87ef63debc0cbe44afc1e6cf9ab9717e` matches Zenodo)
- **Who:** 70 healthy adults (age ~29 ± 9, 27 female), Lausanne CHUV, fully anonymized
- **What’s inside:** for each person, at 5 Lausanne cortical scales (**68 / 114 / 219 / 448 / 1000** regions):
  - **SC** = structural connectome (DSI tractography, fiber density, length- and area-normalized)
  - **FC** = resting-state fMRI correlation (can be positive or negative)
- Note: Zenodo text mentions 83/129/… (with subcortical). This file is the **cortical-only** cousins (68 = 34 left + 34 right, etc.), same family as the Lausanne consensus matrices.

**What we measured (interhemispheric = corpus callosum edges):**

| scale | frac of SC-inter edges that are **homotopic** (mean ± sd across 70) | range | homotopic mean weight / heterotopic mean weight |
|------:|-------------------------------------------------------------------:|------:|-----------------------------------------------:|
| 68    | **0.124 ± 0.014** (range 0.095–0.161) | ~88% hetero | **~10×** stronger |
| 114   | 0.196 ± 0.022 | ~80% | ~6× |
| 219   | 0.265 ± 0.025 | ~73% | ~4× |
| 448   | 0.333 ± 0.027 | ~67% | ~3× |
| 1000  | 0.396 ± 0.030 | ~60% | ~2.4× |

- Homotopic = same region name left↔right. Heterotopic = different regions across hemispheres.
- Fine scales count more same-name subparcels as “homotopic,” so the homotopic **fraction rises** with scale, but **heterotopic still wins** at every scale.
- Person-to-person spread at scale 68 is modest (homotopic fraction stays in a ~10–16% band).

**Structure–function on callosal SC edges (scale 68):**
- Of SC-present cross-hemisphere links, FC is **positive ~87%** of the time, negative ~12% (mean FC ≈ +0.37).
- SC weight and FC magnitude agree moderately (Spearman ≈ **0.32 ± 0.10** across people).
- So for a signed bridge: mostly **excitatory**, with a small inhibitory minority.

**Group “majority” bridge (scale 68, edge present in ≥35/70 people):**
- 115 undirected callosal edges: **18 homotopic (15.7%)**, 97 heterotopic (84.3%)
- Homotopic still **~7× stronger** SC weight
- FC on those edges: **112 positive / 3 negative → ~97% excitatory**
- Homotopic FC mean ≈ 0.75; heterotopic ≈ 0.36 (both positive on average)
- Saved as `griffa70_scale68_bridge.npz` (+ `griffa70_scale68_callosum.npz` per-subject) for the RNN test

### 2. Rosen & Halgren — HCP-MMP 360 group SC
- Zenodo 10150880; `rosen_halgren_averageConnectivity_Fpt.csv`
- 360 regions (180 L + 180 R). Interhemispheric edges are **~99% heterotopic by count**; homotopic are fewer but stronger (top-1% inter edges ≈15% homotopic).
- Matches Griffa’s story: hetero by count, homo by strength.

### 3. Lausanne Consensus (group average, same atlas family)
- Zenodo 3067849; `Consensus_Connectomes.mat` + `lausanne/scale*_sc.npy`
- Same 5 cortical scales. Homotopic fraction of inter edges rises from ~0.11 (scale 68) to ~0.45 (scale 1000) — same trend as Griffa individuals.

### 4. Braingraph OASIS-3 Scale5 average
- ~1058 nodes; inter edges **~99% heterotopic**. Independent (non-HCP) confirmation.

## Big takeaways for our brain modules

1. **A “realistic callosum” is mostly heterotopic** (~85–99% of cross links), not mirror-only. Our old `CALLOSUM` arm was 100% homotopic — that was wrong.
2. **Homotopic links are the strong ones** (several× the typical hetero weight). Keep a minority of strong mirror links.
3. **Signs from FC:** nearly all callosal SC edges sit under **positive** FC → treat bridge as mostly excitatory (~3–12% inhibitory is enough).
4. **Delays:** hetero / longer-range links should be slower (myelin-style); short/homotopic faster.
5. **People differ a bit** (Griffa 70): worth an `INDIVIDUAL_CALLOSUM` arm that samples several people’s maps and reports the spread.
6. **Ceiling problem:** old split-XOR hit 1.0 even with `NO_BRIDGE` because the readout saw both halves. New test must **read out from one hemisphere only** (and/or use delayed cross-hemisphere XOR) so the bridge is actually required.

## Files in this folder
- `Individual_Connectomes.mat` — raw (local only)
- `griffa70_callosum_stats.json` — numbers above
- `griffa70_scale68_callosum.npz` — reordered L|R SC/FC group + per-subject
- `griffa70_scale68_bridge.npz` — majority mask, homo mask, mean SC/FC inter blocks
- `NOTES_callosum_data.md` — this file

Credit: Griffa, Alemán-Gómez, Hagmann (2019); Rosen & Halgren; Lausanne consensus / Hagmann lab; Braingraph OASIS.

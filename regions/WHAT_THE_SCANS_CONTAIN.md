# What the local human scans / connectomes contain

Inventory of every dataset under `connectomes/human/` (local only — **never push** these binaries).
Numbers measured on disk 2026-10-30 (America/New_York). No FA / MD microstructure maps are present in any of these files.

## Quick plain-English map

| Dataset | What it is | White-matter | Grey / regions | Functional | Subjects / scale |
|---|---|---|---|---|---|
| **Griffa70** | 70 people’s brain wiring + activity | Fiber **density** SC (not FA) | Lausanne cortex lh/rh | rs-fMRI **Pearson FC** | 70 × scales 68/114/219/448/1000 |
| **Lausanne Consensus** | Group-average multi-scale | Weighted SC + **fiber length (mm)** | Same Lausanne + RSN labels | Group FC | Consensus of ~70 young adults; 5 scales |
| **Rosen & Halgren** | HCP-MMP group SC | Log fractional connection probability **Fpt** | 360 MMP parcels (180L+180R) | none | Group (~1000 HCP); 360×360 |
| **Braingraph OASIS Scale5** | Averaged OASIS graph | Fiber-**count** edge weights | 1058 nodes (527L+527R+4 central) | none | Group average |
| **Yeh HCP1065 xlsx** | Tract → region table | Named tracts (AF, cingulum, CST…) loadings | Desikan / MMP region rows | none | HCP1065 atlas; **not** region×region |
| Derived NPZs | Bridge / whole-brain exports we built | Copied SC + lengths + signs | scale-68 | FC signs | For experiments only |

**None of the local matrices store FA, MD, or other voxel microstructure.** “Myelin” in our experiments means **using fiber length as a delay prior**, not a measured myelin map.

---

## 1. Griffa / Alemán-Gómez / Hagmann 2019 — `griffa70/`

- **Source:** https://zenodo.org/records/2872624 (DOI 10.5281/zenodo.2872624), CC BY 4.0  
- **Raw:** `Individual_Connectomes.mat` (~701 MB) — `connMatrices.SC` / `connMatrices.FC`  
- **Derived:** `griffa70_scale68_*.npz`, `griffa70_callosum_stats.json`, notes  

### White matter
- **SC = fiber density** (streamline-based weight), cortical Lausanne only (no subcortical in this release).  
- **No fiber lengths inside the .mat** — lengths come from the Lausanne Consensus file (same atlas).  
- Scale-68 group SC density (weight>0): **~0.70** (many weak edges); consensus binary dens ~0.28.  
- Callosum (inter-hemi) at scale 68, per subject (undirected):  
  - ~154 ± 23 inter edges; **homotopic fraction 0.124 ± 0.014** (hetero ~87.6%)  
  - Mean SC weight homo / hetero ≈ **10×** (majority-bridge group ratio ≈ **7.4×**)  
- Majority bridge (≥35/70 subjects, scale 68): **115** undirected inter edges, **18 homo / 97 hetero** (15.7% homo).

### Grey matter / regions
- 5 cortical scales: **68, 114, 219, 448, 1000** parcels with lh/rh labels and region names.

### Functional
- Resting-state **Pearson FC** per subject, same parcels.  
- On SC-present callosal edges: **~87% positive** FC (majority bridge **~97%** excitatory).  
- Mean FC on SC-inter ≈ 0.37 (scale 68); SC–FC Spearman on inter ≈ 0.32.

### Fiber length (paired Consensus FL on same topology)
- Mean callosal FL ≈ **122 mm** (median ~111)  
- Mean within-hemisphere FL ≈ **71 mm** (median ~61)  
→ Callosal paths are ~1.7× longer on average than within-hemi paths at scale 68.

---

## 2. Lausanne Consensus — `Consensus_Connectomes.mat` + `lausanne/`

- **Source:** https://zenodo.org/records/3067849 , CC BY 4.0  
- **Matrices per scale (Readme):** (1) weighted SC adj (2) **fiber length** (3) FC (4) coordinates (5) ROI name / short / RSN / hemisphere  

| Scale | Regions | SC dens (weight>0) | nnz | Mean FL (mm) | Max FL (mm) |
|------:|--------:|-------------------:|----:|-------------:|------------:|
| 68 | 68 | 0.277 | 1282 | 84.1 | 380 |
| 114 | 114 | 0.201 | 2612 | 79.6 | 387 |
| 219 | 219 | 0.119 | 5722 | 74.5 | 392 |
| 448 | 448 | 0.058 | 11728 | 69.0 | 393 |
| 1000 | 1000 | 0.024 | 24302 | 64.6 | 395 |

Exports: `lausanne/scale68_sc.npy`, `scale68_labels.csv`, `scale1000_*`.

### Strongest scale-68 SC edges (examples)
- `superiorfrontal(lh)–superiorfrontal(rh)` — **homotopic callosal**, w≈0.123, FL≈**83 mm**  
- `posteriorcingulate–paracentral` (within hemi), w≈0.16, FL≈30 mm  
- `pericalcarine–cuneus` (within), w≈0.073, FL≈15 mm  

Inter-hemi edges at scale 68: **167 undirected** (334 directed); frac homotopic of inter ≈ **0.11** (see README table for all scales).

---

## 3. Rosen & Halgren — HCP-MMP 360

- **Files:** `rosen_halgren_averageConnectivity_Fpt.csv` (~1.0 MB), `rosen_halgren_parcelOrder.mat` (v7.3 HDF5)  
- **Source:** https://zenodo.org/records/10150880 , CC BY 4.0 (HCP-derived; acknowledge HCP)  

### White matter
- **Fpt = log fractional connection probability** (more positive = stronger; values roughly −6.2 … −0.84).  
- Finite dens ≈ **0.997**; almost fully dense probabilistic matrix (not a sparse streamline count).  
- Inter finite edges 64800; **180/180** homotopic L↔R pairs present.  
- Mean Fpt: **homotopic −3.25** (stronger) vs **heterotopic inter −4.26** vs overall −3.90.  
- By edge count, inter links are **~99.4% heterotopic**; strength still concentrates on homotopic callosal links (top-1% inter ≈15% homo — see README).

### Grey matter
- 360 HCP-MMP1 parcels (180 left + 180 right). No FC in this download.

---

## 4. Braingraph OASIS-3 Averaged Scale5

- **Files:** `braingraph_OASIS_Averaged_Scale5.graphml` (~34 MB), `braingraph_oasis_scale5.npz`  
- **Source:** https://braingraph.org (OASIS-derived; not HCP checkbox)  

### Contents
- **1058 nodes:** hemi codes {0: 4 central, 1: 527 left, 2: 527 right}  
- **327722** directed weighted edges; weights = fiber **counts**  
- Inter-hemi edge fraction ≈ **0.27**  
- Strongest edges in the NPZ are **subcortical–subcortical** (weights ~500–1070); cortical callosal edges are weaker and numerous.  
- No FL, FA, or FC in this export.

---

## 5. Yeh HCP1065 tract-to-region tables

- **Files:** `tract_to_region_connectome.xlsx`, `tract_to_region_connectome_MMP.xlsx`, `abbreviation.xlsx`  
- **Source:** https://github.com/data-others/atlas/releases/tag/hcp1065 ; https://brain.labsolver.org/hcp_trk_atlas.html  
- **License:** CC BY-SA 4.0 + HCP acknowledgment  

### What it is
- **Region × named tract** loadings (not region×region SC).  
- MMP sheet: **179 regions × 26 tracts × L/R** (52 numeric columns).  
- Header: side (LEFT/RIGHT) × tract code (FAT, AF, MdLF, SLF_I/II/III, C_FP, C_FPH, C_PHP, C_PH, ILF, IFOF, UF, VOF, CST, CBT, CStr_*, CTh_*, OR, F/Fornix, …).

### Abbreviation sheet lists (including commissural)
- **Association:** AF (arcuate), cingulum splits (C_FP, C_FPH, C_PHP, C_PH, C_PO), FAT, IFOF, ILF, MdLF, SLF1–3, UF, VOF, …  
- **Projection:** CST, OR, thalamic/striatal radiations, fornix, …  
- **Commissural:** `AC` Anterior_Commissure, **`CC` Corpus_Callosum**  
- **Cerebellum / cranial nerves** also listed  

**Important:** the MMP/Desikan tract-region sheets’ columns are association/projection tracts only — **no CC column** in the numeric tables (CC exists in the abbreviation list and as a volume mask on Zenodo 3627772, not as a matrix here).

### Strongest MMP tract columns by summed region loading (this file)
ILF (R/L), AF (L), IFOF (R/L), SLF_II, corticostriatal superior, thalamic superior, SLF_III, cingulum frontal-parietal, …

---

## 6. Derived experiment files (ours)

| File | Role |
|---|---|
| `griffa70/griffa70_scale68_bridge.npz` | Callosal bridge prior for `realistic_callosum` |
| `griffa70/griffa70_scale68_callosum.npz` | Per-subject inter matrices |
| `griffa70/griffa70_scale68_whole.npz` | Group SC + mean FC + Consensus FL for `human_wholebrain` |
| `griffa70/griffa70_callosum_stats.json` | Precomputed homo/hetero/FC stats |

These are thin numpy views of the sources above — still **local-only**.

---

## What we do *not* have locally

- Voxelwise **FA / MD / T1w/T2w myelin** maps  
- Full HCP-YA BALSA subject packs (TB-scale; open-access terms accepted, not downloaded)  
- A ready-made “callosum-only” streamline file with FA  
- Yeh **CC** as a region×region or region×tract numeric column (abbr only)

## How this feeds the brain tests

- **Topology + weights:** Griffa/Consensus SC fiber density  
- **Delays (“myelin” arm):** Consensus **fiber length** → delay bins (callosal ~122 mm vs within ~71 mm)  
- **E/I signs:** Griffa FC sign (~97% + on majority callosal bridge)  
- **Named tracts (Yeh):** documentation / future tract-masked arms — not yet wired into OscRNN  
- **Rosen 360:** denser probabilistic prior if we scale past 68 parcels  

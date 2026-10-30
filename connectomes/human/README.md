# Human connectomes (local only — do not push data)

Small open region×region structural connectomes with interhemispheric (corpus callosum) wiring for `/workspace/flybrain` experiments.

## Best picks (verified)

### 1. Rosen & Halgren — HCP-MMP 360 probabilistic SC (RECOMMENDED)
- Files: `rosen_halgren_averageConnectivity_Fpt.csv` (~1.0 MB), `rosen_halgren_parcelOrder.mat`
- URL: https://zenodo.org/records/10150880 (`averageConnectivity_Fpt.csv`)
- License: CC BY 4.0; derived from HCP (acknowledge HCP open-access terms; no interactive accept needed for Zenodo)
- Contents: group-average log fractional probability (Fpt) 360×360 matrix (180 L + 180 R, HCP-MMP1)
- Verification (undirected finite edges; L=0..179, R=180..359):
  - regions: 360 (180 L + 180 R)
  - intra: 32220 | inter: 32400
  - homotopic finite: 180/180 | heterotopic inter: 32220
  - frac of INTER that are homotopic: **0.0056** | heterotopic: **0.9944**
  - mean Fpt: homotopic **−3.25** (stronger) vs heterotopic inter **−4.26** vs intra L **−3.54**
  - among top 1% inter by Fpt: **15.4%** are homotopic (strength concentrates on homotopic callosal links)

### 2. Lausanne Consensus — multi-scale SC (70 young adults)
- File: `Consensus_Connectomes.mat` (~10.7 MB); exports in `lausanne/scale68_*`, `lausanne/scale1000_*`
- URL: https://zenodo.org/records/3067849
- License: CC BY 4.0
- Contents: 5 scales × (weighted SC, fiber length, FC, coords, ROI labels with **lh/rh**)
- Verification (weight>0 undirected; homotopic = same region name across lh/rh):

| scale | n (lh+rh) | inter | homo | hetero | frac homo of inter |
|------:|----------:|------:|-----:|-------:|-------------------:|
| 68    | 34+34     | 167   | 18   | 149    | 0.108 |
| 114   | 57+57     | 282   | 54   | 228    | 0.191 |
| 219   | 111+108   | 564   | 162  | 402    | 0.287 |
| 448   | 225+223   | 1099  | 429  | 670    | 0.390 |
| 1000  | 499+501   | 2288  | 1037 | 1251   | 0.453 |

### 3. Braingraph OASIS-3 group-averaged Scale5
- Files: `braingraph_OASIS_Averaged_Scale5.graphml` (33 MB; server file missing `</graphml>` close tag), `braingraph_oasis_scale5.npz`
- URL: https://braingraph.org/static/Averaged_graph_Scale5.graphml (listed at https://braingraph.org/cms/download-pit-group-connectomes/ under OASIS — no HCP checkbox)
- License: site cites papers; OASIS-derived (not HCP). Keep local.
- Contents: 1058-node averaged graph (527 left, 527 right, 4 central)
- Verification (undirected, no self-loops; fiber-count weight):
  - undirected edges: 163861 | intra 120296 | inter 40923
  - homotopic (same base name L–R): 288 | heterotopic inter: 40635
  - frac of INTER homotopic: **0.007** | heterotopic: **0.993**


### 4. Griffa / Alemán-Gómez / Hagmann 2019 — 70 individual SC+FC (NEW)
- Folder: `griffa70/` — `Individual_Connectomes.mat` (669 MB, md5 matches Zenodo), stats + bridge exports
- URL: https://zenodo.org/records/2872624 — DOI 10.5281/zenodo.2872624 — **CC BY 4.0**
- Contents: 70 young adults × 5 cortical Lausanne scales (68/114/219/448/1000) × (SC fiber density, rs-fMRI FC)
- Plain-English notes: `griffa70/NOTES_callosum_data.md` (also `NOTES_callosum_data.md` here)
- Scale-68 majority bridge (≥35/70): 115 undirected inter edges, **15.7% homotopic / 84.3% heterotopic**; homo ~7× stronger; FC on those edges **~97% positive** (excitatory prior)
- Per-subject homo fraction of SC-inter at scale 68: **0.124 ± 0.014** (range 0.095–0.161)
- **Local only — never push the `.mat`**

## Ancillary (not region×region adjacency)

### Yeh HCP1065 tract-to-region (downloaded)
- `tract_to_region_connectome_MMP.xlsx`, `tract_to_region_connectome.xlsx`, `abbreviation.xlsx`
- URL: https://github.com/data-others/atlas/releases/tag/hcp1065
- License: CC BY-SA 4.0 + HCP acknowledgment (https://brain.labsolver.org/hcp_trk_atlas.html)
- Shape: region × named tracts (not region×region). Abbreviation lists `CC` / Corpus_Callosum, but the MMP sheet columns are association/projection tracts only (FAT, AF, …, F=Fornix) — **no CC column**. Volume atlas on Zenodo 3627772 includes `Corpus_Callosum.nii.gz` (tract mask, not connectome matrix).

## Checked but not usable under constraints

| Source | Result |
|--------|--------|
| HCP842 Connectivity matrix Excel (brain.labsolver.org → Pitt SharePoint) | SharePoint redirects to **Authenticate.aspx** — skipped (no registration/login) |
| Brainnetome `BNA_matrix_binary_246x246.csv` | atlas.brainnetome.org / GitHub / jsDelivr 404 or HTML; NITRC returns login HTML |
| Wiley Brain Behav 2026 doi:10.1002/brb3.71306 | Abstract says atlas “will be freely available”; **no supplementary matrix URL found** this hunt |
| BALSA HCP-YA project files | Per-person TB-scale packs only (prior session); no small group-average SC |
| MICA-MICs OSF (j532r) | Metadata/license only at root; group SC not as a small standalone file; full derivatives are large |
| Braingraph HCP packs / Averaged_1015.graphml | Require interactive HCP data-use checkbox — skipped |
| CCmapping.org | Fetch returned HTTP 500 |

## Homotopic vs heterotopic takeaway for callosum modules
Real group SC is **overwhelmingly heterotopic by edge count** (~85–99% of interhemispheric edges), but **homotopic links are disproportionately strong** (Rosen–Halgren top-1% inter edges ≈15% homotopic; Griffa70 scale-68 majority homo weight ~7× hetero). A realistic callosum prior is ~12–16% homotopic by count at coarse scales (Griffa70 individuals 0.124±0.014 at scale 68), rising when same-name subparcels are counted at fine Lausanne scales, with heterotopic majority retained. FC on SC-present callosal edges is **mostly positive** (~87–97%), so a mostly-excitatory signed bridge is the right default.

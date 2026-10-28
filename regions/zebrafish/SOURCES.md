# Zebrafish connectome sources (research only — no module built)

**Date:** 2026-10-28  
**Question:** which public zebrafish datasets have a neuron↔neuron synapse/edge list usable as a graph?

Summary: **one clear whole-brain edge-list resource exists today (Fish1 / Petkova–Januszewski 2025 via CAVE).** Older EM volumes are image + partial reconstructions, not downloadable full edge lists. mapZebrain / FishExplorer are atlases / portals, not edge-list dumps.

---

## 1. Fish1 — Petkova, Januszewski et al. 2025 (Harvard / Google / Engert–Lichtman)

| Field | Value |
|---|---|
| **URL** | https://fish1-release.storage.googleapis.com/index.html |
| **Paper** | https://www.biorxiv.org/content/10.1101/2025.06.10.658982 (bioRxiv; also PMC12259123) |
| **Age / prep** | 7 dpf larval zebrafish, CLEM (confocal vglut2a / gad1b + ssEM) |
| **Neurons** | ~187,053 segmented somata (brain + anterior spinal cord + ganglia); >40,000 molecularly annotated (vglut2a+ ≈26.9k, gad1b+ ≈14.5k) |
| **Synapses** | ~30M axon→dendrite (paper: 29.5M ax→de + 9.5M ax→ax); ~21M with inferred E/I/mod polarity |
| **Edge list?** | **YES** — CAVE materialization table `synapses_axde_label` with `pre_pt_root_id` / `post_pt_root_id` (+ E/I tag). Soma table with lore ID ↔ root ID. |
| **Format** | CAVE / caveclient queries (datastack `fish1_v250915`). Not a single public flatfile dump of all 30M edges; API is query-oriented (bulk full-table download discouraged; contact data admins for mass export). Companion site: https://zebrafishconnectome.vercel.app/data |
| **Tutorials** | https://fish1-release.storage.googleapis.com/tutorials.html ; notebooks in https://github.com/VCG/cave-scripts |
| **License / policy** | Open-access resource; independent publication allowed; **citation required** (Petkova, Januszewski et al. 2025 bioRxiv). No co-authorship required for independent discoveries. Tokened CAVE access (Google account). Policy: https://fish1-release.storage.googleapis.com/data_policy.html |
| **Usable as reservoir graph?** | Best candidate. Needs CAVE token + subsetting (whole 30M is huge). E/I labels available for many synapses. |

---

## 2. Svara et al. 2022 — whole-brain larval SBEM + automated synapses (Baier / Denk)

| Field | Value |
|---|---|
| **URL / portal** | https://www.mapzebrain.org/zebrafishatlas/main_page (EM section + tutorial) |
| **Paper** | https://www.nature.com/articles/s41592-022-01621-0 (Nature Methods; open access) |
| **Age / prep** | 5 dpf larva, SBEM 14×14×25 nm³, whole brain minus retinae (~7.33 TV) |
| **Neurons** | FFN segmentation of the volume; **208** motion-processing neurons fully skeletonized + proofread in the paper; broader automatic segmentation exists for community proofreading |
| **Synapses** | Automated brain-wide chemical-synapse detection (SyConn2); paper reports **1,079** synapses among the 208 reconstructed cells; two tectal SINs mapped to **407** partners |
| **Edge list?** | **Partial / queryable, not a simple public full-brain edge dump.** Neurons+synapses live in a digital “address book” via mapZebrain / Knossos+SyConn agglomeration tools. No documented single downloadable `pre,post,weight` file covering the whole auto-detected synapse set. |
| **Format** | mapZebrain browser + Knossos (GPL-2) + agglomeration proofreading backend; SyConn processing snapshot on GitLab (MPCDF). |
| **License** | Paper open access (Nature). Tools: Knossos GPL-2; agglomeration backend AGPL-2. Dataset access via mapZebrain terms / instructions on the site. |
| **Usable as reservoir graph?** | Good for **small proofread subgraphs** (e.g. 208-cell visual-motion network). Not drop-in whole-brain edges without heavy tooling. |

---

## 3. Hildebrand et al. 2017 — whole-brain ssEM projectome (not synapse connectome)

| Field | Value |
|---|---|
| **URL** | https://www.neurodata.io/data/hildebrand17/ ; legacy http://zebrafish.link/hildebrand16 ; CloudVolume `s3://open-neurodata/hildebrand/...` |
| **Paper** | Nature 2017; PMC5594570 |
| **Age / prep** | 5.5 dpf larva; multi-resolution ssEM of anterior quarter including complete brain |
| **Neurons** | Draft **myelinated projectome** (reticulospinal, lateral-line afferents, etc.) — not a dense chemical synapse catalog |
| **Synapses** | High-res patches can resolve synapses; **no published brain-wide synapse edge list** |
| **Edge list?** | **NO** (not as neuron↔neuron chemical synapse graph). Image volume + myelinated axon reconstructions. |
| **Format** | PNG tile hierarchy / CloudVolume; CATMAID viewing historically |
| **License** | Open Database License (ODbL) for the NeuroData mirror; attribute davidh@fas.harvard.edu + database |
| **Usable as reservoir graph?** | No — wrong product (projectome / EM images). |

---

## 4. mapZebrain (Max Planck Zebrafish Brain Atlas)

| Field | Value |
|---|---|
| **URL** | https://www.mapzebrain.org/zebrafishatlas/main_page ; downloads API https://www.mapzebrain.org/downloads |
| **Contents** | Multimodal **light-microscopy atlas** (~6 dpf): region ontology, >4,000 registered single-neuron morphologies, transgenes / HCR / antibodies. Hosts access path to the **Svara 2022** EM volume. |
| **Edge list?** | **NO whole-brain synapse edge list.** Morphology + region co-occurrence can give **region-level** or morphology-inferred connectivity hints, not chemical synapse edges. |
| **Format** | Browser atlas + JSON/API stack links for markers / individual fish |
| **License** | Site / contributor terms; cite mapZebrain papers and contributing labs |
| **Usable as reservoir graph?** | Not for synapse graphs. Useful later for region labels if grafting a Fish1 / Svara subgraph. |

---

## 5. FishExplorer (2025 atlas platform)

| Field | Value |
|---|---|
| **URL** | Platform paper: https://www.biorxiv.org/content/10.1101/2025.07.14.664689 ; Fish1 portal links FishExplorer as anatomical-atlas integration; confirmed platform URL https://fishexplorer.zib.de/sandbox/ ; downloads https://fishexplorer.zib.de/sandbox/downloads ; code planned at https://github.com/zibneuro/fishexplorer |
| **Contents** | Multimodal cellular atlas (LM / EM / µCT) in a shared reference brain; CAVE / Neuroglancer bridge for synapse-resolved views; neuron DB with morphology, neurochemistry, connectivity, function |
| **Edge list?** | **Portal, not a file dump.** Connectivity queries go through integrated CAVE reconstructions (e.g. Fish1). No standalone public flat edge list called out. |
| **Format** | Web platform + CAVE; SWC upload / registration workflow |
| **License** | Open platform / FAIR principles per preprint; cite the FishExplorer preprint + underlying EM releases |
| **Usable as reservoir graph?** | Use as **browser / join layer** on top of Fish1 CAVE edges, not as the edge source itself. |

---

## Recommendation for a future `regions/hybrid_zebrafish` (not built now)

1. **Primary:** Fish1 via caveclient (`fish1_v250915`), pull a **region-bounded** synapse subgraph (e.g. hindbrain integrator / tectum) with E/I tags → CSR module matching our mouse/human/fly npz schema.  
2. **Secondary / small:** Svara 2022 proofread 208-cell visual-motion network if a clean export can be obtained from mapZebrain.  
3. **Skip for edges:** Hildebrand 2017, bare mapZebrain morphologies, FishExplorer alone.

**No download of EM volumes or full synapse tables was performed for this note** — metadata / paper / portal only, per task scope.

---

## Access attempt 2026-10-28 (this agent)

Tried `caveclient.CAVEclient("fish1_v250915")` with no token → **AuthException** (DAF global info API). Searched for public flat synapse exports (GCS release, VAST `.vsvi` only, FishExplorer, companion vercel portal) → **none**. Per task rules: **stopped**; see `SOURCE_fish1.json` + `RESULTS_zebrafish.md`. No synthetic fish module.

# Zebra finch Area X / HVC connectome sources

**Dir:** `/workspace/flybrain/regions/finch/`  
**Date:** 2026-10-28  
**Verdict on Step 1:** **Real neuron↔neuron synapse lists ARE downloadable** for Area X dataset **j0251** (preferred over j0126). Module built from those edges. No synthetic Area X arm in the primary experiment.

---

## 1. j0251 — adult zebra finch Area X (USED)

| Field | Value |
|---|---|
| Region | Area X (basal ganglia, song learning) |
| Animal | adult male zebra finch |
| Volume | 256 × 256 × 384 µm |
| Voxel | 10 × 10 × 25 nm |
| Synapses (filtered array) | **18,361,224** |
| Neurons (filtered array) | **4,261,298** (includes fragments / glia / axons) |
| Fully reconstr. locals in our module | ~7k (see `data/areax_module_meta.json`) |
| Cell types (v8) | DA, LMAN, HVC, MSN, STN, TAN, GPe, GPi, LTS, INT1–4, ASTRO, OLIGO, OPC, MICRO, MIGR, FRAG |
| Format | NumPy structured arrays + REST API JSON |
| License / use | Public download + REST; cite Rother et al. 2025. Site: best-effort reconstructions. Analysis code repo GPL-2.0. Data presented as public research release (no separate restrictive EULA on the download page). |
| Citation | Rother A, Januszewski M, Jain V, Fee MS, Kornfeld J (2025). *The songbird basal ganglia connectome.* bioRxiv https://doi.org/10.1101/2025.10.25.684569 |

### Download URLs (v8 cell types — what we used)

Base: `https://syconn.esc.mpcdf.mpg.de`

- Synapses (613 MB): `/connectome/data_v8/j0251_synapses.npy`
- Neurons (94 MB): `/connectome/data_v8/j0251_neurons.npy`
- Cell-type map: `/connectome/data_v8/celltype_mapping.txt`
- Post-morph map: `/connectome/data_v8/post_morph_mapping.txt`
- Version stamp: `/connectome/data_v8/version.txt` → `j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_v8_20260923`
- Agent guide: `/connectome/connectome-guide.md`
- Portal: https://syconn.esc.mpcdf.mpg.de/ and https://syconn.esc.mpcdf.mpg.de/data.html
- Related pubs hub: https://songbird-connectomics.org/publications
- Analysis scripts: https://github.com/StructuralNeurobiologyLab/areaX-connectomic-analysis (GPL-2.0)

### REST API (same connectome)

- `GET /j0251/versions`
- `GET /j0251/celltypes?version=<full>`
- `GET /j0251/synapses/json?...` (filters: pre/post cell type, prob, size, morph, skeleton lengths)
- `GET /j0251/neurons/json?...`
- Version string required: `72_seg_20210127_agglo2_syn_20220811_celltypes_v8_20260923`

### Synapse schema (edge list)

`pre_id`, `post_id`, `size` (µm²), `prob`, `x,y,z` (voxels), `post_morph`, `pre_celltype`, `post_celltype`

### Neuron schema

`id`, `cell_type`, `celltype_certainty`, `skel_length_axon/soma/dendrite` (µm)

### Our module cut (`data/areax_module.npz`)

- Locals: MSN/GPe/GPi/STN/TAN/LTS/INT1–4 with axon≥200 µm & dendrite≥200 µm & certainty≥0.7, plus rare types loosened to ≥100 µm / 0.6 so GPe/TAN/LTS survive.
- Afferents: top HVC 300 / LMAN 150 / DA 150 axons by total synapse size onto locals (prob≥0.6, min total size 0.5 µm²).
- Weight = Σ synapse size × presynaptic Dale-style sign (GABA −1; HVC/LMAN/STN/TAN/DA +1).
- Counts locked in `data/areax_module_meta.json`.

---

## 2. j0126 — adult zebra finch Area X (NOT used for edges)

| Field | Value |
|---|---|
| Volume | ~96 × 98 × 114 µm (Nature Methods FFN paper volume) |
| What is public as edges? | **No full filtered synapse/neuron NumPy connectome on the SyConn download page** (j0251 only). Viewer lists cell types `exc, modulatory, MSN, LMAN, HVC, GP, INT`. torch-em docs: “Labels: neuron instance segmentation only” for j0126. |
| Segmentation / EM | Google bucket / HuggingFace `pytc/zebrafinch-j0126` (images, GT subvolumes, skeletons) — **not a ready synapse graph**. |
| Paper | Dorkenwald et al. 2017 *Nature Methods* (FFN); SyConn processing Schubert/Kornfeld/Dorkenwald. |
| HF | https://huggingface.co/datasets/pytc/zebrafinch-j0126 |
| Cloud volume (EM) | `precomputed://gs://j0126-nature-methods-data/.../rawdata_realigned` |

**Decision:** do not fake a j0126 edge list. Prefer j0251.

---

## 3. HVC dataset — Kornfeld et al. 2017 eLife (NOT used)

| Field | Value |
|---|---|
| Paper | Kornfeld et al. 2017. EM connectomics reveals axonal target variation in a sequence-generating network. *eLife* https://doi.org/10.7554/eLife.24364 |
| Volume | 166 × 166 × 77 µm HVC; 34 BDA-labeled HVC(RA) somata; 12 partial dendritic skeletons |
| Public data | https://github.com/jmrk84/HVC_paper — KNOSSOS config + skeleton `.k.zip` annotations in release `data.zip`. Raw EM streaming via KNOSSOS; not a dense neuron↔neuron synapse table for the whole HVC. |
| Usable as reservoir module? | **No** complete synapse edge list comparable to j0251. Manual partial reconstructions only. |

---

## 4. What we did / did not invent

- Used **real** j0251 synapses and cell types.
- Did **not** invent Area X edges.
- No synthetic Area X arm in primary arms. (If a synthetic module is ever added later it must be a clearly flagged separate arm, per project rules.)

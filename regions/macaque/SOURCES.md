# Macaque connectivity sources

Primary wiring for this experiment is **retrograde tract-tracing FLNe** (Markov / Kennedy), not diffusion MRI. Secondary dMRI sources are noted below for completeness and were **not** used as the area FLN matrix.

## PRIMARY (used): Markov et al. FLNe — netneurotools `ds-famous_gmat`

- **Papers:** Markov et al., “Cortical high-density counterstreaming and directed hierarchy in macaque,” *PNAS* 2013 (doi:10.1073/pnas.1308498110); Markov et al., “A weighted and directed interareal connectivity matrix for macaque cerebral cortex,” *Cerebral Cortex* 2014 (doi:10.1093/cercor/bhs270).
- **Download (this run):** OSF project `udpv8`, file id `67327344a6dba6adff62f48c` (`famous_gmat.tar.gz` ~5.9 MB), mirrored by netneurotools `ds-famous_gmat`. Extracted `macaque_markov/{conn,dist,labels,ref}.csv`.
- **Local copies:** `data/raw/markov2013_{conn,dist,labels,ref}.*` and curated `data/markov_fln29.npz` (29×29 injected-area subgraph) + `data/markov_fln93x29.npz` (full 93 sources × 29 targets).
- **What we use:** FLNe weights among the **29 injected** cortical areas as a directed area matrix (density ≈0.72); inter-areal distances in mm from the same package for edge delays. Intrinsic (within-area) connectivity is **not** in FLNe — we synthesize local recurrent graphs when expanding to neurons (documented in PREREG / builder).
- **SLN (feedforward/feedback):** The downloadable famous_gmat Markov pack does **not** include an SLN matrix. core-nets.org and BALSA mirrors require login (not used). FF/FB laminar targeting in the neuron expansion therefore uses a **documented hierarchy-rank proxy** on the 29 areas (posterior/sensory → anterior/associative), flagged `sln_proxy=hierarchy_rank`, not measured SLN.
- **License / cite:** Research reuse with citation of Markov et al. 2013/2014; see paper terms (often CC BY-NC for related CoCoMac/core-nets materials). We cite the papers and the OSF/netneurotools redistribution.

## Checked, not used as primary wiring

### EBRAINS / MEBRAINS macaque atlases
- URL checked: https://ebrains.eu/data-tools-services/brain-atlases/macaque-brain  
- **What it is:** MEBRAINS cytoarchitectonic / receptor / template atlas resources for macaque, linked via the EBRAINS Knowledge Graph — **not** a downloadable Markov-style FLN connectivity matrix.  
- Public KG search without auth returned errors; no FLNe matrix pulled.  
- **Role here:** Noted only. Not used for reservoir wiring.

### OpenNeuro ds001875 v1.0.3 — TheVirtualBrain Macaque MRI
- URL: https://openneuro.org/datasets/ds001875/versions/1.0.3  
- **Name:** “TheVirtualBrain Macaque MRI” (Shen, Gati, Menon, Everling, McIntosh).  
- **Content:** Raw + preprocessed **T1, DWI, resting-state fMRI** from 9 macaques. License **CC0**. DOI `10.18112/openneuro.ds001875.v1.0.3`.  
- **Size:** snapshot ≈ **11.5 GB** — **not downloaded** (policy: no files >2 GB).  
- **Role here:** Secondary modality only (dMRI/fMRI). Prefer tract-tracing for wiring. README points to Zenodo derivatives for a modelling connectome (below).

### Zenodo 1471588 — TVB macaque connectome (dMRI secondary)
- DOI: https://doi.org/10.5281/zenodo.1471588 (Shen et al.; related Neuroimage 2019 / bioRxiv 2018).  
- **Downloaded (small):** `weights.txt` (82×82), `tract_lengths.txt`, `TVBmacaque_RM_LUT.txt` → `data/raw/tvb_shen2019_*` and `data/tvb_shen2019.npz` (~25 MB total archive; individual weight files ≪2 GB).  
- **Role here:** **Secondary only** — diffusion tractography–based weights, not tract-tracing. Kept for reference; **not** used to build the MACAQUE FLN expansion (Markov FLNe preferred).

### Attempted / blocked
- **core-nets.org** — login wall.  
- **BALSA** — login wall.  
- **Oxford DC2 / figshare macaque dumps** — Cloudflare 403 or noisy matches; not used.  
- **openconnecto.me** — dead/redirected.

## Local artifacts

| Path | Role |
|---|---|
| `data/markov_fln29.npz` | Primary 29×29 FLNe + Dist_mm + area names |
| `data/markov_fln93x29.npz` | Full 93×29 FLNe (not expanded; audit) |
| `data/tvb_shen2019.npz` | Secondary dMRI weights (unused for wiring) |
| `data/raw/` | Original CSVs / TXT as downloaded |
| `data/macaque_module.npz` | Neuron-level expansion (~2000) for reservoirs |


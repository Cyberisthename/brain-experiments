# C. elegans connectome sources

## Witvliet et al. 2021 (Nature)

- Paper: Witvliet et al., "Connectomes across development reveal principles of brain maturation", Nature 596:257–261 (2021). https://doi.org/10.1038/s41586-021-03778-8
- Edge-list Excel files (chemical + electrical) for datasets 1–8 downloaded from WormWiring:
  - https://wormwiring.org/pages/witvliet.html
  - Files: `witvliet_2020_{1..8} {L1|L2|L3|adult}.xlsx` → stored as `data/raw/witvliet_2020_*_*.xlsx`
- Also mirrored by OpenWorm Connectome Toolbox (`cect/data/witvliet_2020_*.xlsx`).
- Supplementary synapse-count matrices (chemical only): GitHub `dwitvliet/nature2021` `tables/synapse_count_matrices.xlsx` (not primary; WormWiring edge lists preferred because they include gap junctions).
- Neuron class table: `dwitvliet/nature2021` `tables/_neurons.csv`.
- **License / reuse:** data released with the paper for research reuse; cite Witvliet et al. 2021. EM volumes at https://bossdb.org/project/witvliet2020. Viewer: https://www.nemanode.org/
- **Gap junctions:** WormWiring/NemaNode note that Witvliet gap-junction annotation is **not exhaustive** and should not be treated as complete. We still include listed electrical edges as symmetric.

## Cook et al. 2019 (Nature)

- Paper: Cook et al., "Whole-animal connectomes of both Caenorhabditis elegans sexes", Nature 571:63–71 (2019). https://doi.org/10.1038/s41586-019-1352-7
- Adjacency matrices: WormWiring `SI 5 Connectome adjacency matrices, corrected July 2020.xlsx`
  - https://wormwiring.org/pages/adjacency.html → `data/raw/Cook2019_SI5_corrected.xlsx`
- Hermaphrodite neuron–neuron edge list (OpenWorm mirror of Cook tables): `cect/data/herm_full_edgelist.csv` → `data/raw/herm_full_edgelist.csv`
- Hermaphrodite neuron name list (302): extracted from OpenWorm `cect/Cells.py` `NEURON_NAMES` → `data/raw/NEURON_NAMES.txt` / `PREFERRED_HERM_NEURON_NAMES.txt`
- **Male:** parsed from SI5 sheets `male chemical` and `male gap jn symmetric`; neuron filter = cells that are not muscle-like (see build_graphs.py). Cite Cook et al. 2019.
- Weights in SI5 = total EM serial sections of connectivity (synapse number × size), not raw synapse counts.
- **License / reuse:** published supplementary data; cite Cook et al. 2019. Corrections July 2020 applied.

## E/I

No reliable per-neuron excitatory/inhibitory sign is bundled in these adjacency tables for our use. Reservoir weights are **unsigned** (positive). `ei_status="unknown"` in each `.npz`.

## Local artifacts

Built graphs: `data/{W1..W8,COOK_H,COOK_M}.npz` plus `data/manifest.json`.

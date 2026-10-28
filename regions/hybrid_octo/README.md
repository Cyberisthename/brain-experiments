# hybrid_octo

**No `octo_module.npz`.**

## Lichtman Lab vertical lobe (the lead)
- Paper: Bidel & Meirovitch et al., *eLife* 2023 e84257 — VL memory circuit, **not** optic lobe.
- Portal: https://lichtman.rc.fas.harvard.edu/octopus_connectomes → Neuroglancer.
- Public GCS `gs://octopus-connectomes/vertical_lobe/`: **only** `img/` (EM) + `sgm/` (segmentation).
- **No** synapse table / edge list / graph dump in the bucket or eLife supplements.
- In-paper annotations (~13k post / 2.4k pre on 516 processes) were **not** released as downloadable edges.

## Optic lobe / other
Macroscale 32-region graph; Ca imaging; scRNA; pygmy-squid FIBSEM in progress — none are neuron-level OL edges for download.

See `data/SOURCE.json`.

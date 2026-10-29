# Sources — ei_cross_species E/I labels

## Included (have usable E/I)

- **Mouse module** (`mouse_module.npz`): MICrONS cortical module used in frankenstein/hybrid. `sign` ∈ {+1,−1} from cell-type excitatory/inhibitory annotation in the module extract. N≈1871, E≈1680, I≈191.
- **Human H01 patch** (`human_module.npz`): H01 / hybrid_human extract. `sign` from cell-type / proofread labels. N≈2087, E≈1736, I≈351.
- **Zebra finch Area X** (`areax_module.npz`, j0251 module): `sign` mapped from `cell_type` (e.g. MSN vs INT*). N≈7576, E≈648, I≈6928. Certainty scores stored but not thresholded here.
- **Fly CX** (for DESIGN_EI_ONLY stats only): FlyWire CX module; E/I from cell type. Used as the empirical E/I density template for the synthetic DESIGN_EI arm (same recipe as `designed_brain.DESIGN_EI_ONLY`).

## Skipped — E/I unavailable

- **C. elegans** Witvliet 2021 (W1–W8) and Cook 2019 (COOK_H / COOK_M): chemical+electrical graphs in `regions/celegans/data/*.npz` have `ei_status="unknown"` and `sign` all +1 (unsigned). Witvliet electrical annotations incomplete; Cook SI tables as ingested here carry no neurotransmitter → E/I map. Per prereg: **skip** species without E/I. Artifacts `data/W5.npz` and `data/COOK_H.npz` are shipped only to document the skip.
- No open neurotransmitter→sign table was applied in this study; do not invent labels.

## Controls

ER / SHUF / EI_SHUF / SBM_EI definitions: see PREREG. Helpers aligned with `frankenstein.py` and `xor_cx.ei_class_shuffle`.

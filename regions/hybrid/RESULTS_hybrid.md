# Fly–mouse hybrid results

Prereg: `PREREGISTRATION_hybrid.md`.

## Data source (plain) — no CAVE login

CAVE `minnie65_public` requires a Google-linked API token; phone verification was unavailable, so **CAVE was not used**.

Instead, connectivity came from a **credential-free static export**:

| Field | Value |
|---|---|
| Source | Zenodo [10.5281/zenodo.16744240](https://zenodo.org/records/16744240) `microns_mm3_connectome_v1412.h5` (condensed) |
| Underlying | MICrONS Consortium (Nature 2025) minnie65, materialization **v1412** |
| Also used | GCS static CSVs `gs://mat_dbs/public/minnie65_phase3_v1/v1412/` (proofreading + mtypes) — no auth |
| Download date | 2026-10-27 |
| CAVE token | **not required / not used** |
| Proofread module | **N=1871** neurons (axon+dendrite status=t), **97585** condensed edges |
| E / I | 1680 / 191 (from `synapse_class`) |
| Layers | L2/3=557, L4=542, L5=301, L6=392, unk=79 |
| Weights | synapse_count × pre_sign (EXC=+1, INH=−1) |

Module artifact: `data/mouse_module.npz`. Provenance: `data/SOURCE.json`.

## Experiment status

Queued **after** plasticity → myelination in `run_queue_after_v2.sh` (which itself waits on EMPN v2). Arms FLY / HYBRID / ER / SHUF × tasks MC, T0, NARMA, PC, maze × 8 seeds; Bonferroni on HYBRID vs ER/SHUF/FLY. This file will be overwritten by `analyze_hybrid.py` when runs finish.

## Controls / graft (as preregged)

- Obs → mouse L4-like; mouse L5/L2-3 → fly E-PG/FB-ish CX inputs.
- ER: same-size random module, matched E/I signs on sampled edges.
- SHUF: degree-preserving rewire, signs preserved.

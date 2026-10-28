# Frankenstein results (smoke)

Prereg: `PREREGISTRATION_frankenstein.md`. Full 8-seed queued after human hybrid.

Octopus: **not grafted** — Lichtman VL portal is EM + segmentation only (no neuron-level synapse/edge list). See `../hybrid_octo/`.

## Smoke MC (seeds 0,1; WORKERS=1)

| arm | seed0 | seed1 | mean |
|---|---:|---:|---:|
| FLY_CX | 5.011 | 6.238 | 5.625 |
| FRANKEN | 8.384 | 5.699 | 7.042 |
| ER | 7.597 | 5.836 | 6.717 |
| SHUF | 8.510 | 4.444 | 6.477 |
| DENS_MATCHED | 5.065 | 4.687 | 4.876 |
| BEST_OF | 8.384 | 5.699 | 7.042 |

## Smoke T0 (seed 0)

- FRANKEN: 0.0757
- ER: 0.0703
- DENS_MATCHED: 0.0590

## BEST_OF picks (tune seeds 90–92, MC)

All three blocks picked **real** over dens:

```json
{
  "mouse": {
    "pick": "real",
    "tune_real": 4.738305005357158,
    "tune_dens": 3.306762921926247,
    "tune_seeds": [
      90,
      91,
      92
    ]
  },
  "human": {
    "pick": "real",
    "tune_real": 4.114880342939249,
    "tune_dens": 1.312694934932378,
    "tune_seeds": [
      90,
      91,
      92
    ]
  },
  "fly_cx": {
    "pick": "real",
    "tune_real": 5.259105698784793,
    "tune_dens": 1.6945272071682371,
    "tune_seeds": [
      90,
      91,
      92
    ]
  }
}
```

N=6833 (mouse 1871 + human 2087 + fly_cx 2875). Bridges only via fly hub; no mouse↔human; no octo.
Raw: `out/smoke.jsonl`.

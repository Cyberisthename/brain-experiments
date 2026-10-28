# RESULTS: Zebra finch Area X song-sequence reservoir

**Date:** 2026-10-28  
**Prereg:** `PREREGISTRATION_finch.md`  
**Data:** real j0251 Area X edges — see `SOURCES.md`. **No synthetic Area X edges.**  
**Module:** N=7576, nnz=245174 (`data/areax_module.npz`).  
**Lower NRMSE is better.** Eval seeds 0–7; α tuned on 90–92 only.

## Means ± SEM

| Arm | SEQ | TEMPO | ABLATE |
|---|---:|---:|---:|
| `REAL_NOD` | 0.0526±0.0007 | 0.8645±0.0029 | 19.6420±1.7986 |
| `ER_NOD` | 0.0208±0.0003 | 0.9464±0.0014 | 11.5328±0.6559 |
| `SHUF_NOD` | 0.0326±0.0011 | 0.9228±0.0045 | 21.3446±1.3559 |
| `DENS_NOD` | 0.0121±0.0005 | 1.0469±0.0042 | 13.4570±0.5285 |
| `REAL_DEL` | 0.0308±0.0008 | 0.9821±0.0073 | 13.4418±1.4484 |
| `ER_DEL` | 0.0082±0.0002 | 0.9812±0.0035 | 6.4324±0.4384 |
| `SHUF_DEL` | 0.0164±0.0004 | 1.0166±0.0042 | 12.2617±0.8538 |
| `DENS_DEL` | 0.0062±0.0002 | 0.9997±0.0049 | 7.1717±0.8542 |

## Primary: REAL_DEL vs delay-matched controls (Bonferroni ×9)

| Contrast | Metric | Δ (REAL−ctrl) | p_raw | p_corr | call |
|---|---|---:|---:|---:|---|
| `REAL_DEL` vs `ER_DEL` | SEQ | +0.0226 | 0.007812 | 0.07031 | n.s. |
| `REAL_DEL` vs `SHUF_DEL` | SEQ | +0.0144 | 0.007812 | 0.07031 | n.s. |
| `REAL_DEL` vs `DENS_DEL` | SEQ | +0.0246 | 0.007812 | 0.07031 | n.s. |
| `REAL_DEL` vs `ER_DEL` | TEMPO | +0.0009 | 0.6406 | 1 | n.s. |
| `REAL_DEL` vs `SHUF_DEL` | TEMPO | -0.0345 | 0.01562 | 0.1406 | n.s. |
| `REAL_DEL` vs `DENS_DEL` | TEMPO | -0.0177 | 0.1094 | 0.9844 | n.s. |
| `REAL_DEL` vs `ER_DEL` | ABLATE | +7.0094 | 0.007812 | 0.07031 | n.s. |
| `REAL_DEL` vs `SHUF_DEL` | ABLATE | +1.1801 | 0.4609 | 1 | n.s. |
| `REAL_DEL` vs `DENS_DEL` | ABLATE | +6.2701 | 0.007812 | 0.07031 | n.s. |

## Secondary: REAL_NOD vs no-delay controls (Bonferroni ×9)

| Contrast | Metric | Δ (REAL−ctrl) | p_raw | p_corr | call |
|---|---|---:|---:|---:|---|
| `REAL_NOD` vs `ER_NOD` | SEQ | +0.0319 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `SHUF_NOD` | SEQ | +0.0201 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `DENS_NOD` | SEQ | +0.0405 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `ER_NOD` | TEMPO | -0.0819 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `SHUF_NOD` | TEMPO | -0.0583 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `DENS_NOD` | TEMPO | -0.1823 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `ER_NOD` | ABLATE | +8.1092 | 0.007812 | 0.07031 | n.s. |
| `REAL_NOD` vs `SHUF_NOD` | ABLATE | -1.7026 | 0.4609 | 1 | n.s. |
| `REAL_NOD` vs `DENS_NOD` | ABLATE | +6.1850 | 0.03906 | 0.3516 | n.s. |

## Exploratory: REAL_DEL vs REAL_NOD (raw p only)

- **SEQ:** Δ=-0.0218 (neg favors DEL), p_raw=0.007812
- **TEMPO:** Δ=+0.1175 (neg favors DEL), p_raw=0.007812
- **ABLATE:** Δ=-6.2002 (neg favors DEL), p_raw=0.007812

## Honest verdict

**Honest null / mixed.** No metric has `REAL_DEL` significantly better than all three delay-matched controls after Bonferroni ×9. On **SEQ**, density-matched and ER controls have lower NRMSE than real wiring (raw p≈0.008, p_corr≈0.07 — close but does not survive). **TEMPO** is hard for everyone (~NRMSE 0.9–1.0). **ABLATE** (50% HVC channels zeroed at test) produces large NRMSE for all arms (distribution shift); relative ranking still favors ER/DENS, not REAL.

Within REAL, FIXED delays **help SEQ** (0.053 → 0.031, exploratory). Same pattern as the rest of the project: delays help; real wiring does not beat fair random controls on this HVC→syllable-sequence proxy.

Secondary note: `REAL_NOD` has lower TEMPO error than NOD controls (raw p≈0.008, p_corr≈0.07) — interesting but does **not** meet the prereg claim bar.

## Locked alphas
```json
{
  "REAL_NOD": 1e-08,
  "REAL_DEL": 1e-08,
  "ER_NOD": 1e-08,
  "ER_DEL": 1e-08,
  "SHUF_NOD": 1e-08,
  "SHUF_DEL": 1e-08,
  "DENS_NOD": 1e-08,
  "DENS_DEL": 1e-08
}
```


## Notes

- Fairness: in-edges onto drive ports zeroed in all arms (HVC axon-only).
- W orientation: `W[post,pre]`; transposed once after initial build (see meta).
- HVC 2017 eLife / j0126: no usable full edge list; documented in SOURCES.md, not used.
- Optional FLY_CX graft: not run (out of primary scope).
- No GitHub push.

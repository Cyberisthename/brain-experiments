# Macaque FLN reservoir results

Prereg: `PREREG_macaque.md` (locked 2026-10-29 07:39 America/New_York, before eval).
Rows kept: 760. Eval seeds: 600–623 (n=24).

## Design in plain words

Markov FLNe among 29 injected macaque cortical areas expanded to N=2001 (69/area). Within-area sparse Dale recurrent (p=0.12, frac_E=0.8). Inter-area p∝√FLNe with hierarchy-proxy FF/FB laminar targeting (measured SLN not in download). Delays from Dist_mm → 1..8.

- N=2001, nnz=39244, frac_E=0.7971014492753623, fln_dens=0.7170035671819263, sln_proxy=hierarchy_rank, layers={'L2/3': 696, 'L4': 493, 'L5': 493, 'L6': 319}.
- Primary source: Markov FLNe 29×29 (tract-tracing). OpenNeuro ds001875 MRI (~11.5 GB) not downloaded; TVB dMRI secondary unused for wiring. See `SOURCES.md`.

## Ridge α (90–92)

| Arm | XOR | NARMA | MC | PC_HARD |
|---|---:|---:|---:|---:|
| `MACAQUE` | 1e-06 | 1e-08 | 1e-08 | 0.1 |
| `MACAQUE_ER` | 1e-08 | 1e-08 | 1e-08 | 0.1 |
| `MACAQUE_SHUF` | 1e-08 | 1e-08 | 1e-06 | 0.1 |
| `DENS` | 0.0001 | 1e-08 | 0.0001 | 1.0 |
| `FLY_CX` | 1e-08 | 1e-08 | 1e-08 | 0.0001 |
| `FRANKEN` | 1e-08 | 1e-08 | 1e-08 | 1e-08 |
| `FRANKEN+MACAQUE` | 1e-08 | 1e-08 | 1e-06 | 1.0 |
| `FRANKEN+ER_addon` | 1e-08 | 1e-08 | 1e-08 | 1.0 |

Composite z (tune): μ_xor=0.7521, σ_xor=0.2380, μ_narma=0.3377, σ_narma=0.0297.

## Task means (eval seeds)

| Arm | XOR ↑ | NARMA ↓ | MC ↑ | PC_HARD ↑ | composite |
|---|---:|---:|---:|---:|---:|
| `MACAQUE` | 0.419±0.021 | 0.399±0.007 | 17.906±0.160 | 0.896±0.008 | -1.738±0.115 |
| `MACAQUE_ER` | 0.892±0.004 | 0.319±0.005 | 25.656±0.121 | 0.891±0.011 | 0.604±0.079 |
| `MACAQUE_SHUF` | 0.309±0.057 | 0.374±0.005 | 16.858±0.319 | 0.908±0.010 | -1.560±0.154 |
| `DENS` | 0.583±0.016 | 0.309±0.006 | 24.536±0.271 | 0.912±0.009 | 0.134±0.096 |
| `FLY_CX` | 0.976±0.000 | 0.353±0.005 | 24.630±0.274 | 0.851±0.010 | 0.204±0.079 |
| `FRANKEN` | 0.954±0.001 | 0.340±0.005 | 24.156±0.179 | 0.854±0.013 | 0.387±0.079 |
| `FRANKEN+MACAQUE` | 0.893±0.004 | 0.316±0.005 | 25.063±0.212 | 0.917±0.008 | 0.664±0.085 |
| `FRANKEN+ER_addon` | 0.923±0.003 | 0.311±0.004 | 25.158±0.095 | 0.911±0.009 | 0.804±0.068 |

## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon)

| H | Contrast | mean_a | mean_b | Δ | p_raw | p_bonf | pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| H1 | `MACAQUE` > `MACAQUE_ER` | -1.7380 | 0.6039 | -2.3420 | 1 | 1 | no |
| H2 | `MACAQUE` > `MACAQUE_SHUF` | -1.7380 | -1.5598 | -0.1782 | 0.795 | 1 | no |
| H3 | `FRANKEN+MACAQUE` > `FRANKEN+ER_addon` | 0.6645 | 0.8044 | -0.1399 | 0.9944 | 1 | no |

**Verdict: NOT SUPPORTED**

## Notes

- Reservoir performance ≠ biological claim about macaque cortex.
- SLN was not in the downloadable Markov pack; FF/FB used hierarchy-rank proxy.


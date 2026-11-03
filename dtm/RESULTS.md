# RESULTS — DTM (Neuro-Vines) + MPS + SKF

**Classical biology-inspired grow/prune + MPS + Spatial Kinetic Folding (SKF). Not quantum.**

Prereg: `PREREG_DTM.md`. Out: `/workspace/flybrain/dtm/out`.

## Discrete NARMA (NRMSE ↓)

| Method | mean ± std (n) | mean seconds | mean peak RSS MB |
|--------|----------------|--------------|------------------|
| dtm | 0.8196 ± 0.0000 (n=1) | 3.244 | 247.7 |
| dtm_mps_warp | 0.8074 ± 0.0000 (n=1) | 3.133 | 248.1 |
| dtm_wave | 0.7673 ± 0.0000 (n=1) | 3.046 | 248.3 |
| dtm_both | 0.7581 ± 0.0000 (n=1) | 3.598 | 248.4 |
| long_random | 0.7241 ± 0.0000 (n=1) | 2.995 | 249.8 |
| skf | 0.8997 ± 0.0000 (n=1) | 5.665 | 251.8 |
| skf_k1 | 0.9329 ± 0.0000 (n=1) | 4.738 | 252.0 |
| pso | 0.9140 ± 0.0000 (n=1) | 2.294 | 252.0 |
| sbqa | 0.8590 ± 0.0000 (n=1) | 2.608 | 252.0 |
| sa | 0.8586 ± 0.0000 (n=1) | 2.228 | 252.0 |
| random | 0.8172 ± 0.0000 (n=1) | 2.256 | 252.0 |
| greedy | 0.9067 ± 0.0000 (n=1) | 2.503 | 252.0 |
| set | 0.8414 ± 0.0000 (n=1) | 2.228 | 252.0 |
| rigl | 0.7485 ± 0.0000 (n=1) | 5.292 | 254.3 |

### Tests (Wilcoxon, first method better = lower)

- **dtm_vs_sa**: 0.8196 vs 0.8586, wins 1/1, p=1
- **dtm_vs_random**: 0.8196 vs 0.8172, wins 0/1, p=1
- **dtm_vs_greedy**: 0.8196 vs 0.9067, wins 1/1, p=1
- **dtm_vs_set**: 0.8196 vs 0.8414, wins 1/1, p=1
- **dtm_vs_sbqa**: 0.8196 vs 0.8590, wins 1/1, p=1
- **dtm_vs_rigl**: 0.8196 vs 0.7485, wins 0/1, p=1
- **dtm_vs_long_random**: 0.8196 vs 0.7241, wins 0/1, p=1
- **dtm_vs_dtm_mps_warp**: 0.8196 vs 0.8074, wins 0/1, p=1
- **dtm_vs_dtm_wave**: 0.8196 vs 0.7673, wins 0/1, p=1
- **dtm_vs_dtm_both**: 0.8196 vs 0.7581, wins 0/1, p=1
- **skf_vs_pso**: 0.8997 vs 0.9140, wins 1/1, p=1
- **skf_vs_skf_k1**: 0.8997 vs 0.9329, wins 1/1, p=1
- **skf_vs_dtm**: 0.8997 vs 0.8196, wins 0/1, p=1
- **skf_vs_random**: 0.8997 vs 0.8172, wins 0/1, p=1
- **skf_vs_set**: 0.8997 vs 0.8414, wins 0/1, p=1
- **dtm_both_vs_long_random**: 0.7581 vs 0.7241, wins 0/1, p=1
- **dtm_both_vs_dtm**: 0.7581 vs 0.8196, wins 1/1, p=1

## Continuous NARMA

- dtm: 0.6717 ± 0.0000 (n=1)
- adam: 1.0305 ± 0.0000 (n=1)
- sbqa: 1.0305 ± 0.0000 (n=1)
- set: 0.9435 ± 0.0000 (n=1)
- sgld: 0.9862 ± 0.0000 (n=1)

## NNZ traces (DTM family)

- dtm: nnz 890 → 4452
- dtm_mps_warp: nnz 890 → 4452
- dtm_wave: nnz 890 → 4452
- dtm_both: nnz 890 → 4452
- long_random: nnz 4452 → 4452
- skf: nnz 890 → 4452
- skf_k1: nnz 890 → 4452

## Verdicts

- **H1 (DTM < RANDOM & SET):** NOT SUPPORTED (α=0.0056).
- **MPS success (DTM_BOTH < LONG_RANDOM):** NOT SUPPORTED.
- **SKF success (SKF < PSO):** NOT SUPPORTED.
- Efficiency/nnz: descriptive — see table & NNZ section.
- Smoke / n=1 is underpowered; trust full Kaggle n=8.

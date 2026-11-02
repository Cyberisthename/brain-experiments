# Engine v3 benchmarks (2026-11-02, America/New_York)

Machine under load: callosum_scaled eval 2 workers left running; benches used
`nice -n 19`, `OMP/MKL/torch threads=1`.

## Forward microbench (synthetic ER, B=16, T=48)

| N | nnz | v3 fwd (ms) |
|---:|---:|---:|
| 512 | 2537 | 17.7 |
| 1024 | 5078 | 36.8 |
| 2048 | 10202 | ~51–61 |

## vs callosum_scaled v2 OscRNN (REAL_SC wiring, B=16, T=48)

| N_label | v2 ms | v3 ms | speedup |
|---:|---:|---:|---:|
| 512 | 97.1 | 91.2 | **1.06×** |
| 1024 | 155.4 | 152.9 | **1.02×** |

Ring buffer removes `torch.roll` allocator churn but the Python T-loop + per-delay
`index_add_` still dominate. Plan Phase-2 (seed batching, fused delay gather,
`torch.compile`, GPU) still needed for the ≥1.5× end-to-end gate before migrating
default region scripts.

## Parity

- callosum forward features-off: **max abs err = 0** (PASS)
- wholebrain dense→sparse port: **max abs err ≈ 1.5e-7** (PASS)
- synced train (identical init + batches): **loss/out diff = 0** (PASS)

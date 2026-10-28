# Pipeline speedup (human + frankenstein; mouse untouched)

## Changes (copy-on-write isolation)
- New module `compress/fast_kernels.py` — **not** imported by running mouse `hybrid.py`.
- Patched only `hybrid_human.py` and `frankenstein.py` (+ fast runner).
- Mouse workers keep old in-memory `compress` / `hybrid` code.

## Optimizations
1. **Dual ridge** when n_samples < n_features (exact algebra; float64).
2. **Cached spectral radius** per process.
3. **SciPy CSR matvec** reservoir loop (numba CSR was *slower* here — dropped).
4. **Numba Oja** in-place CSR update (no per-step `np.repeat`).
5. **`put_Wm` without full `tolil`**.
6. **T0 static loop** as `(n, batch)` CSR `@` (same numerics).
7. BLAS already `OMP/OPENBLAS/MKL_NUM_THREADS=1` per worker; human default workers→7, frankenstein waiter→6 (fast pass used 2).

## Equivalence (seed 0, FRANKEN / HUMAN_LAYERED)
| Check | max |diff| / notes |
|-------|----------------|
| Reservoir state old vs new | 2.2e-16 |
| T0 static state | 0 (exact) |
| MC score old vs dual-ridge | 7.9e-08 |
| T0 MSE old vs new | 2.8e-11 |
| HUMAN_LAYERED T0 value | 0.03329073604 → 0.03329073603 |
| FLY T0 value | 0.08559926226 → 0.08559926223 |

Dual-ridge *weights* can differ ~1e-4 from primal under λ=1e-6 (ill-conditioned Gram); **task metrics** match to ≤1e-7.

## Speedup (same box, nice 19, 1 thread)
| Task | Before | After | Speedup |
|------|--------|-------|---------|
| HUMAN_LAYERED T0 | ~19.0 s | 2.49 s | **~7.6×** |
| FLY T0 | ~10.3 s | 2.16 s | **~4.8×** |
| HUMAN_LAYERED+HEBB T0 | (mouse cousins 8k–28k s under thrash) / ~19 s local | 4.04 s | **≳5×** (local) |
| HUMAN_LAYERED MC | ~17–18 s (mouse) | 2.51 s | **~7×** |
| FRANKEN T0 (primal ridge dominated) | ~7–13 s smoke | 1.91 s | **~4–7×** |
| Maze | not in human/franken TASKS; Oja path sped for HEBB maze if used | — | — |

Dominant win: dual ridge removes O(N³) solve on N≈7k–11k.

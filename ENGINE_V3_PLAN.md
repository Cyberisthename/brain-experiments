# Engine v3 plan — quality + speed for OscRNN / connectome experiments

Machine time: 2026-11-02 (America/New_York). **Plan only — do not implement yet.**
Active eval `callosum_scaled` (PID 362279, 2 workers) must not be disturbed.

Companion science doc: `regions/SYNAPSE_RESEARCH.md`.

---

## 1. Current stack (what “v2” means here)

| Layer | Path | Role |
|---|---|---|
| Classic reservoir | `flybrain_reservoir.py` + `compress/fast_kernels.py` | CSR leaky-tanh, ridge tasks |
| SPEEDUP2 | `fastcore2.py` | Batched CSR, graph/state cache; verified ≤1e-6 vs old (~1.05–1.15× end-to-end under load; micro ~1.5–1.9× at B=8) |
| Region OscRNN (dense) | `regions/human_wholebrain/human_wholebrain.py`, `oscillation_gating/`, older myelin scripts | Dense `N×N` params × masks; `torch.roll` delay buffer; ProcessPool workers |
| Region OscRNN (sparse) | `regions/callosum_scaled/callosum_scaled.py` | Edge-list `index_add_` scatter; delay selectors; left/hemi I/O |
| Spiking prototype | `engine2/` | AdEx-lite, Dale, delay **ring**, optional TM STP, float32, numba; **not** wired into OscRNN evals |

**v3 target:** one shared **rate** engine module used by callosum / wholebrain / realistic_callosum / oscillation_gating, absorbing SPEEDUP2 lessons + engine2 delay/STP patterns, without requiring spikes for the first milestone.

---

## 2. Where time goes (profile notes)

Light single-thread microbench (nice 19; callosum eval left running), OscRNN sparse
`forward_sequence`, B=16, T=48, ~8 out-edges/neuron:

| N | Edges | Sparse fwd (ms) |
|---:|---:|---:|
| 160 | ~1.2k | ~13 |
| 512 | ~4.1k | ~26 |
| 1024 | ~8.2k | ~52 |

Roughly **linear in N** for sparse path. Dominant costs observed / inferred from code:

1. **Python step loop** over T with full autograd graph when training (biggest wall in evals).
2. **Per-delay scatter**: `for d in 1..DMAX` → `index_add_` (callosum_scaled) or dense `x @ W.t()` with mask (human_wholebrain).
3. **`torch.roll` on delay history** each step — allocates / moves `(DMAX+1, B, N)`.
4. **Dense OscRNN** (`human_wholebrain`): stores and multiplies full `N×N` even when nnz ≪ N²; at N=1024 this is ~1M weights × 2 matrices + softplus on inter — memory and compute waste.
5. **ProcessPool** over (arm, task, seed): good for throughput, but each worker rebuilds wiring; no cross-seed batching of the **same** W.
6. Training hyperparameters (epochs × steps × batch) dominate over pure forward; speeding forward 2× often yields ~1.5–1.8× end-to-end unless epochs drop.

engine2 already shows CSR + delay buckets + float32 can do **~2e4 steps/s** at N=2875 (B=1) on CPU — different model, but proves sparse delayed delivery need not be slow.

SPEEDUP2 already proved: same-W multi-column batch B=8 → ~1.46× CSR / ~1.91× delay kernels; graph cache ~10×; state reuse across readouts ~9× — but **end-to-end** gains stayed ~1.1× when experiments rebuild graphs and train separately. v3 must **change the eval loop**, not only the matvec.

---

## 3. v3 architecture (proposed)

```
flybrain/engine3/
  rate_network.py   # shared OscRNN: CSR/edge params, ring delays, optional STP / dual-τ / Dale
  wiring.py         # parcel→unit expand, masks, myelin delays (move out of each region script)
  tasks.py          # XOR_HARD, CHAR, NARMA (NaN-safe), TRANSFER, …
  train.py          # Adam loop, metric, resume-safe jsonl
  batching.py       # multi-seed same-W; optional torch.compile
  validate_v2.py    # bit/near-bit parity harness
```

Region scripts become thin: build wiring dict → `engine3.train.run_job(...)`.

Keep **engine2** as the spiking track; do not merge LIF into v3 rate path in v1 of this plan.

---

## 4. Speedups

| # | Change | Expected gain | Risk |
|---|---|---|---|
| S1 | **Unify sparse edge OscRNN**; retire dense `N×N` in wholebrain | Memory O(E) vs O(N²); at N=1024 train **~3–10×** vs dense masks | Must match metrics within tol after port |
| S2 | **Ring-buffer delays** (`buf[head]`, head=(head+1)%D) instead of `torch.roll` | ~1.2–1.5× on forward; less allocator churn | Easy; engine2 pattern |
| S3 | **Fused delay delivery**: one scatter of `state_hist[src] * w` using `delay`-indexed gather (or bucket CSR like engine2) | ~1.3–2× vs Python `for d` | Careful indexing |
| S4 | **Batch seeds** sharing topology (multi-column state `(N,B_seeds)` or batched first dim) | ~1.5–3× when arm fixed across seeds | Different Win/b per seed → stack Win |
| S5 | **float32 everywhere** (already default in OscRNN) + `torch.set_num_threads(1)` per worker | Stability; less RAM → more workers later | OK |
| S6 | **`torch.compile`** on `forward_sequence` (PyTorch 2.x) | ~1.3–2× after warmup on long runs | First-iter slow; validate numerics |
| S7 | Optional **numba/CSR** path for inference-only ridge arms (fast_kernels lineage) | For non-Adam experiments | Separate from OscRNN train |
| S8 | **Kaggle GPU**: move OscRNN train to CUDA; keep wiring on CPU | ~5–15× train step on T4/P100 for N≥512 | Need CUDA smoke; don't assume box GPU |
| S9 | **Resume-safe job queue** (skip done keys; atomic jsonl append) — already partial | Wall-clock via crash recovery | Must not corrupt running callosum files |
| S10 | Prefetch / cache expanded wirings per (arm, n_local, seed) on disk | Avoid rebuild; like FASTCORE2 graph cache | Disk OK |

**Stacked estimate (CPU, callosum-like sparse train eval):**
- Conservative: **~2–3×** end-to-end (S1–S4 + less roll)
- Optimistic if GPU used for train: **~8–20×** vs current box CPU dense/sparse mix
- Re-enables **N=2048** jobs previously cut from callosum_scaled

Do **not** raise `--workers` above 2 while callosum_scaled is running.

---

## 5. Quality improvements

| # | Feature | Source | Notes |
|---|---|---|---|
| Q1 | **TM STP** (facilitation default on long-range E) | SYNAPSE_RESEARCH §1; engine2 `use_stp` | Opt-in flag; identical dynamics on shuffle arms |
| Q2 | **Dale's law + E/I β** | §3 | Neuron sign; softplus magnitudes |
| Q3 | **Dual synaptic τ** (fast/slow) | §2 | Two filtered currents into tanh |
| Q4 | **NaN / Inf guards** | human_wholebrain NARMA fix | Clip NARMA targets; abort + log if loss non-finite; don't write poisoned metrics |
| Q5 | **Deterministic seeds** | — | `torch.manual_seed`, `cuda.manual_seed_all`, numpy; document worker init |
| Q6 | **Resume-safe logging** | callosum jsonl pattern | Schema version field `engine: v3` |
| Q7 | **Homeostatic gain** (slow) | SYNAPSE §5 | Optional; stabilizes large N |
| Q8 | **Myelin/FA delay priors** | microstructure/ + Caminiti | Quality of biology, not speed |

STDP / full dendritic compartments: **phase 2** after STP + dual-τ + Dale land.

---

## 6. Validation plan (v3 must reproduce v2 before science use)

### 6.1 Unit tests
- Ring buffer ≡ `torch.roll` history (max abs < 1e-6 float32; 0 float64 reference)
- Sparse edge forward ≡ dense masked forward on same nnz pattern (N≤128)
- STP off → bit-identical to no-STP buffers
- Dale projection: no sign flips across a neuron's out-edges

### 6.2 Parity eval (frozen)
Rerun a **mini** of existing v2 configs with engine3 flags off (STP/Dale/dual-τ disabled):

| Suite | Arms | Tasks | Seeds | Tolerance |
|---|---|---|---|---|
| A | REAL_SC, WEIGHT_SHUFFLE, MYELIN_ONLY (wholebrain) | XOR_HARD, NARMA | 4 | metric mean abs diff ≤ 0.02; pairwise seed Pearson ≥ 0.99 or rank match |
| B | callosum_scaled REAL_SC @ n_label=512 | TRANSFER, NARMA | 4 | same |
| C | SPEEDUP2-style CSR ridge (if exposed) | MC / NARMA10 | 4 | rel ≤ 1e-5 like SPEEDUP2 |

Fail → fix before enabling Q1–Q3.

### 6.3 Speed report
Write `engine3/BENCH.md`: forward ms vs N; end-to-end 4-seed wall vs v2; GPU if available.
Require ≥**1.5×** end-to-end on suite B before migrating default scripts.

### 6.4 Science gate
Only after parity pass: prereg new STP/Dale/dual-τ arms (see SYNAPSE_RESEARCH §9).
Never claim biology wins from an unvalidated engine swap.

---

## 7. Implementation phases (when building is approved)

**Phase 0** — Docs only (this file + SYNAPSE_RESEARCH). ← *current*  
**Phase 1** — `engine3` sparse OscRNN + ring delays + jsonl train; parity A/B.  
**Phase 2** — Seed batching + compile; BENCH ≥1.5×; port callosum_scaled / wholebrain wrappers.  
**Phase 3** — Opt-in STP, dual-τ, Dale; prereg micro-study.  
**Phase 4** — Kaggle GPU path; optional N=2048 callosum follow-up.  
**Phase 5** — Dendrites / STDP / tighter engine2 coupling (optional).

---

## 8. Non-goals (v3.0)

- Replacing engine2 spiking as default for fly CX ridge tasks  
- Raising worker count during active callosum_scaled  
- Pushing connectomes or changing GitHub remotes  
- Silent numerics changes without `validate_v2`  

---

## 9. One-page summary

v3 = **shared sparse delayed rate RNN** with engine2-style ring delays and SPEEDUP2-style
batching, plus optional **STP / dual-τ / Dale** from the synapse research note.
Expect **~2–3× CPU** end-to-end after Phase 2, **~8–20×** with GPU; unlock N=2048.
Ship only after **parity with v2** within stated tolerances.

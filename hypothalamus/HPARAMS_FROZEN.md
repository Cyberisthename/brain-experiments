# Frozen hyperparameters — hypothalamus A/B/C

**Frozen:** 2026-11-05 America/New_York, **before** any A/B/C training outcomes.
**Rule:** do not edit after looking at RESULTS. Motif *forms* come from RESEARCH.md §6; numeric β/κ/K were not in PREREG_ABC and are locked here.

## Shared (from PREREG_ABC / PREREG_v3_features)
- Topology: callosum_scaled REAL_SC, N_label=512 (n_local=8), dmax=5
- Seeds: `range(3400, 3408)`
- Tasks: XOR_HARD ↑, NARMA ↓, CHAR ↓, TRANSFER ↑
- HP: lr=5e-3, leak=0.4, epochs=25, steps_per_epoch=10,
  batch_xor=32, batch_char=24, batch_narma=24, batch_xfer=32,
  seq_xor=48, seq_char=40, seq_narma=64, seq_xfer=48,
  char_train_chars=20000, emb_dim=32, wd=0
- Ops: `torch.set_num_threads(1)`, workers=1, `nice -n 19`
- Engine features STP/DUAL_TAU/DALE/HOMEO: **all off** (motif arms only)

## Motif numerics (locked)
| Symbol | Value | Used by |
|---|---|---|
| β (slow filter) | 0.05 | A, B, C |
| κ (gain amp) | 0.5 | A, C |
| s* (set point) | 0.3 | A |
| var EMA rate | = β | A RAND_MOD |
| mutual inhibit γ | 0.5 | B |
| drive softmax temp τ | 1.0 | B |
| n_PI reserved nodes | 8 | C |
| K synaptic edges | 32 | C SYN |
| gain clamp | g ∈ [1−κ, 1+κ] via tanh | A, C |
| e clamp | [−10, 10] | A, B, C |

## Arm definitions (executable)
### A
- **HOME_O_ERR:** `err = mean(|x|) − s*`; `e ← (1−β)e + β·err`; `g = 1 + κ·tanh(e)`; multiply **external input drive** by g.
- **RAND_MOD:** track `v ← (1−β)v + β·err²` from the same err; `e ← (1−β)e + β·η` with `η ~ N(0, v)`; same g form. Structure of err destroyed; drive variance matched.
- **BASE:** no modulator.

### B
- Two partitions = left hemi units / right hemi units (fixed by wiring).
- Slow drives `d0,d1` updated **from train loss** each step (not inside forward):  
  `hunger_raw = loss`; `explore_raw = relu(prev_loss − loss)`;  
  `d0 ← (1−β)d0 + β·(hunger_raw − γ·d1)`;  
  `d1 ← (1−β)d1 + β·(explore_raw − γ·d0)`;  
  `g0,g1 = softmax([d0,d1]/τ)` then affine `g ← 1 + κ·(2g−1)` so mean gain ≈1.
- Gains multiply input drive on each partition.
- **RIVAL:** as above.
- **SINGLE:** one drive `d0` only; `g0 = 1 + κ·tanh(d0)` on all units; `d1` unused (non-learned buffers; OscRNN nn.Parameter count identical across arms).
- **SHUF_RIVAL:** same dynamics as RIVAL; unit→partition assignment randomly permuted once per seed.

### C
- Reserve first `n_PI` left-hemi units as PI stub (no FlyWire IPC pull this batch).
- `m = mean(x[:, PI])`; `e ← (1−β)e + β·m`; `δ = κ·tanh(e)`.
- **VOL:** add `δ` to **all** units' pre-activation (broadcast).
- **SYN:** send `δ` only along `K=32` random existing edges (intra∪inter), chosen once per seed; scale messages so batch-mean `‖Δh‖₂` matches a detached VOL-sized broadcast `‖δ·1‖₂` (matched L2 effect).
- **BASE:** no modulator.

## Planned confirmatory contrasts (Bonferroni)
α = 0.05 / 12 = 0.004167 over:
1–4. A: HOME vs RAND, HOME vs BASE × {XOR_HARD, NARMA}
5–8. B: RIVAL vs SINGLE, RIVAL vs SHUF_RIVAL × {XOR_HARD, TRANSFER}
9–12. C: VOL vs SYN, VOL vs BASE × {NARMA, XOR_HARD}

CHAR/TRANSFER (A,C) and NARMA/CHAR (B) reported as exploratory.
With n=8, Wilcoxon two-sided floor ≈0.0078 > α — confirmatory bar unreachable; mark exploratory vs confirmatory explicitly in RESULTS (per PREREG_ABC).

## Outputs
- `results/abc_runs.jsonl` (resume-safe)
- `results/SCRIPT_HASHES.txt` (sha256 before first metric write)
- `RESULTS.md`

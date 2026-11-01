# FIX_NOTES: NARMA NaN on seed 2207

Date: 2026-11-01 (America/New_York)

## Symptom
All 7 arms returned `metric: NaN` for `task=NARMA`, `seed=2207`. Training curves
were healthy until epoch 33, then `train_loss` and `metric` became NaN for every
arm (arm-independent). Other seeds (2200–2206) finished with finite NARMA scores.

## Root cause
`make_narma_batch` drives the standard Atiya & Parlos (2000) NARMA-10 system with
`u ~ Uniform[0, 0.5]`. That recurrence is only marginally stable: a small fraction
of input sequences make the target `y` explode to `+Inf` in float32.

Reproducing the training RNG (`default_rng(seed + 17)`, 20 steps/epoch):

- seed 2207, epoch 33, step 7, series index 22: `y` grows through O(10) → O(1e5)
  → O(1e16) → `Inf`.
- The Inf target makes MSE / NRMSE non-finite → Adam step writes NaNs into weights
  → every later epoch and the final eval stay NaN.
- Same training-data RNG is shared across arms, so every arm failed together.
- Seed 2204 has one mildly large finite trajectory (`|y|≈10.3`) that did **not**
  NaN training; seeds 2200–2203, 2205–2206 stay O(1).

This is a task-generation bug, not a wiring/arm bug.

## Fix (task-level, arm-independent)
In `make_narma_batch`:

1. Keep literature drive `u ~ Uniform[0, 0.5]` and the standard NARMA-10 update.
2. After drawing a batch of `u`, simulate each series; if any step is non-finite or
   `|y| > 50`, **resample that series' `u`** (up to 50 tries). Bound 50 is far
   above the nominal O(1) NARMA operating scale and far below float32 overflow of
   `y²` in the MSE, so it only catches explosive trajectories.
3. Last-resort clip if all tries fail (should not happen with `u∈[0,0.5]`).

First draw of `u` is still one vectorized `uniform(size=(T,batch,1))`, matching the
old RNG consumption whenever no series is rejected. Verified bit-identical NARMA
batches through full training for seeds 2200–2206; only seed 2207 diverges from
the old stream (at the epoch-33 explosion, which is now resampled).

## Rerun scope
Because the fix does **not** change task generation for seeds 2200–2206, only
**seed 2207 × NARMA × all 7 arms** is rerun. Other tasks/seeds untouched.
`RESULTS.md` → `RESULTS_v1.md` before rewriting analysis with n=8 finite NARMA.

## References
- Atiya & Parlos (2000), IEEE TNN — NARMA-10 definition, `u∈[0,0.5]`.
- Common RC practice: discard / regenerate unstable NARMA trajectories rather than
  train against non-finite targets.

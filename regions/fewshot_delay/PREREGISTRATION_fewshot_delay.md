# Preregistration: Few-shot + noise with matched delays

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any fewshot_delay eval scores are written  
**Dir:** `/workspace/flybrain/regions/fewshot_delay/`

## Motivation

Without delays (`regions/robust`):
- At **100 training samples**, MC was **FLY 3.32 > FRANKEN 2.14 ≫ ER 0.84 / SHUF 0.73 / DENS 0.66**.
- Under **input noise**, mean MC retention was **FLY 0.40, FRANKEN 0.37** vs random **0.21–0.27**.

With delays (`regions/best_fly`, `regions/myelin_real`):
- Delays nearly triple MC; **random wiring given the same delays matches or beats real wiring at full train size**.

**Key question:** does real wiring’s few-shot and noise advantage **survive** when every arm gets the **same delay rule**?

## Hypothesis (locked)

**H1 (few-shot):** At small `n_train` (esp. 25/50/100), `FRANKEN_DEL` and/or `FLY_CX_DEL` beat size-matched delay controls on MC (and secondarily NARMA / PC_HARD), after Bonferroni.

**H2 (noise):** With delays, real arms keep a higher fraction of clean MC under input noise than delay-matched random arms.

**H3 (mechanism):** Real arms have **lower** state participation ratio (more low-dimensional / dominant eigenmodes), which predicts better few-shot readout; the PR gap should correlate with the few-shot MC gap.

**H0 we respect:** If delay-matched random equals or beats real at small `n_train` after correction, the earlier few-shot win was an artifact of the no-delay protocol (or of untuned ridge), not a general property of connectome wiring.

## Arms (10)

| Arm | Wiring | Delays |
|---|---|---|
| `FLY_CX_NOD` | real fly CX | all d=1 |
| `FLY_CX_DEL` | real fly CX | FIXED Uniform{1..dmax}, seed-tied |
| `FRANKEN_NOD` | mouse+human+fly CX | all d=1 |
| `FRANKEN_DEL` | mouse+human+fly CX | human-pre: real H01 myelin→delay; other-pre: FIXED |
| `ER_NOD` / `ER_DEL` | ER matched to frankenstein blocks | NOD: d=1; DEL: **identical delay rule as FRANKEN_DEL** (myelin on human index range + FIXED elsewhere) |
| `SHUF_NOD` / `SHUF_DEL` | degree-preserving shuffle per block | same delay rule split |
| `DENS_NOD` / `DENS_DEL` | dens-matched (N+nnz, \|w\| bootstrap) | same delay rule split |

Builders / scaling: `frankenstein.build_arm` — block ρ=1, bridge σ=0.05, global ρ=0.9.  
`dmax=5`. Myelin: `data/human_myelin.npz`; `d = 1 + round((1−clip(mf,0,1))*(dmax−1))`; NaN→median fill.  
Leak = 0.2. Drive = info["drive"] (L4 / union of L4s).

## Protocol

### Ridge α (fairness)
- Grid: `{1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0}`.
- **Tune seeds 90, 91, 92 ONLY.** For each arm, pick α maximizing mean **MC at n_train=100** over tune seeds.
- Lock to `out/alpha_choice.json` **before** writing any eval (seeds 0–7) rows.
- Locked α used for MC, NARMA, and PC_HARD ridge on eval.

### Learning curves
- Wash / max_train / test = **200 / 3000 / 500** (T=3700). MC lags D=100.
- `n_train ∈ {25, 50, 100, 200, 500, 1000, 3000}` — readout fit on first `n_train` post-wash samples; test always the held-out 500.
- Tasks: **MC** ↑, **NARMA-10 NRMSE** ↓, **PC_HARD** ↑ (cue_frac=0.35, noise=1.5).
- One reservoir rollout per (arm, seed, task); score all `n_train` from the same states.

### Input-noise sweep
- Channel: **input** only (u ← u + σ·ε); decode clean targets.
- σ ∈ {0.0, 0.25, 0.5, 1.0, 2.0}.
- Tasks: MC, NARMA. Readout trained at **n_train=3000** with locked α (clean training states; noisy test rollout — train on clean X[wash:wash+3000], evaluate on noisy X_test; for σ=0 reuse clean).
- Simpler locked protocol: for each σ, full re-sim with noisy drive; ridge on noisy train window with clean targets (matches `regions/robust` noise). Retention = score(σ)/score(0) (MC) or clipped score(0)/score(σ) (NARMA).
- Summary: mean retention over σ>0; AUC of retention vs σ via `np.trapezoid`.

### Mechanism (exploratory, same eval seeds)
- From clean MC rollout states X[wash:wash+3000], compute **participation ratio**  
  `PR = (∑λ)² / ∑λ²` of the covariance eigenvalues (or of X⊤X / (T−1)).
- Also report top-10 variance fraction and spectral flatness.
- Correlate (across arms) mean PR with MC at n_train=100.

### Eval seeds
- Seeds **0–7**. No peeking: tune artifacts only under `out/tune_*.jsonl`.

## Stats

- Paired Wilcoxon signed-rank (two-sided) across the 8 seeds; all-zero diffs → p=1.
- **Primary family (Bonferroni ×9):** at `n_train=100`, `FRANKEN_DEL` vs `{ER_DEL, SHUF_DEL, DENS_DEL}` × `{MC, NARMA, PC_HARD}`.
- **Secondary family (Bonferroni ×9):** same contrasts for `FRANKEN_NOD` (replicates robust few-shot under this α protocol).
- **Delay-survival family (Bonferroni ×3):** for MC at n_train=100, paired gap  
  `Δ_DEL = FRANKEN_DEL − ER_DEL` vs `Δ_NOD = FRANKEN_NOD − ER_NOD` (and vs SHUF, vs DENS) — i.e. does the real−random gap shrink under delays? Test on per-seed gaps with Wilcoxon on (gap_DEL − gap_NOD); report direction.
- **Noise family (Bonferroni ×6):** mean input-noise MC retention (σ>0) for `FRANKEN_DEL` and `FLY_CX_DEL` vs `{ER_DEL, SHUF_DEL, DENS_DEL}` (FLY vs franken-sized controls is size-confounded — marked exploratory for FLY).
- Exploratory: full learning curves; FLY_CX_* vs controls; n90; mechanism correlations. Raw p only.

Direction: MC/PC_HARD higher better; NARMA lower better (Δ = real − ctrl; for NARMA negative Δ favors real).

## Predictions (locked)

1. **P1:** `FRANKEN_NOD` / `FLY_CX_NOD` beat delay-free random at n_train=100 MC (replicate robust).
2. **P2 (primary):** `FRANKEN_DEL` still beats `ER_DEL`/`SHUF_DEL`/`DENS_DEL` on MC at n_train=100 after Bonf — real few-shot edge **survives** delays.
3. **P3:** Input-noise retention: `FRANKEN_DEL` > delay-matched random after Bonf.
4. **P4:** If P2 fails, honest verdict = delays erase the connectome few-shot advantage (capacity becomes delay-dominated).

## Ops / honesty

- Fast kernels: `compress/fast_kernels.py` + bucketed delay CSR (same as `best_fly` / `myelin_real`).
- `nice -n 19`, WORKERS≤2 (share box with mouse hybrid + mature + thinking).
- No GitHub push. Report negatives. Do not claim biology wins from NOD-only or from beating FLY_CX_PLAIN.

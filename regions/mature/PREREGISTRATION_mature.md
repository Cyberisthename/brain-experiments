# Preregistration: Developmental maturation — does growing up help real wiring more than random?

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any mature result is written  
**Motivation:** On intact MC / NARMA / PC_HARD, same-size ER often matches or beats FRANKEN. User hypothesis: networks are "immature" — real connectomes may need a developmental phase (Hebbian refinement, pruning, use-dependent myelination-like delays, homeostasis) before topology pays off. Random graphs get the **identical** maturation protocol; the claim is a **maturation × wiring interaction**, not that maturation helps in absolute terms.

## Arms (identical graft to frankenstein; sped-up kernels)

Reuse `frankenstein.build_arm` / block builders. Block ρ=1 before bridges; bridge σ=0.05; global ρ=0.9 after stacking. Drive = union of L4 / L4-proxy. Leak default 0.2. Use `compress/fast_kernels.py` (see `regions/frankenstein/out/SPEEDUP_REPORT.md`). Frankenstein DATA symlinked under `mature/data/`.

| Arm | Definition |
|---|---|
| FLY_CX | fly CX alone |
| FRANKEN | real mouse + real human + real fly CX |
| ER | same-size nnz-matched \|N(0,1)\| ER per block |
| SHUF | degree-preserving shuffle per block |
| DENS_MATCHED | same N+nnz, \|w\| bootstrapped from real \|W.data\| per block |

No BEST_OF, no maze, no easy PC (saturated).

## Maturation protocol (identical rule / rate / budget for every arm)

Unsupervised exposure to **structured developmental streams** (not MC / NARMA / PC_HARD / lesion stimuli):

- Mix of (i) smooth low-frequency sinusoids, (ii) periodic multi-frequency phase patterns, (iii) sparse Poisson-like event pulses on random drive subsets.
- Multi-channel drive into the same L4/drive mask used at test time.
- **No new edges.** Signs of existing edges are preserved (updates that would flip sign are clipped to ε·sign(w₀)).

Components:

| code | rule |
|---|---|
| (a) Hebb | **Oja** on existing CSR edges only (`fast_kernels.oja_update_csr_fast`). Primary locked rule. BCM is a prereg'd fallback only if Oja is unstable on tune seeds (documented deviation). |
| (b) Prune | Every `prune_every` steps, drop the weakest `prune_frac` fraction of edges by recent \|w\|·usage score (usage = EMA of \|pre·post\|); never prune below `min_keep_frac` of initial nnz. |
| (c) Delays | Myelin **ADAPT** rule: integer delays ∈[1, dmax=5]; shorten on above-median coactivity; lengthen on bottom 20%; structure preserved (no ADAPT-SHUF). Delay-bin CSR matvec during maturation + delayed eval when condition includes (c). |
| (d) Homeostasis | After each age checkpoint (and after prune), rescale W so spectral radius = target ρ=0.9 (`radius_cached`). |

### Conditions

| condition | active |
|---|---|
| `none` | drive only; no (a–d). W unchanged across ages (scores identical; still logged per age for schema). |
| `hebb` | (a)+(d) |
| `hebb_prune` | (a)+(b)+(d) |
| `full` | (a)+(b)+(c)+(d) |

### Ages (checkpoints of one continuous run)

`AGES = [0, 1000, 5000, 20000, 50000]` steps. Age 0 = pre-maturation baseline (evaluate immediately after build, before any developmental stream).

## Hyperparameters (tune seeds ONLY)

- Tune seeds: **{90, 91, 92}** — never test seeds 0–7.
- Tuned: Oja η, prune_frac, prune_every, delay EMA rates (if needed). Grid / successive narrowing until: (i) finite MC on FRANKEN+FLY_CX+ER for `hebb` at age 5k; (ii) nnz after prune ≥ 50% of initial for `hebb_prune`; (iii) no NaN NARMA flags.
- Locked defaults before tune (will be overwritten in `out/tune.json`): η=1e-4, prune_frac=0.02 per prune event, prune_every=1000, dmax=5, leak=0.2, ρ=0.9.

## Evaluation at each checkpoint

Same wash/tr/te / PC_HARD hyperparams as frankenstein fast:

1. **MC** ↑ (200/1500/500, D=100)
2. **NARMA** ↓ NRMSE (200/1500/500); flag nonfinite
3. **PC_HARD** ↑ (cue_frac=0.35, noise_std=1.5)
4. **Lesion-20% retention** on MC: random 20% neuron silence; retrain readout; `MC_lesion / MC_intact` (higher = more robust)

For `full`, evaluation uses the matured delay vector (ADAPT). For other conditions, no delays (standard CSR reservoir).

## Seeds / stats

- Test seeds: **0–7**
- Primary quantity: **improvement** Δ(age) = score(age) − score(0)  
  - MC, PC_HARD, lesion-ret: higher Δ better  
  - NARMA: improvement = NRMSE(0) − NRMSE(age) (higher Δ = more improvement)
- **Key contrast (maturation × wiring):**  
  `ΔΔ = mean_Δ(FRANKEN) − mean_Δ(control)` for control ∈ {ER, SHUF, DENS_MATCHED, FLY_CX}  
  at ages {5k, 20k, 50k} under conditions `{hebb, hebb_prune, full}`  
  (skip `none` for ΔΔ — trivially 0).
- Also report absolute scores at age 0 and age 50k.
- Test: Mann–Whitney U on per-seed Δ, two-sided.
- **Bonferroni family:** all FRANKEN-vs-control ΔΔ contrasts declared above  
  = 4 controls × 3 ages × 3 conditions × 4 metrics = **144** planned contrasts.  
  Report raw p and p_bonf = min(1, p×144). Survivors only if p_bonf < 0.05.
- Secondary (exploratory, labeled): which condition maximizes ΔΔ; age–response curves.

## Predictions (locked before run)

1. **H1 (interaction, primary):** Under `full` and/or `hebb_prune`, ΔΔ(FRANKEN − ER) on **lesion-20% retention** and/or **PC_HARD** at age ≥20k is **positive** (real wiring gains more from maturation than ER). Absolute MC may still favor ER.
2. **H2:** Under `hebb` alone (no prune/delays), ΔΔ on MC is **≤ 0** or null — Hebbian whitening without pruning does not create a topology advantage on echo-memory.
3. **H3:** `none` shows flat scores across ages (sanity). If not, implementation bug.
4. **H4:** FLY_CX may show similar (weaker) interaction vs its own size-matched nulls; FRANKEN vs FLY_CX ΔΔ on lesion retention is unclear (no directional claim).
5. **H5:** DENS_MATCHED is the toughest control; if ΔΔ vs DENS_MATCHED fails but vs ER succeeds, claim is limited to “beats fat-spectrum ER,” not “topology beyond density+weights.”

## Compute policy

Nice 19, **WORKERS=1** while mouse `hybrid.py` (≥3 procs) or robust lesion/noise workers run; may rise to 2 after hybrid idle. Must not displace hybrid or robust. AetherLink / thinking stay lower priority.

## Deviations

None at prereg time. Any post-hoc change labeled in RESULTS_mature.md.

# Preregistration: XOR_DELAY confirmatory replication (FRANKEN)

**Locked:** 2026-10-28 15:47 America/New_York (EDT), BEFORE any `xor_rep` eval scores.  
**Parent (exploratory, NOT confirmatory):** `regions/trained_bridges` FIXED FRANKEN XOR_DELAY d=5  
mean 0.525 vs ER 0.213 vs SHUF −0.159 (seeds 0–7, raw p=0.0078; primary ES-gain hypothesis there FAILED).  
**Dir:** `/workspace/flybrain/regions/xor_rep/`  
**Goal:** confirmatory test that real FRANKEN wiring beats fair random controls on delayed XOR.

## Question

Does FRANKEN (mouse|human|fly_CX, real blocks, FIXED bridges σ=0.05, real H01 myelin delays on human-pre edges) score higher on XOR_DELAY (d=5) than size-/density-matched random controls under 24 fresh seeds?

## Recipe (frozen; identical to trained_bridges FIXED path)

- Builder: `frankenstein.build_arm` / `graft_blocks` (block ρ=1, bridges σ=0.05, global ρ=0.9).
- Delays primary: **REALMYELIN** — FIXED Uniform{1..dmax} on non-human-pre edges; real H01 myelin→delay on human-pre (`dmax=5`, `human_myelin.npz`). Same map as trained_bridges / best_fly.
- Leak=0.2. Drive = union of L4 ports (single-block arms: that block’s L4).
- XOR task: **bit-exact** `xor_delay_score` from `trained_bridges.py`  
  (2-input delayed XOR; y_t = ±1 from a_{t-d} XOR b_{t-d}; metric = 1 − NRMSE; wash/tr/te = 200/1500/500).
- Ridge α **per arm** tuned on seeds **90, 91, 92** only (maximize mean XOR_DELAY d=5 on tune seeds;  
  α grid `{1e-8,1e-6,1e-4,1e-2,1e-1,1.0,10.0}`; never peek at eval seeds). Lock `out/alpha_choice.json` before eval.
- Eval seeds: **200–223** (24 NEW; never 0–7, never 90–92, never myelin_real_rep 100–123).
- Speedup: `fastcore2` opt-in if importable; numerics must match.

## Arms

| Arm | Definition |
|---|---|
| `FRANKEN` | real mouse + human + fly_CX |
| `ER` | each block → ER matched N / nnz / sign (same bridges) |
| `SHUF` | each block → degree-preserving shuffle |
| `DENS` | each block → dens_match (N/nnz + \|w\| bootstrap); frankenstein `DENS_MATCHED` |
| `FLY_CX` | fly_CX alone (component check) |
| `HUMAN` | human module alone (component check) |
| `SPEC` | secondary sanity: dens_match blocks, then **temper** edge weights so each block’s top-64 \|λ\| mean matches the real block’s (ARPACK), then rescale block ρ=1 and restack — checks whether spectral bulk (not biology) explains XOR |

Primary family uses FRANKEN / ER / SHUF only. DENS, FLY_CX, HUMAN, SPEC are secondary / component / sanity.

## Primary hypothesis (locked)

- **H1:** On XOR_DELAY **d=5**, REALMYELIN delays:  
  `mean(FRANKEN) > mean(ER)` **and** `mean(FRANKEN) > mean(SHUF)`.
- Test: paired Wilcoxon signed-rank **one-sided** (greater), n=24 seeds, same seed paired across arms.
- **Bonferroni ×2** over the two primary contrasts (FRANKEN>ER, FRANKEN>SHUF).
- **SUPPORTED** iff both contrasts have p_bonf < 0.05 **and** mean(FRANKEN) exceeds both controls.
- Else **NOT SUPPORTED**.

## Secondary (report; do not expand primary Bonferroni)

1. Same arms on XOR_DELAY at **d ∈ {2, 3, 8, 10}** (REALMYELIN).
2. **NODELAY** (all d=1) and **FIXED** (all edges Uniform{1..5}, no myelin map) vs REALMYELIN at d=5 for FRANKEN/ER/SHUF.
3. DENS, FLY_CX, HUMAN, SPEC at d=5 REALMYELIN (component / density / spectral bulk).
4. Effect sizes: paired Cohen’s d_z = mean(Δ)/sd(Δ).

## Mechanism probe (ONLY if H1 SUPPORTED)

Lesion / ablate contribution at d=5 REALMYELIN, seeds 200–207 (8 seeds, exploratory):

| Probe | Change |
|---|---|
| `LES_FLY` | zero fly_CX diagonal block (keep bridges+other blocks) |
| `LES_HUMAN` | zero human diagonal block |
| `LES_BRIDGES` | zero all inter-block bridge edges |

Compare drop vs intact FRANKEN. Which lesion kills XOR most?

## Ops / honesty

- WORKERS≤2, `nice`, nohup CPU. Do not kill user’s Kaggle sessions or unrelated jobs.
- Push code+prereg+results to `Cyberisthename/brain-experiments` (no big data, no secrets) after RESULTS.
- Exploratory parent used seeds 0–7; this file locks before xor_rep scores exist.
- Do not claim “brain computes XOR” — only reservoir metric 1−NRMSE on delayed XOR under this recipe.
- Report negatives honestly.

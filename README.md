# brain-experiments

Preregistered reservoir-computing experiments on real connectomes: FlyWire central complex (CX), grafts from mouse MICrONS and human H01 cortex, and synthetic add-ons (hippocampus-like, myelination delays, EMPN organs).

**Status:** active research scratchpad. Full hybrid / Frankenstein 8-seed runs may still be in progress. Headline numbers below are taken only from committed `RESULTS*.md` files — smoke tests and incomplete arms are labeled as such.

## Core setup (rounds 1–3)

- **Circuit:** FlyWire v783 central complex — 2,875 neurons, ~300k signed synapses (Shiu et al. 2024 model release).
- **Model:** echo-state / rate reservoir; spectral radius fixed per arm; only a linear readout is trained (unless a round explicitly trains weights).
- **Controls:** Erdős–Rényi (ER) and degree-preserving shuffle (DEG / SHUF).
- **Seeds:** typically 8–12 test seeds; Mann–Whitney / Wilcoxon as preregistered; Bonferroni when the RESULTS file says so.
- **Preregistrations:** `PREREGISTRATION.md`, `_2.md`, `_3.md` written before those runs.

### Round 1 (fixed hyperparams) — `RESULTS.md`

| metric | fly CX | ER | degree-shuffle |
|---|---:|---:|---:|
| Memory capacity ↑ | **8.84 ± 0.41** | 4.27 ± 0.36 | 6.53 ± 0.64 |
| NARMA-10 NRMSE ↓ | 0.504 | 0.466 | 0.443 |
| T0 MSE ↓ | 0.062 | 0.036 | 0.043 |

Fly CX remembers past inputs ~2× better than ER at this setting, but is **not** better on NARMA/T0. Adapter tags did not help (H4 supported).

### Round 2 (fair tuning) — `RESULTS.md`

After each wiring got its own best (leak, ρ), **ER beat fly on MC** (17.5 vs 13.6). The fly advantage was largely a hyperparameter artifact. What remains: fly wiring is **more robust** across the leak/ρ grid (ER collapses at many settings). Cell-type knockouts were **not** run (prereg gate failed).

### Round 3 (train weights) — `RESULTS.md`

Short Adam + truncated BPTT. Fly stays more robust after training; on T0 both fly and nulls improve, but nulls end much better. A learning-rate sanity check cut fly T0 error ~30× but fly still lagged ER/DEG.

## Later experiments (headline only)

| Experiment | Folder | Headline from RESULTS |
|---|---|---|
| Grow CX with real fly regions | `regions/` | S1–S2 (~10k): real ≈ random add. S3–S4 (32–40k): real MC **worse** than size-matched ADD/SHUF. Robustness vs SHUF holds. |
| Synthetic hippocampus | `regions/hippo/` | Extra neurons help NARMA/T0, but **random module ≈ hippocampus**. HIPPO beats CX on NARMA (Bonferroni); RAND beats HIPPO on MC. |
| EMPN organ-fix maze (v2b) | `regions/empn/` | Learner bugs (CFS, +TR) fixed; full EMPN food-finding recovered (~56%). Extra organs **do not** beat same-size RANDMOD (70%); no prereg contrast survives Bonferroni ×13. |
| Myelination (delays) | `regions/myelin/` | FIXED delays MC **45.1** > ADAPT **43.0** > NODELAY **40.4** > ADAPT-SHUF **37.7** (8 seeds; ADAPT vs SHUF Bonferroni-sig). Maze NaN (path bug). |
| Compression + adapters | `compress/` | Mag/consol/rand prune keep most MC at f=0.25–0.5; LoRA/adapt recover MC and crush T0 vs frozen compressed nets. |
| Fly–mouse hybrid | `regions/hybrid/` | Module built (N=1871 proofread MICrONS). Full 8-seed RESULTS not yet finalized in-repo (queued). |
| Fly–human hybrid | `regions/hybrid_human/` | Module built (N=2087). Smoke: sparse human cut → **RANDOM_LAYERED MC > human** (mean 8.40 vs 5.78 over 4 seeds). See `out/DIAGNOSIS_random_8_85.md`. |
| Frankenstein (fly+mouse+human) | `regions/frankenstein/` | Smoke MC (2 seeds): FRANKEN 7.04 vs ER 6.72 vs dens-matched 4.88. Octopus **not** included. Full 8-seed pending. |
| Thinking / AetherLink / Plasticity | respective folders | **NO DATA** — import/API bugs documented in RESULTS (not biological nulls). |

## How to fetch data (not in this repo)

Connectome binaries are **excluded** (multi‑GB). Rebuild from public sources:

1. **Fly CX (FlyWire v783)**  
   Shiu et al. 2024 Drosophila brain model:  
   https://github.com/philshiu/Drosophila_brain_model  
   Place under `data/Drosophila_brain_model/` as used by `flybrain_reservoir.py`.

2. **Mouse MICrONS (hybrid)** — Zenodo condensed connectome  
   DOI: [10.5281/zenodo.16744240](https://zenodo.org/records/16744240) (`microns_mm3_connectome_v1412.h5`)  
   Underlying: MICrONS minnie65 materialization **v1412**.  
   Optional proofreading/mtype CSVs: `gs://mat_dbs/public/minnie65_phase3_v1/v1412/`  
   Provenance: `regions/hybrid/data/SOURCE.json`.  
   Build: `regions/hybrid/build_mouse_module.py` → `mouse_module.npz`.

3. **Human H01 (hybrid_human)**  
   Official release: https://h01-release.storage.googleapis.com/landing.html (Shapson-Coe et al. Science 2024).  
   This build used public GCS somas + a derived edge list (HF `NathanRoll/h01-cortex-snn`); see `regions/hybrid_human/data/SOURCE.json`.  
   **Caveat:** only ~8/2087 neurons overlap the public proofread set — far less trustworthy than the MICrONS module.  
   Build: `regions/hybrid_human/build_human_module.py`.

4. **Octopus** — **no usable neuron-level edge list** found (EM/Neuroglancer only). See `regions/hybrid_octo/`.

## Honest caveats

- One rate model, synapse counts as weights, predicted transmitters as signs — not a biophysical fly.
- “Smarter than the fly” claims are wrong: these are abstract reservoir scores, not behavior.
- Many early “wins” reverse under fair tuning or density-matched controls.
- Sparse grafts can make **random** ER add-ons look strong (more mid-size eigenvalues → higher MC); dense grafts can do the opposite. Always report dens-matched / SHUF / ER.
- Incomplete arms (thinking, aetherlink, plasticity) failed for code bugs — do not cite as null results.
- Bonferroni and prereg contrasts are what we treat as confirmatory; uncorrected p-values are exploratory.

## Layout

```
flybrain_reservoir.py   # core CX reservoir
round2.py / round3.py   # tuning + weight training
compress/               # prune / LoRA / SAG
regions/                # grafts, EMPN, myelin, hybrids, …
  */PREREGISTRATION*.md
  */RESULTS*.md
  */data/SOURCE.json    # provenance only (no binaries)
```

Shell helpers (`run_*.sh`) queue long jobs; do not commit `out/`, `*.npz`, or logs (see `.gitignore`).

## License / credit

Upstream connectomes retain their own licenses (FlyWire / Shiu model; MICrONS / Zenodo; H01 CC BY 4.0). Experiment code here is research scaffolding unless otherwise noted.

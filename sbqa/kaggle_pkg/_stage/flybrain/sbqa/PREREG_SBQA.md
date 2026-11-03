# PREREG: Simulated Bio-Quantum Annealing (SBQA)

**Frozen at:** 2026-11-03 ~08:00 America/New_York (UTC-5), **before** held-out eval seeds are scored.  
**Path:** `flybrain/sbqa/`  
**Note:** SBQA is **classical / quantum-inspired** (WKB-style tunneling acceptance on a CPU/GPU). It does **not** use real qubits.

## Question

On a **discrete, bumpy** connectome wiring-search landscape (and as a control on a **smooth continuous** weight-training landscape), does **SBQA** (gradient steps + WKB tunneling jumps constrained to a biological sparsity mask) beat classical simulated annealing and plain Adam / greedy search under an equal evaluation budget?

## Data (local + Kaggle package)

- Griffa et al. 2019 scale-68 group SC only: `data/griffa70_scale68_group_minimal.npz`  
  Zenodo 2872624, **CC BY 4.0**. Private Kaggle dataset only. See `data/DATA_LICENSE.md`.
- No FlyWire / large connectomes in the package. Mask = bio2art-style expansion of Griffa SC to `N = 68 × N_LOCAL`.

## Network (frozen)

- Rate reservoir: `x ← (1−α)x + α tanh(W⊙M · x + Win u + b)`, α=0.4, spectral radius target ρ=0.9 after each topology change.
- Default size: **N_LOCAL=4 → N=272** (full / Kaggle). Smoke: N_LOCAL=2 → N=136.
- Mask `M` = expanded Griffa SC (intra-parcel all-to-all; inter-parcel where SC>0). Self-loops forbidden.
- Torch implementation; `device=auto` (cuda if available else cpu).

## Experiment A — Discrete wiring search (primary)

**State:** binary edge set inside mask with fixed budget `E_budget = round(0.35 × nnz(M))` (or current nnz if smaller).  
**Score:** held-out task metric after ridge readout on a **fixed** (Win, b, task stream) per seed.  
**Primary task:** NARMA-10 NRMSE (↓). Secondary: XOR_HARD accuracy (↑, d=8).  
**Methods (equal eval budget B):**

| Method | Proposal |
|--------|----------|
| **SBQA** | When loss plateaus (grad-proxy = recent score variance flat), propose non-local edge flips (k edges), estimate barrier ΔE by sampling loss at midpoints along Hamming path, accept with `P = exp(−w √ΔE₊)` (WKB); else greedy accept if improves |
| **SA** | Classical simulated annealing, Metropolis on ΔE, geometric cooling |
| **RANDOM** | Uniform random feasible wirings, keep best |
| **GREEDY** | Hill-climb: accept only improvements; random restart if stuck |

Budget B (full): **80** reservoir evals / method / seed. Smoke: 12.  
Seeds (held-out): **4100–4107** (n=8). Smoke seed: 91 only.

## Experiment B — Continuous weight training (control)

Train masked recurrent weights + readout with equal forward-eval budget (or wall steps).  
Tasks: NARMA NRMSE (↓), XOR_HARD acc (↑).  
Methods: **SBQA** (Adam + WKB tunneling jumps when grad-norm variance plateaus), **Adam**, **Adam+restarts**, **SGLD** (Adam + isotropic noise).  
Full: N=272, epochs=40, batch=16, seq=32. Smoke: N=136, epochs=6, batch=8, seq=24.  
Seeds: **4200–4207**. Smoke: 91.

## Ablation (on discrete NARMA, same seeds/budget)

1. SBQA-WKB + mask (main)  
2. Same jumps, **Metropolis** acceptance  
3. WKB jumps **without** mask (dense proposals clipped to N×N no-self)  

## Stats

- Primary metric: held-out task score (NARMA NRMSE ↓; XOR_HARD ↑).  
- Paired Wilcoxon signed-rank across 8 seeds.  
- Bonferroni over the **primary family** (α_family=0.05):  
  1. SBQA < SA on discrete NARMA  
  2. SBQA < RANDOM on discrete NARMA  
  3. SBQA < GREEDY on discrete NARMA  
  4. SBQA < Adam on continuous NARMA *(expect FAIL — smooth landscape)*  
- Success criterion (prereg): SBQA beats **both** SA and RANDOM on discrete NARMA (tests 1–2). Continuous is a negative-control prediction: Adam ≥ SBQA.

## Hypotheses

- **H1 (primary):** On discrete wiring search, SBQA beats SA and RANDOM (and ≥ GREEDY) on NARMA.  
- **H2 (negative control):** On continuous weight training, Adam matches or beats SBQA; tunneling does not help smooth slopes.  
- **H3 (ablation):** Mask helps (WKB+mask ≥ WKB no-mask); WKB vs Metropolis — no strong prior, report both.

## Outputs

- `out/discrete_runs.jsonl`, `out/continuous_runs.jsonl`, `out/ablation_runs.jsonl`  
- `out/summary.json`, `RESULTS.md`  
- Kaggle: `/kaggle/working/out/` via `kaggle_pkg/`

## What this is NOT

Real qubits, entanglement messaging, or energy-free training. SBQA = classical stochastic search with a WKB-inspired accept rule + connectome mask.

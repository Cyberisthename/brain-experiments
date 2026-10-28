# Preregistration: Zebra finch Area X song-sequence reservoir

**Date:** 2026-10-28  
**Status:** preregistered BEFORE any eval scores are written  
**Dir:** `/workspace/flybrain/regions/finch/`  
**Data:** real j0251 Area X edges (`SOURCES.md`, `data/areax_module.npz`)

## Motivation

Generic tasks (MC/NARMA/PC) reward fat random spectra; real wiring rarely wins once FIXED delays and per-arm ridge α are matched. Area X is the songbird basal ganglia nucleus required for **song learning**, with HVC and LMAN axons driving MSN/pallidal circuitry (Rother et al. 2025). This experiment asks whether **real Area X wiring** produces / generalizes a syllable sequence from sparse HVC-like bursts better than ER / degree-shuffle / dens-matched controls, with and without FIXED delays.

## Module (locked before eval)

- Source: SyConn j0251 v8 arrays (18.36 M synapses).  
- Locals + top afferents as in `data/areax_module_meta.json` (N≈7576, nnz≈245k).  
- Weight = Σ synapse size (µm²) × Dale-style presynaptic sign.  
- No synthetic Area X edges.

## Arms (eval seeds 0–7)

| Arm | Wiring | Delays | HVC drive set |
|---|---|---|---|
| `REAL_NOD` / `REAL_DEL` | real Area X module | NOD (all d=1) / FIXED Uniform{1..dmax} | **real HVC indices** |
| `ER_NOD` / `ER_DEL` | ER same N+nnz, signed by random E/I match | NOD / DEL | random set size n_HVC |
| `SHUF_NOD` / `SHUF_DEL` | degree-preserving shuffle of real W | NOD / DEL | random set size n_HVC |
| `DENS_NOD` / `DENS_DEL` | dens-matched (same N+nnz, \|w\| bootstrap) | NOD / DEL | random set size n_HVC |

Control HVC sets are seed-tied (`seed + 4242`), same cardinality as real HVC, independent of real HVC IDs.
**Fairness lock:** incoming recurrent edges onto the drive set are zeroed in **all** arms (HVC axon-only ports). Controls still use a random same-size drive set.


**Out of primary scope (not run unless time):** Area X grafted onto FLY_CX vs random grafts — exploratory only if primary finishes.

## Task: HVC-driven syllable sequence

**Motif:** 4 syllables, period `P` steps (default P=80). Each syllable `s` has a target feature vector of dim `K=6` (smooth raised-cosine “spectrogram-like” channels phased across the motif).

**Drive:** `n_HVC` input channels grouped into 4 equal blocks. At the onset of syllable `s`, that block receives a brief pulse (width `W_pulse=4` steps, amplitude 1). This mimics sparse HVC sequential bursts into Area X.

**Readout:** ridge (bias column) from full reservoir state → K-dim song features. α tuned **per arm** on seeds **90–92 only** (criterion: argmin mean SEQ NRMSE). Locked to `out/alpha_choice.json` before eval.

### Metrics (all ↓ better; NRMSE)

1. **SEQ** — train+test at tempo 1. NRMSE on test window vs target features.  
2. **TEMPO** — train at tempo 1; **test at 1.5×** (motif period ⌊P/1.5⌋, same syllable order). NRMSE on test.  
3. **ABLATE** — train intact; **test with 50% of HVC channels zeroed** (random half, seed-tied). NRMSE on test.

NRMSE = RMSE / (std(target on window) + 1e-12), averaged over the K channels (or computed on flattened multi-dim; we use flattened joint NRMSE).

### Protocol constants

- Leak = 0.2; W scaled to ρ = 0.9 (`fast_kernels.radius_cached`).  
- `dmax = 5` FIXED delays (same recipe as `regions/myelin` / `regions/best_fly`).  
- wash / train / test = **200 / 1200 / 400**.  
- α grid: `{1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0}`.  
- Fast kernels: `compress/fast_kernels.py` + bucketed delay CSR (compass / best_fly).  
- No Hebbian, no reward-modulated tutor imitation in this prereg (too many free knobs; can be a follow-up).  
- WORKERS≤2, `nice -n 19`. No GitHub push.

## Stats

- Per-seed paired Wilcoxon signed-rank (two-sided; all-zero diff → p=1).  
- **Primary family (Bonferroni ×9):** `REAL_DEL` vs `{ER_DEL, SHUF_DEL, DENS_DEL}` × `{SEQ, TEMPO, ABLATE}`.  
  Direction: lower NRMSE favors REAL. Report Δ = REAL − ctrl (negative favors REAL).  
- **Secondary family (Bonferroni ×9):** `REAL_NOD` vs `{ER_NOD, SHUF_NOD, DENS_NOD}` × same three metrics.  
- Exploratory: DEL vs NOD within REAL; raw p only.  
- **Verdict rule:** claim “real Area X wins its song-proxy task” only if ≥1 primary contrast survives Bonferroni **and** mean REAL_DEL is better (lower) than all three delay-matched controls on that metric. Otherwise honest null / mixed.

## Predictions (locked)

- **P1:** REAL_DEL beats delay-matched controls on TEMPO (tempo generalization) — Area X / BG circuits are often linked to timing.  
- **P2:** SEQ is easier; advantage may shrink once delays are matched.  
- **P3:** ABLATE (partial HVC loss) may favor real recurrent MSN/pallidal structure.  
- **P4 (null we respect):** If REAL_DEL ≈ controls after correction, delays/readout dominate and Area X wiring is not special under this proxy.

## Honesty / ops

- Report negatives. Do not claim biology from REAL > NOD-only comparisons.  
- No GitHub push.  
- Tune seeds 90–92 never appear in eval tables.

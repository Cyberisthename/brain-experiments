# Preregistration: Real brain chemicals (FlyWire neuromodulators as gain)

**Locked:** 2026-10-29 08:08 America/New_York (EDT), BEFORE any `neuromod_real` eval scores.  
**Dir:** `/workspace/flybrain/regions/neuromod_real/`  
**Parent ask:** user-approved “real brain chemicals” test — DA / 5-HT / OA as slow gain, not fast spikes.

## Question

Do **real** FlyWire-predicted modulatory neurons (dopamine, serotonin, octopamine) and their **real outgoing targets** improve a context-switch reservoir task relative to (a) no neuromod, (b) shuffled NT labels, (c) size-matched ER with the same mechanism?

## Neurotransmitter source (locked)

- Local file: `/workspace/flybrain/data/ann.tsv` (= FlyWire annotations  
  https://github.com/flyconnectome/flywire_annotations  
  commit `1a83b2776d60d5764cef36b927f5f9679c16c47a2`,  
  `supplemental_files/Supplemental_file1_neuron_annotations.tsv`).
- Columns used: `root_id`, `top_nt`, `top_nt_conf`.
- `top_nt` = per-neuron ML-predicted primary transmitter (FlyWire Codex / Eckstein et al. 2024 ultrastructure classifier):  
  acetylcholine, glutamate, gaba, dopamine, serotonin, octopamine.
- Joined to `regions/frankenstein/data/fly_cx_module.npz` `root_id` (CX subgraph, N=2875, FlyWire / Shiu Connectivity_783 v783).
- Cached as `data/fly_cx_nt.npz` + `data/NT_SOURCE.json` (no full ann.tsv in git).

**CX counts (locked):** ACh 1697, Glu 632, GABA 336, **DA 61**, **5-HT 147**, **OA 1**, UNK 1 → **n_mod = 209**.

Existing module `sign` still comes from Shiu `Excitatory x Connectivity` (Dale-style signed weights). This experiment uses `top_nt` only to mark which neurons are **modulatory** vs **fast**.

## Model (frozen)

Leaky tanh reservoir + FIXED synaptic delays + ridge readout.

- Fast network: ACh / Glu / GABA (and UNK) neurons keep their signed fast synapses.
- **Modulatory** neurons (DA, 5-HT, OA):
  - still **receive** fast input and update state;
  - their **outgoing** fast synapses are **zeroed** (they do not drive the fast field);
  - their activity is low-pass filtered: `m ← (1−τ) m + τ |x|` with **τ = 0.1** (fixed);
  - diffusion: `s = Â_mod @ (m ⊙ 1_mod)` where `Â_mod` is the absolute outgoing adjacency of modulatory neurons, **row-normalized** (each postsynaptic row sums to 1; empty rows → 0);
  - multiplicative gain on recurrent+drive before tanh:  
    `y = tanh( g ⊙ (h_delay + drive) )`, `g = clip(1 + γ s, 0.05, 8)`.
- Leak a = 0.2 (scalar); ρ = 0.9 after building W_fast (or full W for NO_NM); DMAX=5 FIXED delays (seed-tied Uniform{1..5}); wash/tr/te = 200/1500/500; drive = L4 / CX drive set from frankenstein `layer_sets`.
- **γ** (modulation strength) and ridge **α** chosen on tune seeds **90–92 only**.  
  α grid: `{1e-8,1e-6,1e-4,1e-2,0.1,1,10}`.  
  γ grid (NM arms only): `{0.25, 0.5, 1.0, 2.0, 4.0}`.  
  Tune criterion: maximize mean **CTX_SWITCH** score on 90–92. Lock `out/alpha_choice.json` before eval.

## Arms

| Arm | Definition |
|---|---|
| `NM_REAL` | real CX W; real mod labels; real mod→target edges for Â_mod |
| `NM_LABEL_SHUF` | real CX W; **permute** NT labels among all CX neurons (same n_mod); rebuild mod set + Â_mod |
| `NM_TARGET_SHUF` | real mod neurons; **rewire** their outgoing posts preserving each mod neuron’s out-degree; W_fast still zeros real mod columns |
| `NO_NM` | **current model**: full signed W including modulatory as fast synapses; no gain path |
| `NM_ER` | ER size/nnz-matched to CX; same n_mod fraction (random mod set, seed-tied); same NM mechanism |

## Tasks

- **Primary:** `CTX_SWITCH` (defined below). Metric: 1−NRMSE ↑.
- **Secondary:** `XOR_DELAY` d=5 (1−NRMSE ↑); `NARMA-10` (NRMSE ↓); `MC` (sum r² ↑, D=100); `PC_HARD` (cue=0.35, noise=1.5, 1−normSSE ↑).

### CTX_SWITCH (locked)

Neuromodulation should matter most when the rule flips.

- Inputs (3-D): bit `a_t ∈ ±1`, bit `b_t ∈ ±1`, context cue `c_t ∈ ±1`.
- Block length **K = 40**. Cue starts at +1 and **flips every K steps** (deterministic alternation; phase tied to seed only via the bit streams, not the flip times).
- Delay **d = 5** for the XOR rule.
- Target:
  - if `c_t = +1` (XOR mode): `y_t = ±1` from `a_{t−d} XOR b_{t−d}`;
  - if `c_t = −1` (COPY mode): `y_t = a_t`.
- Metric: test **1 − NRMSE** (same wash/tr/te). Higher better.

## Seeds / compute

- Tune: **90, 91, 92** only (α, γ). Never peek at eval.
- Eval: **1000–1023** (24 fresh). Paired by seed across arms.
- `WORKERS=2`, `nice`, CPU (N=2875 ≪ 20k; Kaggle GPU previously slower at this size).

## Primary hypotheses (one-sided Wilcoxon, Bonferroni ×3)

All on **CTX_SWITCH**, n=24 paired seeds:

1. **H1:** mean(`NM_REAL`) > mean(`NO_NM`)
2. **H2:** mean(`NM_REAL`) > mean(`NM_LABEL_SHUF`)
3. **H3:** mean(`NM_REAL`) > mean(`NM_ER`)

`p_bonf = min(1, 3·p_raw)`.  
**SUPPORTED** iff all three have `p_bonf < 0.05` **and** the mean inequalities hold.  
Else **NOT SUPPORTED**.

## Secondary (report; not in primary Bonferroni)

- NM_REAL vs NM_TARGET_SHUF on CTX_SWITCH (targets matter?).
- Same arm contrasts on XOR_DELAY, NARMA, MC, PC_HARD.
- Locked γ, α; n_mod; fraction of neurons with nonzero modulatory fan-in.

## Honesty

- Reservoir gain ≠ biological neuromodulation / behavior.
- `top_nt` is ML-predicted, not ground-truth wet-lab chemistry for every cell.
- Push code + prereg + results (no `ann.tsv`, no big `.npz` secrets) to `Cyberisthename/brain-experiments`.
- Report negatives.

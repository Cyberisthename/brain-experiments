# PREREGISTRATION — oscillation_gating (FULL PROTOCOL)

**Frozen at:** 2026-10-30 07:34:00 EDT (America/New_York)  
**Repo path:** `regions/oscillation_gating/`  
**Status:** preregistered before any evaluation scores on held-out seeds 1100–1111.  
**Supersedes:** fast pilot prereg (2026-10-29; n=6, XOR+CHAR only, N=128, ~25 epochs / 2 steps — undertrained null).

## Question

1. **CTC (primary):** Do multi-region rate-RNNs communicate better when sparse long-range links are gated by phase-synchronized low-frequency oscillations (communication-through-coherence), versus incoherent phases, always-on links, or a density-matched random net — on delayed XOR, char LM, and NARMA-10?
2. **Myelin (secondary):** Do heterogeneous long-range delays from the real H01 myelin→delay map (RULE_FF_DEPTH assignment) help alone, and does myelin + coherent gating beat either alone?
3. **Corpus callosum (secondary):** With a two-hemisphere split and hemispheric input split, do homotopic CALLOSUM bridges beat a same-count random bridge and no bridge (and how do they compare to dense all-to-all)?

## Model

- Rate RNN: \(x_{t+1}=(1-\alpha)x_t+\alpha\tanh(r_t+W_{\mathrm{in}}u_t+b)\).
- **4 modules**, \(N_{\mathrm{local}}=40\) → \(N=160\).
- **Intra-region:** dens≈0.12, ungated, delay 0 (instant).
- **Inter-region:** dens≈0.02; delays on inter edges.
  - Non-myelin modular arms: uniform delay \(D=2\).
  - Myelin arms: per-edge delays in \(\{1..5\}\) from H01 `myelin_frac`→delay map, assigned by **RULE_FF_DEPTH** (isotonic on \(|\Delta\mathrm{region}|\); deeper long-range → larger delay), matching `regions/myelin_mechanism` / `myelin_real_rep`.
- **Phase gate (CTC):** \(e_r(t)=\tfrac12(1+\cos(2\pi t/P+\phi_r))\), \(P=8\). Gate \(g_{ij}=e_i e_j\), **mean-renormalized to 1** over one period.
- Trainable weights under frozen masks; Adam + BPTT. Workers **≤ 2**.

## Arms

### Primary / CTC family (parameter-matched: same \(N\), same total nnz ±1%, same train budget)

| Arm | Wiring / gate / delay |
|-----|------------------------|
| `COHERENT` | Modular; all \(\phi_r=0\); uniform D=2 |
| `INCOHERENT` | Same masks; \(\phi_r=2\pi r/4\); D=2 |
| `NO_GATE` | Same masks; \(g\equiv 1\); D=2 |
| `HUB` | Same nnz; region-0 hub; coherent; D=2 (**secondary**) |
| `RAND_MATCH` | ER random, same nnz; no gate; D=2 |

### Myelin (secondary)

| Arm | Spec |
|-----|------|
| `MYELIN_ONLY` | Modular masks; no gate; H01/RULE_FF_DEPTH delays on inter |
| `MYELIN_COHERENT` | Modular; coherent phases; H01/RULE_FF_DEPTH delays |

### Corpus callosum (secondary; separate wiring)

Split into LEFT/RIGHT halves (\(N/2\) each). Intra-hemi dens≈0.12. Input split: XOR uses 2 channels (left←ch0, right←ch1, target=ch0 XOR ch1); NARMA/CHAR drive left half only.

| Arm | Bridge |
|-----|--------|
| `CALLOSUM` | Homotopic reciprocal links (budget \(2\cdot N/2\)) |
| `RAND_BRIDGE` | Same edge count, random cross-hemi |
| `NO_BRIDGE` | Zero cross links |
| `DENSE_BRIDGE` | All-to-all cross-hemi (higher nnz; param-matched only on \(N\) + train budget) |

Graph template seed: **7**.

## Tasks

1. **XOR** — modular: delayed d=5 accuracy ↑; callosum: instantaneous 2-bit hemispheric-split XOR accuracy ↑.
2. **CHAR** — next-char CE on `data/alice.txt` ↓.
3. **NARMA-10** — supervised NRMSE ↓.

## Hyperparameters

Tuned **only** on seeds **90, 91** (never 1100–1111), requiring **NO_GATE XOR ≥ 0.70** (task learnable) and **RAND_MATCH XOR ≥ 0.55** (clearly above chance; ER random is harder on delayed XOR). Grid over lr/leak. Frozen in `out/hparams.json` before eval. Same hparams every arm. Eval epochs=45, steps_per_epoch=20 (full budget; tune may use shorter).

## Evaluation

- Seeds **1100–1111** (n=12), one run per (arm, seed, task).
- Record learning curves.
- ≤2 CPU workers; target wall ≲ 3 h.

## Primary endpoints (Bonferroni)

Family of **9** paired Wilcoxon tests over 12 seeds (α_fam=0.05; per-test α=0.05/9):

1. `COHERENT` vs `INCOHERENT` × {XOR greater, CHAR less, NARMA less}
2. `COHERENT` vs `NO_GATE` × {XOR greater, CHAR less, NARMA less}
3. `COHERENT` vs `RAND_MATCH` × {XOR greater, CHAR less, NARMA less}

**CTC supported** if COHERENT beats INCOHERENT on **all three** tasks after correction.  
**Overall win** if COHERENT also beats NO_GATE and RAND_MATCH on all three after correction.

## Secondary endpoints (descriptive / uncorrected + reported raw p)

- **Myelin:** MYELIN_ONLY vs NO_GATE; MYELIN_COHERENT vs MYELIN_ONLY; MYELIN_COHERENT vs COHERENT — on XOR↑, NARMA↓.
- **Callosum:** CALLOSUM vs RAND_BRIDGE, NO_BRIDGE, DENSE_BRIDGE — on XOR↑ (split task). CHAR/NARMA descriptive.

## Analysis

`analyze_oscillation_gating.py` → `RESULTS_oscillation_gating.md` + `figs/oscillation_gating.png`.

## Honesty

Null allowed. No peeking at 1100–1111 before freezing hparams. Fast pilot was undertrained (chance XOR); this protocol first verifies baselines learn XOR on tune seeds.

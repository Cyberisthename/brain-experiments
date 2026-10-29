# PREREGISTRATION — oscillation_gating (FAST PILOT)

**Frozen at:** 2026-10-29 13:57:30 EDT (America/New_York)  
**Repo path:** `regions/oscillation_gating/`  
**Status:** preregistered before any evaluation scores on held-out seeds.  
**Scope note:** This is a **fast pilot** (target ~20–30 min wall). Relative to a full protocol we shrink N, epochs, drop NARMA, and use **n=6** held-out seeds (1100–1105) instead of 12. Primary logic (CTC contrast, Bonferroni, param matching, tune≠eval seeds) is unchanged; treat positive hits as provisional pending a full-n replication.

## Question

Do multi-region rate-RNNs communicate **better** when sparse long-range (“white matter”) links are **gated by phase-synchronized low-frequency oscillations** (communication-through-coherence / CTC), versus incoherent phases, always-on links, a hub+feedback layout, or a density-matched random sparse net — on delayed XOR and a tiny character LM?

## Biological motivation (user brief)

Regions talk via axon bundles / white matter, action potentials + synapses, **neural oscillations with phase synchronization** (shared low-frequency windows for high-frequency payload), broadcast hubs, and feedback loops. This pilot isolates the **oscillation / CTC** piece (+ a hub arm).

## Model

- Rate RNN: \(x_{t+1}=(1-\alpha)x_t+\alpha\tanh(r_t+W_{\mathrm{in}}u_t+b)\).
- **4 modules**, \(N_{\mathrm{local}}=32\) → \(N=128\).
- **Intra-region:** denser local mask (dens≈0.15), ungated.
- **Inter-region (“white matter”):** sparse directed (pairwise dens≈0.025); shared delay buffer \(D=2\) on inter links (myelin stand-in; same for gated/modular arms).
- **Phase gate (CTC):** \(e_r(t)=\tfrac12(1+\cos(2\pi t/P+\phi_r))\), \(P=8\). Gate \(g_{ij}(t)=e_i(t)e_j(t)\), **renormalized over one period to mean 1** (matched gain; only alignment differs).
- Trainable weights under frozen masks; Adam + BPTT. Workers **≤ 2**.

## Arms (parameter-matched: same \(N\), same total nnz ±1%, same train budget)

| Arm | Wiring / gate |
|-----|----------------|
| `COHERENT` | Modular masks; all \(\phi_r=0\). |
| `INCOHERENT` | Same masks; \(\phi_r=2\pi r/4\) (quadrature). Mean gate matched. |
| `NO_GATE` | Same masks; \(g\equiv 1\). |
| `HUB` | Same nnz; region 0 broadcast hub + reciprocal feedback; coherent phases. |
| `RAND_MATCH` | ER random mask, same total nnz; no modules; \(g\equiv 1\). |

Graph template seed: **7**.

## Tasks (pilot)

1. **delayed XOR**, d=5 (accuracy; **primary**).
2. **char next-char** on `data/alice.txt` (CE; **primary**).
3. NARMA-10: **deferred** (not run in this fast pilot).

## Hyperparameters

Tuned **only** on seeds **90, 91** (never 1100–1105): lr, leak. Grid on `COHERENT`+`INCOHERENT`, XOR+CHAR. Frozen in `out/hparams.json` before eval. Same hparams every arm. Pilot epochs ≈ 25.

## Evaluation

- Seeds **1100–1105** (n=6), one run per (arm, seed, task).
- Record learning curves.

## Primary endpoints (Bonferroni)

Family of **6** paired Wilcoxon tests over 6 seeds (α_fam=0.05; per-test α=0.05/6):

1. `COHERENT` vs `INCOHERENT` × {XOR greater, CHAR less} — **key CTC contrast**
2. `COHERENT` vs `NO_GATE` × {XOR greater, CHAR less}
3. `COHERENT` vs `RAND_MATCH` × {XOR greater, CHAR less}

**CTC supported (pilot)** if COHERENT beats INCOHERENT on **both** after correction.  
**Overall win** if COHERENT also beats NO_GATE and RAND_MATCH on both after correction.  
`HUB` is **secondary** (not in primary family).

## Analysis

`analyze_oscillation_gating.py` → `RESULTS_oscillation_gating.md` + `figs/oscillation_gating.png`.

## Honesty

Null allowed. No peeking at 1100–1105 before freezing hparams. n=6 has low power; null/positive both provisional.

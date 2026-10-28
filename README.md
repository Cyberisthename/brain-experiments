# brain-experiments

Preregistered reservoir-computing experiments on real connectomes: FlyWire central complex (CX), grafts from mouse MICrONS and human H01 cortex, plus C. elegans, zebrafish (Fish1), and zebra finch Area X probes.

**Status:** active research scratchpad. Headline numbers below come only from committed `RESULTS*.md` files. See also `ANALYSIS_cross_species.md` for an evidence-only cross-species write-up.

## Honest results summary (current)

- **Delays roughly triple memory for every wiring type.** Fixed / myelin-style delays lift MC from ~6 to ~15–17 on fly/Frankenstein-sized nets; the same boost appears on ER / shuffle / density-matched controls. Delays are the main win so far, not species-specific wiring.
- **Real H01 myelin placement beat shuffled myelin on NARMA in the human patch** (`regions/myelin_real/`, HUMAN_LAYERED NARMA: p_bonf ≈ 0.002). That contrast did **not** transfer to the Frankenstein graft in the same run. **Replication held:** 24 new seeds, REAL NARMA **0.393** vs SHUF **0.557**, p_one_sided ≈ **2×10⁻⁹**, `claim_ok=True` (`regions/myelin_real_rep/`). LAYER_SHUF is in between (0.441); EI_SHUF ≈ SHUF.
- **Frankenstein’s gains are mostly explained by size.** Best-of fly (Frankenstein + real myelin) beats plain fly+delays on pattern completion in raw means, but size-/recipe-matched ER/shuffle controls match or beat it after Bonferroni (`regions/best_fly/`).
- **Earlier few-shot “real wiring learns from fewer examples” advantage was a ridge-α artifact** and was corrected: with per-arm tuned α, real vs random few-shot MC ties (`regions/fewshot_delay/`).
- **Compass / path-integration test was null**; all arms (fly and controls) were near-failing (~1.2 rad RMSE ≈ 70°), so the task did not discriminate wiring (`regions/compass/`).
- **Zebra finch Area X song-sequence test was null after Bonferroni** (0/9 REAL_DEL vs delay-matched controls). Density/ER controls beat REAL on sequence NRMSE; delays still help (`regions/finch/`).

Level framing: we are at **Level 1** for generic wiring/composition. The myelin-placement NARMA result is the main **Level-2 candidate** and is now **replicated** on new seeds (`myelin_real_rep`). Cross-species stitching still fails size-/delay-matched controls (`ANALYSIS_cross_species.md`).

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
| Synthetic hippocampus | `regions/hippo/` | Extra neurons help NARMA/T0, but **random module ≈ hippocampus**. |
| EMPN organ-fix maze (v2b) | `regions/empn/` | Extra organs **do not** beat same-size RANDMOD after Bonferroni. |
| Myelination (synthetic delays) | `regions/myelin/` | FIXED delays MC **45.1** > ADAPT **43.0** > NODELAY **40.4** > ADAPT-SHUF **37.7**. |
| **Real H01 myelin placement** | `regions/myelin_real/` | HUMAN_LAYERED: REAL beats SHUF on NARMA (p_bonf≈0.002). FRANKEN: no REAL vs SHUF win. |
| **Myelin REAL vs SHUF replication** | `regions/myelin_real_rep/` | **REPLICATED.** REAL 0.393 vs SHUF 0.557 NARMA (n=24, p≈2e-9). LAYER/EI shuffles show placement matters. |
| **Developmental maturation** | `regions/mature/` | Hebb/prune/full maturation; many Bonf survivors on ΔΔ — see RESULTS; absolute scores often degrade. |
| **Cross-species analysis** | `ANALYSIS_cross_species.md` | Size/spectrum, delays, weak bridges (σ=0.05), myelin win, consistent nulls, next tests. |
| **Smartest fly (best-of)** | `regions/best_fly/` | Delays dominate; BEST ≈ Frankenstein+real myelin; ER_BEST raw-best; no BEST vs control contrast survives Bonferroni. |
| **Few-shot + matched delays** | `regions/fewshot_delay/` | Per-arm ridge α removes earlier “few-shot real>random” claim. Delays still help all wirings. |
| **Fly compass / heading** | `regions/compass/` | Null: 0/9 primary contrasts; all arms ~1.2 rad RMSE (task too hard / wrong readout). |
| **C. elegans (8 worms)** | `regions/celegans/` | Individual-variation Witvliet graphs; see RESULTS for real-vs-random across worms/ages. |
| Zebrafish Fish1 | `regions/zebrafish/` | **Blocked:** needs CAVE/DAF auth; no public synapse table. Provenance in `SOURCES.md`. No invented graph. |
| **Zebra finch Area X** | `regions/finch/` | Null after Bonferroni (0/9). ER/DENS beat REAL on SEQ NRMSE; delays help all arms. |
| Fly–mouse / human hybrids | `regions/hybrid/`, `hybrid_human/` | Modules built; see earlier RESULTS / diagnoses. |
| Frankenstein (fly+mouse+human) | `regions/frankenstein/` | Cross-species graft; size explains much of the gain once delays are matched. |

## How to fetch data (not in this repo)

Connectome binaries are **excluded** (multi‑GB). Rebuild from public sources:

1. **Fly CX (FlyWire v783)** — Shiu et al. 2024: https://github.com/philshiu/Drosophila_brain_model  
2. **Mouse MICrONS** — Zenodo DOI [10.5281/zenodo.16744240](https://zenodo.org/records/16744240); provenance `regions/hybrid/data/SOURCE.json`  
3. **Human H01** — https://h01-release.storage.googleapis.com/landing.html ; provenance `regions/hybrid_human/data/SOURCE.json`  
4. **C. elegans** — Witvliet et al. 2021 (8 developmental reconstructions); see `regions/celegans/SOURCES.md`  
5. **Zebrafish Fish1** — https://fish1-release.storage.googleapis.com/index.html (CAVE token required for edges); see `regions/zebrafish/SOURCES.md`  
6. **Zebra finch** — Area X / HVC EM samples; see `regions/finch/SOURCES.md`  
7. **Octopus** — no usable neuron-level edge list (`regions/hybrid_octo/`)

## Honest caveats

- One rate model, synapse counts as weights, predicted transmitters as signs — not a biophysical animal.
- “Smarter than a fly” claims are wrong: these are abstract reservoir scores, not behavior.
- Many early “wins” reverse under fair tuning, matched delays, or density-matched controls.
- Incomplete arms that failed for code/auth bugs are labeled as such — do not cite as biological nulls.
- Bonferroni and prereg contrasts are confirmatory; uncorrected p-values are exploratory.
- Do not commit secrets, CAVE tokens, `.npz`/`.npy` raw graphs, or huge logs.

## Layout

```
flybrain_reservoir.py   # core CX reservoir
round2.py / round3.py   # tuning + weight training
compress/               # prune / LoRA / SAG
figs/                   # summary figures for key runs
regions/                # grafts, EMPN, myelin, hybrids, compass, worms, …
  */PREREGISTRATION*.md
  */RESULTS*.md
  */SOURCES.md          # provenance (no binaries / no tokens)
```

Shell helpers (`run_*.sh`) queue long jobs; see `.gitignore` for excluded artifacts.

## License / credit

Upstream connectomes retain their own licenses (FlyWire / Shiu model; MICrONS / Zenodo; H01 CC BY 4.0; Witvliet 2021; Fish1 authors). Experiment code here is research scaffolding unless otherwise noted.

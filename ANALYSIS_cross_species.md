# ANALYSIS: How cross-species brains and brain segments behave in our reservoirs

**Scope.** Evidence-only write-up of what our own RESULTS files and the Frankenstein / human-block eigenvalue diagnoses show.  
**Sources used:** `regions/*/RESULTS*.md`, `regions/hybrid_human/out/DIAGNOSIS_random_8_85.md`, `figs/franken_figs_meta.json` (franken_eigs work).  
**Not sources:** speculation from outside papers, unpublished local runs without a RESULTS file, or CAVE/Fish1 edges we never obtained.

Facts are labeled **[FACT]**. Interpretations are labeled **[HYPOTHESIS]** or **[INTERPRETATION]**.

---

## 0. What we built

**[FACT]** We turn real connectome patches into echo-state / rate reservoirs (spectral radius fixed per arm; linear readout trained unless a round says otherwise). Controls are Erdős–Rényi (ER), degree-preserving shuffle (SHUF), and density-matched random graphs (DENS), size-matched where claimed.

**[FACT]** Cross-species “Frankenstein” grafts stack mouse MICrONS + human H01 + fly CX blocks (N≈6833: mouse 1871, human 2087, fly_cx 2875). Bridges go through the fly hub only (no mouse↔human); bridge weight scale σ = 0.05; global ρ ≈ 0.9 (`frankenstein.py`, `franken_figs_meta.json`).

**[FACT]** Tasks in most arms: memory capacity (MC ↑), NARMA prediction error (↓), pattern completion (PC_HARD ↑). Later arms add few-shot curves, compass heading RMSE, worm smell tracking, finch song-sequence NRMSE, and maturation conditions.

---

## 1. Size and spectral breadth explain most generic-task gains

**[FACT — franken_eigs]** After scaling each matrix to spectral radius 1, Frankenstein has only **~0.48%** of eigenvalues with |λ| > 0.5 (33 / 6833), while a same-size ER graft has **~24.5%** (1676 / 6833). Mean |λ| is ~0.066 (Franken) vs ~0.334 (ER). Fly CX alone is similarly sparse in the bulk (~0.63% above 0.5).

**[FACT — DIAGNOSIS_random_8_85]** In the isolated human H01 block (N=2087, matched nnz), real wiring has **11** eigenvalues with |λ| > 0.5 vs **769** for ER; effective rank of L4-driven states ≈ 1.51 (real) vs 3.07 (ER); active neuron fraction ≈ 0.29 vs 0.88. Extra seeds keep RANDOM_LAYERED MC ~2–3 points above HUMAN_LAYERED (mean MC 5.78 vs 8.40).

**[FACT — best_fly]** With delays matched, BEST (Frankenstein + real myelin) MC 16.96 / NARMA 0.485 / PC 0.881 vs ER_BEST 19.92 / 0.384 / 0.926. **No BEST-vs-control contrast survives Bonferroni ×9.** Secondary: BEST beats fly+fixed delays on PC_HARD (p_bonf ≈ 0.047).

**[INTERPRETATION]** Generic reservoir scores reward many weakly coupled memory modes. Sparse biological blocks collapse to a few dominant loops after ρ-normalization, so random graphs with the same N and nnz often supply more independent channels and win MC / NARMA / PC. Cross-species stacking helps mainly by **increasing N** (and sometimes by adding delays), not by creating a new integrated computation that beats size-matched ER.

**[HYPOTHESIS]** Real modules are optimized for specific dynamical motifs (compass rings, laminar pathways), which look “low-rank” under a linear readout on white-noise / NARMA probes — the opposite of what MC maximizes.

---

## 2. Delays help any wiring (they add temporal dimensions)

**[FACT]** Across arms, fixed or myelin-style delays roughly **triple** MC vs no-delay baselines, for real and random alike:

| Setting | Approx. MC (real) | Approx. MC (random / ER) | Source |
|---|---:|---:|---|
| Fly / Franken, no delay | ~6–7 | ~6–8 | `best_fly`, `mature` age-0 `none` |
| Same + delays | ~15–17 | ~18–20 | `best_fly` |
| Worms NOD → DEL | ~5 → ~13 | ~5–6 → ~14–15 | `celegans` |
| Few-shot @ n=100 NOD → DEL | ~3.3 → ~8–9 | similar | `fewshot_delay` |

**[FACT]** Few-shot with **per-arm** ridge α: delays still dominate; FRANKEN_DEL vs delay-matched controls is **0/9** Bonferroni wins on MC/NARMA/PC (`fewshot_delay`). Earlier “real learns from fewer examples” result does **not** replicate under fair α (that win was a fixed-α overfitting confound on high-dimensional random arms).

**[INTERPRETATION]** Delays expand the temporal feature space of *every* reservoir. They are the largest, most reproducible engineering lever in this project. They are **not** evidence that biological adjacency is special, because ER/SHUF/DENS get the same boost when given the same delay recipe.

---

## 3. Weak bridges (σ = 0.05) → parallel blocks, not one integrated brain

**[FACT]** Frankenstein bridges use σ = 0.05 between fly hub and mouse/human L4/L5 ports; there are **no** mouse↔human edges (`BRIDGE_SIGMA`, `franken_figs_meta.json`).

**[FACT]** Frankenstein’s spectrum remains sparse-bulk like the real blocks (~0.5% |λ|>0.5), not ER-like (~24%). In `myelin_real`, REAL vs SHUF myelin helps **HUMAN_LAYERED** NARMA but **not** FRANKEN NARMA (Δ ≈ −0.0009, n.s.) — the graft does not inherit the human-only placement win.

**[INTERPRETATION]** At this bridge scale the species blocks behave largely as **parallel, weakly coupled reservoirs** whose states are concatenated for a linear readout. That is useful for “more neurons,” but it is a weak test of whether *combining* species circuits yields complementary computation. A true integration claim would need stronger / anatomically motivated long-range pathways and tasks that require both blocks.

**[HYPOTHESIS]** Raising bridge σ, adding directed long-range motifs, or training a nonlinear / multi-readout head that must mix blocks would change this picture; we have not preregistered that test yet.

---

## 4. The one tissue-specific win: real H01 myelin placement

### Parent result

**[FACT — myelin_real]** On HUMAN_LAYERED, REAL myelin vs SHUF myelin NARMA: Δ = −0.130, p_raw ≈ 3.1×10⁻⁴, **p_bonf ≈ 0.0019** (Bonferroni ×6). MC and PC do not clear the bar. Same contrast on FRANKEN: null.

### Replication (new seeds)

**[FACT — myelin_real_rep]** Preregistered primary: REAL vs SHUF NARMA at leak=0.2, one-sided, **n=24 new seeds**.

| Arm | NARMA (mean) |
|---|---:|
| REAL | **0.3934** |
| SHUF (3 draws/seed) | **0.5575** |
| LAYER_SHUF | 0.4408 |
| EI_SHUF | 0.5560 |
| NODELAY | 0.5916 |

- Δ (REAL−SHUF) = **−0.164**; p_one_sided ≈ **1.97×10⁻⁹**; **claim_ok = True**.  
- Holds at leak 0.1 / 0.2 / 0.3 (all p < 10⁻⁷).  
- REAL vs LAYER_SHUF: Δ = −0.047, p_raw ≈ 3.6×10⁻⁵.  
- EI_SHUF ≈ global SHUF (class-level myelin mass is not enough).  
- MC rises with *any* delays; REAL vs SHUF on MC remains small; PC near-tied.

**[INTERPRETATION]** This is the first result in the project that (a) is about **real tissue structure** (myelin placement on H01), (b) beats a matched shuffle of the same delays, and (c) **replicates** on held-out seeds under a written prereg. It is still a **single patch + prediction task** result — not a cross-species composition breakthrough.

**[HYPOTHESIS — mechanism]** Myelin in cortex is not uniform noise: it is biased toward specific long-range / myelinated projection neurons. Preserving that map gives **structured delay diversity** aligned with projection roles. Global shuffle keeps the delay *histogram* but destroys which neurons get which delays; layer shuffle keeps laminar averages but destroys within-layer assignment (partial loss); E/I-class shuffle destroys almost everything. Structured delay diversity can help NARMA (needs timed nonlinear memory) more than raw MC (which likes many equal channels).

**[FACT]** We have **not** yet shown the same placement win in a second species’ myelin map (Fish1 blocked on CAVE auth; worms/finch used generic or fixed delays, not measured myelin).

---

## 5. Consistent null: real wiring vs random on generic (and several “native”) tasks

**[FACT]** After fair size / delay / regularization matching, real adjacency does **not** beat ER/SHUF/DENS on the main generic suite:

| Brain / arm | Headline | Source |
|---|---|---|
| Fly CX (fair tune) | ER beats fly on MC | `RESULTS.md` R2 |
| Frankenstein / best-fly | 0/9 BEST vs controls after Bonf | `best_fly` |
| Few-shot + delays | 0/9; prior few-shot win was α artifact | `fewshot_delay` |
| Human block MC | ER > real (~8.4 vs 5.8) | `DIAGNOSIS_random_8_85` |
| C. elegans (8 Witvliet + Cook) | REAL MC **below** ER in every worm; delays triple both | `celegans` |
| Zebra finch Area X | 0/9 REAL_DEL vs controls; ER/DENS lower SEQ NRMSE | `finch` |
| Fly compass | 0/9; all arms ~1.2 rad RMSE (~70°) | `compass` |

**[FACT — limits of the null]**

1. **Tasks** are mostly generic reservoir benchmarks (MC, NARMA, PC) or weak proxies (song NRMSE with linear readout; compass with the same linear setup).  
2. **Readout** is linear; native circuits may need their own dynamics / attractors (compass ring), not a least-squares map. Compass near-failure for *all* arms means the test did not discriminate wiring.  
3. **Worms** use **unsigned** weights (E/I unknown) — a serious biological misspecification.  
4. **Zebrafish Fish1** never entered the scoreboard: no public synapse table without a CAVE token (`zebrafish` RESULTS = blocked).  
5. **Maturation** (`mature`) finds many Bonferroni survivors on ΔΔ metrics, but absolute scores often degrade with Hebb/prune; treat as exploratory developmental dynamics, not a generic-task win for real adjacency over ER.

**[INTERPRETATION]** The honest cross-species story on *these* probes is: **delays ≫ wiring identity ≫ species stitching**. Stitching mainly adds neurons; random graphs with the same N, nnz, and delays usually match or beat real adjacency.

---

## 6. Concrete next tests (toward Level 2 / 3)

Only tests that close a real gap above:

1. **Native dynamics, not just linear readouts** — e.g. fly CX heading with an explicit ring / bump attractor; re-run compass only if the dynamics can succeed at all.  
2. **Realistic long-range bridges** — replace σ=0.05 random hub links with pathway-inspired projections; preregister an integration task that *requires* both blocks.  
3. **Myelin placement in a second dataset** — H01 replication is done; next is Fish1 (after CAVE token) or another myelinated volume with per-neuron myelin proxies, same REAL vs SHUF vs LAYER_SHUF design.  
4. **Keep fair α / size / delay matching** — never again claim few-shot or composition wins without per-arm regularization and size-matched controls.  
5. **Signed worm graphs** if transmitter predictions become available — current unsigned nulls are informative but incomplete.

---

## Level check (same framing as the lab notes)

| Level | Meaning | Where we are |
|---|---|---|
| 1 | Engineering / scaling: bigger + delays help; biology not required | **Here** for almost all generic tasks |
| 2 | Biological structure beats matched randomization on relevant tasks | **Candidate:** H01 myelin REAL > SHUF NARMA, **replicated** (`myelin_real_rep`) |
| 3 | Cross-species composition yields complementary generalization | **Not shown** — bridges weak; Franken loses to ER when size/delays matched |

---

## One-paragraph summary

Cross-species reservoirs in this repo behave like **weakly coupled parallel modules**: sparse real spectra (~0.5% of eigenvalues above 0.5 vs ~24% for ER) explain why random wiring often wins memory and prediction; **delays** add temporal channels for every wiring type; **σ=0.05 bridges** do not forge an integrated multi-species brain. The sole tissue-specific claim that cleared a preregistered bar and a new-seed replication is **real H01 myelin placement** (NARMA 0.393 vs shuffled 0.557, p≈2×10⁻⁹), with layer and E/I shuffles showing that *which* neurons are myelinated matters. Everything else — fly, mouse, human adjacency, eight worms, finch Area X, compass — is a careful null or a size/delay story once controls are fair.

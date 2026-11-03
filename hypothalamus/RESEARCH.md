# Hypothalamus research notes (for flybrain / callosum / JARVIS)

**Written:** 2026-11-03 ~08:45 America/New_York  
**Scope:** What the hypothalamus does, which circuit motifs transfer to our engines, the fly analogue (pars intercerebralis / neurosecretory cells), open human connectivity data, and 3 falsifiable tests we can run on `engine_v3` without inventing a new stack.  
**Honesty bar:** classical / bio-inspired control, not real qubits. Motifs below are abstractions; a win on a reservoir task is *not* proof we built a hypothalamus.

---

## 1. Core functions (human / mammal)

The hypothalamus is the densest hub of **homeostatic set-point control** in the brain: sensors → compare to a neural set point → adjust autonomic, endocrine, and behavioral output → feedback closes the loop.

| Function | What it means | Key nuclei / pathways | Primary sources |
|---|---|---|---|
| **Homeostasis & set points** | Baseline firing rates encode targets (temp, osmolarity, HR, etc.); deviations change hypothalamic outflow to restore balance | Many nuclei; PVN, preoptic, arcuate, VMH, LH | UT Houston Neuroscience Online Ch.1: https://nba.uth.tmc.edu/neuroscience/m/s4/chapter01.html |
| **Drives / motivation** | Hunger, thirst, thermoregulation, sex, defense — states that bias behavior and suppress rivals | ARC AgRP/POMC, LH orexin/MCH, VMH | Hunger competition: https://www.cell.com/neuron/fulltext/S0896-6273(16)30525-6 ; AgRP review: https://pmc.ncbi.nlm.nih.gov/articles/PMC9040143/ |
| **Circadian (SCN)** | Master clock (~24 h). Light via retinohypothalamic tract entrains clock genes (*Per*, *Cry*, *Bmal*). SCN times slave oscillators (hormones, sleep, metabolism) | Suprachiasmatic nucleus (SCN) → medial hypothalamus | Same UT chapter; SCN outputs review: https://doi.org/10.1177/0748730406293854 |
| **Neuromodulation / neuropeptides** | Peptides (orexin, oxytocin, vasopressin, α-MSH, NPY, AgRP, CRH…) act mostly via GPCRs on seconds–minutes timescales; often **extrasynaptic** | Sparse peptidergic populations with brain-wide reach | Volume transmission review: https://www.nature.com/articles/s41380-024-02608-3 ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11540752/ |
| **HPA axis (stress hormones)** | PVN CRH/AVP → pituitary ACTH → adrenal glucocorticoids; GCs feedback-inhibit PVN/pituitary. SCN gates basal HPA tone | PVN ↔ pituitary ↔ adrenal | Ulrich-Lai & Herman: https://pmc.ncbi.nlm.nih.gov/articles/PMC4240627/ ; systems HPA+circadian: https://pmc.ncbi.nlm.nih.gov/articles/PMC8900069/ |

**Takeaway for AI:** hypothalamus-like modules are **slow controllers** (set points, gains, learning rates, which “drive” wins), not fast spike-by-spike computers. That maps cleanly onto our OscRNN flags (leak, bias, LR schedules) more than onto adding more local synapses.

---

## 2. Circuit motifs worth stealing

### 2.1 Setpoint feedback control
Error = measured − set point → corrective command → plant → feedback. Classic negative-feedback homeostat (UT Neuroscience Online).  
**Engine map:** a slow state `e_t` tracking mean activity or loss; adjust bias / gain / LR to drive `|error| → 0`. We already have a seed of this: v3 arm **HOMEO** = slow bias homeostasis toward `|x̄|=0.3` (`engine_v3/PREREG_v3_features.md`).

### 2.2 Slow global modulatory broadcast
Monoamines and neuropeptides do **volume transmission**: release from dense-core vesicles, diffuse in extracellular space, hit GPCRs without needing a synapse. Spatiotemporal scale ≫ 1 ms synaptic transmission (Nature Mol Psychiatry 2024). Dendritic oxytocin/vasopressin release is explicitly “population → population” messaging (Phil Trans B: https://royalsocietypublishing.org/doi/10.1098/rstb.2014.0182).  
**Engine map:** one (or few) slow scalars broadcast as multiplicative gain / additive bias / LR multiplier on *all* reservoir units — not a sparse adjacency change.

### 2.3 Drive competition (not pure WTA, but rivalry)
ARC **AgRP** (orexigenic) vs **POMC** (anorexigenic) form a reciprocal system. AgRP can suppress competing drives (thirst, fear, social) when food is available (Neuron 2016). Downstream, NPY and α-MSH **compete** over cAMP in PVH MC4R neurons; feeding resolves the competition over minutes (Nature 2024: https://www.nature.com/articles/s41586-024-08164-8). Reciprocal AgRP↔POMC control of feeding/metabolism: https://doi.org/10.1038/s42255-024-00987-z.  
**Engine map:** 2–K slow “drive” units with mutual inhibition + a shared resource (gain budget). Soft rivalry / softmax is closer to biology than hard argmax WTA.

### 2.4 Volume / paracrine transmission
Same as 2.2, but stress the mismatch: peptide axons and receptors can be millimeters apart; specificity is receptor identity, not wiring. Fly clock → neurosecretory signaling is also sparse monosynaptic + peptide paracrine (FlyWire clock paper below).  
**Engine map:** do **not** require a connectome edge for the modulator; apply a global field. Synaptic-only ablation = control that forces the same signal through random local edges.

---

## 3. Fly equivalent

| Mammal | Fly analogue | Notes | Sources |
|---|---|---|---|
| Hypothalamus (neuroendocrine + drive hub) | **Pars intercerebralis (PI)** — often called protohypothalamic | Sleep, feeding, peripheral cycling; heterogeneous neurosecretory peptides | PNAS 2021: https://www.pnas.org/doi/10.1073/pnas.2019826118 ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7958456/ |
| Insulin / metabolic hormones | **Insulin-producing cells (IPCs)** in PI; Drosophila insulin-like peptides (**DILP2/3/5**) | ~14–18 adult m-NSC DILP; metabolic homeostasis, growth, aging, feeding | eLife neurosecretory connectome: https://doi.org/10.7554/eLife.102684 ; PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11384003/ ; IPC modulation: https://pmc.ncbi.nlm.nih.gov/articles/PMC11893105/ |
| CRH-like stress / activity peptides | **DH44+** PI neurons (CRF ortholog) | Rest:activity rhythms | PNAS / PMC above |
| SCN clock | **Clock neuron network** (s-LNv, l-LNv, LNd, DN1/2/3, …) | ~240 neurons complete in FlyWire; hemibrain has **64** identified clock cells (missing mostly DN3 + contralateral) | Nat Commun 2024: https://www.nature.com/articles/s41467-024-54694-0 |
| Volume / peptide broadcast | Clock → NSC often **paracrine** (receptors present; sparse synapses) | Connectome ≠ full story for neuromodulation | Same Nat Commun + eLife papers |

### Do FlyWire / hemibrain include them?
- **Yes for annotations:** FlyWire Codex annotates medial neurosecretory cells including **m-NSC DILP / IPC** and related NSC classes; Virtual Fly Brain term FBbt_00004011 (insulin-secreting cell of PI): https://www.virtualflybrain.org/term/insulin-secreting-cell-of-pars-intercerebralis-fbbt_00004011/ ; Codex: https://codex.flywire.ai/
- **Hemibrain:** partial clock set (64 cells); incomplete contralateral; still usable for ipsilateral PI/NSC wiring checks (Scheffer et al. 2020 hemibrain; clock comparison in Nat Commun 2024).
- **Caveat for us:** synaptic graphs undercount peptide volume transmission. A “PI subgraph” experiment should treat IPC/DH44 outputs as **slow broadcast** candidates, not only as adjacency rows.

---

## 4. Open human hypothalamus connectivity data

| Resource | What it is | Hypothalamus? | URL |
|---|---|---|---|
| **HCP-MMP 1.0** | Cortical surface atlas (~180 areas/hemisphere) | **No** — cortical; hippocampus is the only “subcortical-ish” exception in many pipelines. FreeSurfer aseg lumps hypo into **ventral diencephalon**, not a clean hypo parcel | Overview: https://github.com/tannerjared/HCP-MMP1 ; FreeSurfer aseg: https://surfer.nmr.mgh.harvard.edu/fswiki/SubcorticalSegmentation |
| **Neudorfer et al. 2020 hypothalamic MRI atlas** | High-res in vivo atlas from **990 HCP** T1/T2 scans; **13** hypothalamic nuclei + 12 nearby structures per hemisphere | **Yes** — nuclei (SCN, PVN, ARC, VMH, LH, …) | Paper: https://www.nature.com/articles/s41597-020-00644-6 ; Zenodo: https://doi.org/10.5281/zenodo.3942115 |
| **Probabilistic hypothalamic pathway atlas (HCP dMRI)** | **22** hypo-centered bundles (11/hemisphere): fornix, stria terminalis, MTT, MFB segments, DLF, etc. | **Yes** — tract *pathways* to/from hypo, not nucleus→nucleus synapse matrix | ScienceDB: https://doi.org/10.57760/sciencedb.45751 |
| **Yeh HCP1065 / HCP842 tractography atlases** | Population tractograms; some hypo-related tracts (e.g. DLF) | Partial / coarse | https://brain.labsolver.org/hcp_trk_atlas.html ; Zenodo HCP-YA: https://zenodo.org/records/3627772 |
| **Our callosum / Lausanne / MMP graphs** | Cortical (and some subcortical) SC we already use | **Hypothalamus not a first-class node** in HCP-MMP / typical Lausanne cortical graphs | Local: `connectomes/human/` |

**Practical verdict for our lab:**
1. We **cannot** honestly claim a “real hypothalamus connectome wiring” test from HCP-MMP callosum graphs — the atlas does not include hypo as cortex.
2. We **can** download Neudorfer nuclei ROIs + the probabilistic pathway atlas and build a **coarse hypo-centered SC prior** (pathways as soft long-range channels). That is MRI tractography scale, not synapse scale — same honesty level as Griffa/callosum work.
3. For **motif** tests (below), we do **not** need hypo SC: broadcast / homeostat / drive rivalry are functional abstractions that run on callosum_scaled or fly wiring we already have.

---

## 5. Implications for JARVIS (from-scratch LM)

Hypothalamus motifs are a poor fit for “add more RAG tokens.” Better fits:
- **Slow gain on retrieval vs generation** (drive: “look up” vs “babble”) — mirrors AgRP-style suppression of competing policies.
- **Setpoint on answer confidence / contradiction** — if retrieval agreement is low, raise hybrid weight (homeostat).
- **Circadian / session phase** — optional later; not urgent.
- Do **not** expect peptide biology to appear as a transformer layer; at best a **scalar controller** outside the LM core (same spirit as pack router + CAD already winning on JARVIS evals).

---

## 6. Proposed falsifiable tests (engine_v3)

All use existing stack: sparse OscRNN, callosum_scaled REAL_SC @ N_label=512 (or fly wiring), tasks XOR_HARD / NARMA / CHAR / TRANSFER, paired Wilcoxon on matched seeds, Bonferroni across planned contrasts. Prereg before looking.

**Already related:** v3 **HOMEO** arm (slow bias toward `|x̄|=0.3`). Treat tests below as *extensions*, not duplicates — if HOMEO already wins, Test A is the natural next step.

### Test A — Error-driven homeostat vs matched random modulator
**Idea:** Setpoint feedback (motif 2.1 + 2.2).  
**Arms:**
1. **HOME_O_ERR:** slow state `e ← (1−β)e + β(‖h‖₁/N − s*)`; broadcast `g = 1 + κ·tanh(e)` as multiplicative gain on reservoir input (or LR multiplier). `s*` fixed set point (e.g. 0.3).
2. **RAND_MOD:** same filter β, κ, but drive `e` with **Gaussian noise** matched for variance of `e` (not task error). Same broadcast form.
3. **BASE:** no modulator.  
**Hypothesis (prereg):** HOME_O_ERR beats RAND_MOD and BASE on XOR_HARD and/or NARMA (paired, seeds 3400–3407).  
**Falsify if:** HOME_O_ERR ≤ RAND_MOD (means any slow gain is enough) or both ≤ BASE.

### Test B — Two-drive rivalry (AgRP/POMC-style) vs single drive vs shuffled rivalry
**Idea:** Drive competition (motif 2.3).  
**Setup:** two slow units `d_hunger`, `d_explore` updated from task-phase features (e.g. recent loss ↑ → hunger; recent reward ↑ → explore). Mutual inhibition; softmax → gains `g0,g1` on two fixed partitions of units (or on two readout heads).  
**Arms:**
1. **RIVAL:** mutual inhibition + competition.
2. **SINGLE:** one drive only (matched #params).
3. **SHUF_RIVAL:** same dynamics but gains applied to **randomly permuted** unit partitions each seed (breaks structured competition).  
**Hypothesis:** RIVAL > SINGLE and RIVAL > SHUF_RIVAL on TRANSFER and XOR_HARD.  
**Falsify if:** SHUF_RIVAL ≈ RIVAL (structure irrelevant) or SINGLE ≥ RIVAL.

### Test C — Volume broadcast vs forced-synaptic delivery (paracrine motif)
**Idea:** Volume transmission (motif 2.4); inspired by fly clock→NSC peptide signaling.  
**Arms:**
1. **VOL:** one slow modulator scalar from a tiny PI-like unit (mean of 8 reserved nodes); **broadcast** to all units (no edge required).
2. **SYN:** same modulator amplitude forced through **K random existing edges** only (matched L2 effect size on Δh).
3. **BASE.**  
**Optional fly flavour:** reserve nodes = annotated IPC-like / high-out-degree neurosecretory proxies if we pull FlyWire labels; else random 8-node “PI stub.”  
**Hypothesis:** VOL > SYN on NARMA (slow memory / integration) and ≥ on XOR_HARD.  
**Falsify if:** SYN ≥ VOL (local synapses enough) or neither > BASE.

### Optional later (not in first batch)
- **SCN phase:** multiplicative circadian-like sinusoid on LR; control = phase-randomized same spectrum. Only if A/B/C show signal.
- **Human pathway prior:** inject Neudorfer/pathway soft channels into a hybrid graph — only after motif tests, and labeled as MRI-scale not synaptic.

### Stats / ops (match lab norms)
- 8 seeds, paired Wilcoxon vs BASE and vs matched control; Bonferroni over planned contrasts.
- Wall: reuse `run_ablation.py` pattern; Kaggle zip if impatient.
- Do **not** stop callosum_scaled or live ablations; schedule after or on free GPU.

---

## 7. Bottom line

1. Hypothalamus = **slow homeostatic / drive / peptide broadcast** controller, not a mini-cortex.  
2. Fly **PI + IPCs + clock→NSC** is the right analogue; FlyWire has the cells; peptides still leak outside the synapse graph.  
3. HCP-MMP **does not** give us hypo; Neudorfer atlas + hypo pathway atlases **do**, at MRI scale.  
4. Best next science: Tests **A/B/C** on `engine_v3` — falsifiable, cheap, and aligned with motifs that already rhyme with our HOMEO flag and JARVIS’s router/CAD wins.


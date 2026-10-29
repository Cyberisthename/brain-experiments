# Integration research: how to make Frankenstein modules work together

**Written:** 2026-10-29 ~11:40 America/New_York (EDT)  
**Scope:** literature + concrete reservoir designs. **No new experiments in this file.**  
**Dir:** `regions/integration_research/`  
**Repo:** https://github.com/Cyberisthename/brain-experiments

## Why this note exists

Stitching real connectome patches (fly CX, mouse MICrONS, human H01, finch, worm) into a "Frankenstein" leaky-tanh reservoir with **fixed random bridges** (σ≈0.05) and a **single ridge readout** has repeatedly failed size-/density-matched random controls. Key project facts that any integration fix must respect:

| Finding | Where |
|---|---|
| CX alone XOR_DELAY ≈ **0.66**; ER/SHUF/DENS ≈ 0.03–0.16 | `xor_cx` SUPPORTED |
| E/I class shuffle of CX ≥ real CX on XOR (≈0.70) | `xor_cx` mechanism |
| TEAM (CX⊕HUMAN, zero bridges, concat ridge) XOR **0.46** vs CX **0.64** — ridge dilutes the specialist | `team_franken` NOT SUPPORTED |
| TEAM_ER beat real TEAM on composite (NARMA-driven) | `team_franken` |
| ES-trained bridges ≈ fixed bridges | `trained_bridges` null |
| Real H01 myelin placement helps NARMA (REAL 0.393 vs SHUF 0.557) | `myelin_real_rep` REPLICATED |
| Putting E/I rule + myelin delays in one net wrecked XOR (delays piled near task delay) | `designed_brain` |

**Diagnosis:** brains do not mash modules through dense random axons into one linear decoder. Biology uses **sparse, selective, gated** routes. ML uses **gating / experts / hierarchy**. Our Frankenstein used neither.

---

## What biology and ML actually do (evidence map)

### 1. Communication subspaces (selective population routing)

**Claim:** Inter-areal communication uses a **low-dimensional subspace** of source population activity; many strong within-area fluctuations stay private and are not relayed.

- Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M., Kohn, A. (2019). *Cortical Areas Interact through a Communication Subspace.* **Neuron** 102:249–259.  
  https://doi.org/10.1016/j.neuron.2019.01.026  
  PDF: https://users.ece.cmu.edu/~byronyu/papers/SemedoNeuron2019.pdf  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6449210/

V1→V2 (and awake V1→V4) interactions needed ~2 predictive dimensions vs more within V1; predictive dims ≠ dominant within-area dims. Code: https://github.com/joao-semedo/communication-subspace

- Kohn, A., Jasper, A.I., Semedo, J.D., Gokcen, E., Machens, C.K., Yu, B.M. (2020). *Principles of Corticocortical Communication: Proposed Schemes and Design Considerations.* **Trends in Neurosciences**.  
  https://doi.org/10.1016/j.tins.2020.07.001  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7484382/

Reviews CTC, subspaces, pulvinar relay, source-structure vs target-sensitivity schemes — the design checklist for inter-area mechanisms.

**Implication for us:** bridges should be **rank-k** maps (k≪N), not dense Gaussian σ=0.05 blocks. Private dims keep CX's XOR dynamics from being overwritten by HUMAN / mouse noise.

### 2. Thalamic / pulvinar gating (external router)

**Claim:** Higher-order thalamus (esp. pulvinar) synchronizes / gates cortex–cortex transmission based on attention; transthalamic routes can be opened or closed.

- Saalmann, Y.B., Pinsk, M.A., Wang, L., Li, X., Kastner, S. (2012). *The Pulvinar Regulates Information Transmission Between Cortical Areas Based on Attention Demands.* **Science** 337:753–756.  
  https://doi.org/10.1126/science.1223082  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3714098/

- Sherman, S.M., Guillery, R.W. (2002 / related). Driver vs modulator distinction; thalamic relay as gated path. Classic statement: Sherman & Guillery, *Thalamic Relay Functions…* **Neuron** (2001) https://doi.org/10.1016/S0896-6273(01)00582-7 ; PNAS drivers/modulators https://www.pnas.org/doi/10.1073/pnas.95.12.7121

**Implication:** a small **hub reservoir** (or gain vector) that multiplies bridge efficacy from a task cue — not more dense cortex↔cortex edges.

### 3. Communication through coherence (oscillatory phase gating)

**Claim:** Effective connectivity is high only when pre/post groups are phase-aligned; gamma ≈ feedforward, alpha/beta ≈ feedback/control.

- Fries, P. (2005). *A mechanism for cognitive dynamics: neuronal communication through neuronal coherence.* **Trends in Cognitive Sciences** 9:474–480.  
  https://doi.org/10.1016/j.tics.2005.08.011

- Fries, P. (2015). *Rhythms for Cognition: Communication through Coherence.* **Neuron** 88:220–235.  
  https://doi.org/10.1016/j.neuron.2015.09.034  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4605134/

- Bastos, A.M., Vezoli, J., Fries, P. (2015). *Communication through coherence with inter-areal delays.* **Current Opinion in Neurobiology**.  
  https://doi.org/10.1016/j.conb.2014.11.001

**Implication:** time-varying bridge gain `g(t)=[1+cos(ωt+φ)]/2` (or band-limited) can implement selective coupling without learning bridge weights. Risk in our discrete leaky-tanh: CTC may help ER as much as real tissue unless phases are tied to tissue-specific timescales (leak / myelin).

### 4. Attention as gain modulation

**Claim:** Attention multiplies drive (and interacts with normalization), changing effective routing without rewriting anatomy.

- Reynolds, J.H., Heeger, D.J. (2009). *The Normalization Model of Attention.* **Neuron** 61:168–185.  
  https://doi.org/10.1016/j.neuron.2009.01.002  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2752446/

**Implication:** task-conditioned **per-module gain** on states (or on bridge inputs) before the shared ridge — cheaper than training bridges.

### 5. Basal ganglia–style action selection

**Claim:** Parallel channels compete; the most salient wins via disinhibition (select one pathway, suppress others).

- Gurney, K., Prescott, T.J., Redgrave, P. (2001). *A computational model of action selection in the basal ganglia. I. A new functional anatomy.* **Biological Cybernetics** 84:401–410.  
  https://doi.org/10.1007/PL00007984  
  Part II: https://doi.org/10.1007/PL00007985  
  PubMed: https://pubmed.ncbi.nlm.nih.gov/11417052/

- Bogacz, R., Gurney, K. (2007). *The basal ganglia and cortex implement optimal decision making between alternative actions.* **Neural Computation**.  
  https://pubmed.ncbi.nlm.nih.gov/17206871/

**Implication:** hard or soft **winner-take-all over module readouts** (or over bridge channels) per task — exactly what TEAM lacked when ridge blended CX and HUMAN.

### 6. Mixture-of-experts / modular RNNs (ML gating)

**Claim:** A gating net assigns input (or context) to specialized experts; blending hurts specialization.

- Jacobs, R.A., Jordan, M.I., Nowlan, S.J., Hinton, G.E. (1991). *Adaptive Mixtures of Local Experts.* **Neural Computation** 3:79–87.  
  https://doi.org/10.1162/neco.1991.3.1.79  
  PDF: https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf

- Yang, G.R., et al. (2019). *Task representations in neural networks trained to perform many cognitive tasks.* **Nature Neuroscience**.  
  https://doi.org/10.1038/s41593-018-0310-2  
  (modular / compositional structure emerges under multi-task training)

**Implication:** **per-task expert ridge** on one module + learned/hard gate. TEAM's concat ridge is the anti-pattern (Jacobs: blending encourages non-local experts).

### 7. Hierarchical / multi-reservoir ESNs

**Claim:** Stacking reservoirs (DeepESN) or hierarchical coupling creates multi-timescale features; later layers see slower / more abstract dynamics.

- Gallicchio, C., Micheli, A., Pedrelli, L. (2017). *Deep Reservoir Computing: A Critical Experimental Analysis.* **Neurocomputing**.  
  https://doi.org/10.1016/j.neucom.2016.12.089  
  Survey: Gallicchio & Micheli, *Deep Echo State Network (DeepESN): A Brief Survey* https://arxiv.org/abs/1712.04323

- Gallicchio, C., Micheli, A. (2017). *Echo State Property of Deep Reservoir Computing Networks.* **Cognitive Computation**.  
  https://doi.org/10.1007/s12559-017-9461-9

- Moon, J., et al. / hierarchical RC review (2021). *Hierarchical architectures in reservoir computing.* **Neuromorphic Computing and Engineering**.  
  https://iopscience.iop.org/article/10.1088/2634-4386/ac1b75

- Manneschi, L., et al. (2021). *Exploiting Multiple Timescales in Hierarchical Echo State Networks.* **Frontiers in Applied Mathematics and Statistics**.  
  https://doi.org/10.3389/fams.2020.616658

**Implication:** **unidirectional** CX → HUMAN cascade (memory specialist drives prediction specialist), not symmetric mash. Match leaks / delays to role (CX FIXED; HUMAN real myelin) — but do **not** put myelin delays inside the XOR path (designed_brain lesson).

### 8. Sparse weighted inter-area anatomy (Markov FLN)

**Claim:** Macaque cortex has many pathways, but weights are **lognormal**; most mass is local/moderate; long-range links are weak but specific; hubs exist.

- Markov, N.T., et al. (2014). *A Weighted and Directed Interareal Connectivity Matrix for Macaque Cerebral Cortex.* **Cerebral Cortex** 24:17–36.  
  https://doi.org/10.1093/cercor/bhs270  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3862262/

- Markov, N.T., et al. (2013). *The role of long-range connections on the specificity of the macaque interareal cortical network.* **PNAS**.  
  https://doi.org/10.1073/pnas.1218972110  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3612613/

**Implication:** bridges should be **sparse + lognormal weights + hub neurons**, not dense Gaussian. Density-match and weight-distribution-match the random control.

### 9. Plasticity for inter-area learning (FORCE / reward-modulated Hebbian)

**Claim:** Readouts (and recurrent feedback) can be shaped online; reward-modulated Hebb can reorganize without full BPTT.

- Sussillo, D., Abbott, L.F. (2009). *Generating Coherent Patterns of Activity from Chaotic Neural Networks.* **Neuron** 63:544–557.  
  https://doi.org/10.1016/j.neuron.2009.05.025  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2756108/

- Legenstein, R., Chase, S.M., Schwartz, A.B., Maass, W. (2010). *A Reward-Modulated Hebbian Learning Rule Can Explain Experimentally Observed Network Reorganization in a Brain Control Task.* **Journal of Neuroscience** 30:8400–8410.  
  https://doi.org/10.1523/JNEUROSCI.4284-09.2010  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2917246/

**Implication:** if bridges must learn, prefer **FORCE/RLS on a low-rank bridge** or **RM-Hebb on sparse hubs**, not ES on dense σ=0.05 (already null in `trained_bridges`). Keep plasticity **off** inside modules that already win (CX E/I; HUMAN myelin).

---

## Design principles distilled (for our framework)

1. **Do not blend specialists in one ridge** when tasks conflict (XOR vs NARMA). Gate or hard-select.
2. **Keep private dimensions** inside each module (rank-k bridges ≪ N).
3. **Sparse, hubby, lognormal** inter-module edges — Markov-style — with density-matched ER controls.
4. **Hierarchy / role order matters:** memory path should not inherit prediction delays.
5. **Train only the router** (gate, low-rank bridge, hub) — not the whole Frankenstein.
6. **Every mechanism must also run on random modules** of the same sizes. A win that ER also gets is not tissue-specific.

Standard project stats still apply if/when these are run: preregister, 24 eval seeds, ridge α on 90–92 only, Bonferroni, ER + SHUF + DENS (and mechanism-matched) controls.

---

## Six concrete designs (ranked by chance of a **tissue-specific** win)

Chance = P(real modules + mechanism beat the **same mechanism on size-matched random modules**, after fair controls). Not P(beats no-mechanism baseline).

### Design 1 — Task-gated mixture of specialist readouts (MoE / BG selection)  
**Rank #1 — highest tissue-specific chance (~55–70%)**

**Mechanism:** Basal-ganglia / MoE style. Each module has its own ridge expert. A gate (hard task-ID or soft softmax on a tiny cue net) picks which expert(s) write the output. No bridges required for v1.

**Implement (leaky-tanh + ridge):**
- Blocks: real `FLY_CX` (FIXED delays) + real `HUMAN` (REALMYELIN), block-diag, **bridge σ=0**.
- Shared drive into each block's input ports (same as TEAM).
- Train **separate** ridge W_cx, W_hum on tune seeds 90–92 **per task**.
- Gate: for XOR_DELAY → use only W_cx; for NARMA → only W_hum; for composite → hard switch by task. Soft gate optional: `y = g·y_cx + (1−g)·y_hum` with g from a 1-layer logistic on a 1-hot task cue (cue not given to reservoir).
- Primary metrics: XOR non-inferiority to CX alone (margin 0.03); NARMA ≤ HUMAN alone + ε; composite vs TEAM_ER_GATED.

**Fair controls (same gate on random modules!):**
- `GATE_ER`: ER-size-matched CX-block + ER-size-matched HUMAN-block, same gate, shuffled myelin on human-sized block.
- `GATE_SHUF` / `GATE_DENS`: wiring shuffles / density match inside blocks.
- `GATE_EI`: E/I-shuffle CX + real HUMAN (mechanism of XOR).
- Ablation: concat ridge (old TEAM) as negative control.

**Predicted outcome:** Real GATE matches CX on XOR (~0.66) **and** HUMAN on NARMA (~0.39), beating GATE_ER on composite. If GATE_ER still wins NARMA (as TEAM_ER did), tissue win fails on NARMA half — report honestly.

**Cost:** ~0.5–1× TEAM (~15–30 min on box, 2 workers). No bridge matvecs. Easiest code change.

**Why top:** We already know the specialists exist. TEAM failed because ridge **blended** them. This is the minimal biological/ML fix (Jacobs 1991; Gurney 2001).

---

### Design 2 — Communication-subspace bridges (Semedo rank-k)  
**Rank #2 (~40–55%)**

**Mechanism:** Inter-module edges = low-rank map `B = U V^T / z` with k ∈ {2,4,8}, columns of V from within-source PCA/FA on drive-only states (private dims zeroed). Mirrors Semedo: only a few source patterns reach the target.

**Implement:**
- Build block-diag CX⊕HUMAN (and optional mouse).
- Estimate per-module top-k PCs of states under shared drive (tune seeds only; freeze before eval).
- Bridge: project source → k coeffs → expand into target with random orthonormal U (or target-PC basis). Scale so Frobenius(B) matches old σ=0.05 dense bridge energy (energy-matched control).
- Optional asymmetric: CX→HUMAN k small; HUMAN→CX = 0 (protect XOR private space).
- Delays: FIXED on CX side; REALMYELIN only on human-internal + human-bound synapses (not on XOR-critical CX recurrence).
- Single ridge on concat states **or** gated readout from Design 1 (combine for max power).

**Fair controls:**
- `SUB_ER`: same rank-k recipe on ER modules (identical k, energy).
- `SUB_RAND_BASIS`: real modules but random k-planes (not PCA) — tests whether **learned/structure-aligned** subspace matters.
- `SUB_FULL`: dense bridges energy-matched (old FRANKEN).
- `SUB_SHUF_PC`: shuffle PC loadings across neurons within module.

**Predicted outcome:** Real SUB > SUB_ER on XOR retention (less dilution than dense) and ≥ TEAM on NARMA. Biggest risk: PCA subspace from drive may not be the "right" communication plane (Semedo's were fit to target activity).

**Cost:** ~1.5–2× TEAM (PCA once + sparse-ish matvec). k=4 preferred.

---

### Design 3 — Pulvinar / thalamic gain hub  
**Rank #3 (~35–50%)**

**Mechanism:** Saalmann/Sherman-style external router. A small hub (N_h=64–128, ER or ring) receives a task cue + compressed summaries of modules; its state multiplies **bridge gains** (and/or module gains) elementwise.

**Implement:**
- Modules block-diag; bridges exist but `W_bridge_eff(t) = g_hub(t) ⊙ W_bridge` with g = σ(W_h h_t + b) ∈ (0,1)^n_bridge or one scalar per directed module pair.
- Hub update: standard leaky-tanh; input = [task_cue | mean-pool(module states) | optional].
- Freeze modules+W_bridge; train only hub→gain weights with FORCE/RLS on tune seeds (Sussillo 2009), target = task error.
- Gains default 0 (isolated specialists) and open only when helpful.

**Fair controls:**
- Same hub+FORCE on ER modules (`HUB_ER`).
- Hub with shuffled task cue (`HUB_SHUF_CUE`).
- Fixed g=1 (always open) and g=0 (TEAM).

**Predicted outcome:** Real modules + learned gate recover CX XOR while opening HUMAN path for NARMA; HUB_ER opens both but lacks specialist quality → real wins composite.

**Cost:** ~2–3× TEAM (online FORCE during tune). More moving parts → higher bug risk; prereg must lock FORCE α, hub size, seed split.

---

### Design 4 — Role-ordered DeepESN cascade (no mash)  
**Rank #4 (~25–40%)**

**Mechanism:** Hierarchical ESN (Gallicchio): input → CX (memory / XOR) → HUMAN (prediction / myelin), **unidirectional** sparse feedforward, different timescales. No HUMAN→CX feedback (protects XOR).

**Implement:**
- Layer 1: FLY_CX, FIXED delays, leak=0.2, receives external input.
- Layer 2: HUMAN, REALMYELIN, leak∈{0.1,0.2} sweep on tune only, receives **only** rank-k or sparse projection of layer-1 states (no raw input, or weak input).
- Readout: concat both layers with **Design-1 gate**, or separate experts.
- Do **not** put myelin delays inside layer 1 (designed_brain failure mode).

**Fair controls:**
- `CASCADE_ER`: ER→ER same sizes, same k, shuffled delays on layer 2.
- `CASCADE_SWAP`: HUMAN→CX order (should hurt XOR if role order matters).
- `CASCADE_PARALLEL`: both get input, zero FF (TEAM).

**Predicted outcome:** Real cascade beats CASCADE_ER on XOR (layer-1 tissue) and beats SWAP; NARMA may or may not beat HUMAN alone. Tissue-specific win hinges on whether CX→HUMAN features beat ER→ER features — uncertain given TEAM_ER's NARMA strength.

**Cost:** ~1–1.5× TEAM.

---

### Design 5 — Markov-FLN sparse hubs + FORCE only on hubs  
**Rank #5 (~20–35%)**

**Mechanism:** Anatomy-like bridges: few hub neurons per module, lognormal inter-module weights, long-range sparse; plasticity only on hub edges (FORCE or RM-Hebb), modules frozen.

**Implement:**
- Pick n_hub = round(0.02 N) highest out-degree (or random for control) neurons per module as hubs.
- Inter-module edges: only hub↔hub (or hub→all), nnz and weight lognormal matched to a scaled Markov FLN motif (or simple lognormal draw with p≪ dens(σ=0.05 dense)).
- Train hub edges with FORCE targeting task error on tune seeds; freeze before eval.
- Contrast to `trained_bridges` (dense ES): here capacity is tiny → less overfitting noise into CX.

**Fair controls:**
- Same hub fraction + FORCE on ER modules.
- Random hubs vs degree hubs on real modules.
- Frozen sparse hubs (no FORCE).

**Predicted outcome:** Mild improvement over dense ES null; tissue win only if real hub identity matters vs random hubs. Prior ES null makes this a lower prior.

**Cost:** ~2–4× TEAM (FORCE loops).

---

### Design 6 — CTC-style phase gating of bridges  
**Rank #6 — lowest tissue-specific chance (~10–25%)**

**Mechanism:** Fries CTC. Each module has a phase oscillator φ_m(t)=ω_m t; bridge from i→j scaled by `[1+cos(φ_i−φ_j−ψ)]/2`. ω tied to leak or myelin timescale.

**Implement:**
- Discrete-time multiplier on existing bridges; ω_cx vs ω_hum from {slow, mid, fast} grid on tune seeds only.
- Optional: task cue selects phase offset ψ (attention-like).

**Fair controls:** same (ω,ψ) schedule on ER modules; phase-shuffled; always-on bridges.

**Predicted outcome:** May help all wirings equally (generic effective-connectivity knob). Tissue-specific win unlikely unless real myelin/leak create unique coherent windows — speculative in rate reservoirs.

**Cost:** ~1.2× TEAM. Include mainly as a preregistered negative/ablation relative to Designs 1–3.

---

## Ranking summary

| Rank | Design | Tissue-specific chance | Cost | Depends on |
|---:|---|---|---|---|
| 1 | Task-gated MoE readouts | high | low | known specialists |
| 2 | Semedo rank-k bridges | med-high | med | private dims |
| 3 | Pulvinar gain hub + FORCE | medium | med-high | learned router |
| 4 | DeepESN role cascade | med-low | low-med | hierarchy + roles |
| 5 | FLN sparse hubs + FORCE | low-med | high | hub plasticity |
| 6 | CTC phase gates | low | low | oscillations in rate model |

**Recommended run order (when experiments resume):** Design 1 + Design 8 (task-gated experts **with** neuromod gain) → Design 2 subspace bridges → Design 7 SELF_CANCEL native task → Design 3/4 → Design 9 only on a separate embodiment/rhythm track → Design 5/6 last.

---

## What would count as a real Frankenstein win

All of the following, preregistered:

1. Mechanism + **real** modules beat mechanism + **ER/SHUF/DENS** modules (tissue-specific).
2. XOR stays within 0.03 of CX alone (non-inferiority) **or** composite clearly wins with XOR drop preregistered as acceptable (prefer non-inferiority).
3. NARMA ≤ HUMAN alone + small ε **or** clear composite win vs gated-ER.
4. Hold after Bonferroni; 24 new seeds; α locked on 90–92.
5. Negative controls reported (concat ridge, always-open bridges, swapped hierarchy).

Reservoir scores still ≠ animal intelligence. Cite wiring/mechanism claims only against the listed controls.

---

## Key sources (quick links)

1. Semedo et al. 2019 Neuron — communication subspace — https://doi.org/10.1016/j.neuron.2019.01.026  
2. Kohn et al. 2020 TINS — corticocortical schemes — https://doi.org/10.1016/j.tins.2020.07.001  
3. Saalmann et al. 2012 Science — pulvinar gating — https://doi.org/10.1126/science.1223082  
4. Fries 2015 Neuron — CTC — https://doi.org/10.1016/j.neuron.2015.09.034  
5. Reynolds & Heeger 2009 Neuron — attention gain — https://doi.org/10.1016/j.neuron.2009.01.002  
6. Gurney, Prescott, Redgrave 2001 Biol Cybern — BG action selection — https://doi.org/10.1007/PL00007984  
7. Jacobs et al. 1991 Neural Comp — mixture of experts — https://doi.org/10.1162/neco.1991.3.1.79  
8. Gallicchio & Micheli DeepESN — https://arxiv.org/abs/1712.04323  
9. Markov et al. 2014 Cereb Cortex — FLN weights — https://doi.org/10.1093/cercor/bhs270  
10. Sussillo & Abbott 2009 Neuron — FORCE — https://doi.org/10.1016/j.neuron.2009.05.025  
11. Legenstein et al. 2010 J Neurosci — reward-modulated Hebb — https://doi.org/10.1523/JNEUROSCI.4284-09.2010  
12. Crapse & Sommer 2008 Nat Rev Neurosci — corollary discharge — https://doi.org/10.1038/nrn2457  
13. Burke et al. 2012 Nature — fly OA/DA reward — https://doi.org/10.1038/nature11614  
14. Marder & Calabrese 1996 Physiol Rev — CPG principles — https://doi.org/10.1152/physrev.1996.76.3.687  
15. Takemura et al. MANC VNC connectome — https://doi.org/10.1101/2023.06.05.543757  

---

## User add-ons (2026-10-29): efference copy, neuromodulation, CPG

Three extra mechanisms the user asked to evaluate. Ranked relative to Designs 1–6 for **our current no-body, open-loop leaky-tanh + ridge setup**.

### Design 7 — Efference copy / corollary discharge (closed-loop cancellation)
**Tissue-specific chance in our setup: medium (~30–45%) if we add a closed-loop task; low on MC/NARMA/XOR alone**

**Mechanism:** A copy of the motor command (or readout output) is routed back into sensory / reservoir input so the system can cancel self-caused signals and keep external signals (Crapse & Sommer taxonomy: sensory filtration / stability).

**Sources:**
- Crapse, T.B., Sommer, M.A. (2008). *Corollary discharge across the animal kingdom.* **Nature Reviews Neuroscience** 9:587–600.  
  https://doi.org/10.1038/nrn2457  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5153363/
- Crapse, T.B., Sommer, M.A. (2008). *Corollary discharge circuits in the primate brain.* **Current Opinion in Neurobiology**.  
  https://doi.org/10.1016/j.conb.2008.09.017  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2702467/
- Classical: von Holst & Mittelstaedt (1950) Efferenzkopie; Sperry (1950) corollary discharge (cited in the reviews above).

**Does it apply without a body?** **Partially yes.** We do not need muscles if we treat the **ridge readout** as the "motor command" and feed `ŷ_t` (or a fixed random projection of it) back into designated input ports with delay d∈{1..5}. That is a real closed loop. What we **cannot** claim without an effector is biological motor CD from VNC→brain.

**Concrete task (required):**  
`SELF_CANCEL`: input `u_t = e_t + s_t` where `s_t = α·ŷ_{t−d}` is the reafferent (self-caused) component and `e_t` is external target-relevant noise/signal. Target = `e_t` (or a function of e only). Metric: NRMSE on e (↓). A system with a good efference path should cancel s.  
Secondary: keep XOR/NARMA on open-loop arms unchanged (CD off).

**Implement:**
- Modules: CX ± HUMAN as before.
- New input channel: `W_in_cd @ ŷ_{t−d}` into a subset of neurons (prefer sensory-like / L4 ports), scale α locked on tune seeds.
- Arms: `CD_REAL` (real modules + CD path), `CD_ER` (ER modules + same CD path), `NO_CD` (α=0), `CD_SHUF` (CD weights shuffled each trial).
- Optional: CD only into HUMAN (sensory) not CX (protect XOR) — matches "motor copy → sensory structure".

**Fair controls:** same CD architecture on ER/SHUF/DENS modules; CD weight shuffle; open-loop (α=0).

**Predicted outcome:** CD helps all wirings on SELF_CANCEL; tissue win only if real CX/HUMAN geometry makes cancellation cheaper (uncertain). **Do not expect a Frankenstein win on XOR/NARMA from CD alone.**

**Cost:** ~1.5× TEAM (closed-loop state depends on readout → must refit carefully; avoid train/test leakage of ŷ).

**Honesty:** Best as a **new native task**, not a patch for stitching. Closest biological tests (electric fish, fly saccade CD) all have effectors; our version is a computational analogue.

---

### Design 8 — Global neuromodulator (DA / OA / 5-HT–like) gating modules
**Tissue-specific chance: high (~50–65%) — best overlap with Design 1, and directly fits no-body reservoirs**

**Mechanism:** A diffuse scalar (or low-dim) signal changes **gain, leak, and/or plasticity** of synapses/modules by context or reward, unlocking different behaviors from the same wiring (Tsuda-style multiplicative synaptic gain; fly OA/DA reward layering; STDP under neuromodulatory control).

**Sources:**
- Tsuda, B., et al. (2021+ modeling line). Neuromodulators as multiplicative factors on synaptic transmission enabling overlapping memory regimes in RNNs.  
  https://cnl.salk.edu/~terry/BIPN147-21/Tsuda.neuromodulation.21.pdf  
  (treat as computational evidence for gain-unlocking, not a connectome claim)
- Brzosko, Z., Mierau, S.B., Paulsen, O. (2019). *Neuromodulation of Spike-Timing-Dependent Plasticity: Past, Present, and Future.* **Neuron**.  
  https://doi.org/10.1016/j.neuron.2019.05.041
- Burke, C.J., Huetteroth, W., Owald, D., Perisse, E., Krashes, M.J., Das, G., Gohl, D., Silies, M., Certel, S., Waddell, S. (2012). *Layered reward signalling through octopamine and dopamine in Drosophila.* **Nature** 492:433–437.  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3528794/  
  DOI: https://doi.org/10.1038/nature11614
- Boto, T., et al. / MB octopamine reviews — e.g. *Octopamine in the mushroom body circuitry for learning and memory* (2024 review).  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11199948/
- Marder / Harris-Warrick: neuromodulation essential to CPG parameter setting — Harris-Warrick (2011) *Neuromodulation and Flexibility in Central Pattern Generator Networks.*  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3171584/

**Does it apply without a body?** **Yes — strongly.** A task cue or reward scalar `m_t ∈ [0,1]` that multiplies per-module recurrent gain (`W ← m_c · W_c`) or leak (`α_leak ← α0·(1+β m)`) needs no effector. This is exactly how to "gate which module dominates per task" without blending ridge weights.

**Implement (preferred v1 — overlaps Design 1):**
- Two modulators: `m_xor` high / `m_narma` low when task=XOR; reverse for NARMA (hard schedule from task ID on tune+eval — no peeking at labels beyond the task identity the agent already knows).
- Effect A (gain): scale CX recurrent weights by `(1 + γ m_xor)`, HUMAN by `(1 + γ m_narma)`; freeze γ from {0.5,1,2} on tune only.
- Effect B (leak): raise CX leak under XOR (faster/forgetting tradeoff grid), lower HUMAN leak under NARMA.
- Effect C (optional plasticity): allow FORCE/RM-Hebb **only** on bridges when `m_reward` pulses after error drop (Legenstein 2010); modules frozen.
- Readout: still prefer **per-module expert ridges** (Design 1); modulator makes the *dynamics* specialize, gate picks the expert.

**Fair controls:**
- Same `m(t)` schedule on ER modules (`MOD_ER`).
- Modulator shuffled across time (`MOD_SHUF`).
- Uniform m=1 (no modulation).
- Gain on wrong module (swap) as negative control.

**Predicted outcome:** Real MOD preserves CX XOR ≈0.66 and HUMAN NARMA ≈0.39 better than concat TEAM; MOD_ER may still win NARMA half (as TEAM_ER) — tissue claim needs XOR+composite vs MOD_ER.

**Cost:** ~1× TEAM (gain rescale is cheap). Highest ROI add-on.

**Honesty:** Fly OA/DA biology is about *learning reinforcement* in mushroom body, not about Frankenstein grafts. We borrow the **computational motif** (diffuse gain unlock), not the claim that our HUMAN patch has dopaminergic innervation.

---

### Design 9 — Sensorimotor gating + CPG module (VNC/MANC or model CPG)
**Tissue-specific chance on current open-loop suite: low (~10–20%). With a new rhythm/body task: medium (~30–45%)**

**Mechanism:** CPGs generate rhythmic motor patterns without rhythmic input; descending commands and neuromodulators recruit them; sensory feedback gates and shapes the rhythm (Marder & Calabrese). Fly VNC now has connectomes (MANC/FANC/…) and a proposed walking CPG motif from connectome simulations.

**Sources:**
- Marder, E., Calabrese, R.L. (1996). *Principles of rhythmic motor pattern generation.* **Physiological Reviews** 76:687–717.  
  https://doi.org/10.1152/physrev.1996.76.3.687
- Calabrese, R.L., Marder, E. (2024). *Degenerate neuronal and circuit mechanisms important for generating rhythmic motor patterns.* **Physiological Reviews**.  
  https://doi.org/10.1152/physrev.00003.2024  
  PubMed: https://pubmed.ncbi.nlm.nih.gov/39453990/
- Harris-Warrick, R.M. (2011). *Neuromodulation and Flexibility in Central Pattern Generator Networks.* PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3171584/
- Takemura, S., et al. (2023/2024). *A Connectome of the Male Drosophila Ventral Nerve Cord* (MANC). bioRxiv / eLife.  
  https://doi.org/10.1101/2023.06.05.543757  
  Janelia: https://www.janelia.org/project-team/flyem/manc-connectome  
- Public MANC used to identify a walking CPG motif in:
- Pugliese, S.M., et al. (2025/2026). *Connectome simulations identify a central pattern generator circuit for fly walking.* **Nature** / bioRxiv.  
  https://www.nature.com/articles/s41586-026-11195-y  
  bioRxiv: https://www.biorxiv.org/content/10.1101/2025.09.12.675944  
  PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/  
  Code/data note: https://github.com/smpuglie/Pugliese_2026

**Does it apply without a body?** **Mostly no for "sensorimotor gating."** Gating by definition needs sensory consequences of movement. Our MC/NARMA/XOR/PC_HARD suite has **no effector**, so grafting a CPG into Frankenstein and scoring open-loop reservoirs tests "does a rhythm generator help abstract memory?" — usually a confused question.

**What we *can* test honestly:**
1. **Rhythm-native task:** attach a CPG block (real MANC subcircuit if downloadable, else classic half-center / 3-neuron E-I CPG matched to Pugliese motif size) that must produce a target oscillation; brain modules provide descending drive; metric = rhythm freq/phase error. Controls: ER-matched CPG graph, phase-shuffled, disconnected CPG.
2. **Toy body loop (minimal):** 1-D "leg angle" `θ_{t+1}=θ_t+ω·motor_t`; proprioception `θ` fed back as input; task = track a speed setpoint. Then CD (Design 7) + CPG + neuromod (Design 8) become jointly meaningful.
3. **Do not** expect CPG to fix XOR dilution in TEAM — wrong hypothesis class.

**Implement (if pursued):**
- Prefer **model CPG first** (half-center, N≈50–200, known to oscillate) size-matched to a MANC extract later.
- Descending drive from CX (command-like) → CPG; optional weak CPG→brain corollary (ties to Design 7).
- Fair controls: ER CPG, shuffled signs (break E-I motif), frozen random oscillator input (not a network).

**Predicted outcome:** Real/model CPG wins rhythm task vs ER-broken motif; unlikely to beat ER on XOR/NARMA. Frankenstein "smarter brain" claim should **not** rest on this until a body/rhythm task is preregistered.

**Cost:** high (data download, new task code, closed loop). Weeks if MANC auth/size bites; days for model CPG + toy body.

---

### Honest applicability table (no-body reservoir)

| Idea | Fits open-loop MC/NARMA/XOR? | Needs new task? | Needs body/effector? | Priority vs Designs 1–6 |
|---|---|---|---|---|
| Neuromod gain/leak (D8) | **Yes** — best fit | No (task-ID cue enough) | No | **Do next with Design 1** |
| Efference / CD (D7) | Weak alone | **Yes** (SELF_CANCEL) | No for analogue; yes for biology claim | After D1; new native task |
| CPG + sensorimotor (D9) | **No** (wrong metrics) | **Yes** (rhythm / toy body) | **Yes** for gating claim | Only if user wants embodiment track |

**Updated recommended order:** Design 1 (gated experts) **+ Design 8 (modulator gain)** as one prereg → Design 2 (subspace bridges) → Design 7 (SELF_CANCEL native task) → Design 3/4 → Design 9 only on an embodiment branch → Design 5/6 last.

---

## Honesty / limits

- This document is **research synthesis**, not results. No scores claimed.
- Citations checked via publisher/PMC/DOI/author PDFs (2026-10-29). Do not invent extra papers.
- Macaque FLN is area-level, not neuron-level; Design 5 uses it as a **weight motif**, not a claim we have macaque synapses in the Frankenstein.
- CTC and thalamic gating are richer in spiking/LFP data than in our rate reservoirs; Designs 1–2 are the most faithful to what we can actually test.

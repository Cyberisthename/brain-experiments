# Synapse research for brain-module RNN / OscRNN experiments

Machine time: 2026-11-02 (America/New_York). Literature + cheap hooks for our harness
(`callosum_scaled`, `human_wholebrain`, `realistic_callosum`, `oscillation_gating`).
**Do not build engine v3 yet** — see sibling `../ENGINE_V3_PLAN.md`.

Citations are real papers / reviews. Numbers are typical cortical / connectomic values,
not magic constants for our tanh units.

---

## 0. What our harness does today (gap)

| Feature | OscRNN (wholebrain / callosum_scaled) | engine2 (spiking, separate) |
|---|---|---|
| Neuron | leaky `tanh` rate | adaptive LIF + refractory |
| Synapse | instantaneous weight × state | exp current kernels; optional TM STP |
| Sign / Dale | FC-derived `esign` on **edges** (not per-neuron Dale) | per-neuron Dale from NT |
| Delays | integer steps on inter / callosal edges (`torch.roll` history) | axonal delay **ring** + CSR buckets |
| Plasticity | train `W` / edge weights with Adam | homeostatic thr; STP optional |
| Prior STP experiment | `regions/plasticity/` — **NO DATA** (API bug) | —

So synapse ideas below target **cheap rate-RNN hooks** first; engine2 already prototypes
several of them for a later spiking path.

---

## 1. Short-term plasticity (Tsodyks–Markram)

### What it is
Presynaptic resources `x` and utilization `u` make each synapse's effective weight
`w · u · x` history-dependent on a ~100 ms–1 s scale (Tsodyks & Markram 1997;
Tsodyks, Pawelzik & Markram 1998; Scholarpedia: Short-term synaptic plasticity).

Depression (STD): large `U`, long recovery `τ_rec` → synapses weaken after use.
Facilitation (STF): small `U`, long `τ_fac` → synapses strengthen with repeated spikes.

### Numbers
| Regime | U | τ_rec | τ_fac | Notes |
|---|---:|---:|---:|---|
| Depressing (sensory cortex-ish) | ~0.4–0.5 | ~200–800 ms | ~0–50 ms | Markram et al. 1998; NEST `tsodyks_synapse` defaults often U=0.5, τ_rec=800 ms |
| Facilitating (PFC E→E) | ~0.15–0.3 | ~50–200 ms | ~500–1500 ms | Mongillo, Barak & Brunel 2008 *Science* (synaptic WM); example tF≈1.5 s, tD≈0.2 s |
| Mixed / memory-friendly | U≈0.02 | τ_rec≈100 ms | τ_fac≳20 ms | Mejias & Torres / related Hopfield+STP analyses (~50% capacity recovery vs depression-only) |

Mongillo et al. 2008: working memory can live in **synaptic facilitation state**, refreshed by sparse spikes — not only persistent firing.

RC relevance: STP adds a hidden synaptic state → richer temporal kernel (Buonomano & Maass 2009 framing; STD can keep nets near criticality while long-term weights change — e.g. Neural Networks 2019 on STD expanding operational range). Homeostatic / delay-coupled RC: up to ~70% memory-capacity gain, ~23% better NARMA-10 NMSE (Nachstedt et al. 2015, *Neural Computation*).

### Cheap hook in our harness
Per **edge** (or per callosal edge only): keep `u_e`, `x_e` as buffers.

Rate-proxy update each step `t` (dt ≈ 1 “tick” ≈ leak time unit; map τ to steps):

```
u ← u + U*(1-u)*|pre|          # or softplus(pre)
x ← x * (1 - u*|pre|)          # consume
eff = w * u * x
x ← 1 - (1-x)*exp(-dt/τ_rec)
u ← U + (u-U)*exp(-dt/τ_fac)   # or decay to 0 then reset; match TM discrete form
```

Start with **STF on excitatory callosal / long-range edges only** (few edges → cheap).
Disable STP on shuffle arms' weights identically so topology controls stay fair.

engine2 already has TM fields (`SynapseParams.U`, `tau_rec_ms`, `tau_fac_ms`) behind `use_stp`.

### Tasks expected to help
- **XOR_HARD** (lag-12 memory): STF buffer holds lag without needing huge N
- **TRANSFER / DELAYED_MATCH**: bridge must remember RH cue until LH acts
- **NARMA**: multi-timescale synaptic state matches nonlinear autoregression
- **CHAR**: milder; may help word-scale context if τ_fac mapped to many steps

---

## 2. Synaptic time constants (AMPA / NMDA / GABA_A / GABA_B)

### What it is
Postsynaptic currents are filtered, not delta impulses. Different receptors = different τ.

### Numbers (cortical models)
| Receptor | Rise | Decay | Role |
|---|---|---|---|
| **AMPA** | ~0.2–0.8 ms | **~2–5 ms** (pyramidal ~5 ms) | Fast excitation (Hestrin 1993; Destexhe kinetic reviews) |
| **NMDA** | ~2–20 ms | **~25–150 ms** (often dual; Izhikevich 2008 uses 150 ms) | Slow E; coincidence / dendrite nonlinearity; Mg block |
| **GABA_A** | ~0.3–1 ms | **~5–10 ms** (~6–8 ms common) | Fast inhibition |
| **GABA_B** | ~25–50 ms | **~100–300 ms** (+ slow ~0.5–1 s) | Slow inhibition (Destexhe & Paré; Neuronal Dynamics Ch.3) |

Izhikevich & Edelman 2008 thalamocortical appendix: τ = 5 / 150 / 6 / 150 ms for AMPA/NMDA/GABA_A/GABA_B.

### Cheap hook
Replace single leak with **two filtered currents** into the tanh pre-activation:

```
g_fast ← (1-α_f)*g_fast + α_f * (W_fast @ x)   # AMPA+GABA_A-like
g_slow ← (1-α_s)*g_slow + α_s * (W_slow @ x)   # NMDA+GABA_B-like
rec = g_fast + κ_nmda * g_slow                 # κ_nmda ~ 0.3–1.0
```

Or split E edges into 70% AMPA / 30% NMDA weight share (fixed). Map α from τ/dt with dt≈1–5 ms of biological time per RNN step (document the mapping in prereg).

### Tasks
- **NARMA**, **CHAR**: slow NMDA-like τ expands memory horizon
- **XOR_HARD**: moderate help if lag spans slow τ
- **TRANSFER**: slow excitatory callosal component ≈ myelin+NMDA combo

---

## 3. Dale's law / E–I balance

### What it is
Each neuron is E **or** I, not both (Dale's principle). Cortex ≈ **4:1** E:I cells;
I synapses stronger so net balance holds (often ~1:4 E:I strength ratio in models).

Song et al. / Brunel balanced networks; RC: balanced or slightly inhibition-dominated
regimes maximize memory & prediction — excitation-dominated reservoirs saturate
(Nature Communications 2025 “Boosting reservoir computing with … E-I balance”: up to
**~130%** memory / time-series gains with adaptive E–I; also note shuffling E/I
*assignments* sometimes did **not** hurt, so **balance dynamics** may matter more
than hard Dale alone). Maximal MC near edge-of-chaos balanced E–I (e.g. *Neural
Computation* “Maximal Memory Capacity Near the Edge of Chaos in Balanced Cortical
E-I Networks”).

### Numbers
- f_I ≈ 0.2; |w_I| / |w_E| ≈ 4 for global balance β≈0
- Target rates ~few Hz in spiking; in rate RNNs: keep mean |rec| away from ±1 saturation

### Cheap hook
1. Assign each unit E/I with p_I=0.2 (fixed across seeds for REAL; reshuffle for control).
2. Zero any E→ with negative trained weight (project: `w = softplus(w_raw) * sign_neuron`).
3. Optional: **global β** dial on I strength; sweep β ∈ [-2, 0.5] like the 2025 RC paper.

We already have edge `esign` from FC — that is **not** Dale. Promote to **column-constant
sign** (neuron Dale) as a new arm: `DALE_EI` vs `SIGN_SHUFFLE`.

### Tasks
- Stability on **NARMA** / large-N callosum (fewer NaN / blow-ups)
- **XOR_HARD** memory capacity in balanced regime
- Less clear for **CHAR**; still worth locking as a control for biology claims

---

## 4. Dendritic nonlinearity

### What it is
Pyramidal dendrites act as nonlinear subunits (NMDA spikes): a neuron ≈ two-layer net
before the soma (London & Häusser 2005 *Annu Rev Neurosci* “Dendritic computation”).
PFC models: dendritic NMDA supports persistent activity with smaller networks
(e.g. work building on Legenstein & Maass / Papoutsi et al. dendritic PFC microcircuits).

### Numbers
- NMDA dendritic spike threshold ~ local depolarization; effective “AND-like” integration
within ~50–100 µm of dendrite
- MICrONS: excitatory cells receive most synapses on **spines** (~70%); I cells on shafts (~80%) — compartment structure is real at synapse scale

### Cheap hook (rate)
Two compartments per unit: `x_d`, `x_s`

```
x_d ← (1-λ) x_d + λ tanh(W_d @ x_s + …)
x_s ← (1-leak) x_s + leak tanh(W_s @ softplus(x_d)^2 + Win u)  # quadratic NMDA-ish
```

Or single state with `rec = W @ x + γ (W @ x)_+²` (Poirazi-style). Cost: ~1.5–2× matvec.

### Tasks
- **XOR_HARD** / **SPLIT_BIND**: nonlinear binding across features
- **TRANSFER**: less primary; use if memory arms stall
- Skip first if STP + multi-τ already move the needle (costlier)

---

## 5. STDP and homeostatic plasticity

### STDP
Spike-timing-dependent plasticity: pre-before-post → LTP; reverse → LTD
(Bi & Poo 1998; Song, Miller & Abbott 2000 *Nat Neurosci*).

Numbers: Song 2000 often τ₊=τ₋=**20 ms**; Bi & Poo data ≈ 17±9 ms (LTP), 34±13 ms (LTD).
Competitive STDP → sparse strong synapses.

### Homeostasis / synaptic scaling
Turrigiano et al. 1998 *Nature*: activity blockade → mEPSC amplitudes scale up ~1.5–2× over **~48 h**; measurable changes within **4–6 h**. Stabilizes rates so Hebbian rules don't run away (Turrigiano 2011 review).

### Cheap hook
- **Online STDP in rate form** (trace eligibility): expensive + confounds “architecture vs learning” — **defer** for connectome topology studies; OK later as trained-bridge arm.
- **Homeostatic gain** (cheap, recommended): every K steps, rescale rows/columns so mean |activity| → target; or slow bias `b ← b + η (r* - |x|)`. Matches engine2 `homeo_rate` idea.

### Tasks
- Homeostasis: all tasks at large N (keeps NARMA finite; we already saw NaN blow-ups)
- STDP: not for prereg topology tests; optional follow-up on `trained_bridges`

---

## 6. Synaptic / axonal delays

### What it is
Conduction + synaptic latency. Callosal fibers longer than intra-hemispheric
(~122 mm vs ~71 mm in Lausanne consensus → ~1.7×; Griffa70 / Vohryzek CTRL similar).
Caminiti et al. velocities by callosal sector (we already encode Method-3 velocity
multipliers in `callosum_scaled`).

### Numbers
- Cortical local synaptic delay ~0.5–2 ms + axonal 0.1–few ms/mm depending on myelination
- Callosal crossing often quoted ~**5–15 ms** human (sector-dependent); our discrete `DMAX`
  bins should keep physical order: local < mid < callosal

### Cheap hook
Already present. Improvements:
1. Wire **real myelin / FA → delay** (queued “myelin_real” arm) instead of length quantiles only
2. Replace `torch.roll` history with **ring index** (engine2 / v3) — quality same, speed up
3. Separate AMPA (delay+0) vs NMDA (delay+1) on same edge if multi-τ added

### Tasks
- **NARMA**, **TRANSFER**, **DELAYED_MATCH** (already the win domain in human_wholebrain)

---

## 7. Sparse connectivity stats (MICrONS / FlyWire)

### Numbers
| Dataset | Scale | Synapses / connections | Sparsity |
|---|---|---|---|
| **MICrONS** mouse visual | ~200k cells, **~524M** synapses (~1 mm³) | ~**2.6k synapses/cell** avg; proofread column ~3.0–3.5k in / cell (E ~3.1k, I ~5.8k) | pairwise connect prob ≪1%; related minnie65 ~**0.13%** |
| **FlyWire** whole fly brain | ~128k neurons, ~2.6M thresholded connections (≥5 syn) | mean degree ~**20.5**; ~12.6 syn/connection; P(connect)≈**1.6e-4** | extremely sparse; 71% of edges within 50 µm |

### Cheap hook
Keep **edge lists / CSR**; never dense N×N at N≥512. Degree targets: local mean degree
O(10–50), not O(N). Our callosum_scaled sparse OscRNN already matches this spirit;
`human_wholebrain` still allocates dense `W_intra/W_inter` — migrate to edge params.

### Tasks
All large-N arms; enables 1024–2048 tests without RAM blow-up.

---

## 8. Reservoir / RNN papers — STP or E/I help memory & timing

| Paper | Finding | Relevance |
|---|---|---|
| Mongillo, Barak, Brunel 2008 *Science* | Synaptic facilitation as WM buffer | XOR_HARD / TRANSFER |
| Nachstedt et al. 2015 *Neural Comput* | Homeostatic plasticity in delay-coupled RC: MC ↑~70%, NARMA-10 ↑~23% | NARMA |
| STD + RC (Neural Networks 2019) | STD widens usable LTP range / criticality | training stability |
| Nat Commun 2025 E–I adaptive RC | Balanced / slightly inhibited best; up to ~130% gains | Dale / β dial |
| Neural Comput balanced E–I MC | Max memory near edge of chaos + E–I balance | spectral radius + I strength |
| Rodriguez et al. 2022 ICML “STP Neurons…” | STP state helps dynamic tasks | CHAR / NARMA |
| London & Häusser 2005 | Dendrites = nonlinear subunits | XOR binding |

---

## 9. Priority shortlist (what to test first)

Ordered by (expected task gain × cheapness × fairness of controls):

1. **TM facilitation on long-range / callosal E edges** — XOR_HARD, TRANSFER, NARMA  
2. **Dual synaptic τ (fast + slow E; fast I)** — NARMA, CHAR  
3. **Neuron-level Dale + β E/I dial** — stability + XOR_HARD  
4. **Homeostatic rate / gain control** — NaN-proof large-N  
5. **Myelin/FA-calibrated delays** (already planned) — TRANSFER, NARMA  
6. *(Later)* dendritic quadratic / two-compartment — XOR binding  
7. *(Later)* STDP on bridges only — not for topology prereg  

### Suggested prereg skeleton (after callosum_scaled finishes)
- Arms: `BASE`, `STP_FAC`, `DUAL_TAU`, `DALE_BAL`, `STP+DUAL`  
- Controls: weight shuffle with **same** STP dynamics; STP-off  
- Primary: XOR_HARD, NARMA, TRANSFER  
- N ∈ {512, 1024}; seeds ≥ 8; Bonferroni over primary endpoints  

---

## 10. References (key)

1. Tsodyks MV, Markram H (1997). The neural code between neocortical pyramidal neurons… *PNAS*.  
2. Tsodyks M, Pawelzik K, Markram H (1998). Neural networks with dynamic synapses. *Neural Comput*.  
3. Markram H et al. (1998). Differential signaling via the same axon… *PNAS*.  
4. Mongillo G, Barak O, Brunel N (2008). Synaptic theory of working memory. *Science* 319:1493. PMID 18339943.  
5. Bi GQ, Poo MM (1998). Synaptic modifications in cultured hippocampal neurons… *J Neurosci*.  
6. Song S, Miller KD, Abbott LF (2000). Competitive Hebbian learning through STDP. *Nat Neurosci*.  
7. Turrigiano GG et al. (1998). Activity-dependent scaling of quantal amplitude… *Nature*.  
8. London M, Häusser M (2005). Dendritic computation. *Annu Rev Neurosci*.  
9. Destexhe / Neuronal Dynamics (EPFL) Ch.3 — GABA_A/B, AMPA, NMDA time courses.  
10. Izhikevich EM, Edelman GM (2008). Large-scale model of mammalian thalamocortical systems. *PNAS* (τ = 5/150/6/150).  
11. MICrONS / Nature 2025 functional connectomics (~524M synapses).  
12. Lin et al. / Nature 2024 FlyWire network statistics (degree ~20.5, P≈1.6e-4).  
13. Nachstedt et al. (2015). Homeostatic plasticity for single node delay-coupled RC. *Neural Comput*.  
14. Nature Communications (2025). Boosting RC with adaptive E–I balance.  
15. Caminiti et al. — callosal conduction sector velocities (already used in harness).  
16. Ringo et al. (1994). Time is of the essence: callosal delay & hemisphere specialization.


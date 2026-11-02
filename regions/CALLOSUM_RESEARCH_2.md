# Corpus callosum research note 2 — more samples + next experiment design

Companion to `CALLOSUM_RESEARCH.md`. Written after hunting additional **open, local-only** callosum / interhemispheric datasets and recomputing comparable stats. **Do not run the bigger experiment until the user says go.** Never push connectome data.

Machine time: 2026-11-02 (America/New_York). Disk after downloads: ~23 G used / ~105 G free.

---

## 1. Datasets found and saved (this pass)

All under `/workspace/flybrain/connectomes/`. Local only. Licenses checked; anything needing a DUA or login was skipped.

### Downloaded / kept

| Path | What | Size | License / cite | Callosum use |
|---|---|---|---|---|
| `human/vohryzek54/` | 27 CTRL + 27 SCHZ Lausanne SC (density, fiber count, length, gFA, ADC) + FC @ scales 83–1015 | ~361 MB | Zenodo 3758534, **CC BY 4.0** | Per-subject callosal edges + fiber lengths; CTRL normative |
| `human/geometric_renorm/` | 40 UL + 44 HCP multi-layer edgelists + coords (Bettinardi et al.) | ~271 MB | Zenodo 3766139, **CC BY 4.0** | Per-subject binary topology; layer4 ≈ scale-83 lh_/rh_ |
| `mouse/` | Allen Oh 2014 213×213 **contra** + **ipsi** structure matrices | ~4.9 MB | Oh et al. 2014 Nature; neuroinformatics.nl mirror | Contralateral = commissural analogue |
| `macaque/` | Markov 2014 FLN (ipsi-only) + **TVB Shen 2019** bilateral weights + tract lengths | ~0.4 MB | Markov 2014; TVB macaque templates | Only TVB has usable inter-hemi edges |
| `human/kang2026_bimanual/` | Kang, Snyder, Mooshagian 2026 posterior-CC inactivation behavior + neural | ~21 MB | Zenodo 19341112, **CC BY 4.0** | Task/mechanism prior (not SC) |
| `human/probabilistic_atlas/` | Sample tables + bundcount/TDI NIfTIs from multi-scale WM atlas | ~37 MB | Zenodo 4919132 | Tables only; full 5.2 GB zip **not** pulled |

### Already on disk (prior pass; still the core human SC stack)

- `human/griffa70/` — Zenodo 2872624, CC BY, 70 subjects SC+FC
- `human/` Rosen & Halgren HCP-MMP 360 Fpt — Zenodo 10150880, CC BY
- `human/` Lausanne `Consensus_Connectomes.mat` + `lausanne/` — Zenodo 3067849
- Braingraph OASIS Scale5; Yeh HCP1065 tract×region; `human/microstructure/` (myelin, FA, JHU/Yeh CC masks)

### Skipped (with reason)

| Candidate | Why skipped |
|---|---|
| HCP continuous / BALSA group packs | Open Access DUT already accepted, but redistribution/Aspera packs are huge and restricted terms; use Zenodo derivatives instead |
| MICA-MICs OSF | ~70 GB / API not clean JSON for this pass |
| FlyWire / hemibrain full | ~852 MB+; Drosophila commissures useful later, not needed for human callosum design now |
| Owen AgCC 2013 raw | Paper used; no open Zenodo deposit of subject connectomes found |
| Core-nets / some CCmapping bulk | Login wall or web-only |
| Full `probconnatlas.zip` (5.2 GB) | Too large this pass; sample tables kept |
| Markov FLN alone for inter-hemi | **Ipsilateral only** — do not treat as callosal |

---

## 2. Cross-dataset callosal stats (comparable)

Computed by `/tmp/callosum_cross_stats.py` → `connectomes/callosum_stats_cross_dataset.json`.

Definitions:

- **Homotopic fraction** = share of undirected *inter*-hemispheric edges that link matched L/R regions.
- **Homo/hetero weight ratio** = mean weight of homotopic edges ÷ mean weight of heterotopic inter edges.
- **FL ratio** = mean fiber length of inter edges ÷ mean length of intra edges (when available).

| Dataset | Regions | Frac homo | Homo/hetero wt | Inter dens. | FL inter/intra | Notes |
|---|---:|---:|---:|---:|---:|---|
| Lausanne consensus 68 | 68 | ~0.11 | ~**18×** | (sparse) | **~1.73** (122 / 71 mm) | Group fiber lengths |
| Griffa70 (mean±across subjects) | ~68–83 | **0.124 ± 0.014** | **~10×** | sparse | (see NOTES) | Strong subject consistency |
| Vohryzek 27 CTRL scale83 | 83 | **0.093** | **~11×** | sparse | **~1.83×** | Matches Griffa pattern |
| Geometric renorm UL/HCP L4 | ~83 | UL ~0.17 / HCP ~0.06 | 1.0 (binary) | binary | — | Topology only |
| Rosen & Halgren 360 (exp Fpt) | 360 | n/a (near-dense) | **~3.7×** | ~1.0 | — | Probabilistic; homo still stronger |
| Mouse Allen Oh 2014 | 213 | — | **~6.4×** (homo/hetero of *contra*) | contra/ipsi wt ~**0.38** | — | Contra weaker than ipsi overall |
| Macaque TVB Shen | bilateral | **0.054** | **~3.3×** | sparse | **~1.52×** | Inter paths longer, homo stronger |
| Markov FLN 29 | 29 | — | — | — | — | **Ipsi-only** — not callosal |

### Pattern that holds across species / labs

1. **Most callosal *edges* are heterotopic** (human ~85–95% of inter edges).
2. **Homotopic edges are few but much stronger** (~3–18× mean weight).
3. **Interhemispheric paths are longer** (~1.5–1.8× intra) → natural delay diversity.
4. **Subject-to-subject callosal layout is stable** (Griffa homo fraction CV ≈ 0.11).
5. **Contralateral is weaker than ipsilateral overall** (mouse contra/ipsi ≈ 0.38) — bridges are sparse by design.
6. Prior FC work (Griffa): ~87% of callosal FC positive (majority bridge ~97%+) — but motor IHI literature says functional *effect* can still be inhibitory via local circuits. Model E/I carefully; do not equate SC sign with FC sign.

---

## 3. Literature lessons (real citations only)

### Why a callosum can matter

- **Conduction delay & brain size (Ringo conjecture).** Larger brains → longer callosal axons → longer delays → pressure to specialize within a hemisphere rather than wait for the other side.  
  Ringo JL et al. (1994) *Cereb Cortex* 4:331–343. https://doi.org/10.1093/cercor/4.4.331 · PMID 7950307.

- **Measured human delays.** Caminiti et al.: callosal delays on order of ~6–10+ ms, stretch with length; humans keep large-diameter motor fibers but delay ranges widen vs macaque.  
  Caminiti et al. (2009) *PNAS* https://doi.org/10.1073/pnas.0907655106 · (2013) *J Neurosci* https://www.jneurosci.org/content/33/36/14501.

- **Interhemispheric inhibition (IHI), especially motor.** TMS: IHI of hand M1 travels mainly through posterior midbody/trunk; conduction ~10–20 ms (mean ~15.5). FA in that segment tracks IHI strength.  
  Meyer et al. (1998) *Ann Neurol* PMID 9506553 · Wahl et al. (2007) *J Neurosci* https://www.jneurosci.org/content/27/45/12132.

- **Topography is ordered, not random.** Genu ↔ prefrontal; midbody ↔ sensorimotor/auditory; splenium ↔ visual/posterior association. Fiber diameter (speed) varies by segment.  
  Aboitiz et al. (1992) *Brain Res* https://doi.org/10.1016/0006-8993(92)90178-C · Fabri & Polonara (2023) *Neuropsychologia* https://doi.org/10.1016/j.neuropsychologia.2023.108533.

- **Agenesis ≠ virtual cut.** AgCC: lower global efficiency, higher local clustering / intra-hemi routes, Probst bundles, high inter-subject variance; virtual callosotomy on a normal SC does not recreate AgCC.  
  Owen et al. (2013) *NeuroImage* PMID 23268782 · https://doi.org/10.1016/j.neuroimage.2012.12.031.

- **Behavioral transfer tasks.** Poffenberger paradigm: crossed−uncrossed RT difference (CUD) ~2–4 ms intact; rises to ~36–61 ms after complete callosotomy — classic measure of callosal transfer.  
  Meta-analysis: https://doi.org/10.1007/s11065-022-09569-w · PMC10769931.

- **Posterior CC & bimanual coordination.** Kang, Snyder, Mooshagian (2026) reversible posterior-CC inactivation impairs bimanual coordination / related neural coupling (Zenodo 19341112). Use as a **task prior**: bimanual / dual-hand binding should load the posterior bridge.

- **Reservoir / BioESN size scaling.** Biological connectome topology in echo-state networks only pays off when N is large enough and **interareal weights stay heterogeneous**; pure rank-preserving biology can *hurt* small reservoirs; homogeneous interareal weights fail to recover the gain at scale.  
  Damicelli et al. (2022) *PLOS Comput Biol* https://doi.org/10.1371/journal.pcbi.1010639 · bio2art scale-up: https://github.com/AlGoulas/bio2art.

### Mapping onto our prior results

| Prior result | How literature + new data explain it |
|---|---|
| Realistic callosum layout ≈ shuffle at N≈160 | BioESN: too small / insufficient diversity for hetero interareal prior to help |
| Sparse bridge > none > dense | Matches sparse callosum + AgCC compensation story |
| Whole-human SC beats shuffles on **NARMA timing** (SUPPORTED after seed fix) | Timing tasks care about delay structure; FL inter/intra ~1.7× + myelin give delay diversity |
| Whole-human SC **not** better on hard memory at that size | Memory capacity in small reservoirs dominated by spectral / randomness, not callosal map |
| Myelin-alone strong on memory | Delay diversity alone is a big lever — keep it, then add real map at larger N |

---

## 4. Proposed next experiment (design only — **do not run yet**)

**Goal:** Give the real callosal / whole-brain prior a fair chance where BioESN says it should help: **larger N**, **heterogeneous interareal weights**, **real delays**, and **tasks that require L↔R communication**.

### 4.1 Network / wiring arms (preregister)

1. **REAL_SC** — Griffa CTRL mean or Lausanne consensus (or Vohryzek CTRL mean) at Lausanne 68/83, expanded with **bio2art-style multi-unit-per-region** so reservoir N ∈ {512, 1024, 2048} (preregister exact set).
2. **REAL_SC + DELAYS** — same topology; edge delays from fiber length (Consensus / Vohryzek FL) × myelin prior (`microstructure/delay_prior_suggestions.json` / Caminiti segment speeds).
3. **SHUFFLE_TOPO** — degree-preserving shuffle of inter + intra (or full weight shuffle); **same delay distribution** as (2) so delays are matched.
4. **WEIGHT_SHUFFLE** — topology fixed, interareal weights reshuffled (tests Damicelli hetero-weight claim).
5. **SPARSE_BRIDGE** — two hemispheres + random sparse callosal bridge matched for edge count / strength to REAL; no topography.
6. **NONE / DENSE** — no bridge; full bipartite dense bridge (replication of earlier qualitative finding).
7. **Optional AgCC-like** — zero callosum + boosted intra-hemi clustering (Owen-inspired), not as a main claim.

Primary contrast for “does the **map** matter?”: **REAL_SC+DELAYS vs SHUFFLE_TOPO (matched delays)** at each N.  
Secondary: REAL vs WEIGHT_SHUFFLE (hetero weights); REAL vs SPARSE_BRIDGE (topography vs sparsity).

### 4.2 Tasks (must include cross-hemisphere load)

Keep XOR-memory / CHAR / NARMA for continuity, **plus** at least two transfer tasks:

1. **Crossed sensory→motor (Poffenberger-like):** stimulus injected into LH visual/auditory ROI; readout from RH motor ROI (and reverse). Metric: transfer lag / accuracy vs ipsilateral control.
2. **Split-binding / bimanual:** two independent streams (L and R); target depends on a feature that only exists after combining both (Kang posterior-CC motivation). Include a “cut posterior CC only” lesion arm if segments available.
3. **Interhemispheric delayed match:** cue in one hemisphere, probe in the other after a delay spanning callosal conduction (~10–20 ms scaled to discrete steps).

Preregister: primary endpoint = transfer-task score of REAL_SC+DELAYS vs SHUFFLE_TOPO at largest N; Bonferroni or hierarchical testing declared up front. Seeds ≥ 8.

### 4.3 Delays / E/I

- Delay ∝ fiber length; scale so mean callosal delay ≈ 10–15 ms in model time (Caminiti / Meyer).
- Optional myelin modulation: faster on high-FA / high-myelin parcels (`microstructure/`).
- Sign: start with **excitatory SC + local inhibition**, plus an optional **IHI arm** that flips or attenuates callosal motor→motor edges (Wahl/Meyer). Do not silently mix FC sign into SC without a named arm.

### 4.4 Scale / compute

- Start N=512; escalate to 1024/2048 only if trends appear (BioESN asymptote).
- Use Vohryzek CTRL or Griffa mean as the structural prior (small, open, already local).
- Keep runs local; results JSON + prereg MD only → `brain-experiments` when GitHub auth works. **Never push** `.mat` / edgelists / microstructure volumes.

### 4.5 Success / failure criteria (honest)

- **SUPPORTED (map matters):** REAL_SC+DELAYS beats SHUFFLE_TOPO on the preregistered transfer task at the largest N, Wilcoxon + correction, all planned seeds.
- **PARTIAL:** delays or sparse-bridge win, but topography does not → report “bridge + delay, not map.”
- **NOT SUPPORTED:** same as N=160 — say so; next step would be even larger N or different E/I, not quiet redefinition.

---

## 5. Files touched this pass

- New data dirs: `connectomes/human/vohryzek54/`, `geometric_renorm/`, `mouse/`, `macaque/`, `kang2026_bimanual/`, `probabilistic_atlas/` (+ READMEs)
- Stats: `connectomes/callosum_stats_cross_dataset.json`
- This note: `regions/CALLOSUM_RESEARCH_2.md`

## 6. What we still do **not** have

- Open AgCC subject connectomes (paper only)
- Full HCP group myelin/FA already partly in `microstructure/`; full TB subject packs still too big
- FlyWire commissure extract (skipped this pass)
- Logged-in Core-nets / CCmapping bulk

---

*End of research note 2. Experiment not started.*

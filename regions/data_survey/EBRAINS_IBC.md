# EBRAINS + Individual Brain Charting (IBC) data survey

**Written:** 2026-10-29 ~08:05 America/New_York (EDT)  
**Scope:** research only — no experiments, no downloads >500 MB.  
**Dir:** `regions/data_survey/`  
**Repo:** https://github.com/Cyberisthename/brain-experiments  
**Context:** leaky-tanh reservoir + ridge; real wiring vs random controls; wins so far = fly CX E/I on delayed XOR, H01 myelin on NARMA; current idea = **task-/neuromodulator-gated modules**.

URL checks (HTTP HEAD/GET, 2026-10-29): all primary links below returned **200** (or 302→200) except CoCoMac `cocomac.g-node.org` which returned **503** at survey time — noted inline.

---

## 0. What we need from this survey

| Need | Ideal data shape | Why |
|---|---|---|
| Reservoir graph | Area×area or neuron×neuron connectivity matrix (weights ± optional delays) | Build / expand Frankenstein modules |
| “Brain chemicals” | Per-region receptor densities (DA, 5-HT, ACh, NE, Glu, GABA, …) | Static **gain map** over modules / bridges |
| Task-dependent routing | Same brains, many tasks → per-task region activation | Data-driven **gating** (which module gets turned up) |

Hard constraint this turn: **no file >500 MB**. Prefer matrices / tables / NeuroVault z-maps over raw fMRI volumes.

---

## 1. EBRAINS survey (connectivity + receptors)

Entry points (verified load):

- Platform: https://ebrains.eu/
- Knowledge Graph search: https://search.kg.ebrains.eu/
- Find data hub: https://ebrains.eu/data-tools-services/data-knowledge/find-data
- Atlases overview: https://ebrains.eu/data-tools-services/brain-atlases
- Python client (siibra): https://ebrains.eu/data-tools-services/brain-atlases/apis/python-client  
  Docs: https://siibra-python.readthedocs.io/

**Honest summary of EBRAINS for our project:** excellent for **human area-level** structural / anatomo-functional matrices and **receptor fingerprints**; weak for **neuron-level** cross-species connectomes we already have elsewhere (fly FlyWire/FAFB, worm, MICrONS). Macaque page is mostly **atlas + cyto/receptor**, not a downloadable Markov-style FLN (we already used Markov 2014 outside EBRAINS; macaque reservoir test flopped).

### 1.1 Datasets usable as reservoir graphs or gain maps

#### A. 1000BRAINS structural connectivity (Julich-Brain / Schaefer)

| Field | Detail |
|---|---|
| **Name** | 1000BRAINS study, connectivity data (v1.1) |
| **Species** | Human |
| **Type** | Area-level **structural connectivity** matrices (streamline counts from dwMRI / MRtrix CSD) |
| **Size** | Per-subject + mean matrices for Julich-Brain and Schaefer parcels. Via siibra: **349** StreamlineCounts (and StreamlineLengths) features for cohort `1000BRAINS`. One matrix is a modest DataFrame (≪500 MB); full cohort is larger — fetch **mean / one subject only**. |
| **License / access** | **Controlled access**; cite DOI; Data Use Agreement. Not click-and-go public. |
| **URL** | KG: https://search.kg.ebrains.eu/instances/Dataset/83407c06-b494-4307-861e-d06a5aecdf8a · DOI: https://doi.org/10.25493/6640-3XH · siibra example: https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/006_connectivity_matrices.html |
| **How we'd use it** | Mean Julich streamline-count matrix → sparse weighted **human module** (or inter-area bridges) in the Frankenstein; density- + weight-distribution-matched ER/SHUF controls. Pair with Julich receptor maps (below) for region-wise gains. |

#### B. HCP structural connectivity via siibra (same StreamlineCounts API)

| Field | Detail |
|---|---|
| **Name** | HCP cohort StreamlineCounts / StreamlineLengths (exposed by siibra on Julich 2.9) |
| **Species** | Human |
| **Type** | Area-level structural connectivity (dwMRI streamlines), 200 subjects |
| **Size** | 200 matrices; each matrix small. Prefer **one subject or cohort mean**. |
| **License / access** | Through EBRAINS/siibra (EBRAINS account typically required for programmatic KG). Underlying HCP data has its own terms. |
| **URL** | Same siibra connectivity notebook as above; search KG for HCP + connectivity. |
| **How we'd use it** | Same as 1000BRAINS — alternate / replicate human area graph. Do **not** pull full HCP imaging. |

#### C. F-TRACT anatomo-functional connectivity (with latencies)

| Field | Detail |
|---|---|
| **Name** | F-TRACT: probabilistic atlas of anatomo-functional connectivity (e.g. F-TRACT_P_11_v2307) |
| **Species** | Human (SEEG epilepsy cohort; 625 implantations / 613 patients in v2307) |
| **Type** | Area×area **effective / anatomo-functional** matrices: connection probability, amplitude stats, **propagation delays / speeds** (CCEP-based), projected to many parcellations including **Julich-Brain 3.0** |
| **Size** | Matrices / feature tables — suitable under 500 MB if we take selected features (e.g. `probability`, `peak_delay__median`) for one parcellation |
| **License / access** | **Free access** on KG (custodian O. David) |
| **URL** | Example KG page (verified 200): https://search.kg.ebrains.eu/instances/ea46bcd6-d19b-479a-8d45-ecc486e3c8b1 · related DOI listed on that page: https://doi.org/10.25493/5AM4-J3F · siibra: `siibra.features.connectivity.AnatomoFunctionalConnectivity` on Julich 3.0 |
| **How we'd use it** | (1) Weight inter-area bridges by CCEP probability. (2) Set **discrete delays** from latency features — rare human delay data that matches our delayed reservoir better than pure streamline counts. Caveat: patient cohort ≠ healthy young adult. |

#### D. Zilles / Palomero-Gallagher receptor density fingerprints (neuromodulator maps)

| Field | Detail |
|---|---|
| **Name** | Quantitative in vitro receptor autoradiography fingerprints (project: “Quantitative receptor data for human, non-human primate, and rodent brains”) |
| **Species** | Primarily **human** Julich-Brain areas; related macaque/rodent receptor work exists in the same lab / EBRAINS ecosystem |
| **Type** | Per-region **mean ± std densities** (fmol/mg protein) for ~14–16 receptors: AMPA, kainate, NMDA, GABA_A/B/BZ, M1–M3, α4β2, α1/α2, 5-HT1A/2, **D1**, etc. Some areas also have laminar profiles. |
| **Size** | Tiny tables (KB–MB). Via siibra: `ReceptorDensityFingerprint` / `ReceptorDensityProfile` for Julich 2.9 — ~20+ anchored regions in the public example (V1, 44, 45, CA1–3, …). |
| **License / access** | Many instances **free access**, **CC BY-NC-SA 4.0** (example CA3: DOI https://doi.org/10.25493/XFHR-X41; V1 example DOI https://doi.org/10.25493/p8sd-jmh) |
| **URL** | Overview: https://julich-brain-atlas.de/atlas/neurotransmitter-receptors · siibra tutorial: https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/001_receptor_densities.html · KG search: receptor + autoradiography filters on https://search.kg.ebrains.eu/ |
| **How we'd use it** | Normalize receptor vector (esp. **D1**, 5-HT, ACh, NE) per Julich area → **static gain** on that area’s reservoir neurons or on bridges leaving/entering it. This is the cleanest EBRAINS fit to “brain chemicals” without inventing a soup. Pair with task gating (IBC) so chemicals = bias, task = switch. |

#### E. Multilevel macaque (MEBRAINS) atlas

| Field | Detail |
|---|---|
| **Name** | Multilevel Macaque Monkey Brain Atlas |
| **Species** | Macaque |
| **Type** | Template + cytoarchitectonic maps + **14 receptor densities** (mesoscale); connectivity advertised as a facet, but **no Markov-style FLN matrix** surfaced as a simple download on the atlas page |
| **Size** | Atlas volumes can be large — **skip bulk NIfTI**; receptor tables / siibra features only if available |
| **License / access** | EBRAINS atlas services (account for some APIs) |
| **URL** | https://ebrains.eu/data-tools-services/brain-atlases/macaque-brain (verified 200) |
| **How we'd use it** | Low priority after macaque FLN flop. Receptor fingerprints for macaque areas could still inform gains if we ever revisit monkey modules. For wiring, keep using **Markov 2014** (outside EBRAINS) or CoCoMac (below). |

#### F. Rat Waxholm + tract-tracing point clouds

| Field | Detail |
|---|---|
| **Name** | Waxholm Space rat atlas + EBRAINS-registered tract-tracing (e.g. corticopontine point clouds) |
| **Species** | Rat |
| **Type** | **Atlas** (222+ regions NIfTI) + selected **point-cloud** labelling of projections — **not** a whole-brain connectivity matrix |
| **Size** | Full atlas NIfTI may approach our size limit; point JSON can be small |
| **License / access** | Open atlas on NITRC; EBRAINS KG for linked experiments |
| **URL** | https://ebrains.eu/data-tools-services/brain-atlases/rat-brain · exploration note: https://ebrains.eu/impact/ebrains-in-use/explorations/explore-neural-connectivity · NITRC: https://www.nitrc.org/projects/whs-sd-atlas/ |
| **How we'd use it** | Only if we need rat-specific paths. Not a drop-in reservoir graph. Skip unless a specific experiment needs pontine / cerebellar routing. |

#### G. Mouse (Allen via EBRAINS / AllenSDK)

| Field | Detail |
|---|---|
| **Name** | Allen Mouse Brain Connectivity Atlas (EBRAINS hosts atlas linkage; matrices from Allen) |
| **Species** | Mouse |
| **Type** | AAV projection → structure-level **unionize** matrices (projection density / volume) |
| **Size** | Structure-level tables are manageable; voxel grids are large — **tables only** |
| **License / access** | Allen terms; EBRAINS as discovery layer |
| **URL** | EBRAINS mouse atlas: https://ebrains.eu/data-tools-services/brain-atlases (Mouse Brain Atlas card) · AllenSDK connectivity: https://allensdk.readthedocs.io/en/stable/connectivity.html |
| **How we'd use it** | We already have MICrONS neuron-level mouse tissue. Allen area matrix is only useful as a **coarse** mouse module / bridge prior — optional. |

#### H. CoCoMac (macaque tract-tracing meta-DB; not EBRAINS-hosted)

| Field | Detail |
|---|---|
| **Name** | CoCoMac 2 |
| **Species** | Macaque |
| **Type** | Literature-aggregated axonal tract-tracing connectivity statements / matrices |
| **Size** | Database extracts — small |
| **License / access** | Open research resource |
| **URL** | https://cocomac.g-node.org/cocomac2/main/index.php — **HTTP 503 at survey time**; treat as flaky. Alternate literature: Markov FLN already used. |
| **How we'd use it** | Backup macaque wiring if site recovers. Not required for gated-team work. |

### 1.2 What EBRAINS is *not* giving us

- No ready **fly / worm / finch** neuron connectomes (we already have those elsewhere).
- Macaque “connectivity” on the atlas page ≠ downloadable FLN; do not re-download atlas volumes hoping for synapses.
- Full **1000BRAINS / HCP imaging** cohorts exceed the 500 MB rule — take **matrices only** through siibra.
- Receptor maps cover **selected** Julich areas, not every parcel — expect missing regions → impute with atlas mean or leave gain = 1.

---

## 2. Individual Brain Charting (IBC) — the multi-task collection the user described

### 2.1 Identification

**Confirmed:** the collection described (“individuals scanned across dozens of different tasks — including mathematics, spatial navigation, working memory, and emotion recognition”) is **Individual Brain Charting (IBC)** (Pinho, Thirion, Hertz-Pannier, et al.).

Evidence:

- Task battery explicitly includes **MathLanguage** (math), **SpatialNavigation**, **HcpWm** (working memory), **HcpEmotion / EmoReco** (emotion), plus dozens more (ARCHI, HCP set, MTT, TheoryOfMind, …).  
  Task docs: https://individual-brain-charting.github.io/docs/tasks.html (verified 200)
- OpenNeuro accession **ds002685** (versions through **2.0.0** documented in community catalogs): https://openneuro.org/datasets/ds002685 (verified 200)
- Hosting overview: https://individual-brain-charting.github.io/docs/data_hosting.html · project data page: https://project.inria.fr/IBC/data/ (verified 200)
- Papers: first release Sci Data 2018 https://doi.org/10.1038/sdata.2018.105 ; second release Sci Data 2020 https://doi.org/10.1038/s41597-020-00670-4 (verified 200); fifth release PMC abstract matches math / navigation / emotion / WM wording.

Fixed cohort (~12 participants; releases vary slightly), **same people**, many tasks → ideal for “how one brain re-routes by problem.”

### 2.2 What’s downloadable (and size honesty)

| Product | Where | Approx size | Usable under 500 MB? |
|---|---|---|---|
| Raw + preprocessed fMRI (BIDS) | OpenNeuro ds002685; also EBRAINS KG collections | **~1.1 TB** combined (Sci Data 2020) | **No** — do not pull |
| Statistical **contrast / z-maps** (volume + surface) | **NeuroVault** collection **6618** (releases combined); earlier **4438** | Map collection (individual NIfTIs; selective download) | **Yes**, if we pull **selected contrasts / subjects**, not the whole vault blindly |
| Preprocessed 3 mm resampled | EBRAINS (linked from IBC hosting docs) | Still large | Prefer NeuroVault contrasts |
| Protocols / event files / analysis code | GitHub `individual-brain-charting/public_protocols`, `public_analysis_code` | Small | Yes |

NeuroVault links (verified 200):

- https://neurovault.org/collections/6618/
- https://neurovault.org/collections/4438/
- Identifiers: https://identifiers.org/neurovault.collection:6618 · https://identifiers.org/neurovault.collection:4438

EBRAINS also hosts IBC raw / preprocessed / contrast collections (account required) — see hosting page. Prefer NeuroVault for our size budget.

### 2.3 How IBC could help the gated Frankenstein

**Best use (fits current MoE / neuromod idea):**

1. Pick a Julich (or Schaefer) parcellation.
2. For each task family that maps to our tests (e.g. WM / math-ish → prediction-ish; spatial / navigation → memory / XOR-ish; emotion → optional third expert), average z-maps → **per-region activation vector** \(a_r(\mathrm{task})\).
3. Softmax or normalize \(a_r\) → **module / area gains** \(g_r(\mathrm{task})\) multiplying that area’s reservoir state (or its specialist readout) — **data-driven neuromodulation / gating**.
4. Control: shuffle region labels; scramble task→gain assignment; uniform gains.

**Secondary use:** compare human task-dependent correlation structure of activation maps (RSA already in IBC papers) to which of our gated experts fires — qualitative only.

### 2.4 Honest mismatch (do not oversell)

| Human IBC | Our reservoir |
|---|---|
| fMRI BOLD, **region / voxel**, seconds | Leaky-tanh units, **ms-ish steps**, synapse-level or area-proxy graphs |
| Same 12 people, cognitive tasks | Abstract XOR_DELAY / NARMA / composite |
| Contrast maps = statistical effects | Gains = engineered multipliers |

So IBC can **inspire and set priors for gates**, not prove that our routing “is” human routing. Any write-up must say: **region-level BOLD ≠ neuron-level connectome dynamics**.

---

## 3. Top 3 usable datasets (for us, now)

### #1 — Julich receptor density fingerprints (Zilles / Palomero-Gallagher via siibra)

- **Why #1:** Tiny download, free (CC BY-NC-SA on many instances), directly implements “brain chemicals” as **per-area gain** without a body or wet chem simulation.
- **Plain-words plan:** Pull D1 / 5-HT / ACh / NE fingerprints for overlapping Julich areas with our human module parcels; normalize; multiply that area’s neurons (or bridges) by \(1 + \beta \cdot \tilde{\rho}\). Control: shuffle densities across areas. Combine with task gate so chemicals = bias, task = switch.
- **Links:** https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/001_receptor_densities.html · https://julich-brain-atlas.de/atlas/neurotransmitter-receptors · example DOI https://doi.org/10.25493/p8sd-jmh

### #2 — IBC NeuroVault contrast maps (not OpenNeuro TB)

- **Why #2:** Exact dataset the user pointed at; unique same-subject multi-task routing signal for **gated specialists**.
- **Plain-words plan:** Download only z-maps for HcpWm, MathLanguage (or ARCHI computation), SpatialNavigation / MTT, HcpEmotion; parcel-average; build task→gain vectors for the gated readout already sketched in `regions/integration_research/RESEARCH.md`. Never mirror the 1.1 TB OpenNeuro dump.
- **Links:** https://openneuro.org/datasets/ds002685 (catalog / metadata) · https://neurovault.org/collections/6618/ · https://individual-brain-charting.github.io/docs/tasks.html · https://doi.org/10.1038/s41597-020-00670-4

### #3 — F-TRACT + (optional) one 1000BRAINS/HCP mean streamline matrix

- **Why #3:** Human **area graph + delays** in one free atlas (F-TRACT); streamline counts as alternate weights if DUA/account allows. Better bridge anatomy than dense Gaussian σ=0.05.
- **Plain-words plan:** Take Julich-parcellated F-TRACT probability + median peak delay → sparse lognormal-ish bridges with discrete delays between human areas (or between Frankenstein modules mapped to Julich). Control: rewired same sparsity / delay histogram. Add receptor gains from #1 on top. Skip full subject cohorts.
- **Links:** https://search.kg.ebrains.eu/instances/ea46bcd6-d19b-479a-8d45-ecc486e3c8b1 · siibra connectivity notebook https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/006_connectivity_matrices.html · 1000BRAINS DOI https://doi.org/10.25493/6640-3XH (controlled)

---

## 4. Recommended next step (still research → then a small experiment)

1. **siibra script** (later): dump receptor fingerprints + one StreamlineCounts mean matrix + selected F-TRACT delay/probability features to CSV under `regions/data_survey/` — keep outputs ≪500 MB.  
2. **NeuroVault-only IBC**: 4–6 contrast families → parcel gain table `task × region`.  
3. Wire gains into the **specialist-readout / gated_team** design (Design 1 in integration research) — chemicals as multiplicative bias, IBC as task gate — with ER modules getting the **same** gate/gain machinery.

No downloads or training in this commit — survey text only.

---

## 5. URL checklist (verified this survey)

| URL | Status |
|---|---|
| https://ebrains.eu/ | 200 |
| https://search.kg.ebrains.eu/ | 200 |
| https://ebrains.eu/data-tools-services/brain-atlases | 200 |
| https://ebrains.eu/data-tools-services/brain-atlases/macaque-brain | 200 |
| https://julich-brain-atlas.de/atlas/neurotransmitter-receptors | 200 |
| https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/001_receptor_densities.html | 200 |
| https://siibra-python.readthedocs.io/en/latest/examples/03_data_features/006_connectivity_matrices.html | 200 |
| https://search.kg.ebrains.eu/instances/Dataset/83407c06-b494-4307-861e-d06a5aecdf8a | 200 |
| https://search.kg.ebrains.eu/instances/ea46bcd6-d19b-479a-8d45-ecc486e3c8b1 | 200 |
| https://doi.org/10.25493/6640-3XH | 302 → KG |
| https://openneuro.org/datasets/ds002685 | 200 |
| https://neurovault.org/collections/6618/ | 200 |
| https://neurovault.org/collections/4438/ | 200 |
| https://individual-brain-charting.github.io/docs/data_hosting.html | 200 |
| https://individual-brain-charting.github.io/docs/tasks.html | 200 |
| https://project.inria.fr/IBC/data/ | 200 |
| https://www.nature.com/articles/s41597-020-00670-4 | 200 |
| https://cocomac.g-node.org/cocomac2/main/index.php | **503** (flaky) |

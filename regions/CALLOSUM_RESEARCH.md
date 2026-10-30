# Corpus callosum — research notes for JARVIS / brain-experiments

Written for the human-wholebrain and realistic_callosum experiments.
Sources are real papers/sites only (links below). No fabricated citations.

## Plain-English takeaways

1. **The callosum is a map, not a random wire.** Front (genu) mostly links prefrontal association cortex; midbody links motor/somatosensory/auditory in front-to-back order; splenium links visual and posterior association areas ([Aboitiz 1992 review](http://biologiachile.cl/biological_research/VOL25_1992/N2/F_ABOITIZ.pdf); [Fabri & Polonara 2023](https://www.sciencedirect.com/science/article/pii/S0028393223000672)).

2. **Fiber size and speed differ by region.** Sensory/motor segments have more large, heavily myelinated, fast axons; prefrontal/association segments are denser with thin, slower fibers ([Aboitiz et al. 1992 Fiber composition](https://doi.org/10.1016/0006-8993(92)90178-C); [Caminiti et al. 2013](https://www.jneurosci.org/content/33/36/14501)). Estimated human callosal delays are roughly ~6–10 ms and stretch with axon length; humans keep large-diameter motor fibers but delay ranges widen vs macaque ([Caminiti et al. 2009 PNAS](https://doi.org/10.1073/pnas.0907655106)).

3. **Cross-hemisphere effects are often inhibitory in motor cortex.** TMS work shows interhemispheric inhibition (IHI) of hand M1 travels mainly through the posterior half of the callosal trunk; estimated transcallosal conduction ~10–20 ms (mean ~15.5 ms) ([Meyer et al. 1998](https://pubmed.ncbi.nlm.nih.gov/9506553/)). Callosal motor FA tracks IHI strength ([Wahl et al. 2007 J Neurosci](https://www.jneurosci.org/content/27/45/12132)).

4. **Most structural callosal links are heterotopic (different regions), not only mirror spots.** Our Griffa/HCP-scale maps (~85–99% heterotopic edge counts, with homotopic edges often stronger) match modern connectome reports and the Wiley atlas layout we used earlier. Homotopy is still important functionally (especially primary sensory/motor fusion).

5. **Born without a callosum ≠ just “cut the wires.”** In agenesis of the corpus callosum (AgCC), global efficiency falls, local within-hemisphere wiring and clustering rise, person-to-person variance rises, and odd Probst bundles appear; a computer “virtual callosotomy” on a normal brain does not fully recreate AgCC ([Owen et al. 2013 NeuroImage](https://doi.org/10.1016/j.neuroimage.2012.12.031); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4127170/)). Split-brain surgical cases show durable disconnection syndromes and lost IHI when the critical midbody segment is cut ([Meyer et al. 1998](https://pubmed.ncbi.nlm.nih.gov/9506553/)).

6. **Development & plasticity.** Callosal pruning and myelination continue into adolescence; AgCC brains compensate with stronger intra-hemispheric routes. That lines up with our finding that *having a sparse bridge* beats *none* or *dense*, and that half-brain modularity itself helps even when the exact bridge pattern does not.

7. **Why tiny RNNs often ignore the real map.** BioESN / bio2art work shows biological connectome topology only matches random reservoirs when enough randomness/diversity remains; rank-preserving “pure biology” maps can *hurt* small reservoirs; heterogeneous inter-areal patterns help more as size grows ([Damicelli et al. 2022 PLOS Comp Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010639); [bio2art](https://github.com/AlGoulas/bio2art)). That is a concrete explanation for our realistic_callosum result (REAL ≈ SHUFFLED at N≈160): the network is too small and too uniformly trained to need the human heterotopic prior.

8. **Modeling implication for us.** Prefer (a) myelin/length-based *delay diversity*, (b) sparse not dense bridges, (c) E/I from FC or known IHI, (d) larger N / multi-unit-per-region expansion, and (e) tasks that *require* cross-hemisphere binding — before claiming the human SC pattern itself is a win. That is exactly what `human_wholebrain` preregisters.

## Key sources (clickable)

| Topic | Citation | Link |
|---|---|---|
| Fiber diameters / topography | Aboitiz et al. 1992 Brain Res | https://doi.org/10.1016/0006-8993(92)90178-C |
| Review of callosal systems | Aboitiz 1992 Biol Res | http://biologiachile.cl/biological_research/VOL25_1992/N2/F_ABOITIZ.pdf |
| Functional topography review | Fabri & Polonara 2023 Neuropsychologia | https://doi.org/10.1016/j.neuropsychologia.2023.108533 |
| IHI topography (lesions) | Meyer et al. 1998 Ann Neurol | https://pubmed.ncbi.nlm.nih.gov/9506553/ |
| FA ↔ IHI (motor CC) | Wahl et al. 2007 J Neurosci | https://www.jneurosci.org/content/27/45/12132 |
| Conduction delays macaque→human | Caminiti et al. 2009 PNAS; 2013 J Neurosci | https://doi.org/10.1073/pnas.0907655106 · https://www.jneurosci.org/content/33/36/14501 |
| AgCC structural connectome | Owen et al. 2013 NeuroImage | https://doi.org/10.1016/j.neuroimage.2012.12.031 |
| BioESN / connectome reservoirs | Damicelli et al. 2022 PLOS Comp Biol | https://doi.org/10.1371/journal.pcbi.1010639 |
| Griffa 70 individual SC+FC | Zenodo 2872624 | https://zenodo.org/records/2872624 |
| Rosen & Halgren HCP-MMP SC | Zenodo 10150880 | https://zenodo.org/records/10150880 |
| HCP | humanconnectome.org | https://www.humanconnectome.org/ |

## Data we keep local (never push)

- `connectomes/human/griffa70/` — 70-subject SC+FC (Lausanne scales) + derived NPZs
- `connectomes/human/lausanne/` — consensus scale matrices
- `connectomes/human/` Rosen CSV / HCP-MMP group Fpt
- Consensus fiber-length matrices from Lausanne Consensus_Connectomes.mat


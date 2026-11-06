
## Post-run interpretation audit (no rerun or tuning)

- E answered 50/52 direct questions correctly and abstained on the two Homer questions (The Iliad / The Odyssey); the same two retrieval misses occurred in every arm. They were not fixed after scoring.
- E and the bag baseline each got 31/31 association questions. The degree-shuffled control averaged 5.625/31 correct across seeds (range 1/31 to 10/31). This establishes a dependence on intact source-card wiring for this particular graph implementation, not an advantage over ordinary local-card extraction.
- E's overall 90/92 versus bag's 88/92 comes entirely from correctly abstaining on the two pre-existing conflict/weak probes. All three boundary-aware arms also passed those probes; correct typed labels were not necessary for that raw-value/weak-marker safeguard.
- Seven topic clusters were fixed before outcomes. Their independence is not externally established: NEURON-4, NEURON-5 and QMS-1 describe related memory research and may share underlying benchmark provenance. H2's adjusted p=0.015625 is conditional on the preregistered topic-cluster construction, not a robust independent-source replication. No alternate clustering was selected after seeing results.
- Primary outcome remains NOT SUPPORTED: adding graph spreading did not improve on the no-link baseline, and real labels tied label-shuffled labels. High absolute accuracy must not be described as making JARVIS generally smarter.

Original source hashes and all 14 frozen artifact hashes were independently checked after the run; none changed. All 3,680 prediction rows cover 92 unique questions, five arms and eight seeds (not 3,680 independent examples).

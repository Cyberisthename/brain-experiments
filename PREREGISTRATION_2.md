# Preregistration 2 — tuning sweep + cell-type knockouts
Written 2026-10-27 BEFORE running any of the experiments below (after seeing Results round 1).

## (a) Does the MC advantage survive per-condition tuning?
Sweep leak a ∈ {0.1,0.2,0.3,0.5,0.7,0.9} × spectral radius ρ ∈ {0.5,0.7,0.9,0.95,0.99,1.1} for REAL, ER, DEG.
Tuning: 2 tuning seeds (seeds 900,901; one fixed null graph each), fit readout on train (1500 steps), score on a separate validation block (500 steps).
Pick the best (a,ρ) PER WIRING and PER TASK (MC max, NARMA-10 NRMSE min, T0 MSE min, using val 256 samples for T0).
Test: 12 fresh seeds (0..11, new null graph per seed, held-out test block) at the chosen params. Mann–Whitney two-sided, REAL vs ER and REAL vs DEG.
- H2a: REAL keeps a significantly higher tuned MC than ER (p<0.05) but the gap shrinks from ~2x to <1.5x.
- H2b: REAL vs DEG tuned MC gap shrinks to non-significant or small (<15%).
- H2c: After tuning, REAL is no better than nulls on NARMA-10 and T0.
Full MC heatmaps (mean of the 2 tuning seeds, validation MC) reported for every wiring.

## (b) Which cell types drive it? (only if H2a holds, i.e. REAL tuned MC > ER and DEG with p<0.05)
At REAL's MC-tuned (a,ρ). Groups = FlyWire `cell_type` collapsed to family prefix (e.g. FB*, PFN*, FS*, ER*, FC*, vDelta*, hDelta*, EPG, PEN, PEG, PFL, PFR, PFG, Delta7, FR, ExR, IbSpsP, SA, EL), groups with ≥20 neurons.
Knockout: zero rows+cols, re-normalize spectral radius on remaining matrix, measure MC (12 seeds).
Control: remove the same number of uniformly random CX neurons (new random set per seed), same procedure.
Effect = MC(control) − MC(type KO), seed-paired; Wilcoxon signed-rank, Bonferroni over number of groups.
- H2d: ring neurons (ER*) and/or the EPG/PEN/Delta7 compass loop produce the largest drop (recurrent attractor circuits); large FB tangential groups have effects no bigger than random.

## Structural checks (descriptive, no tests promised)
Reciprocity, counts of directed 2- and 3-cycles (trace of A^2, A^3 on binary adjacency), E/I synapse-weight balance per neuron, eigenvalue spectrum (fraction of eigenvalues with |λ|>0.5ρ, after scaling) for REAL vs one ER and one DEG graph.
- Prediction: REAL has much higher reciprocity and short-loop counts than both nulls.

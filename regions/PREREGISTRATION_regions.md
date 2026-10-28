# Preregistration: growing the fly network with real brain regions vs random neurons
Written 2026-10-27 before any region results were computed.

## Question
When the CX reservoir (2,875 neurons) grows by adding REAL brain regions with their real connections, do performance and robustness improve more than when it grows by the same number of random neurons?

## Design
- Data: FlyWire v783 Connectivity_783.parquet (signed "Excitatory x Connectivity"), annotations Supplemental_file1 (local copies in ../data).
- Cumulative steps (by `cell_class` / `super_class`):
  S0 CX; S1 + mushroom body (Kenyon_Cell, MBON, DAN); S2 + olfactory→LH path and CX input (ALPN, LHLN, LHCENT, TuBu); S3 + all remaining `central` neurons; S4 + `visual_projection` neurons (only if time allows).
  Edges: all connectome edges among the included neurons.
- Conditions per step:
  REAL = induced subgraph.
  ADD = CX subgraph kept exactly; all edges touching non-CX neurons get their postsynaptic endpoints shuffled among themselves (keeps each neuron's in/out degree, edge count and weights; destroys real region wiring and real CX↔region connections).
  SHUF = whole grown graph degree-shuffled (round-1 `deg`).
  REAL is one fixed graph; ADD/SHUF get a new graph per seed.
- Reservoir: same leaky tanh rate model as round 2, sparse W scaled to unit spectral radius (scipy eigs) times rho; input to all neurons.
- Readout: to keep cost and readout size equal across sizes, the ridge readout uses a random subset of 2,875 neurons (all neurons at S0); same subset rule for all conditions.
- Tuning (reduced from round 2 for cost): leak ∈ {0.2, 0.5, 0.9} × rho ∈ {0.7, 0.95, 1.1}; 2 tuning seeds on validation split; best cell per condition×task; then 8 fresh test seeds.
- Metrics: memory capacity (MC), NARMA-10 NRMSE, T0 MSE (teacher from user's braid_vs_gd_benchmark.py). Robustness: across the 9 grid cells, worst-cell MC and std of MC, and the fraction of cells within 50% of the best.
- Stats: two-sided Mann–Whitney, REAL vs ADD and REAL vs SHUF at each step (8 vs 8). No correction; report how many survive Bonferroni.

## Predictions
- P1: Growing helps everyone a bit on MC/NARMA (bigger reservoir), but with fixed 2,875-neuron readout gains are modest.
- P2: REAL will NOT beat ADD/SHUF on tuned performance (same as round 2) — probably lose on NARMA/T0.
- P3: REAL stays more robust (higher worst-cell MC) at every step, as in round 2.
- P4: "Real regions help more than random" (REAL gain from S0 > ADD gain) — I predict no.

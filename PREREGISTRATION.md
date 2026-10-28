# Preregistration — fly central-complex connectome as a reservoir
Written 2026-10-27 BEFORE any experiment was run (only data loading / edge counts done).

## Data
- Connectivity: Shiu et al. 2024 (Nature) model repo, https://github.com/philshiu/Drosophila_brain_model
  commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960, file `Connectivity_783.parquet` (FlyWire materialization v783).
  Weight = column `Excitatory x Connectivity` (signed synapse count; sign from predicted neurotransmitter, Dale's law per presynaptic neuron).
- Cell classes: FlyWire annotations, https://github.com/flyconnectome/flywire_annotations
  commit 1a83b2776d60d5764cef36b927f5f9679c16c47a2, `supplemental_files/Supplemental_file1_neuron_annotations.tsv`.
- Circuit: all neurons with `cell_class == "CX"` (central complex), induced subgraph: N = 2875 neurons, E = 300,234 directed edges
  (163,416 excitatory / 136,818 inhibitory). Full brain (139k neurons) not used — too heavy for repeated reservoir runs on this box.

## Substrate
Leaky rate reservoir: x_{t+1} = (1-a) x_t + a tanh(W x_t + W_in u_t + b), a = 0.5.
W scaled to spectral radius 0.95 in EVERY condition. W_in ~ U(-1,1)*0.5 dense, b ~ U(-0.1,0.1), both drawn from the run seed.
Only a linear ridge readout (lambda = 1e-6 for NARMA/MC, 1e-4 for T0/T1) is trained. Same hyperparameters for all conditions; no tuning per condition.

## Wiring conditions
1. REAL — CX connectome (fixed; seed varies W_in, b, data).
2. ER — Erdős–Rényi, same N and E, edges placed uniformly (no self loops / duplicates), the real multiset of signed weights randomly assigned to edges.
3. DEG — degree-preserving shuffle: each edge keeps its presynaptic neuron, its weight and its sign (so out-degree, Dale's law and weight distribution preserved);
   postsynaptic endpoints are permuted across all edges (preserves every in-degree), duplicates/self-loops re-shuffled until none remain (or a negligible residue is dropped; count reported).
A new random null graph is drawn per seed.

## Tasks / metrics
- NARMA-10 (u ~ U(0,0.5)), 2000 washout-free train steps after 200 washout, 1000 test steps. Metric: test NRMSE (lower better).
- Memory capacity (MC): u ~ U(-1,1), delays 1..100, MC = sum of squared correlation on test set (higher better).
- T0 (user's benchmark): teacher W* from user's `make_regression_target(RNG_SEED)` (6-strand Burau word). Inputs x ~ N(0,I/6), target tanh(x W*).
  Each input held constant for 10 reservoir steps (state reset to 0), final state is the feature. 512 train / 512 test. Metric: test MSE.
- T1 (user's benchmark): same but loss = mean MSE over K=3 teacher targets.
  - WITH adapters: teachers derived exactly as in the user's script (sha256 of first 3 distinct `task_tags` in adapters/*.json).
  - WITHOUT adapters: 3 teacher words drawn from `random_word(RandomState(seed))`.
- T2 (braid invariant matching) and T3 (CHSH) are discrete searches over braid words with no input→output data: a reservoir readout has no meaningful role. They are NOT tested (stated up front).

## Seeds / stats
Seeds 0..11 (12 per condition). Report mean ± std. Two-sided Mann–Whitney U, REAL vs ER and REAL vs DEG per task; adapters vs no-adapters per wiring. alpha = 0.05, no correction claimed beyond noting multiple comparisons (~14 tests).

## Hypotheses & predictions
- H1 (main): REAL beats ER on NARMA-10 / MC. Prediction: small or no difference; ER and REAL within ~10% (honest prior — published connectome-reservoir results are mixed, and spectral-radius normalization removes the biggest difference).
- H2: Differences between REAL and DEG are smaller than REAL vs ER (degree structure explains most of any effect).
- H3: On T0/T1 all wirings perform about the same (task is a 6-D static tanh map; any rich random feature set solves it).
- H4 (adapters): adapters carry no information into the task except via the hash seed of 3 tag strings, so WITH vs WITHOUT adapters will show no systematic benefit for any wiring; any difference reflects the particular 3 teacher draws, not "adapter knowledge".

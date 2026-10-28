# Preregistration 3 — train weight magnitudes on fixed masks
Written 2026-10-27 BEFORE any round-3 run.

## Setup
- Masks: fly CX (N=2875, E=300,234), ER and degree-shuffle (new graph per seed, built exactly as in rounds 1–2).
- Recurrent weights w = sign · softplus(θ). Sign and mask are fixed and θ is trained. θ is initialised so w equals the wiring's original signed synapse-count weights, scaled to spectral radius 0.95. Full 2,875-neuron networks, sparse matmul, no subsampling.
- Rate RNN x_{t+1} = (1−a)x_t + a·tanh(W x_t + W_in u_t + b), a = 0.5 during training. W_in and b are fixed random values per seed. A linear readout is trained jointly.
- Optimizer: Adam, lr 0.01 on θ and 1e-4 on the readout (changed before the main run after a 3-step smoke test diverged with 0.01 on the readout), identical for every condition: STEPS updates (planned 150; if CPU time forces fewer, the same number for all conditions, stated in results), batch 16.
  - Sequence tasks use truncated BPTT over 150 steps (50 washout + 100 loss steps), fresh random sequences each update.
  - T0 uses 10 held-input steps with batch 64.
- Tasks:
  - NARMA-10.
  - DELAY: predict u(t−k) for k = 1..30 during training. Evaluation is the full memory capacity (MC) over 1..100.
  - T0: the user's Burau teacher.
- Evaluation is identical to round 2. The trained W is frozen, a fresh ridge readout is fit on new train data, and the score comes from test data.
  - For the sequence tasks: 200 washout, 2000 train, 1000 test steps.
  - For T0: 512 train and 512 test samples.
- Robustness grid after training, applied to the frozen trained W with the readout refit at each point: leak {0.3, 0.5, 0.7} × W multiplier {0.8, 1.0, 1.2} (9 points). Report the grid mean and the worst-case value.
- Untrained baseline: the same pipeline with 0 updates. The round-2 tuned results are also quoted.
- Weight movement: relative L2 change ‖w_trained − w_init‖/‖w_init‖, and the Spearman correlation between initial and trained magnitudes.
- Seeds 0..7 (8 per condition). Mann–Whitney two-sided for fly vs ER and fly vs degree-shuffle, and Wilcoxon signed-rank for trained vs untrained within each wiring.

## Hypotheses
- H3a: Training improves every wiring on every task relative to untrained (paired p < 0.05).
- H3b: Fly CX gains the most in relative terms but still ends worse than ER and degree-shuffle on NARMA and T0.
- H3c: Fly CX keeps better worst-case robustness (worst grid point relative to its own grid mean) than the nulls after training.
- H3d: Weights move less than 50% in relative L2 for every wiring, so trained fly weights stay correlated with the originals (Spearman > 0.5).

## Compute amendment (made after timing smoke tests, BEFORE the main run)
Full-size sparse BPTT took ~8 s per update for the sequence tasks. To fit the box, every condition uses **60 updates** and truncated BPTT over **60 steps (20 washout + 40 loss steps)**. T0 also uses 60 updates. No subsampling: all 2,875 neurons are kept. The hypotheses are unchanged.

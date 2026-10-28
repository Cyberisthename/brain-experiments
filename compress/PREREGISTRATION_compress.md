# Preregistration: compressing the fly CX reservoir (+ adapter recovery arm)
Written 2026-10-27 BEFORE any compression results were computed.

## Question
Can the CX reservoir (2,875 neurons, 300,234 edges) be made smaller/faster with little accuracy loss, and does the user's AetherLink "consolidation pass" (merge redundant neurons/modes) beat standard compression? Can small trained adapters recover what's lost ("no cost")?

## Wirings
REAL (FlyWire CX), ER (same N, edge count, weights), DEG (degree-preserving shuffle; graphs from ../regions/out/graphs/0_SHUF_*). New ER/DEG graph per seed.

## Compression methods, keep fraction f ∈ {0.5, 0.25, 0.1} of neurons/state size
- FULL: uncompressed.
- RANDP: random neuron subset, induced subgraph.
- MAGP (magnitude pruning): keep neurons with the largest total |in|+|out| synaptic weight, induced subgraph.
- LOWRANK: top-k singular modes (k = f·N) of W; the reservoir runs in mode space z (k-dim): z ← (1−a)z + a·tanh(Vᵀ U S z + W_in u + b). Dense k×k.
- CONSOL (AetherLink consolidation): embed neurons by top-32 left+right singular vectors of W, k-means into m = f·N clusters, merge each cluster into one unit: W_c = P̄ᵀ W P (P membership, P̄ size-normalised). "Recall/expand at readout": the expanded state P z is a linear copy of z, so for a linear ridge readout it is equivalent to reading z; we read z (stated, not hidden).
- All compressed matrices rescaled to spectral radius 1, then × ρ.

## Protocol
Leaky tanh rate reservoir as in regions.py; tasks MC (lags 1..100), NARMA-10 NRMSE, T0 MSE (user's teacher). Readout reads all units (so readout size shrinks with f — part of the cost). Tuning grid leak {0.2,0.5,0.9} × ρ {0.7,0.95,1.1}, 2 tuning seeds; 8 test seeds at best cell. Robustness = worst grid cell. Speed = wall-clock of the state simulation only (median over test jobs, nice'd, 3 parallel workers; noisy, reported as ratio vs FULL of the same wiring).
Stats: Mann–Whitney, each method×size vs FULL (same wiring), and CONSOL vs each other method at the same size (REAL wiring). Bonferroni within each family.

## Adapter arm (REAL wiring, f = 0.25, CONSOL and MAGP; tuned leak/ρ from above)
- NONE: compressed + ridge readout.
- ADAPT (functional version of the user's adapters): the user's adapter is a 32-bit code (y16/z8/x8). Here the bits gate a bank of 32 bottleneck units in a residual module on the reservoir update: h = tanh(Wz + U·diag(g)·tanh(Vᵀ z) + W_in u + b), g = bits; 8 bits on → 8 active units. U (init 0), V trained. (Edited before any adapter run: residual placed inside the tanh so it plugs into the same simulator. Note the user's real adapter file adapter_9250a8ab has only 3 bits on; we use an 8-on code to match LoRA rank 8.)
- LORA: W + A Bᵀ, rank 8 (same 2·m·8 trainable params as ADAPT's active units).
- W_in, b fixed per seed and shared between training and evaluation. DELAY training target = lags 1..30 (as round 3); evaluation = full MC 1..100.
- Training: Adam lr 0.01 on adapter params + a linear head, 100 updates, truncated BPTT 60 steps (20 washout), batch 16 (T0: 10 held steps, batch 64); then freeze, refit ridge readout, evaluate as above. 8 seeds.
- "No cost" = mean within 1 sd of FULL and Mann–Whitney p > 0.05 vs FULL.

## Predictions
- C1: Moderate compression (f=0.5) costs little on NARMA/T0 for every method; MC drops roughly with readout size.
- C2: CONSOL is not better than MAGP/RANDP at matched size (I expect similar or worse; merging averages away dynamics).
- C3: LOWRANK is the fastest but loses the most MC (low-rank modes of the fly CX are few and contracting).
- C4: REAL stays more robust (worst-cell MC) than ER/DEG after compression.
- C5: Adapters recover part but not all of the loss; ADAPT ≈ LORA (no special benefit from the bit gating); "no cost" fails on at least one task.

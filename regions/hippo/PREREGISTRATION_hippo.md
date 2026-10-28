# Preregistration: synthetic hippocampus attached to the fly CX reservoir
Written 2026-10-27 BEFORE any hippocampus results were computed. Flies have no hippocampus; this module is synthetic, built from mammalian anatomy/theory.

## Literature basis
- Marr (1971) Phil Trans R Soc B 262:23 — "Simple memory": archicortex as autoassociative store; recall from partial cues.
- McNaughton & Morris (1987) TINS 10:408; Treves & Rolls (1992) Hippocampus 2:189 / (1994) Hippocampus 4:374 — CA3 recurrent collaterals as autoassociator; DG mossy fibres as strong sparse "detonator" input for storage, perforant path EC→CA3 for retrieval cue.
- O'Reilly & McClelland (1994) Hippocampus 4:661 — DG sparse expansion and pattern separation vs completion trade-off.
- Rolls (2013) Front Syst Neurosci 7:74 — quantitative DG/CA3/CA1 model; Amaral, Ishizuka & Claiborne (1990) Prog Brain Res 83:1 — CA3 recurrent connectivity a few %.
- Jung & McNaughton (1993) Hippocampus 3:165; Chawla et al. (2005) Hippocampus 15:579 — DG granule cells very sparsely active (~1–5%).
- Leutgeb et al. (2007) Science 315:961; Neunuebel & Knierim (2014) Neuron 81:416 — DG pattern separation, CA3 completion.
- Hasselmo & Schnell (1994) J Neurosci 14:3898; Brun et al. (2002) Science 296:2243 — CA1 combines Schaffer collaterals (CA3) with direct EC III input; CA1 → EC/neocortex output.

## Module (≈3,000 neurons; CX 2,875 unchanged)
- EC 200: input from CX (each EC cell 50 random CX inputs) + CA1→EC back-projection (10 per EC cell).
- DG 2,000: each cell 20 EC inputs; k-winners-take-all, k = 2% (40 active) per time step (others 0) — models strong feedback inhibition.
- CA3 400: mossy fibres 10 DG inputs/cell at 4× weight (strong, sparse); perforant path 20 EC inputs/cell; recurrent collaterals 10% (40/cell).
- CA1 400: Schaffer 40 CA3 inputs/cell; direct EC 20/cell. CA1 → CX (30 CX targets per CA1 cell).
- Hippocampal excitatory weights lognormal(0, 0.5) positive; module internal block scaled to spectral radius 1; CX block (real FlyWire CX, radius 1) unchanged; whole W multiplied by ρ.
- Dynamics: x ← (1−a)x + a·f(Wx + W_in u + b), f = tanh, except DG rows: tanh then kWTA. Input u drives CX neurons only.
- Readout: all CX neurons + the module's 400 "output" neurons (CA1 / matched set). Condition (a) reads CX only (noted asymmetry: 400 fewer features).

## Conditions
- CX: CX alone.
- RAND: CX + same-size module (3,000 neurons), ER placement of the same number of internal edges with the same weights, same CX→module / module→CX edge counts, kWTA on a random 2,000-neuron subset (matched activity sparsity), 400 random neurons read out.
- SHUF: hippocampus with internal edges degree-preserving shuffled (pooled across all internal projections); CX↔module edges kept; kWTA on DG kept.
- HIPPO: intact module.
- New random module per seed for all module conditions; CX fixed.
- Optional Hebbian CA3: NOT run in main design (no plasticity); if added later it is exploratory.

## Tasks
- MC (memory capacity, lags 1..100), NARMA-10 NRMSE, T0 (user's Burau teacher, 10 held-input steps) — same as regions.py.
- PC (pattern completion / associative recall): 20 random ±1 patterns of dim 20 per seed. Each trial: cue = one pattern with 50% of entries zeroed and Gaussian noise σ=0.5, presented 5 steps, then 5 blank steps; readout at end trained (ridge) to output the clean pattern. Score = accuracy of nearest-stored-pattern on 400 test trials (train 800).
- Tuning: leak {0.2,0.5,0.9} × ρ {0.7,0.95,1.1}, 2 tuning seeds; best cell per condition×task; 8 test seeds. Robustness = worst grid cell (tuning seeds) per task.
- Stats: Mann–Whitney two-sided HIPPO vs each of CX/RAND/SHUF per task (n=8 vs 8); Bonferroni over 12 tests (α = 0.0042).

## Predictions
- Q1: HIPPO does NOT significantly beat RAND on MC, NARMA or T0 (extra neurons, not structure, matter there).
- Q2: Module conditions beat CX alone on MC (more neurons/readout features).
- Q3: On PC, HIPPO beats CX alone, but I expect HIPPO vs RAND and vs SHUF to be small/non-significant, because without plasticity the CA3 is not storing the patterns — the ridge readout does the "completion".
- Q4: HIPPO worst-case robustness is no worse than CX alone.

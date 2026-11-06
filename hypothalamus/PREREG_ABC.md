# Preregistration — hypothalamus motif tests A/B/C (engine_v3)

**Date (America/New_York):** 2026-11-05  
**Written before any A/B/C code or runs.** Do not edit hypotheses after seeing results.  
**Gate:** engine_v3 feature ablation finished (STP exploratory win). Callosum_scaled eval stays untouched.

## Frozen context
- Engine: `flybrain/engine_v3` sparse OscRNN (ring delays), REAL_SC callosum at N_label=512 (n_local=8), same expand rules as v3 ablation.
- Motifs and arms: `hypothalamus/RESEARCH.md` §6 (Tests A, B, C). This file locks stats/ops only.
- Parallelism: `torch.set_num_threads(1)`, single worker, `nice -n 19`. Do not stop callosum_scaled PID 362279.

## Shared task / seed / hp freeze
- Tasks: XOR_HARD (↑), NARMA (↓), CHAR (↓), TRANSFER (↑)
- Seeds: 8 seeds `range(3400, 3408)` (new block; never reuse 3300–3307 ablation seeds)
- Hyperparameters: identical to `engine_v3/PREREG_v3_features.md` (lr=5e-3, leak=0.4, epochs=25, steps_per_epoch=10, batches/seq as there)
- Stats: paired Wilcoxon within task vs matched control and vs BASE; Bonferroni α=0.05 / (#planned contrasts). Report means ± std; effect signed so + = better.
- Note: with n=8, two-sided Wilcoxon floor is ~0.0078; confirmatory bars tighter than that are unreachable — mark confirmatory vs exploratory explicitly in RESULTS.

## Test A — Error-driven homeostat vs matched random modulator
Arms: HOME_O_ERR, RAND_MOD (variance-matched noise on e), BASE.  
Hypothesis: HOME_O_ERR beats RAND_MOD and BASE on XOR_HARD and/or NARMA.  
Falsify if HOME_O_ERR ≤ RAND_MOD, or both ≤ BASE.

## Test B — Two-drive rivalry vs single vs shuffled
Arms: RIVAL, SINGLE (matched #params), SHUF_RIVAL.  
Hypothesis: RIVAL > SINGLE and RIVAL > SHUF_RIVAL on TRANSFER and XOR_HARD.  
Falsify if SHUF_RIVAL ≈ RIVAL or SINGLE ≥ RIVAL.

## Test C — Volume broadcast vs forced-synaptic delivery
Arms: VOL, SYN (K random existing edges, matched L2 Δh), BASE.  
Hypothesis: VOL > SYN on NARMA and ≥ on XOR_HARD.  
Falsify if SYN ≥ VOL or neither > BASE.

## Ops
- Implement as feature flags / small modules on engine_v3; resume-safe jsonl under `hypothalamus/results/`.
- Run A, then B, then C (or interleaved by seed) only after this file exists.
- No IBM / QPU. No biology-win language beyond motif abstraction.
- Output: `hypothalamus/results/*.jsonl`, `hypothalamus/RESULTS.md`.

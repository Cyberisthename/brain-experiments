# Preregistration — engine v3 feature ablation

**Date (America/New_York):** 2026-11-02  
**Engine:** `flybrain/engine_v3` sparse OscRNN (ring delays).  
**Rule:** this file is written **before** the ablation jsonl is produced. Do not edit hypotheses after seeing results.

## Frozen context
- Callosum_scaled eval (PID 362279) continues untouched; this ablation uses `torch.set_num_threads(1)` and a single worker.
- Validation (`validate_v2.py`) must show callosum forward parity (features off) before interpreting ablation as biology.

## Arms (feature flags only; topology fixed)
| Arm | Flags |
|---|---|
| BASE | all off |
| STP | TM facilitation (U=0.2, τ_rec=100, τ_fac=500 steps) on inter edges |
| DUAL_TAU | fast α=0.5 + slow α=0.05, κ=0.5 |
| DALE | neuron Dale p_I=0.2, I-scale≈4 |
| HOMEO | slow bias homeostasis toward \|x̄\|=0.3 |
| ALL | STP + DUAL_TAU + DALE + HOMEO |

Topology: callosum_scaled **REAL_SC** at **N_label=512** (n_local=8), same expand rules as v2.  
If connectome load fails, fall back to `synthetic_wiring(n=512)` and note it in RESULTS (not for biology claims).

## Tasks (primary)
- XOR_HARD (higher better)
- NARMA (NRMSE, lower better)
- CHAR (CE, short seq; lower better)
- TRANSFER (higher better)

## Seeds
**Declared:** 8 seeds `range(3300, 3308)`.  
If wall-clock forces a cut before finish, stop after completing whole seeds and report n_completed; do not peek then add seeds.

## Hyperparameters (frozen)
```
lr=5e-3, leak=0.4, epochs=25, steps_per_epoch=10,
batch_xor=32, batch_char=24, batch_narma=24, batch_xfer=32,
seq_xor=48, seq_char=40, seq_narma=64, seq_xfer=48,
char_train_chars=20000, emb_dim=32, wd=0
```

## Stats
- Paired Wilcoxon signed-rank (feature vs BASE) within each task, pairing on seed.
- Bonferroni: α=0.05 / (5 features × 4 tasks) = 0.0025 for confirmatory claims.
- Report means ± std; effect = mean(feature − BASE) with sign so “+ = better” per task.

## Success criteria (engineering, not biology)
- No NaN / poisoned metrics on BASE.
- Ablation finishes or leaves resume-safe jsonl.
- Biology win claims require later prereg on held-out seeds after this exploratory pass.

## Output
- `out/ablation_runs.jsonl`
- `RESULTS.md` (means, Wilcoxon, speedup from validate/bench)

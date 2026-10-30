# oscillation_gating (FULL)

Communication-through-coherence (CTC) multi-region rate-RNN experiment, plus myelin-delay and corpus-callosum arms.

- Prereg: `PREREG_oscillation_gating.md`
- Trainer: `oscillation_gating.py`
- Analyzer: `analyze_oscillation_gating.py`
- Results: `RESULTS_oscillation_gating.md`

```bash
python oscillation_gating.py --tune --workers 2
python oscillation_gating.py --eval --workers 2
python analyze_oscillation_gating.py
```

## Secondary FLY followup (separate labeled run)

After main RESULTS: fly CX / EI_SBM wiring (`bio_wired_net` FRAC_E + EI dens, dens-scaled to match modular nnz=1127 at N=160), Dale softplus×esign, delay=1, no gate. Same seeds/epochs/tasks. Compared vs NO_GATE, MYELIN_ONLY, CALLOSUM.

```bash
python oscillation_gating.py --followup-fly --workers 2   # → out/fly_followup_runs.jsonl
python analyze_fly_followup.py                           # → RESULTS_FLY_FOLLOWUP.md
```

Fast pilot (2026-10-29) archived under `out/pilot_fast/` — undertrained null.

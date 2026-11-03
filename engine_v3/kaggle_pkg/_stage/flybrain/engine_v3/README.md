# Engine v3 — shared sparse delayed rate OscRNN

See `../ENGINE_V3_PLAN.md` and `../regions/SYNAPSE_RESEARCH.md`.

```
rate_network.py  OscRNN + ring delays + opt-in STP / dual-τ / Dale / homeostasis
wiring.py        delay / myelin / Dale helpers
tasks.py         XOR_HARD, NARMA (NaN-safe), CHAR, TRANSFER, …
train.py         Adam + resume-safe jsonl
validate_v2.py   parity vs callosum_scaled / human_wholebrain
run_ablation.py  prereg feature ablation
```

```bash
OMP_NUM_THREADS=1 python3 -m pytest engine_v3/tests -q
OMP_NUM_THREADS=1 python3 -u engine_v3/validate_v2.py
OMP_NUM_THREADS=1 python3 -u engine_v3/run_ablation.py --workers 1
```

Do not raise workers while `callosum_scaled` eval is running.

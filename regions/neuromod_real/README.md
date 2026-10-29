# neuromod_real — Real FlyWire neuromodulators as gain

Preregistered test: DA / 5-HT / OA neurons (FlyWire `top_nt`) drive a slow multiplicative gain on their real postsynaptic targets; ACh/Glu/GABA stay fast.

- Prereg: `PREREG_neuromod_real.md` (locked 2026-10-29 08:08 EDT, before eval)
- Results: `RESULTS_neuromod_real.md` — **primary NOT SUPPORTED** (1/3)
- Figure: `figs/neuromod_real.svg`
- Rebuild `data/fly_cx_nt.npz` by joining `data/ann.tsv` (`top_nt`) onto `frankenstein/data/fly_cx_module.npz` root_ids (see `NT_SOURCE.json`).
- Run: `WORKERS=2 MODE=full ./run_neuromod_real.sh`

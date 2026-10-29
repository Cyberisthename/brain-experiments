# engine2 — AdEx-lite spiking brain engine

Replacement for the leaky-tanh rate reservoir (`flybrain_reservoir.py` / `compress/fast_kernels.py`).

## What is different

| | Old engine | engine2 |
|---|---|---|
| Neuron | leaky `tanh` rate unit | adaptive LIF (AdEx-lite): V, w, refractory |
| Communication | continuous rates | spikes + axonal delay ring (myelin-ready) |
| Dale / chemistry | optional sign on W | Dale from FlyWire `top_nt` (ACh/Glu/GABA; Glu inhibitory in fly) |
| Synapses | instantaneous | exp kernels by transmitter class; optional Tsodyks–Markram STP |
| Neuromod | gain dial | DA/5-HT/OA hooks on **adaptation / threshold** |
| Stability | spectral radius scale | auto E/I calibration to 2–10 Hz, homeostatic thr, clamps, NaN/rate abort |
| Readout | state → ridge | filtered spikes → ridge (same tasks) |

## Quick start

```bash
cd /path/to/brain-experiments   # repo root (flybrain layout)
python -m engine2.tests.test_lif_fi
python -m engine2.tests.test_integrator
python -m engine2.tests.test_stability
python -m engine2.bench          # writes engine2/BENCH.md
python -m engine2.smoke_xor      # SMOKE ONLY, 3 seeds FLY_CX vs ER
```

Optional: `pip install numba` (used if present). CuPy path is stubbed/optional.

## Tasks

`MC`, `NARMA-10`, `XOR_DELAY` (d=5), `PC_HARD` via `engine2.tasks.run_task`, with builders `FLY_CX|ER|SHUF|DENS|EI_SHUF` from frankenstein controls.

**Do not treat smoke numbers as a study.** Full preregistered runs are out of scope for this drop.

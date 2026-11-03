# SBQA — Simulated Bio-Quantum Annealing

Classical / **quantum-inspired** trainer (WKB-style tunneling + Griffa connectome mask). **Not real qubits.**

## Quick start

```bash
# local smoke (CPU, ~30s)
python3 run_all.py --smoke --device cpu

# full prereg (prefer GPU / Kaggle T4)
python3 run_all.py --device auto
# or:
bash run_sbqa_kaggle.sh
```

## Kaggle package

`kaggle_pkg/sbqa_kaggle.zip` + `kaggle_pkg/notebook_cell.py`  
Attach zip as a dataset, GPU on, run the notebook cell. Outputs → `/kaggle/working/out/`.

Data in the zip: Griffa scale-68 group SC only (CC BY 4.0). See `data/DATA_LICENSE.md`.

## Prereg / results

- `PREREG_SBQA.md` (frozen before held-out seeds)
- `RESULTS.md` / `out/summary.json`

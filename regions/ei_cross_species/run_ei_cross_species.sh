#!/usr/bin/env bash
# Kaggle / local runner for ei_cross_species
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Optional: copy dataset files into repo paths
if [[ -f scripts/kaggle_setup.py ]]; then
  python scripts/kaggle_setup.py || true
fi
export WORKERS="${WORKERS:-4}"
export MODE="${MODE:-full}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(TZ=America/New_York date "+%Y-%m-%d %H:%M:%S %Z")
echo "START $START WORKERS=$WORKERS MODE=$MODE"
mkdir -p regions/ei_cross_species/out regions/ei_cross_species/figs
python -u regions/ei_cross_species/ei_cross_species.py 2>&1 | tee regions/ei_cross_species/out/run.log
python -u regions/ei_cross_species/analyze_ei_cross_species.py 2>&1 | tee -a regions/ei_cross_species/out/run.log
END=$(TZ=America/New_York date "+%Y-%m-%d %H:%M:%S %Z")
echo "END $END" | tee -a regions/ei_cross_species/out/run.log
echo "- Tune+eval wall: $START → $END (America/New_York); WORKERS=$WORKERS." > regions/ei_cross_species/out/runtime.txt

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
nice -n 10 python3 -u macaque.py
python3 -u analyze_macaque.py

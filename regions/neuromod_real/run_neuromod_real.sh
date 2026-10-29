#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
export PHASE="${PHASE:-all}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 python3 -u neuromod_real.py 2>&1 | tee -a out/run.log

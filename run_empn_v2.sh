#!/bin/bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WORKERS=7
cd /workspace/flybrain/regions/empn
echo "=== TUNE $(date -Is) ===" | tee out/v2_tune.log
python3 empn_v2.py tune 2>&1 | tee -a out/v2_tune.log
echo "=== RUN $(date -Is) ===" | tee out/v2_run.log
python3 empn_v2.py 2>&1 | tee -a out/v2_run.log
echo V2_DONE > out/v2.done
echo "=== DONE $(date -Is) ===" | tee -a out/v2_run.log

#!/bin/bash
# Do not touch mega_franken. Low priority.
set -euo pipefail
cd /workspace/flybrain/regions/trained_bridges
export MODE=full
export WORKERS="${WORKERS:-1}"
export ES_G="${ES_G:-10}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[$(date -Is)] start trained_bridges WORKERS=$WORKERS ES_G=$ES_G" | tee -a out/run.log
nice -n 19 python3 -u trained_bridges.py 2>&1 | tee -a out/run.log
echo "[$(date -Is)] analyze" | tee -a out/run.log
python3 -u analyze_trained_bridges.py 2>&1 | tee -a out/analyze.log
echo "[$(date -Is)] all done" | tee -a out/run.log

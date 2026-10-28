#!/bin/bash
# Mega-frankenstein full run: tune then eval, then analyze.
set -euo pipefail
cd /workspace/flybrain/regions/mega_franken
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-2}"
LOG=out/run.log
echo "=== mega_franken start $(date -Is) workers=$WORKERS ===" | tee -a "$LOG"

# Resume-safe: MODE=full runs tune (cached) then eval
MODE=full nice -n 19 python3 -u mega_franken.py 2>&1 | tee -a "$LOG"
echo "=== analyze $(date -Is) ===" | tee -a "$LOG"
nice -n 19 python3 -u analyze_mega_franken.py 2>&1 | tee -a "$LOG"
echo "=== mega_franken finished $(date -Is) ===" | tee -a "$LOG"

#!/bin/bash
# Wait for human hybrid done, then frankenstein full 8-seed.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DONE=/workspace/flybrain/regions/hybrid_human/out/human.done
LOG=/workspace/flybrain/regions/frankenstein/out/waiter.log
mkdir -p /workspace/flybrain/regions/frankenstein/out
echo "franken waiter start $(date -Is)" | tee -a "$LOG"
while [ ! -f "$DONE" ]; do
  sleep 120
done
echo "human done; starting frankenstein $(date -Is)" | tee -a "$LOG"
cd /workspace/flybrain/regions/frankenstein
export WORKERS="${WORKERS:-3}"
python3 frankenstein.py >> out/run.log 2>&1 || true
python3 analyze_frankenstein.py >> out/run.log 2>&1 || true
echo ok > out/frankenstein.done
echo "frankenstein done $(date -Is)" | tee -a "$LOG"

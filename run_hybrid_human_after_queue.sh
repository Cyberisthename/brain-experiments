#!/bin/bash
# Wait for mouse hybrid queue to finish, then run human hybrid full 8-seed.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DONE=/workspace/flybrain/queue_after_v2.done
LOG=/workspace/flybrain/regions/hybrid_human/out/waiter.log
mkdir -p /workspace/flybrain/regions/hybrid_human/out
echo "human waiter start $(date -Is)" | tee -a "$LOG"
while [ ! -f "$DONE" ]; do
  sleep 120
done
echo "queue done; starting hybrid_human $(date -Is)" | tee -a "$LOG"
cd /workspace/flybrain/regions/hybrid_human
# light workers so we don't thrash if other stuff remains
export WORKERS="${WORKERS:-4}"
python3 hybrid_human.py >> out/run.log 2>&1 || true
python3 analyze_hybrid_human.py >> out/run.log 2>&1 || true
echo ok > out/human.done
echo "human done $(date -Is)" | tee -a "$LOG"

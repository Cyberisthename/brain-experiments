#!/bin/bash
set -e
cd /workspace/flybrain/regions/finch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-1}"
echo "finch start workers=$WORKERS $(date -Is)" | tee -a out/run.log
nice -n 19 /workspace/flybrain/.venv_cave/bin/python3 -u finch.py 2>&1 | tee -a out/run.log
echo "finch done $(date -Is)" | tee -a out/run.log
touch out/finch.done
/workspace/flybrain/.venv_cave/bin/python3 -u analyze_finch.py 2>&1 | tee out/analyze.log
echo "analyze_done $(date -Is)" | tee -a out/run.log

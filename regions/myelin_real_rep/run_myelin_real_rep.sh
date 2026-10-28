#!/bin/bash
set -e
cd /workspace/flybrain/regions/myelin_real_rep
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-2}"
echo "myelin_real_rep start workers=$WORKERS $(date -Is)" | tee -a out/run.log
nice -n 19 /workspace/flybrain/.venv_cave/bin/python3 -u myelin_real_rep.py 2>&1 | tee -a out/run.log
echo "myelin_real_rep done $(date -Is)" | tee -a out/run.log
touch out/myelin_real_rep.done
/workspace/flybrain/.venv_cave/bin/python3 -u analyze_myelin_real_rep.py 2>&1 | tee out/analyze.log
echo "analyze_done $(date -Is)" | tee -a out/run.log

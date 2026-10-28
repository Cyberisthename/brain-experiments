#!/bin/bash
set -e
cd /workspace/flybrain/regions/best_fly
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-2}"
PY=/workspace/flybrain/.venv_cave/bin/python3
echo "best_fly start $(date -Is) workers=$WORKERS" | tee -a out/run.log
# Tune first (locks best_choice.json)
export MODE=tune
nice -n 19 $PY -u best_fly.py 2>&1 | tee -a out/tune.log
echo "tune done $(date -Is)" | tee -a out/run.log
# Full eval
export MODE=full
nice -n 19 $PY -u best_fly.py 2>&1 | tee -a out/run.log
echo "full done $(date -Is)" | tee -a out/run.log
touch out/best_fly.done
nice -n 19 $PY -u analyze_best_fly.py 2>&1 | tee -a out/analyze.log
echo "analyze done $(date -Is)" | tee -a out/run.log

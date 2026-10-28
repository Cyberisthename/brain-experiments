#!/bin/bash
set -euo pipefail
cd /workspace/flybrain/regions/fewshot_delay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
PY=/workspace/flybrain/.venv_cave/bin/python3
echo "fewshot_delay start MODE=$MODE WORKERS=$WORKERS $(date -Is)" | tee -a out/run.log
nice -n 19 "$PY" -u fewshot_delay.py 2>&1 | tee -a out/run.log
echo "fewshot_delay runs done $(date -Is)" | tee -a out/run.log
# analyze with system python (has matplotlib) if venv lacks it
if "$PY" -c "import matplotlib" 2>/dev/null; then
  "$PY" -u analyze_fewshot_delay.py 2>&1 | tee out/analyze.log
else
  python3 -u analyze_fewshot_delay.py 2>&1 | tee out/analyze.log
fi
echo "analyze_done $(date -Is)" | tee -a out/run.log
touch out/all.done

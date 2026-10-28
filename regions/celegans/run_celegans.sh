#!/usr/bin/env bash
set -euo pipefail
cd /workspace/flybrain/regions/celegans
export PYTHONPATH="/workspace/flybrain/compress:${PYTHONPATH:-}"
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
# Prefer project venvs that have numpy/scipy; fall back
if [[ -x /workspace/flybrain/.venv_cave/bin/python ]]; then
  PY=/workspace/flybrain/.venv_cave/bin/python
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi
# Ensure openpyxl available for any rebuild (not needed at run)
echo "PY=$PY WORKERS=$WORKERS MODE=$MODE"
exec "$PY" -u celegans.py

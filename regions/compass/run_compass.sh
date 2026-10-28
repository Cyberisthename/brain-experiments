#!/usr/bin/env bash
set -euo pipefail
cd /workspace/flybrain/regions/compass
export WORKERS="${WORKERS:-1}"
export MODE="${MODE:-full}"
export PYTHONUNBUFFERED=1
nice -n 19 python3 -u compass.py >> out/run.log 2>&1
nice -n 19 python3 -u analyze_compass.py >> out/run.log 2>&1
echo "analyze done" >> out/run.log

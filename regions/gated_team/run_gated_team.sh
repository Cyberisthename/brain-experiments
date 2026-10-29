#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
export PHASE="${PHASE:-all}"
START=$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')
echo "START $START" | tee -a out/run.log
nice -n 10 python3 -u gated_team.py 2>&1 | tee -a out/run.log
END=$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')
echo "END $END" | tee -a out/run.log
echo "- Tune+eval wall: $START → $END (America/New_York); WORKERS=${WORKERS}." > out/runtime.txt
python3 -u analyze_gated_team.py 2>&1 | tee -a out/run.log

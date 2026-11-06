#!/bin/bash
set -euo pipefail
PID="${1:?pid}"
LOG=/workspace/flybrain/hypothalamus/results/waiter.log
echo "waiter start $(TZ=America/New_York date) pid=$PID" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  n=$(wc -l < /workspace/flybrain/hypothalamus/results/abc_runs.jsonl 2>/dev/null || echo 0)
  echo "$(TZ=America/New_York date '+%H:%M:%S') rows=$n" >> "$LOG"
  sleep 120
done
echo "pid $PID exited $(TZ=America/New_York date)" | tee -a "$LOG"
cd /workspace/flybrain
python3 -u hypothalamus/analyze_abc.py \
  --jsonl hypothalamus/results/abc_runs.jsonl \
  --out-json hypothalamus/results/abc_summary.json \
  --out-md hypothalamus/RESULTS.md \
  >> "$LOG" 2>&1
echo "analyze done $(TZ=America/New_York date)" | tee -a "$LOG"
# final counts
wc -l hypothalamus/results/abc_runs.jsonl | tee -a "$LOG"

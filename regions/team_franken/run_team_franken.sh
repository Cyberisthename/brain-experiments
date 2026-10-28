#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export WORKERS="${WORKERS:-2}"
export MODE="${MODE:-full}"
export PHASE="${PHASE:-all}"
nice -n 10 python3 -u team_franken.py

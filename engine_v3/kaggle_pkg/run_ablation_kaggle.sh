#!/usr/bin/env bash
# Full prereg ablation on CUDA, resume-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FLYBRAIN_GRIFFA_WHOLE_NPZ="$ROOT/connectomes/human/griffa70/griffa70_scale68_group_minimal.npz"
export ENGINE_V3_DETERMINISTIC="${ENGINE_V3_DETERMINISTIC:-1}"

OUT="${OUT_DIR:-/kaggle/working/out}"
mkdir -p "$OUT"
# seed resume from package snapshot if working out is empty
if [[ ! -f "$OUT/ablation_runs.jsonl" ]]; then
  if [[ -f "$ROOT/engine_v3/out/ablation_runs.jsonl" ]]; then
    cp "$ROOT/engine_v3/out/ablation_runs.jsonl" "$OUT/ablation_runs.jsonl"
  elif [[ -f "$ROOT/engine_v3/out/runs.jsonl" ]]; then
    cp "$ROOT/engine_v3/out/runs.jsonl" "$OUT/ablation_runs.jsonl"
  else
    : > "$OUT/ablation_runs.jsonl"
  fi
fi

python3 - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

python3 -u engine_v3/run_ablation.py \
  --device cuda \
  --workers 1 \
  --out-dir "$OUT" \
  --jsonl "$OUT/ablation_runs.jsonl"

python3 -u engine_v3/analyze_ablation.py \
  --jsonl "$OUT/ablation_runs.jsonl" \
  --out "$OUT/ablation_summary.json"

echo "DONE out=$OUT"
ls -la "$OUT"

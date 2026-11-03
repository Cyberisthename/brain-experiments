#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/sbqa_vendor:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SBQA_GRIFFA_NPZ="$ROOT/dtm/data/griffa70_scale68_group_minimal.npz"
export FLYBRAIN_GRIFFA_WHOLE_NPZ="$SBQA_GRIFFA_NPZ"

OUT="${OUT_DIR:-/kaggle/working/out}"
mkdir -p "$OUT"

python3 - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

cd "$ROOT/dtm"
# prefer vendored sbqa on path
export PYTHONPATH="$ROOT/sbqa_vendor:$ROOT/dtm:${PYTHONPATH:-}"
python3 -u run_all.py --device auto --out-dir "$OUT"
echo "DONE out=$OUT"
ls -la "$OUT"

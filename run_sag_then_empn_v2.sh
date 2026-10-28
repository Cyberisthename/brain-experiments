#!/bin/bash
export OMP_NUM_THREADS=1 WORKERS=7
cd /workspace/flybrain/compress
python3 sag.py > out/sag.log 2>&1
echo SAG_DONE > out/sag.done
cd /workspace/flybrain/regions/empn
python3 empn_v2.py diag > out/v2_diag.log 2>&1
python3 empn_v2.py tune > out/v2_tune.log 2>&1
python3 empn_v2.py > out/v2_run.log 2>&1
echo V2_DONE > out/v2.done
echo queue done > /workspace/flybrain/queue_sag_v2.done

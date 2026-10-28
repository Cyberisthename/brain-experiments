#!/bin/bash
export OMP_NUM_THREADS=1 WORKERS=7
cd /workspace/flybrain/compress && python3 compress.py >> out/run.log 2>&1
python3 adapt.py > out/adapt.log 2>&1
cd /workspace/flybrain/regions/hippo_plastic && python3 hippo_plastic.py > out/run.log 2>&1
cd /workspace/flybrain/regions/empn && python3 empn.py > out/run.log 2>&1
echo queue done > /workspace/flybrain/queue.done

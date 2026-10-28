#!/bin/bash
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WORKERS=7
cd /workspace/flybrain/regions/empn

# v2 already finished; ensure RESULTS_empn_v2.md exists
if [ ! -s RESULTS_empn_v2.md ]; then
  echo "=== rewrite RESULTS_empn_v2 ==="
  python3 - <<'PY' || true
import json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
OUT=Path('out')
d=pd.read_json(OUT/'runs_v2.jsonl', lines=True)
summ=d.groupby('agent').agg(success=('success','mean'),train_food=('train_food_rate','mean'),threat=('threat_hit_loom','mean'),steps=('steps_to_food','mean'),short_safe=('short_route_when_safe','mean'),short_loom=('short_route_when_loom','mean'),sec=('sec','mean')).sort_values('success',ascending=False)
Path('RESULTS_empn_v2.md').write_text('# EMPN v2 results\n\nSee prior write.\n\n'+summ.to_markdown()+'\n')
PY
fi

echo "=== ORGAN-FIX v2b (CFS+TR fixed) $(date -Is) ==="
python3 empn_v2b.py > out/v2b_run.log 2>&1 || true
python3 - <<'PY' || true
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
p=Path('out/runs_v2b.jsonl')
d=pd.read_json(p, lines=True) if p.exists() else pd.DataFrame()
lines=['# EMPN v2b organ-fix results\n\nPrereg: `PREREGISTRATION_empn_v2_organ_fix.md`.\n\n']
lines.append('## Root cause (plain)\n')
lines.append('1. CFS scored risk-only / stale goals and overrode Q food-seeking (EMPN 0% vs noCFS 50%).\n')
lines.append('2. +TR replaced TD and used unnormalized eligibility so Wq exploded (FLY 46%→+TR 2.5%).\n\n')
lines.append('## Fixes applied\n')
lines.append('- CFS: expected-reward−risk, seen-only bootstrap, rew decay, assist-only (never flip policy argmax).\n')
lines.append('- TR: keep TD; small L2-normalized episodic bonus η_tr=0.02.\n')
lines.append('- NTRAIN×2; CTB gated; no SCNC V in plasticity RPE.\n\n')
if len(d):
    summ=d.groupby('agent').agg(success=('success','mean'),train_food=('train_food_rate','mean'),threat=('threat_hit_loom','mean'),wq=('wq_final','mean'),sec=('sec','mean')).sort_values('success',ascending=False)
    lines.append('## Success (mean over seeds)\n\n')
    lines.append('| Agent | success | train food | threat loom | wq | sec |\n|---|---:|---:|---:|---:|---:|\n')
    for a,r in summ.iterrows():
        lines.append(f'| {a} | {r.success:.3f} | {r.train_food:.3f} | {r.threat:.3f} | {r.wq:.3f} | {r.sec:.1f} |\n')
    lines.append('\n## +TR vs no-TR\n\n| Base | noTR | +TR | Δ |\n|---|---:|---:|---:|\n')
    bases=sorted({a.replace('+TR','') for a in d.agent.unique()})
    for b in bases:
        if b in summ.index and b+'+TR' in summ.index:
            lines.append(f'| {b} | {summ.loc[b,"success"]:.3f} | {summ.loc[b+"+TR","success"]:.3f} | {summ.loc[b+"+TR","success"]-summ.loc[b,"success"]:+.3f} |\n')
    lines.append(f'\n{len(d)} jobs. Raw: `out/runs_v2b.jsonl`.\n')
else:
    lines.append('No runs_v2b.jsonl.\n')
Path('RESULTS_empn_v2b.md').write_text(''.join(lines))
print('RESULTS_empn_v2b.md')
# also stamp done if all 160 present
if len(d)>=160:
    Path('out/v2b.done').write_text('ok\n')
PY

echo "=== THINKING $(date -Is) ==="
cd /workspace/flybrain/regions/thinking
mkdir -p out
python3 thinking.py > out/run.log 2>&1 || true
python3 - <<'PY' || true
import pandas as pd
from pathlib import Path
d=pd.read_json('out/runs.jsonl', lines=True) if Path('out/runs.jsonl').exists() else pd.DataFrame()
lines=['# Thinking-time results\n\nPrereg: `PREREGISTRATION_thinking.md`.\n\n']
if len(d):
    cols=[c for c in ['agent','arm','K','task'] if c in d.columns]
    metric='success' if 'success' in d.columns else 'value'
    lines.append(d.groupby(cols)[metric].mean().reset_index().to_markdown(index=False)+'\n')
lines.append('\nRaw: `out/runs.jsonl`.\n')
Path('RESULTS_thinking.md').write_text(''.join(lines))
PY

echo "=== AETHERLINK $(date -Is) ==="
cd /workspace/flybrain/regions/aetherlink
mkdir -p out
python3 aetherlink.py > out/run.log 2>&1 || true
python3 - <<'PY' || true
import pandas as pd
from pathlib import Path
d=pd.read_json('out/runs.jsonl', lines=True) if Path('out/runs.jsonl').exists() else pd.DataFrame()
lines=['# AetherLink lanes results\n\nPrereg: `PREREGISTRATION_aetherlink.md`.\n\n']
if len(d):
    cols=[c for c in ['meth','task','N','wide'] if c in d.columns]
    lines.append(d.groupby(cols).agg(mean=('value','mean'),sec=('sec','mean')).reset_index().to_markdown(index=False)+'\n')
Path('RESULTS_aetherlink.md').write_text(''.join(lines))
PY

echo "=== ROLES $(date -Is) ==="
cd /workspace/flybrain/regions/roles
mkdir -p out
python3 roles.py > out/run.log 2>&1 || true
python3 - <<'PY' || true
import pandas as pd
from pathlib import Path
d=pd.read_json('out/runs.jsonl', lines=True) if Path('out/runs.jsonl').exists() else pd.DataFrame()
lines=['# Regions doing their real job — results\n\nPrereg: `PREREGISTRATION_roles.md`.\n\n']
if len(d): lines.append(d.head(80).to_markdown(index=False)+'\n')
Path('RESULTS_roles.md').write_text(''.join(lines))
Path('/workspace/flybrain/regions/RESULTS_roles.md').write_text(''.join(lines))
PY

echo "=== PLASTICITY $(date -Is) ==="
cd /workspace/flybrain/regions/plasticity
mkdir -p out
python3 plasticity.py > out/run.log 2>&1 || true
python3 analyze_plasticity.py >> out/run.log 2>&1 || true

echo "=== MYELIN $(date -Is) ==="
cd /workspace/flybrain/regions/myelin
mkdir -p out
python3 myelin.py > out/run.log 2>&1 || true
python3 analyze_myelin.py >> out/run.log 2>&1 || true

echo "=== HYBRID (no-login MICrONS Zenodo) $(date -Is) ==="
cd /workspace/flybrain/regions/hybrid
mkdir -p out
if [ ! -f data/mouse_module.npz ]; then
  /workspace/flybrain/.venv_cave/bin/python build_mouse_module.py > out/build.log 2>&1 || {
    echo "Hybrid skipped: no mouse_module.npz" > RESULTS_hybrid.md; exit 0; }
fi
python3 hybrid.py > out/run.log 2>&1 || true
python3 analyze_hybrid.py >> out/run.log 2>&1 || true

echo QUEUE_DONE > /workspace/flybrain/queue_after_v2.done
echo "=== ALL QUEUED WORK DONE $(date -Is) ==="

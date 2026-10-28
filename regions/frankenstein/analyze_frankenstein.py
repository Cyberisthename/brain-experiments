"""Summarize frankenstein runs.jsonl → RESULTS_frankenstein.md"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
H = Path(__file__).resolve().parent
rows = [json.loads(l) for l in (H/'out'/'runs.jsonl').open()] if (H/'out'/'runs.jsonl').exists() else []
lines = ['# Frankenstein results\n\nPrereg: `PREREGISTRATION_frankenstein.md`.\n\n']
lines.append('Octopus: **not included** (Lichtman VL = EM+seg only; no edge list). See `../hybrid_octo/`.\n\n')
if not rows:
    lines.append('No runs yet.\n')
else:
    d = pd.DataFrame(rows)
    summ = d.groupby(['arm','task']).agg(mean=('value','mean'), std=('value','std'), n=('value','count')).reset_index()
    lines.append('## Means\n\n')
    lines.append(summ.to_markdown(index=False) + '\n\n')
    if (H/'out'/'best_of_choices.json').exists():
        lines.append('## BEST_OF choices (tune seeds 90–92)\n\n```json\n')
        lines.append((H/'out'/'best_of_choices.json').read_text())
        lines.append('\n```\n')
    lines.append(f'\n{len(rows)} rows. Raw: `out/runs.jsonl`.\n')
(H/'RESULTS_frankenstein.md').write_text(''.join(lines))
print('wrote RESULTS')

"""Analyze myelin_real runs: REAL vs SHUF primary, Bonferroni ×6."""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

OUT = Path(__file__).resolve().parent / "out"
rows = []
for line in (OUT / "runs.jsonl").open():
    r = json.loads(line)
    rows.append(r)

by = defaultdict(list)
nans = []
for r in rows:
    if r.get("error"):
        nans.append((r["net"], r["arm"], r["seed"], r["task"], "error:"+r["error"][:80]))
        continue
    v = r.get("value")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        nans.append((r["net"], r["arm"], r["seed"], r["task"], "nan"))
        continue
    by[(r["net"], r["task"], r["arm"])].append(float(v))

NETS = ["HUMAN_LAYERED", "FRANKEN"]
TASKS = ["MC", "NARMA", "PC_HARD"]
ARMS = ["NODELAY", "REAL_MYELIN", "SHUF_MYELIN", "ADAPT"]

print("=== NaNs / errors ===")
print(nans if nans else "(none)")

print("\n=== Means ± sd ===")
summary = {}
for net in NETS:
    summary[net] = {}
    print(f"\n## {net}")
    for task in TASKS:
        summary[net][task] = {}
        print(f"  {task}:")
        for arm in ARMS:
            vals = by.get((net, task, arm), [])
            if not vals:
                print(f"    {arm:12s} n=0"); continue
            m = float(np.mean(vals)); s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[net][task][arm] = {"mean": m, "sd": s, "n": len(vals), "vals": vals}
            print(f"    {arm:12s} {m:.4f} ± {s:.4f} (n={len(vals)})")

# Primary: REAL vs SHUF, Bonferroni m=6
print("\n=== PRIMARY: REAL_MYELIN vs SHUF_MYELIN (Bonferroni m=6) ===")
primary = []
for net in NETS:
    for task in TASKS:
        a = by.get((net, task, "REAL_MYELIN"), [])
        b = by.get((net, task, "SHUF_MYELIN"), [])
        if len(a) < 2 or len(b) < 2:
            print(f"  {net} {task}: insufficient n"); continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        p_bonf = min(1.0, p * 6)
        direction = "REAL>" if np.mean(a) > np.mean(b) else "REAL<"
        # for NARMA lower is better — still report direction on raw value
        row = dict(net=net, task=task, mean_real=float(np.mean(a)), mean_shuf=float(np.mean(b)),
                   delta=float(np.mean(a)-np.mean(b)), p_raw=float(p), p_bonf=float(p_bonf),
                   direction=direction)
        primary.append(row)
        print(f"  {net:14s} {task:7s} REAL={row['mean_real']:.4f} SHUF={row['mean_shuf']:.4f} "
              f"Δ={row['delta']:+.4f} p_raw={p:.4g} p_bonf={p_bonf:.4g} {direction}")

print("\n=== Exploratory: REAL vs NODELAY / ADAPT (raw p only) ===")
exploratory = []
for net in NETS:
    for task in TASKS:
        a = by.get((net, task, "REAL_MYELIN"), [])
        for ctrl in ("NODELAY", "ADAPT"):
            b = by.get((net, task, ctrl), [])
            if len(a) < 2 or len(b) < 2:
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            row = dict(net=net, task=task, ctrl=ctrl, mean_real=float(np.mean(a)),
                       mean_ctrl=float(np.mean(b)), delta=float(np.mean(a)-np.mean(b)),
                       p_raw=float(p))
            exploratory.append(row)
            print(f"  {net} {task} vs {ctrl}: Δ={row['delta']:+.4f} p_raw={p:.4g}")

out = {"summary": {n: {t: {a: {k: summary[n][t][a][k] for k in ("mean","sd","n")}
                        for a in summary[n][t]} for t in summary[n]} for n in summary},
       "primary_real_vs_shuf": primary,
       "exploratory": exploratory,
       "nans": nans}
json.dump(out, open(OUT / "analyze.json", "w"), indent=2)
# also markdown
lines = ["# Real H01 myelin delay results", "", "Prereg: `PREREGISTRATION_myelin_real.md`.", ""]
for net in NETS:
    lines.append(f"## {net}")
    lines.append("")
    lines.append("| task | NODELAY | REAL_MYELIN | SHUF_MYELIN | ADAPT |")
    lines.append("|---|---:|---:|---:|---:|")
    for task in TASKS:
        cells = []
        for arm in ARMS:
            d = summary.get(net, {}).get(task, {}).get(arm)
            cells.append(f"{d['mean']:.4f}±{d['sd']:.4f}" if d else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")
lines.append("## Primary REAL vs SHUF (Bonferroni ×6)")
lines.append("")
for r in primary:
    lines.append(f"- {r['net']} {r['task']}: Δ={r['delta']:+.4f}, p_raw={r['p_raw']:.4g}, p_bonf={r['p_bonf']:.4g} ({r['direction']})")
(OUT.parent / "RESULTS_myelin_real.md").write_text("\n".join(lines) + "\n")
print("\nwrote out/analyze.json and RESULTS_myelin_real.md")

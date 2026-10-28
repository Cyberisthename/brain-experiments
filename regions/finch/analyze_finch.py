"""Analyze finch song-sequence runs per PREREGISTRATION_finch.md."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out"
FIGS = Path("/workspace/flybrain/figs"); FIGS.mkdir(exist_ok=True)
if not (OUT / "runs.jsonl").exists():
    print("no runs yet"); raise SystemExit(0)
rows = [json.loads(l) for l in (OUT / "runs.jsonl").open()]
by = defaultdict(dict)
for r in rows:
    v = r.get("value")
    if v is None or not math.isfinite(float(v)): continue
    by[(r["arm"], r["task"])][r["seed"]] = float(v)

ARMS = [f"{w}_{d}" for w in ("REAL","ER","SHUF","DENS") for d in ("NOD","DEL")]
TASKS = ["SEQ","TEMPO","ABLATE"]

def paired(a_arm, b_arm, task):
    sa, sb = by.get((a_arm, task), {}), by.get((b_arm, task), {})
    seeds = sorted(set(sa) & set(sb))
    if len(seeds) < 2: return None
    A = np.array([sa[s] for s in seeds]); B = np.array([sb[s] for s in seeds])
    d = A - B
    if np.allclose(d, 0): p = 1.0
    else:
        try: p = float(wilcoxon(A, B, alternative="two-sided").pvalue)
        except Exception: p = 1.0
    return dict(n=len(seeds), mean_a=float(A.mean()), mean_b=float(B.mean()),
                delta=float(d.mean()), p_raw=p)

print("=== Means ===")
summary = {}
for arm in ARMS:
    summary[arm] = {}
    for task in TASKS:
        vals = list(by.get((arm, task), {}).values())
        if not vals: continue
        summary[arm][task] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1) if len(vals)>1 else 0), "n": len(vals)}
        print(f"{arm:10s} {task:6s} {summary[arm][task]['mean']:.4f}±{summary[arm][task]['sd']:.4f} n={len(vals)}")

primary = []
print("\n=== PRIMARY REAL_DEL vs controls × tasks (Bonferroni ×9) ===")
for ctrl in ("ER_DEL","SHUF_DEL","DENS_DEL"):
    for task in TASKS:
        r = paired("REAL_DEL", ctrl, task)
        if not r: continue
        r.update(ctrl=ctrl, task=task, p_bonf=min(1.0, r["p_raw"]*9))
        primary.append(r)
        print(f"  vs {ctrl} {task}: Δ={r['delta']:+.4f} p_raw={r['p_raw']:.4g} p_bonf={r['p_bonf']:.4g}")

secondary = []
print("\n=== SECONDARY REAL_NOD vs NOD controls (Bonferroni ×9) ===")
for ctrl in ("ER_NOD","SHUF_NOD","DENS_NOD"):
    for task in TASKS:
        r = paired("REAL_NOD", ctrl, task)
        if not r: continue
        r.update(ctrl=ctrl, task=task, p_bonf=min(1.0, r["p_raw"]*9))
        secondary.append(r)
        print(f"  vs {ctrl} {task}: Δ={r['delta']:+.4f} p_raw={r['p_raw']:.4g} p_bonf={r['p_bonf']:.4g}")

# figure
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
x = np.arange(4); width = 0.35
wirings = ["REAL","ER","SHUF","DENS"]
for ax, task in zip(axes, TASKS):
    nod = [summary.get(f"{w}_NOD", {}).get(task, {}).get("mean", np.nan) for w in wirings]
    del_ = [summary.get(f"{w}_DEL", {}).get(task, {}).get("mean", np.nan) for w in wirings]
    ax.bar(x - width/2, nod, width, label="NOD", color="#888")
    ax.bar(x + width/2, del_, width, label="DEL", color="#2a9d8f")
    ax.set_xticks(x); ax.set_xticklabels(wirings)
    ax.set_title(task); ax.set_ylabel("NRMSE ↓")
axes[0].legend(fontsize=8)
fig.suptitle("Zebra finch Area X song-proxy")
fig.tight_layout()
fp = FIGS / "finch_song_nrmse.png"
fig.savefig(fp, dpi=140); plt.close()
print("wrote", fp)

wins = [r for r in primary if r["p_bonf"] < 0.05 and r["delta"] < 0]
# claim rule from prereg: ≥1 primary survives AND mean REAL_DEL better than all three on that metric
claim = False
for task in TASKS:
    rs = [r for r in primary if r["task"] == task]
    if len(rs) == 3 and all(r["p_bonf"] < 0.05 and r["delta"] < 0 for r in rs):
        claim = True
json.dump(dict(summary=summary, primary=primary, secondary=secondary, claim_real_wins=claim, fig=str(fp)),
          open(OUT / "analyze.json", "w"), indent=2)
lines = ["# Finch Area X results", "", "Prereg: `PREREGISTRATION_finch.md`.", "",
         "## Means (NRMSE ↓)", "",
         "| arm | SEQ | TEMPO | ABLATE |", "|---|---:|---:|---:|"]
for arm in ARMS:
    cells = []
    for t in TASKS:
        d = summary.get(arm, {}).get(t)
        cells.append(f"{d['mean']:.4f}±{d['sd']:.4f}" if d else "—")
    lines.append(f"| {arm} | " + " | ".join(cells) + " |")
lines += ["", "## Primary REAL_DEL vs delay-matched controls (Bonferroni ×9)", ""]
for r in primary:
    lines.append(f"- vs {r['ctrl']} {r['task']}: Δ={r['delta']:+.4f}, p_raw={r['p_raw']:.4g}, p_bonf={r['p_bonf']:.4g}")
lines += ["", f"**claim_real_wins (prereg rule) = {claim}**", "", f"Figure: `{fp.name}`", ""]
(OUT.parent / "RESULTS_finch.md").write_text("\n".join(lines)+"\n")
print("claim_real_wins", claim)

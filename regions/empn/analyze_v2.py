"""Analyze EMPN v2 runs + curves → RESULTS_empn_v2.md"""
import json, os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu as mw

OUT = "out"
rows = [json.loads(l) for l in open(f"{OUT}/runs_v2.jsonl")]
d = pd.DataFrame(rows)
print("n", len(d), "agents", d.agent.nunique(), "seeds", d.seed.nunique())

summ = d.groupby("agent").agg(
    success=("success", "mean"),
    success_sd=("success", "std"),
    train_food=("train_food_rate", "mean"),
    threat=("threat_hit_loom", "mean"),
    steps=("steps_to_food", "mean"),
    short_safe=("short_route_when_safe", "mean"),
    short_loom=("short_route_when_loom", "mean"),
    clutter=("ca3_clutter_relL1", "mean"),
    sec=("sec", "mean"),
    n=("success", "count"),
).round(4)
print(summ.sort_values("success", ascending=False).to_string())

# pairwise: +TR vs noTR for each base; EMPN vs FLY/RANDMOD; ALL3 vs ALL3RAND
tests = []
bases = sorted({a.replace("+TR", "") for a in d.agent.unique()})
for b in bases:
    a0, a1 = b, b + "+TR"
    if a0 in d.agent.values and a1 in d.agent.values:
        x = d.loc[d.agent == a0, "success"]; y = d.loc[d.agent == a1, "success"]
        if len(x) >= 2 and len(y) >= 2:
            p = mw(y, x, alternative="two-sided").pvalue
            tests.append(dict(cmp=f"{a1} vs {a0}", mean0=x.mean(), mean1=y.mean(), p=p, kind="TR"))
# key prereg
for cmp_a, cmp_b, kind in [
    ("RANDMOD", "FLY", "V2-1"), ("RANDMOD", "EMPN", "V2-1"), ("FLY", "EMPN", "V2-1"),
    ("EMPN+TR", "EMPN", "V2-2"), ("EMPN+SCNC+TR", "EMPN+SCNC", "V2-2"),
    ("EMPN+SCNC+TR", "EMPN+TR", "V2-3"),
    ("EMPN+SCNC+ALL3+TR", "EMPN+SCNC+ALL3RAND+TR", "V2-4"),
]:
    if cmp_a in d.agent.values and cmp_b in d.agent.values:
        x = d.loc[d.agent == cmp_b, "success"]; y = d.loc[d.agent == cmp_a, "success"]
        p = mw(y, x).pvalue if len(x) >= 2 and len(y) >= 2 else 1.0
        tests.append(dict(cmp=f"{cmp_a} vs {cmp_b}", mean0=x.mean(), mean1=y.mean(), p=p, kind=kind))
T = pd.DataFrame(tests)
if len(T):
    T["p_bonf"] = (T.p * len(T)).clip(upper=1); T["sig"] = T.p_bonf < 0.05
    print("\n## tests\n", T.round(5).to_string(index=False))

# learning curves
curves = [json.loads(l) for l in open(f"{OUT}/curves_v2.jsonl")] if os.path.exists(f"{OUT}/curves_v2.jsonl") else []
print("curves", len(curves))

# write markdown
lines = ["# EMPN v2 results — eligibility / RPE + shared learner fixes\n",
         "Prereg: `PREREGISTRATION_empn_v2.md` (diagnosis addenda locked before this run).\n",
         f"Arms × ±TR × 8 seeds = {len(d)} completed jobs.\n"]
lines.append("\n## Success and risk (mean over 8 seeds)\n")
lines.append("| Agent | success | train food | threat (loom) | steps to food | short safe | short loom | sec |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
for a, r in summ.sort_values("success", ascending=False).iterrows():
    lines.append(f"| {a} | {r.success:.3f} | {r.train_food:.3f} | {r.threat:.3f} | {r.steps:.1f} | {r.short_safe:.3f} | {r.short_loom:.3f} | {r.sec:.1f} |\n")

lines.append("\n## Prereg predictions\n")
if len(T):
    lines.append("| Comparison | kind | mean A | mean B | p | p_bonf | sig |\n|---|---|---:|---:|---:|---:|---|\n")
    for _, t in T.iterrows():
        lines.append(f"| {t.cmp} | {t.kind} | {t.mean1:.3f} | {t.mean0:.3f} | {t.p:.4g} | {t.p_bonf:.4g} | {t.sig} |\n")

# learning curve table: mean train food by ep bin for key arms
if curves:
    lines.append("\n## Learning curves (train food rate in 20-ep bins, mean over seeds)\n")
    key = ["FLY", "RANDMOD", "EMPN", "EMPN-noCFS", "EMPN+SCNC", "FLY+TR", "RANDMOD+TR", "EMPN+TR", "EMPN-noCFS+TR"]
    # build
    from collections import defaultdict
    bucket = defaultdict(list)  # (agent, ep) -> list of food
    for c in curves:
        for pt in c.get("curve", []):
            bucket[(c["agent"], pt["ep"])].append(1.0 if pt["food"] else 0.0)
    eps = sorted({e for a, e in bucket})
    lines.append("| ep | " + " | ".join(key) + " |\n|" + "---|" * (len(key)+1) + "\n")
    for e in eps:
        row = [str(e)]
        for a in key:
            vals = bucket.get((a, e), [])
            row.append(f"{np.mean(vals):.2f}" if vals else "—")
        lines.append("| " + " | ".join(row) + " |\n")

lines.append("\n## Honest takeaway\n")
lines.append("See narrative written by analyze after inspecting means.\n")
lines.append("\nRaw: `out/runs_v2.jsonl`, curves: `out/curves_v2.jsonl`.\n")
open(f"{OUT}/analysis_v2.md", "w").write("".join(lines))
summ.to_csv(f"{OUT}/summ_v2.csv")
T.to_csv(f"{OUT}/tests_v2.csv", index=False) if len(T) else None
print("wrote analysis_v2.md")

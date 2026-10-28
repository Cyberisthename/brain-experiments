import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
H = Path(__file__).resolve().parent
d = pd.read_json(H/"out"/"runs.jsonl", lines=True)
lines = ["# Synthetic myelination results\n\nPrereg: `PREREGISTRATION_myelin.md`.\n\n"]
mc = d[d.task=="MC"]
if len(mc):
    lines.append("## MC\n\n")
    lines.append(mc.groupby("arm").value.agg(["mean","std"]).reset_index().to_markdown(index=False)+"\n\n")
    lines.append("### Contrasts (Bonferroni ×2: ADAPT vs FIXED, ADAPT vs ADAPT-SHUF)\n\n")
    a = mc[mc.arm=="ADAPT"]["value"].dropna().to_numpy()
    for other in ("FIXED","ADAPT-SHUF"):
        b = mc[mc.arm==other]["value"].dropna().to_numpy()
        if len(a) and len(b):
            u = mannwhitneyu(a,b,alternative="two-sided")
            lines.append(f"- ADAPT vs {other}: Δ={a.mean()-b.mean():+.3f}, p_raw={u.pvalue:.4g}, p_bonf={min(1,u.pvalue*2):.4g}\n")
mz = d[d.task=="maze"]
if len(mz):
    lines.append("\n## Maze (tagged; primary score is MC)\n\n")
    lines.append(mz.groupby("arm").agg(success=("success","mean")).reset_index().to_markdown(index=False)+"\n")
(H/"RESULTS_myelin.md").write_text("".join(lines))
print("RESULTS_myelin.md")

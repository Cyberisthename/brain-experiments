import pandas as pd
from scipy.stats import mannwhitneyu as mw
d = pd.read_json("out/runs.jsonl", lines=True)
M = ["success", "success_no_odour", "steps_to_food", "threat_hit_loom", "short_route_when_safe", "short_route_when_loom"]
T = d.groupby("agent")[M].mean(); W = d.groupby("agent").success.min().rename("worst_seed_success")
o = ["## Means over 8 seeds (eval episodes)\n", T.join(W).round(3).to_markdown()]
def cmp(a, others, k):
    R = []
    for b in others:
        x, y = d[d.agent == a].success, d[d.agent == b].success; R.append((f"{a} vs {b}", x.mean(), y.mean(), mw(x, y).pvalue if x.nunique() + y.nunique() > 2 else 1.0))
    R = pd.DataFrame(R, columns=["comparison", "a", "b", "p"]); R["p_bonf"] = (R.p * k).clip(upper=1); return R
R1 = cmp("EMPN", ["FLY", "RANDMOD", "EMPN-noTBU", "EMPN-noVTM", "EMPN-noCFS", "EMPN-TBUstatic", "EMPN-FBT", "EMPN-RCX"], 8)
R2 = cmp("EMPN+SCNC", ["EMPN+SCNC-Vmean", "EMPN+SCNC-Vshuf", "EMPN+RAND500"], 3)
o += ["\n## EMPN comparisons on success (Bonferroni x8)\n", R1.round(4).to_markdown(index=False), "\n## SCNC comparisons (Bonferroni x3)\n", R2.round(4).to_markdown(index=False)]
open("out/analysis.md", "w").write("\n".join(o)); print("\n".join(o))

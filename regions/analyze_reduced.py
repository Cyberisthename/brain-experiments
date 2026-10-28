import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
d = pd.read_json("out/jobs.jsonl", lines=True); tst = d[~d.val]; sw = d[d.val]
steps = json.load(open("out/steps.json")); out = []
for (k, t), g in tst.groupby(["step", "task"]):
    r = dict(step=steps[k]["step"], N=steps[k]["neurons"], task=t)
    for c in ["REAL", "ADD", "SHUF"]: v = g[g.cond == c].value; r[c] = f"{v.mean():.4g} ± {v.std():.2g} (n={len(v)})"
    for c in ["ADD", "SHUF"]: r[f"p_vs_{c}"] = mannwhitneyu(g[g.cond == "REAL"].value, g[g.cond == c].value).pvalue
    out.append(r)
T = pd.DataFrame(out); print(T.to_markdown(index=False, floatfmt=".3g"))
rob = []
for (k, c), g in sw[sw.task == "MC"].groupby(["step", "cond"]):
    m = g.groupby(["leak", "rho"]).value.mean(); rob.append(dict(step=k, cond=c, worst_MC=m.min(), std_MC=m.std(), frac_within50=(m >= 0.5 * m.max()).mean()))
print(); print(pd.DataFrame(rob).pivot(index="step", columns="cond").round(2).to_markdown())
print("nsig_raw", (T[["p_vs_ADD", "p_vs_SHUF"]] < .05).sum().sum(), "of", T[["p_vs_ADD","p_vs_SHUF"]].size, "bonf", (T[["p_vs_ADD", "p_vs_SHUF"]] < .05 / T[["p_vs_ADD","p_vs_SHUF"]].size).sum().sum())

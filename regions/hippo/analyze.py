import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
d = pd.read_json("out/jobs.jsonl", lines=True); te = d[~d.val]; sw = d[d.val]; C = ["CX", "RAND", "SHUF", "HIPPO"]
rows = []
for t, g in te.groupby("task"):
    r = {"task": t}
    for c in C: v = g[g.cond == c].value; r[c] = f"{v.mean():.4g} ± {v.std():.2g}"
    for c in ["CX", "RAND", "SHUF"]: r[f"p H vs {c}"] = mannwhitneyu(g[g.cond == "HIPPO"].value, g[g.cond == c].value).pvalue
    rows.append(r)
T = pd.DataFrame(rows); print(T.to_markdown(index=False, floatfmt=".3g"))
rb = []
for t, g in sw.groupby("task"):
    m = g.groupby(["cond", "leak", "rho"]).value.mean()
    rb.append({"task": t, **{c: round(float(m[c].min() if t in ("MC", "PC") else m[c].max()), 4) for c in C}})
print("\nWorst grid cell (tuning seeds; MC/PC lowest, NARMA/T0 highest):"); print(pd.DataFrame(rb).to_markdown(index=False))
ps = T[[c for c in T if c.startswith("p")]].values.ravel(); print("\nBonferroni (α=0.05/12) survivors:", int((ps < .05 / 12).sum()), "of 12")

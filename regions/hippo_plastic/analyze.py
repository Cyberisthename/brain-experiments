import pandas as pd
from scipy.stats import mannwhitneyu as mw
d = pd.read_json("out/jobs.jsonl", lines=True); d = d[~d.val]
T = d.groupby("cond").agg(mean=("value", "mean"), sd=("value", "std"), worst=("value", "min"), worst_lesion=("worst_lesion", "min"), lesion_mean=("worst_lesion", "mean")).reset_index()
o = ["## Pattern completion, test seeds 0-7\n", T.round(4).to_markdown(index=False)]
x = d[d.cond == "HP_EPG"].value; R = []
for c in ["CX", "HS_EPG", "RANDP_EPG", "SHUFP_EPG", "HP_FBT", "HP_RCX"]:
    y = d[d.cond == c].value; R.append(("HP_EPG vs " + c, x.mean(), y.mean(), mw(x, y).pvalue))
R = pd.DataFrame(R, columns=["comparison", "HP_EPG", "other", "p"]); R["p_bonf"] = (R.p * 6).clip(upper=1)
o += ["\n## Planned comparisons (Bonferroni x6)\n", R.round(5).to_markdown(index=False)]
open("out/analysis.md", "w").write("\n".join(o)); print("\n".join(o))

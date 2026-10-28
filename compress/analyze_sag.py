import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu as mw
d = pd.read_json("out/sag.jsonl", lines=True)
# means with blowups as 1e9 kept for honesty; also clean means
def summary(df):
    rows = []
    for (me, t, a), g in df.groupby(["meth", "task", "arm"]):
        v = g.value; rows.append(dict(meth=me, task=t, arm=a, mean=v.mean(), mean_clean=v[v < 1e8].mean(), n_blow=int((v >= 1e8).sum()),
            sd=v[v < 1e8].std(), worst=v[v < 1e8].min() if t == "MC" else v[v < 1e8].max(),
            sim_sec=g.sim_sec.mean(), train_sec=g.train_sec.mean(), params=g.params.iloc[0], n_tgt=g.n_tgt.iloc[0]))
    return pd.DataFrame(rows)
S = summary(d); print(S.round(4).to_string(index=False))
tests = []
for me in ["CONSOL", "MAGP"]:
    for t in ["MC", "NARMA", "T0"]:
        sag = d[(d.meth == me) & (d.task == t) & (d.arm == "SAG") & (d.value < 1e8)].value
        for other in ["NONE", "ADAPT", "LORA", "LORA15", "SAG_RAND", "SAG_DENSE"]:
            y = d[(d.meth == me) & (d.task == t) & (d.arm == other) & (d.value < 1e8)].value
            if len(sag) < 2 or len(y) < 2: p = 1.0
            else: p = mw(sag, y).pvalue
            tests.append(dict(meth=me, task=t, cmp=f"SAG vs {other}", sag=sag.mean(), other=y.mean(), p=p))
T = pd.DataFrame(tests); T["p_bonf"] = (T.p * len(T)).clip(upper=1); T["sig"] = T.p_bonf < 0.05
print("\n## comparisons", len(T), "survivors", T.sig.sum())
print(T.round(5).to_string(index=False))
open("out/sag_analysis.md", "w").write(S.round(4).to_markdown(index=False) + "\n\n" + T.round(5).to_markdown(index=False))

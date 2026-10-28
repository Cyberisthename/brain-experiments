import os, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "out3"; df = pd.read_json(os.path.join(OUT, "round3_raw.json"))
C = ["REAL", "ER", "DEG"]; T = ["DELAY", "NARMA", "T0"]; res = {}
for t in T:
    d = df[df.task == t]; r = {}
    for c in C:
        a = d[d.cond == c].sort_values("seed")
        r[c] = {k: [float(a[k].mean()), float(a[k].std())] for k in
                ["untrained_test", "trained_test", "trained_gridmean", "trained_worst", "untrained_worst", "rel_L2", "spearman", "loss_first", "loss_last"]}
        r[c]["p_trained_vs_untrained"] = float(wilcoxon(a.trained_test, a.untrained_test).pvalue)
        rob = a.trained_worst / a.trained_gridmean; r[c]["worst_over_mean"] = [float(rob.mean()), float(rob.std())]
    for c in ["ER", "DEG"]:
        for k in ["trained_test", "trained_worst"]:
            r[f"p_REAL_vs_{c}_{k}"] = float(mannwhitneyu(d[d.cond == "REAL"][k], d[d.cond == c][k]).pvalue)
        rr = lambda x: x.trained_worst / x.trained_gridmean
        r[f"p_REAL_vs_{c}_worst_over_mean"] = float(mannwhitneyu(rr(d[d.cond == "REAL"]), rr(d[d.cond == c])).pvalue)
    res[t] = r
json.dump(res, open(os.path.join(OUT, "round3_summary.json"), "w"), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, t in zip(ax, T):
    d = df[df.task == t]
    for i, c in enumerate(C):
        a = d[d.cond == c]
        a_.scatter(np.full(len(a), i - .15), a.untrained_test, c="gray", s=15, label="untrained" if i == 0 else None)
        a_.scatter(np.full(len(a), i + .15), a.trained_test, c="C3", s=15, label="trained" if i == 0 else None)
        a_.errorbar(i + .3, a.trained_gridmean.mean(), yerr=[[a.trained_gridmean.mean() - min(a.trained_grid.apply(min))], [max(a.trained_grid.apply(max)) - a.trained_gridmean.mean()]], fmt="o", c="C0", label="trained grid range" if i == 0 else None)
    a_.set_xticks(range(3), ["fly CX", "ER", "deg-shuffle"]); a_.set_title({"DELAY": "Memory capacity ↑", "NARMA": "NARMA-10 NRMSE ↓", "T0": "T0 MSE ↓"}[t]); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "trained_vs_untrained.png"), dpi=120)
for t in T:
    print(t); [print(" ", k, v) for k, v in res[t].items()]

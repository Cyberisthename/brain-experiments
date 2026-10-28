import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
O = "out/"; st = json.load(open(O + "steps.json")); te = pd.read_csv(O + "test.csv"); sw = pd.read_csv(O + "sweep.csv")
names = [s["step"] for s in st]; C = ["REAL", "ADD", "SHUF"]; T = ["MC", "NARMA", "T0"]; L = []
for t in T:
    L.append(f"\n### {t} ({'higher' if t=='MC' else 'lower'} is better), tuned test, 8 seeds, mean ± std\n| step | neurons | REAL | ADD | SHUF | p REAL vs ADD | p REAL vs SHUF |\n|---|---|---|---|---|---|---|")
    for k, n in enumerate(names):
        d = {c: te[(te.step == k) & (te.cond == c) & (te.task == t)].value.values for c in C}
        f = (lambda x: f"{x.mean():.3f} ± {x.std():.3f}") if t != "MC" else (lambda x: f"{x.mean():.2f} ± {x.std():.2f}")
        p = [mannwhitneyu(d["REAL"], d[c]).pvalue for c in ["ADD", "SHUF"]]
        L.append(f"| {n} | {st[k]['neurons']:,} | {f(d['REAL'])} | {f(d['ADD'])} | {f(d['SHUF'])} | {p[0]:.2g} | {p[1]:.2g} |")
# gain from S0 per condition (seedwise means)
L.append("\n### Gain vs S0 (mean test value at step minus own S0 mean)\n| step | task | REAL | ADD | SHUF |\n|---|---|---|---|---|")
for k in range(1, len(names)):
    for t in T:
        g = {c: te[(te.step == k) & (te.cond == c) & (te.task == t)].value.mean() - te[(te.step == 0) & (te.cond == c) & (te.task == t)].value.mean() for c in C}
        L.append(f"| {names[k]} | {t} | {g['REAL']:+.3f} | {g['ADD']:+.3f} | {g['SHUF']:+.3f} |")
# robustness: per (step,cond,seed) over 9 grid cells on MC
r = sw[sw.task == "MC"].groupby(["step", "cond", "seed"]).value.agg(worst="min", std="std", best="max",
      frac50=lambda v: np.mean(v >= 0.5 * v.max())).reset_index()
L.append("\n### Robustness across the 3x3 leak×radius grid (validation MC; 2 tuning seeds each, mean)\n| step | cond | worst cell | best cell | std over grid | fraction of cells ≥50% of best |\n|---|---|---|---|---|---|")
for (k, c), g in r.groupby(["step", "cond"]):
    L.append(f"| {names[k]} | {c} | {g.worst.mean():.2f} | {g.best.mean():.2f} | {g['std'].mean():.2f} | {g.frac50.mean():.2f} |")
nar = sw[sw.task == "NARMA"].groupby(["step", "cond", "seed"]).value.agg(worst="max").reset_index().groupby(["step", "cond"]).worst.mean()
L.append("\nWorst-cell NARMA NRMSE (validation): " + "; ".join(f"{names[k]} {c}={v:.3g}" for (k, c), v in nar.items()))
open(O + "tables.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5)); x = [s["neurons"] for s in st]
for a, t in zip(ax, T):
    for c in C:
        mm = [te[(te.step == k) & (te.cond == c) & (te.task == t)].value for k in range(len(names))]
        a.errorbar(x, [m.mean() for m in mm], [m.std() for m in mm], marker="o", label=c, capsize=3)
    a.set_xscale("log"); a.set_xlabel("neurons"); a.set_title(f"tuned test {t}"); a.legend()
for c in C:
    g = r[r.cond == c].groupby("step").worst.mean(); ax[3].plot(x, g.values, marker="o", label=c)
ax[3].set_xscale("log"); ax[3].set_title("worst-grid-cell MC (robustness)"); ax[3].legend(); ax[3].set_xlabel("neurons")
plt.tight_layout(); plt.savefig(O + "regions.png", dpi=110)

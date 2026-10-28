import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu as mw
d = pd.read_json("out/jobs.jsonl", lines=True); d = d[~d.val]
hi = {"MC"}; o = []
g = d.groupby(["wire", "meth", "f", "task"])
T = g.value.agg(["mean", "std", "min", "max", "count"]).reset_index(); T["worst"] = np.where(T.task.isin(hi), T["min"], T["max"])
sp_ = g.sim_sec.mean().reset_index(); full = sp_[sp_.meth == "FULL"].set_index(["wire", "task"]).sim_sec
T["speedup"] = [full[(w, t)] / s for w, t, s in zip(T.wire, T.task, g.sim_sec.mean().values)]
o.append("## Test-seed results (8 seeds). Mean ± SD, worst seed, speed-up of simulation vs FULL\n"); o.append(T.round(4).to_markdown(index=False))
# planned comparisons: in REAL, CONSOL vs MAGP, RANDP, LOWRANK at each f/task; each compressed vs FULL; REAL vs ER/DEG for CONSOL
tests = []
for f in [0.5, 0.25, 0.1]:
    for t in ["MC", "NARMA", "T0"]:
        x = d[(d.wire == "REAL") & (d.meth == "CONSOL") & (d.f == f) & (d.task == t)].value
        for m in ["MAGP", "RANDP", "LOWRANK"]:
            y = d[(d.wire == "REAL") & (d.meth == m) & (d.f == f) & (d.task == t)].value; tests.append(("REAL", f"CONSOL vs {m}", f, t, x.mean(), y.mean(), mw(x, y).pvalue))
        for m in ["CONSOL", "MAGP", "RANDP", "LOWRANK"]:
            y = d[(d.wire == "REAL") & (d.meth == m) & (d.f == f) & (d.task == t)].value; z = d[(d.wire == "REAL") & (d.meth == "FULL") & (d.task == t)].value
            tests.append(("REAL", f"{m} vs FULL", f, t, y.mean(), z.mean(), mw(y, z).pvalue))
        for w in ["ER", "DEG"]:
            y = d[(d.wire == w) & (d.meth == "CONSOL") & (d.f == f) & (d.task == t)].value; tests.append(("CONSOL", f"REAL vs {w}", f, t, x.mean(), y.mean(), mw(x, y).pvalue))
S = pd.DataFrame(tests, columns=["scope", "comparison", "f", "task", "mean_a", "mean_b", "p"]); S["p_bonf"] = (S.p * len(S)).clip(upper=1); S["survives"] = S.p_bonf < 0.05
o.append(f"\n## Planned comparisons (Mann-Whitney, Bonferroni over {len(S)})\n"); o.append(S.round(5).to_markdown(index=False))
a = pd.read_json("out/adapt.jsonl", lines=True)
A = a.groupby(["meth", "task", "arm"]).agg(mean=("value", "mean"), sd=("value", "std"), worst_min=("value", "min"), worst_max=("value", "max"), sim_sec=("sim_sec", "mean"), train_sec=("train_sec", "mean"), params=("params", "first"), m=("m", "first")).reset_index()
o.append("\n## Adapter arm (REAL, f=0.25)\n"); o.append(A.round(5).to_markdown(index=False))
fullv = d[(d.wire == "REAL") & (d.meth == "FULL")]; at = []
for (me, t), aa in a.groupby(["meth", "task"]):
    fv = fullv[fullv.task == t].value
    for arm in ["NONE", "ADAPT", "LORA"]:
        x = aa[aa.arm == arm].value; at.append((me, t, arm, "vs FULL", x.mean(), fv.mean(), mw(x, fv).pvalue))
    at.append((me, t, "ADAPT", "vs LORA", aa[aa.arm == "ADAPT"].value.mean(), aa[aa.arm == "LORA"].value.mean(), mw(aa[aa.arm == "ADAPT"].value, aa[aa.arm == "LORA"].value).pvalue))
    at.append((me, t, "ADAPT", "vs NONE", aa[aa.arm == "ADAPT"].value.mean(), aa[aa.arm == "NONE"].value.mean(), mw(aa[aa.arm == "ADAPT"].value, aa[aa.arm == "NONE"].value).pvalue))
AT = pd.DataFrame(at, columns=["meth", "task", "arm", "comparison", "mean_a", "mean_b", "p"]); AT["p_bonf"] = (AT.p * len(AT)).clip(upper=1)
o.append(f"\n## Adapter comparisons (Bonferroni over {len(AT)})\n"); o.append(AT.round(5).to_markdown(index=False))
full_sim = fullv.groupby("task").sim_sec.mean(); o.append("\nFULL REAL sim_sec per task: " + str(full_sim.round(4).to_dict()))
open("out/analysis.md", "w").write("\n".join(o)); print("\n".join(o))

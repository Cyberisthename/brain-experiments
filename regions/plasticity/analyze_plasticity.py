import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
H = Path(__file__).resolve().parent
d = pd.read_json(H/"out"/"runs.jsonl", lines=True)
lines = ["# Plasticity boost results\n\nPrereg: `PREREGISTRATION_plasticity.md`.\n\n"]
g = d.groupby(["arm","eta_mul","tau"]).success.agg(["mean","std","count"]).reset_index()
lines.append(g.to_markdown(index=False)+"\n\n")
picks = json.loads((H/"out"/"picks.json").read_text()) if (H/"out"/"picks.json").exists() else {}
lines.append(f"Tune picks: `{picks}`\n\n")
# Bonferroni: η contrasts at best τ for EMPN+TR
sub = d[(d.arm=="EMPN+TR") & d.success.notna()]
if len(sub):
    best_tau = picks.get("EMPN+TR", [None, None])[1]
    if best_tau is not None:
        s = sub[sub.tau==best_tau]
        base = s[s.eta_mul==1.0]["success"].to_numpy()
        lines.append("## η vs 1× at best τ (Bonferroni)\n\n| η | Δ | p_raw | p_bonf |\n|---:|---:|---:|---:|\n")
        etas = sorted(s.eta_mul.unique())
        m = max(1, len(etas)-1)
        for e in etas:
            if e==1.0: continue
            a = s[s.eta_mul==e]["success"].to_numpy()
            if len(a) and len(base):
                u = mannwhitneyu(a, base, alternative="two-sided")
                lines.append(f"| {e} | {a.mean()-base.mean():+.3f} | {u.pvalue:.4g} | {min(1,u.pvalue*m):.4g} |\n")
(H/"RESULTS_plasticity.md").write_text("".join(lines))
print("RESULTS_plasticity.md")

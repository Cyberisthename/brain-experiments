"""Analyze best_fly: paired Wilcoxon, Bonferroni primary ×9 and secondary ×3."""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parent / "out"
H = Path(__file__).resolve().parent
rows = []
for line in (OUT / "runs.jsonl").open():
    r = json.loads(line)
    rows.append(r)

by = defaultdict(dict)  # (arm, task) -> {seed: value}
errs = []
for r in rows:
    if r.get("error"):
        errs.append(r); continue
    v = r.get("value")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        errs.append(r); continue
    by[(r["arm"], r["task"])][int(r["seed"])] = float(v)

choice = json.loads((OUT / "best_choice.json").read_text()) if (OUT / "best_choice.json").exists() else {}
ARMS = ["FLY_CX_PLAIN", "FLY_CX_FIXED", "FRANKEN_REALMYELIN", "BEST", "ER_BEST", "SHUF_BEST", "DENS_BEST"]
TASKS = ["MC", "NARMA", "PC_HARD"]
SEEDS = list(range(8))


def paired(a_arm, b_arm, task):
    a = by.get((a_arm, task), {})
    b = by.get((b_arm, task), {})
    xs, ys = [], []
    for s in SEEDS:
        if s in a and s in b:
            xs.append(a[s]); ys.append(b[s])
    if len(xs) < 2:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    d = xs - ys
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(xs, ys, alternative="two-sided", zero_method="wilcox")
            p = float(p)
        except ValueError:
            p = 1.0
    return dict(n=len(xs), mean_a=float(xs.mean()), mean_b=float(ys.mean()),
                delta=float(d.mean()), p_raw=p,
                sd_a=float(xs.std(ddof=1)) if len(xs) > 1 else 0.0,
                sd_b=float(ys.std(ddof=1)) if len(ys) > 1 else 0.0)


print("=== Means ± sd ===")
summary = {}
for arm in ARMS:
    summary[arm] = {}
    print(f"\n## {arm}")
    for task in TASKS:
        vals = [by[(arm, task)][s] for s in SEEDS if s in by.get((arm, task), {})]
        if not vals:
            print(f"  {task}: n=0"); continue
        m = float(np.mean(vals)); s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[arm][task] = {"mean": m, "sd": s, "n": len(vals), "vals": vals}
        print(f"  {task:8s} {m:.4f} ± {s:.4f} (n={len(vals)})")

print("\n=== PRIMARY: BEST vs controls (Bonferroni ×9) ===")
primary = []
for ctrl in ("ER_BEST", "SHUF_BEST", "DENS_BEST"):
    for task in TASKS:
        r = paired("BEST", ctrl, task)
        if r is None:
            print(f"  BEST vs {ctrl} {task}: insufficient"); continue
        r["ctrl"] = ctrl; r["task"] = task
        r["p_bonf"] = min(1.0, r["p_raw"] * 9)
        # favor direction
        if task == "NARMA":
            favor = "BEST" if r["delta"] < 0 else ctrl
        else:
            favor = "BEST" if r["delta"] > 0 else ctrl
        r["favor"] = favor
        primary.append(r)
        print(f"  BEST vs {ctrl:10s} {task:7s} BEST={r['mean_a']:.4f} ctrl={r['mean_b']:.4f} "
              f"Δ={r['delta']:+.4f} p_raw={r['p_raw']:.4g} p_bonf={r['p_bonf']:.4g} favor={favor}")

print("\n=== SECONDARY: BEST vs FLY_CX_FIXED (Bonferroni ×3) ===")
secondary = []
for task in TASKS:
    r = paired("BEST", "FLY_CX_FIXED", task)
    if r is None:
        print(f"  BEST vs FLY_CX_FIXED {task}: insufficient"); continue
    r["task"] = task
    r["p_bonf"] = min(1.0, r["p_raw"] * 3)
    if task == "NARMA":
        favor = "BEST" if r["delta"] < 0 else "FLY_CX_FIXED"
    else:
        favor = "BEST" if r["delta"] > 0 else "FLY_CX_FIXED"
    r["favor"] = favor
    secondary.append(r)
    print(f"  BEST vs FLY_CX_FIXED {task:7s} BEST={r['mean_a']:.4f} FIXED={r['mean_b']:.4f} "
          f"Δ={r['delta']:+.4f} p_raw={r['p_raw']:.4g} p_bonf={r['p_bonf']:.4g} favor={favor}")

# exploratory FLY_CX_PLAIN vs FIXED
print("\n=== Exploratory: FLY_CX_FIXED vs FLY_CX_PLAIN (raw p) ===")
exploratory = []
for task in TASKS:
    r = paired("FLY_CX_FIXED", "FLY_CX_PLAIN", task)
    if r is None: continue
    r["task"] = task
    exploratory.append(r)
    print(f"  FIXED vs PLAIN {task}: Δ={r['delta']:+.4f} p_raw={r['p_raw']:.4g}")

# verdict
sig_primary = [r for r in primary if r["p_bonf"] < 0.05 and r["favor"] == "BEST"]
sig_sec = [r for r in secondary if r["p_bonf"] < 0.05 and r["favor"] == "BEST"]
ctrl_wins = [r for r in primary if r["p_bonf"] < 0.05 and r["favor"] != "BEST"]

if sig_primary:
    verdict = ("REAL WIRING WINS after stacking: BEST beat ≥1 random control on a task "
               f"after Bonferroni ({len(sig_primary)} contrasts).")
elif ctrl_wins and not sig_primary:
    verdict = ("RANDOM WIRING WITH SAME TRICKS MATCHES OR BEATS the stacked real recipe "
               f"({len(ctrl_wins)} control-favoring significant contrasts). Real wiring does not win once tricks are shared.")
else:
    verdict = ("NO significant BEST-vs-control difference after Bonferroni. "
               "Honest read: random wiring with the same delay tricks matches the stacked real recipe on these tasks.")

if sig_sec:
    verdict += f" Secondary: BEST also beat FLY_CX_FIXED on {len(sig_sec)} task(s)."
elif secondary:
    verdict += " Secondary: BEST did not significantly beat FLY_CX_FIXED after correction."

print("\n=== VERDICT ===")
print(verdict)

out = {
    "best_choice": choice,
    "summary": {a: {t: {k: summary[a][t][k] for k in ("mean", "sd", "n")}
                for t in summary.get(a, {})} for a in summary},
    "primary": primary,
    "secondary": secondary,
    "exploratory_fixed_vs_plain": exploratory,
    "verdict": verdict,
    "n_errors": len(errs),
}
json.dump(out, open(OUT / "analyze.json", "w"), indent=2, default=float)

# markdown
lines = [
    "# Smartest-fly results",
    "",
    "Prereg: `PREREGISTRATION_best_fly.md`.",
    "",
    f"**BEST recipe locked on tune seeds 90–92:** `{choice.get('candidate', '?')}`",
    "",
    f"Composite: `{choice.get('composite', 'MC/10 - NARMA + PC_HARD')}`.",
    "",
]
if choice.get("scores"):
    lines.append("## Tune scores")
    lines.append("")
    lines.append("| candidate | MC | NARMA | PC_HARD | S |")
    lines.append("|---|---:|---:|---:|---:|")
    for c, d in choice["scores"].items():
        lines.append(f"| {c} | {d['MC']:.4f} | {d['NARMA']:.4f} | {d['PC_HARD']:.4f} | {d['S']:.4f} |")
    lines.append("")

lines.append("## Eval means ± sd (seeds 0–7)")
lines.append("")
lines.append("| arm | MC ↑ | NARMA ↓ | PC_HARD ↑ |")
lines.append("|---|---:|---:|---:|")
for arm in ARMS:
    cells = []
    for task in TASKS:
        d = summary.get(arm, {}).get(task)
        cells.append(f"{d['mean']:.4f}±{d['sd']:.4f}" if d else "—")
    lines.append(f"| {arm} | " + " | ".join(cells) + " |")
lines.append("")

lines.append("## Primary: BEST vs controls (Bonferroni ×9, paired Wilcoxon)")
lines.append("")
for r in primary:
    lines.append(
        f"- BEST vs {r['ctrl']} {r['task']}: Δ={r['delta']:+.4f}, "
        f"p_raw={r['p_raw']:.4g}, p_bonf={r['p_bonf']:.4g}, favor={r['favor']}"
    )
lines.append("")
lines.append("## Secondary: BEST vs FLY_CX_FIXED (Bonferroni ×3)")
lines.append("")
for r in secondary:
    lines.append(
        f"- {r['task']}: Δ={r['delta']:+.4f}, p_raw={r['p_raw']:.4g}, "
        f"p_bonf={r['p_bonf']:.4g}, favor={r['favor']}"
    )
lines.append("")
lines.append("## Verdict")
lines.append("")
lines.append(verdict)
lines.append("")
(H / "RESULTS_best_fly.md").write_text("\n".join(lines))
print("\nwrote out/analyze.json and RESULTS_best_fly.md")

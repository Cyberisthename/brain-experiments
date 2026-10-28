"""Analyze myelin_real_rep: primary REAL vs SHUF NARMA one-sided at leak=0.2."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "out"
FIGS = Path("/workspace/flybrain/figs")
FIGS.mkdir(exist_ok=True)
rows = [json.loads(l) for l in (OUT / "runs.jsonl").open()]

def ok(r):
    if r.get("error"): return False
    v = r.get("value")
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))

# aggregate: for shuffle arms, mean over shuffle_id within (arm,seed,task,leak)
raw = defaultdict(list)
for r in rows:
    if not ok(r): continue
    key = (r["arm"], r["seed"], r["task"], float(r["leak"]))
    raw[key].append(float(r["value"]))

agg = {k: float(np.mean(v)) for k, v in raw.items()}

ARMS = ["NODELAY", "REAL_MYELIN", "SHUF_MYELIN", "LAYER_SHUF", "EI_SHUF"]
TASKS = ["MC", "NARMA", "PC_HARD"]
LEAKS = [0.1, 0.2, 0.3]
PRIMARY_LEAK = 0.2

def vals(arm, task, leak):
    return [agg[(arm, s, task, float(leak))] for s in range(100, 124) if (arm, s, task, float(leak)) in agg]

summary = {}
print("=== Means ± sd (per-seed; shuffle arms averaged over 3 draws) ===")
for leak in LEAKS:
    summary[str(leak)] = {}
    print(f"\n## leak={leak}")
    for task in TASKS:
        summary[str(leak)][task] = {}
        print(f"  {task}:")
        for arm in ARMS:
            v = vals(arm, task, leak)
            if not v:
                print(f"    {arm:12s} n=0"); continue
            m, s = float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
            summary[str(leak)][task][arm] = {"mean": m, "sd": s, "n": len(v)}
            print(f"    {arm:12s} {m:.4f} ± {s:.4f} (n={len(v)})")

# PRIMARY
print("\n=== PRIMARY: REAL vs SHUF NARMA leak=0.2 (one-sided REAL<SHUF) ===")
a = vals("REAL_MYELIN", "NARMA", PRIMARY_LEAK)
b = vals("SHUF_MYELIN", "NARMA", PRIMARY_LEAK)
primary = {}
if len(a) >= 2 and len(b) >= 2:
    u, p = mannwhitneyu(a, b, alternative="less")
    primary = dict(
        mean_real=float(np.mean(a)), mean_shuf=float(np.mean(b)),
        delta=float(np.mean(a) - np.mean(b)), p_one_sided=float(p),
        n_real=len(a), n_shuf=len(b),
        claim_ok=bool(p < 0.05 and np.mean(a) < np.mean(b)),
    )
    print(f"  REAL={primary['mean_real']:.4f} SHUF={primary['mean_shuf']:.4f} "
          f"Δ={primary['delta']:+.4f} p_one_sided={p:.4g} claim_ok={primary['claim_ok']}")
else:
    print("  insufficient data", len(a), len(b))

# secondary
print("\n=== Secondary NARMA leak=0.2 (two-sided raw) ===")
secondary = []
for ctrl in ("LAYER_SHUF", "EI_SHUF", "NODELAY"):
    aa = vals("REAL_MYELIN", "NARMA", PRIMARY_LEAK)
    bb = vals(ctrl, "NARMA", PRIMARY_LEAK)
    if len(aa) < 2 or len(bb) < 2: continue
    u, p = mannwhitneyu(aa, bb, alternative="two-sided")
    row = dict(ctrl=ctrl, mean_real=float(np.mean(aa)), mean_ctrl=float(np.mean(bb)),
               delta=float(np.mean(aa)-np.mean(bb)), p_raw=float(p))
    secondary.append(row)
    print(f"  vs {ctrl}: Δ={row['delta']:+.4f} p_raw={p:.4g}")

# leak robustness for primary contrast
print("\n=== Leak robustness REAL vs SHUF NARMA (one-sided) ===")
leak_rows = []
for leak in LEAKS:
    aa, bb = vals("REAL_MYELIN", "NARMA", leak), vals("SHUF_MYELIN", "NARMA", leak)
    if len(aa) < 2 or len(bb) < 2: continue
    u, p = mannwhitneyu(aa, bb, alternative="less")
    row = dict(leak=leak, mean_real=float(np.mean(aa)), mean_shuf=float(np.mean(bb)),
               delta=float(np.mean(aa)-np.mean(bb)), p_one_sided=float(p))
    leak_rows.append(row)
    print(f"  leak={leak}: REAL={row['mean_real']:.4f} SHUF={row['mean_shuf']:.4f} "
          f"Δ={row['delta']:+.4f} p={p:.4g}")

# figure
fig, ax = plt.subplots(figsize=(7, 4.2))
means = [summary.get("0.2", {}).get("NARMA", {}).get(a, {}).get("mean", np.nan) for a in ARMS]
sds = [summary.get("0.2", {}).get("NARMA", {}).get(a, {}).get("sd", 0) for a in ARMS]
x = np.arange(len(ARMS))
colors = ["#888", "#2a9d8f", "#e76f51", "#f4a261", "#264653"]
ax.bar(x, means, yerr=sds, color=colors, capsize=4, alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=20, ha="right")
ax.set_ylabel("NARMA NRMSE (↓ better)")
ax.set_title("Myelin placement replication — HUMAN_LAYERED, leak=0.2")
ax.axhline(means[ARMS.index("REAL_MYELIN")] if means[1]==means[1] else 0, color="#2a9d8f", ls="--", lw=0.8, alpha=0.5)
fig.tight_layout()
fig_path = FIGS / "myelin_real_rep_narma.png"
fig.savefig(fig_path, dpi=140)
plt.close()
print("wrote", fig_path)

# also MC/PC panel at leak 0.2
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, task, ylab, higher in zip(axes, TASKS,
        ["MC (↑)", "NARMA NRMSE (↓)", "PC_HARD (↑)"], [True, False, True]):
    m = [summary.get("0.2", {}).get(task, {}).get(a, {}).get("mean", np.nan) for a in ARMS]
    s = [summary.get("0.2", {}).get(task, {}).get(a, {}).get("sd", 0) for a in ARMS]
    ax.bar(x, m, yerr=s, color=colors, capsize=3, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=8)
    ax.set_title(task); ax.set_ylabel(ylab)
fig.suptitle("Myelin REAL vs controls — leak=0.2, seeds 100–123")
fig.tight_layout()
fig2 = FIGS / "myelin_real_rep_all_tasks.png"
fig.savefig(fig2, dpi=140)
plt.close()
print("wrote", fig2)

out = dict(summary=summary, primary=primary, secondary=secondary, leak_robustness=leak_rows,
           fig=str(fig_path), fig_all=str(fig2))
json.dump(out, open(OUT / "analyze.json", "w"), indent=2)

# RESULTS md
lines = [
    "# Myelin REAL vs SHUF replication results",
    "",
    "Prereg: `PREREGISTRATION_myelin_real_rep.md`.",
    "",
    "## Primary (REAL vs SHUF NARMA, leak=0.2, one-sided)",
    "",
]
if primary:
    lines += [
        f"- REAL mean NARMA = **{primary['mean_real']:.4f}** (n={primary['n_real']})",
        f"- SHUF mean NARMA = **{primary['mean_shuf']:.4f}** (n={primary['n_shuf']}, 3 draws/seed averaged)",
        f"- Δ (REAL−SHUF) = **{primary['delta']:+.4f}**",
        f"- p_one_sided (REAL < SHUF) = **{primary['p_one_sided']:.4g}**",
        f"- **claim_ok = {primary['claim_ok']}**",
        "",
    ]
else:
    lines += ["- insufficient data", ""]

lines += ["## Means ± sd at leak=0.2", "",
          "| task | NODELAY | REAL | SHUF | LAYER_SHUF | EI_SHUF |",
          "|---|---:|---:|---:|---:|---:|"]
for task in TASKS:
    cells = []
    for arm in ARMS:
        d = summary.get("0.2", {}).get(task, {}).get(arm)
        cells.append(f"{d['mean']:.4f}±{d['sd']:.4f}" if d else "—")
    lines.append(f"| {task} | " + " | ".join(cells) + " |")
lines += ["", "## Secondary NARMA vs other controls (two-sided raw p)", ""]
for r in secondary:
    lines.append(f"- vs {r['ctrl']}: Δ={r['delta']:+.4f}, p_raw={r['p_raw']:.4g}")
lines += ["", "## Leak robustness (REAL vs SHUF NARMA, one-sided)", ""]
for r in leak_rows:
    lines.append(f"- leak={r['leak']}: REAL={r['mean_real']:.4f} SHUF={r['mean_shuf']:.4f} "
                 f"Δ={r['delta']:+.4f} p={r['p_one_sided']:.4g}")
lines += ["", f"Figures: `{fig_path.name}`, `{fig2.name}` under `/workspace/flybrain/figs/`.", ""]
# honest verdict
if primary:
    if primary["claim_ok"]:
        verdict = ("**Verdict: REPLICATED.** Real myelin placement beats global shuffle on NARMA "
                   "with 24 new seeds under the preregistered one-sided test.")
    elif primary["delta"] < 0:
        verdict = ("**Verdict: DIRECTION HOLDS, NOT SIGNIFICANT.** REAL < SHUF on mean NARMA but "
                   "one-sided p ≥ 0.05 under the stricter replication protocol.")
    else:
        verdict = ("**Verdict: DID NOT REPLICATE.** Under 24 new seeds / multi-shuffle / tuned α, "
                   "REAL did not beat SHUF on NARMA.")
else:
    verdict = "**Verdict: INCOMPLETE** — analyze before runs finished."
lines += ["## Honest verdict", "", verdict, ""]
(OUT.parent / "RESULTS_myelin_real_rep.md").write_text("\n".join(lines) + "\n")
print("\n" + verdict)
print("wrote RESULTS_myelin_real_rep.md")

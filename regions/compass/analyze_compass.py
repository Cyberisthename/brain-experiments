"""Analyze compass runs → RESULTS_compass.md + fig."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
OUT = H / "out"
FIGS = Path("/workspace/flybrain/figs")
FIGS.mkdir(exist_ok=True)

ARMS = [
    "FLY_CX_NOD", "FLY_CX_DEL",
    "ER_NOD", "ER_DEL",
    "SHUF_NOD", "SHUF_DEL",
    "DENS_NOD", "DENS_DEL",
]
TASKS = ["HEADING", "DRIFT", "GEN"]
PRIMARY_CTRL = ["ER_DEL", "SHUF_DEL", "DENS_DEL"]
SECONDARY_CTRL = ["ER_NOD", "SHUF_NOD", "DENS_NOD"]


def load_rows():
    rows = []
    if not (OUT / "runs.jsonl").exists():
        return rows
    for line in (OUT / "runs.jsonl").open():
        r = json.loads(line)
        if r.get("error"):
            continue
        v = r.get("value")
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        rows.append(r)
    return rows


def paired_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 1:
        return 1.0
    if np.allclose(d, 0):
        return 1.0
    try:
        # zero_method='wilcox' ; scipy>=1.9 has nan_policy
        stat = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(stat.pvalue)
    except Exception:
        return 1.0


def main():
    rows = load_rows()
    lines = [
        "# Compass (fly CX path integration) results\n\n",
        "Prereg: `PREREGISTRATION_compass.md`.\n\n",
        "Metric: circular RMSE in radians (↓ better).\n\n",
    ]
    if not rows:
        lines.append("No runs yet.\n")
        (H / "RESULTS_compass.md").write_text("".join(lines))
        print("no rows")
        return

    by = defaultdict(dict)  # (task,arm) -> {seed: value}
    for r in rows:
        by[(r["task"], r["arm"])][int(r["seed"])] = float(r["value"])

    # means table
    lines.append("## Means ± sd (n seeds)\n\n")
    lines.append("| Arm | HEADING | DRIFT | GEN |\n|---|---:|---:|---:|\n")
    summary = {}
    for arm in ARMS:
        cells = []
        summary[arm] = {}
        for task in TASKS:
            vals = list(by.get((task, arm), {}).values())
            if not vals:
                cells.append("—")
                continue
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[arm][task] = {"mean": m, "sd": s, "n": len(vals), "vals": vals,
                                  "by_seed": by[(task, arm)]}
            cells.append(f"{m:.3f} ± {s:.3f} (n={len(vals)})")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # primary contrasts
    lines.append("## Primary family: FLY_CX_DEL vs delay-matched controls (Bonferroni ×9)\n\n")
    lines.append("| Metric | Control | Δ (FLY−ctrl) | raw p | bonf p | FLY better? |\n|---|---|---:|---:|---:|---|\n")
    primary_hits = []
    m_bonf = 9
    for task in TASKS:
        fly = summary.get("FLY_CX_DEL", {}).get(task)
        if not fly:
            continue
        for ctrl in PRIMARY_CTRL:
            c = summary.get(ctrl, {}).get(task)
            if not c:
                continue
            # align seeds
            seeds = sorted(set(fly["by_seed"]) & set(c["by_seed"]))
            fv = [fly["by_seed"][s] for s in seeds]
            cv = [c["by_seed"][s] for s in seeds]
            delta = float(np.mean(fv) - np.mean(cv))
            p = paired_p(fv, cv)
            pb = min(1.0, p * m_bonf)
            better = delta < 0
            if better and pb < 0.05:
                primary_hits.append((task, ctrl, delta, pb))
            lines.append(
                f"| {task} | `{ctrl}` | {delta:+.4f} | {p:.4g} | {pb:.4g} | {'yes' if better else 'no'} |\n"
            )
    lines.append("\n")

    # secondary
    lines.append("## Secondary family: FLY_CX_NOD vs no-delay controls (Bonferroni ×9)\n\n")
    lines.append("| Metric | Control | Δ (FLY−ctrl) | raw p | bonf p | FLY better? |\n|---|---|---:|---:|---:|---|\n")
    for task in TASKS:
        fly = summary.get("FLY_CX_NOD", {}).get(task)
        if not fly:
            continue
        for ctrl in SECONDARY_CTRL:
            c = summary.get(ctrl, {}).get(task)
            if not c:
                continue
            seeds = sorted(set(fly["by_seed"]) & set(c["by_seed"]))
            fv = [fly["by_seed"][s] for s in seeds]
            cv = [c["by_seed"][s] for s in seeds]
            delta = float(np.mean(fv) - np.mean(cv))
            p = paired_p(fv, cv)
            pb = min(1.0, p * m_bonf)
            better = delta < 0
            lines.append(
                f"| {task} | `{ctrl}` | {delta:+.4f} | {p:.4g} | {pb:.4g} | {'yes' if better else 'no'} |\n"
            )
    lines.append("\n")

    # exploratory DEL vs NOD within FLY
    lines.append("## Exploratory: FLY_CX_DEL vs FLY_CX_NOD (raw p)\n\n")
    for task in TASKS:
        a = summary.get("FLY_CX_DEL", {}).get(task)
        b = summary.get("FLY_CX_NOD", {}).get(task)
        if not a or not b:
            continue
        seeds = sorted(set(a["by_seed"]) & set(b["by_seed"]))
        fv = [a["by_seed"][s] for s in seeds]
        nv = [b["by_seed"][s] for s in seeds]
        delta = float(np.mean(fv) - np.mean(nv))
        p = paired_p(fv, nv)
        lines.append(f"- {task}: Δ={delta:+.4f} (DEL−NOD), raw p={p:.4g}\n")
    lines.append("\n")

    # verdict
    lines.append("## Verdict\n\n")
    # rule from prereg
    win_metrics = set()
    for task in TASKS:
        fly = summary.get("FLY_CX_DEL", {}).get(task)
        if not fly:
            continue
        beats_all = True
        any_sig = False
        for ctrl in PRIMARY_CTRL:
            c = summary.get(ctrl, {}).get(task)
            if not c:
                beats_all = False
                break
            seeds = sorted(set(fly["by_seed"]) & set(c["by_seed"]))
            fv = [fly["by_seed"][s] for s in seeds]
            cv = [c["by_seed"][s] for s in seeds]
            delta = float(np.mean(fv) - np.mean(cv))
            p = paired_p(fv, cv)
            pb = min(1.0, p * m_bonf)
            if not (delta < 0):
                beats_all = False
            if delta < 0 and pb < 0.05:
                any_sig = True
        if beats_all and any_sig:
            win_metrics.add(task)

    if win_metrics:
        lines.append(
            f"**Real CX wiring wins its native task** on: {', '.join(sorted(win_metrics))} "
            "(FLY_CX_DEL lower error than all three delay-matched controls, ≥1 Bonferroni-significant).\n"
        )
        verdict = "WIN"
    elif primary_hits:
        lines.append(
            f"**Mixed.** Some Bonferroni-significant pairwise wins {primary_hits}, "
            "but FLY_CX_DEL does not beat *all three* delay-matched controls on any single metric.\n"
        )
        verdict = "MIXED"
    else:
        lines.append(
            "**Null / no win.** Under this compass proxy, real CX wiring does **not** beat "
            "delay-matched ER/SHUF/DENS controls after Bonferroni. Same lesson as generic tasks: "
            "once delays and α are matched, random wiring keeps up.\n"
        )
        verdict = "NULL"

    lines.append(f"\nRows: {len(rows)}. Raw: `out/runs.jsonl`. Figure: `figs/compass_heading.png`.\n")
    (H / "RESULTS_compass.md").write_text("".join(lines))

    # figure: grouped bar for each task
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    colors = {
        "FLY_CX": "#c0392b", "ER": "#7f8c8d", "SHUF": "#2980b9", "DENS": "#27ae60",
    }
    xlabels = ["NOD", "DEL"]
    for ax, task in zip(axes, TASKS):
        width = 0.18
        xs = np.arange(2)
        for i, w in enumerate(["FLY_CX", "ER", "SHUF", "DENS"]):
            means, sds = [], []
            for d in ["NOD", "DEL"]:
                arm = f"{w}_{d}"
                s = summary.get(arm, {}).get(task)
                if s:
                    means.append(s["mean"]); sds.append(s["sd"])
                else:
                    means.append(np.nan); sds.append(0)
            ax.bar(xs + (i - 1.5) * width, means, width, yerr=sds, label=w,
                   color=colors[w], alpha=0.9, capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(xlabels)
        ax.set_title(task)
        ax.set_xlabel("delay")
        ax.axhline(math.pi / 2, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("circular RMSE (rad)")
    axes[2].legend(fontsize=8, loc="upper right")
    fig.suptitle("Fly CX compass: heading error (↓ better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "compass_heading.png", dpi=140)
    fig.savefig(H / "out" / "compass_heading.png", dpi=140)
    plt.close(fig)

    # machine-readable summary
    slim = {a: {t: {k: summary[a][t][k] for k in ("mean", "sd", "n")}
                for t in summary[a]} for a in summary}
    (OUT / "summary.json").write_text(json.dumps(
        {"verdict": verdict, "primary_hits": [(t, c, d, p) for t, c, d, p in primary_hits],
         "win_metrics": sorted(win_metrics), "summary": slim}, indent=2, default=float))
    print("verdict", verdict, "hits", primary_hits)
    print("wrote RESULTS + fig")


if __name__ == "__main__":
    main()

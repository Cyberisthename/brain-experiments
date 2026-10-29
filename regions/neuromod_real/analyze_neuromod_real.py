"""Analyze neuromod_real → RESULTS_neuromod_real.md + figs/neuromod_real.png."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
OUT, FIGS = H / "out", H / "figs"
FIGS.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
SEEDS = list(range(1000, 1024))
ARMS = ["NM_REAL", "NM_LABEL_SHUF", "NM_TARGET_SHUF", "NO_NM", "NM_ER"]
PRIMARY = "CTX_SWITCH"
TASKS = [PRIMARY, "XOR_DELAY", "NARMA", "MC", "PC_HARD"]
HIGHER = {"CTX_SWITCH", "XOR_DELAY", "MC", "PC_HARD"}


def load_rows():
    rows = []
    for line in open(JF):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or not np.isfinite(r.get("value", float("nan"))):
            continue
        rows.append(r)
    return rows


def pivot(rows, task):
    out = {a: {} for a in ARMS}
    for r in rows:
        if r.get("task") != task:
            continue
        if r["arm"] not in out:
            continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired(a, b, seeds, alternative="greater"):
    d, xa, xb = [], [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            xa.append(a[s]); xb.append(b[s]); d.append(a[s] - b[s])
    d = np.asarray(d); xa = np.asarray(xa); xb = np.asarray(xb); n = len(d)
    if n < 2:
        return dict(n=n, mean_a=float("nan"), mean_b=float("nan"), mean_d=float("nan"),
                    sd_d=float("nan"), d_z=float("nan"), p_raw=1.0,
                    ci_lo=float("nan"), ci_hi=float("nan"),
                    sem_a=float("nan"), sem_b=float("nan"))
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d, alternative=alternative).pvalue)
        except Exception:
            p = 1.0
    sd = float(np.std(d, ddof=1))
    dz = float(np.mean(d) / sd) if sd > 1e-15 else float("nan")
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return dict(n=n, mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
                sem_a=float(np.std(xa, ddof=1) / math.sqrt(n)),
                sem_b=float(np.std(xb, ddof=1) / math.sqrt(n)),
                mean_d=float(np.mean(d)), sd_d=sd, d_z=dz, p_raw=p,
                ci_lo=float(np.mean(d) - tcrit * se),
                ci_hi=float(np.mean(d) + tcrit * se))


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    nt = json.loads((H / "data" / "NT_SOURCE.json").read_text())
    lines = []
    lines.append("# RESULTS — Real brain chemicals (neuromod_real)")
    lines.append("")
    lines.append("Prereg: `PREREG_neuromod_real.md` (locked 2026-10-29 08:08 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("")
    lines.append("## Neurotransmitter source")
    lines.append("")
    lines.append(f"- FlyWire annotations `ann.tsv` (`top_nt`), repo {nt.get('source_repo')} commit `{nt.get('source_commit_cited_in_prereg')}`.")
    lines.append(f"- Method: {nt.get('top_nt_method')}")
    lines.append(f"- CX counts: `{nt.get('counts')}` → **n_mod = {nt.get('n_mod')}** (DA+5-HT+OA).")
    lines.append("")
    lines.append("## Locked α / γ (tune seeds 90–92, maximize CTX_SWITCH)")
    lines.append("")
    lines.append("| Arm | α | γ | tune CTX mean |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        a = alphas.get("alphas", {}).get(arm, {})
        g = alphas.get("gammas", {}).get(arm, {})
        lines.append(f"| `{arm}` | {a.get('alpha')} | {g.get('gamma')} | {a.get('tune_mean_ctx', float('nan')):.4f} |")
    lines.append("")

    # means table
    lines.append("## Means ± SEM (eval)")
    lines.append("")
    header = "| Arm | " + " | ".join(TASKS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(TASKS)) + "|"
    lines.append(header)
    lines.append(sep)
    pivots = {t: pivot(rows, t) for t in TASKS}
    for arm in ARMS:
        cells = []
        for t in TASKS:
            vals = [pivots[t][arm][s] for s in SEEDS if s in pivots[t][arm]]
            if not vals:
                cells.append("—")
            else:
                m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
                cells.append(f"{m:.3f}±{se:.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("↑ better: CTX_SWITCH, XOR_DELAY, MC, PC_HARD. ↓ better: NARMA.")
    lines.append("")

    # Primary
    lines.append("## Primary (CTX_SWITCH, one-sided Wilcoxon, Bonferroni ×3)")
    lines.append("")
    pv = pivots[PRIMARY]
    tests = [
        ("H1", "NM_REAL", "NO_NM", "greater"),
        ("H2", "NM_REAL", "NM_LABEL_SHUF", "greater"),
        ("H3", "NM_REAL", "NM_ER", "greater"),
    ]
    # For NARMA-like lower-better we'd flip, but CTX is higher-better
    lines.append("| Hyp | Contrast | mean_REAL | mean_ctrl | Δ | d_z | p_raw | p_bonf |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    verdicts = []
    for name, a, b, alt in tests:
        # if NARMA style lower better for ctrl comparison — not here
        st = paired(pv[a], pv[b], SEEDS, alternative=alt)
        p_bonf = min(1.0, 3.0 * st["p_raw"])
        ok = (st["mean_a"] > st["mean_b"]) and (p_bonf < 0.05)
        verdicts.append(ok)
        lines.append(
            f"| {name} | `{a}` > `{b}` | {st['mean_a']:.4f} | {st['mean_b']:.4f} | "
            f"{st['mean_d']:.4f} | {st['d_z']:.3f} | {st['p_raw']:.4g} | {p_bonf:.4g} |"
        )
    supported = all(verdicts) and len(verdicts) == 3
    lines.append("")
    lines.append(f"**Primary verdict: {'SUPPORTED' if supported else 'NOT SUPPORTED'}** "
                 f"({sum(verdicts)}/3 hypotheses pass).")
    lines.append("")

    # Secondary
    lines.append("## Secondary (exploratory)")
    lines.append("")
    st_t = paired(pv["NM_REAL"], pv["NM_TARGET_SHUF"], SEEDS, alternative="greater")
    lines.append(
        f"- CTX_SWITCH NM_REAL vs NM_TARGET_SHUF: Δ={st_t['mean_d']:.4f}, "
        f"p_one={st_t['p_raw']:.4g}, d_z={st_t['d_z']:.3f}"
    )
    lines.append("")
    lines.append("| Task | Contrast | Δ | p_raw (one-sided greater for ↑ tasks; for NARMA: NO_NM/ctrl − REAL style via alt) |")
    lines.append("|---|---|---:|---:|")
    SECONDARY = ["XOR_DELAY", "NARMA", "MC", "PC_HARD"]
    for task in SECONDARY:
        pv_t = pivots[task]
        for a, b in [("NM_REAL", "NO_NM"), ("NM_REAL", "NM_LABEL_SHUF"), ("NM_REAL", "NM_ER")]:
            if task == "NARMA":
                # lower better: test ctrl > real i.e. real smaller → alternative on (ctrl - real) greater? 
                # Report NM_REAL - ctrl with alt less for NARMA
                st = paired(pv_t[a], pv_t[b], SEEDS, alternative="less")
            else:
                st = paired(pv_t[a], pv_t[b], SEEDS, alternative="greater")
            lines.append(f"| {task} | `{a}` vs `{b}` | {st['mean_d']:.4f} | {st['p_raw']:.4g} |")
    lines.append("")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # CTX bar
    ax = axes[0]
    means, sems, labels = [], [], []
    for arm in ARMS:
        vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        labels.append(arm.replace("NM_", "").replace("_", "\n"))
    x = np.arange(len(ARMS))
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#7f7f7f", "#1f77b4"]
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CTX_SWITCH (1−NRMSE) ↑")
    ax.set_title("Primary: context-switch")
    ax.axhline(means[ARMS.index("NM_REAL")], color="#2ca02c", ls="--", lw=0.8, alpha=0.5)

    # Secondary XOR
    ax = axes[1]
    pv_x = pivots["XOR_DELAY"]
    means, sems = [], []
    for arm in ARMS:
        vals = [pv_x[arm][s] for s in SEEDS if s in pv_x[arm]]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("XOR_DELAY d=5 (1−NRMSE) ↑")
    ax.set_title("Secondary: XOR_DELAY")
    fig.suptitle("neuromod_real — FlyWire DA/5-HT/OA as gain", fontsize=11)
    fig.tight_layout()
    fig_path = FIGS / "neuromod_real.png"
    fig.savefig(fig_path, dpi=140)
    plt.close()
    lines.append(f"## Figure")
    lines.append("")
    lines.append(f"![neuromod_real](figs/neuromod_real.png)")
    lines.append("")

    # Plain words
    lines.append("## Plain words")
    lines.append("")
    if supported:
        lines.append(
            "Real modulatory labels and wiring helped the fly circuit switch rules "
            "better than no neuromod, shuffled labels, and a random graph with the same trick."
        )
    else:
        lines.append(
            "Turning real dopamine/serotonin/octopamine neurons into a slow gain dial "
            "did **not** clearly beat the controls on the context-switch test after correction. "
            "Either the chemical map does not help this abstract task, or our simple gain model "
            "is too crude. See the table — report the negatives."
        )
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    lines.append("- Reservoir gain ≠ biological neuromodulation or fly behavior.")
    lines.append("- `top_nt` is ML-predicted chemistry, not wet-lab ground truth for every cell.")
    lines.append("- CPU only (N=2875); γ/α locked on 90–92 before eval 1000–1023.")
    lines.append("")

    text = "\n".join(lines) + "\n"
    (H / "RESULTS_neuromod_real.md").write_text(text)
    summary = dict(supported=supported, n_pass=int(sum(verdicts)),
                   primary=[dict(name=n, pass_=bool(v)) for (n, *_), v in zip(tests, verdicts)],
                   fig=str(fig_path))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", H / "RESULTS_neuromod_real.md", fig_path)


if __name__ == "__main__":
    main()

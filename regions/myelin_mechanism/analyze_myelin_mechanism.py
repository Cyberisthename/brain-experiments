"""Analyze myelin_mechanism → RESULTS + figs."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
OUT, FIGS = H / "out", H / "figs"; FIGS.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
SEEDS = list(range(800, 824))
RULES = ["RULE_LAYER", "RULE_DEGREE", "RULE_EI_LAYER", "RULE_FF_DEPTH"]
ALL = ["REAL", "SHUF", "NODELAY"] + RULES
BONF = 4


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
    out = {a: {} for a in ALL}
    for r in rows:
        if r.get("task") != task or r["arm"] not in out:
            continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired_less(a, b, seeds):
    """one-sided a < b"""
    d = []
    xa, xb = [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            xa.append(a[s]); xb.append(b[s]); d.append(a[s] - b[s])
    d = np.asarray(d); n = len(d)
    if n < 2:
        return dict(n=n, mean_a=float("nan"), mean_b=float("nan"), mean_d=float("nan"), p_raw=1.0)
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d, alternative="less").pvalue)
        except Exception:
            p = 1.0
    return dict(n=n, mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
                mean_d=float(np.mean(d)), p_raw=p)


def mean_sem(pv, arm):
    vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))), len(vals)


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    desc = json.loads((OUT / "desc_corr.json").read_text()) if (OUT / "desc_corr.json").exists() else {}
    pn = pivot(rows, "NARMA"); pmc = pivot(rows, "MC"); ppc = pivot(rows, "PC_HARD")
    m_real, _, _ = mean_sem(pn, "REAL")
    m_shuf, _, _ = mean_sem(pn, "SHUF")
    gap = (m_shuf - m_real) if (m_real is not None and m_shuf is not None) else float("nan")

    lines = []
    lines.append("# Myelin mechanism results")
    lines.append("")
    lines.append("Prereg: `PREREG_myelin_mechanism.md` (locked 2026-10-29 08:01 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Seeds {SEEDS[0]}–{SEEDS[-1]}. Leak=0.2. Graph=HUMAN_LAYERED.")
    lines.append("")
    lines.append("## Plain words")
    lines.append("")
    lines.append(
        "We kept the real human+fly layered graph fixed and only changed **which neuron gets which delay**, "
        "using rules inspired by how real H01 myelin lines up with layer, degree, E/I×layer, and feedforward depth. "
        "Goal: see which rule gets at least halfway from a global myelin shuffle toward real placement on NARMA, "
        "without beating real placement."
    )
    lines.append("")
    lines.append("## Descriptive myelin correlations (pre-eval)")
    lines.append("")
    lines.append(f"- Spearman(mf, indeg)={desc.get('spearman_mf_indeg')}")
    lines.append(f"- Spearman(mf, outdeg)={desc.get('spearman_mf_outdeg')}")
    lines.append(f"- Spearman(mf, L4-BFS depth)={desc.get('spearman_mf_depth')}")
    lines.append(f"- Layer means: {desc.get('layer_means')}")
    lines.append(f"- E/I means: {desc.get('sign_means')}")
    lines.append("")
    lines.append("## NARMA means (↓ better)")
    lines.append("")
    lines.append("| Arm | NARMA | MC | PC_HARD | recovery |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ALL:
        mn, sen, _ = mean_sem(pn, arm)
        mm, sem, _ = mean_sem(pmc, arm)
        mp, sep, _ = mean_sem(ppc, arm)
        if mn is None:
            rec = "—"
        elif arm in ("REAL", "SHUF", "NODELAY") or not np.isfinite(gap) or gap <= 0:
            rec = "—"
        else:
            rec = f"{(m_shuf - mn) / gap:.2f}"
        def fmt(m, se):
            return "—" if m is None else f"{m:.3f}±{se:.3f}"
        lines.append(f"| `{arm}` | {fmt(mn, sen)} | {fmt(mm, sem)} | {fmt(mp, sep)} | {rec} |")
    lines.append("")
    lines.append(f"REAL−SHUF gap (SHUF−REAL on NARMA) = {gap:.4f}" if np.isfinite(gap) else "gap n/a")
    lines.append("")
    lines.append("## Primary rules (Bonferroni ×4)")
    lines.append("")
    lines.append("| Rule | mean | vs SHUF Δ | p_raw | p_bonf | recovery | ≥REAL | pass |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|:---:|")
    passes = []
    hyp = {}
    for arm in RULES:
        st = paired_less(pn[arm], pn["SHUF"], SEEDS)
        p_bonf = min(1.0, BONF * st["p_raw"])
        rec = (m_shuf - st["mean_a"]) / gap if (np.isfinite(gap) and gap > 0) else float("nan")
        ge_real = bool(st["mean_a"] >= m_real - 1e-6) if m_real is not None else False
        ok = bool(p_bonf < 0.05 and rec >= 0.50 and ge_real)
        passes.append(ok)
        hyp[arm] = {**st, "p_bonf": p_bonf, "recovery": rec, "ge_real": ge_real, "pass": ok}
        lines.append(
            f"| `{arm}` | {st['mean_a']:.4f} | {st['mean_d']:.4f} | {st['p_raw']:.4g} | "
            f"{p_bonf:.4g} | {rec:.2f} | {'yes' if ge_real else 'no'} | {'yes' if ok else 'no'} |"
        )
    verdict = "SUPPORTED" if any(passes) else "NOT SUPPORTED"
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("Passing rules: " + (", ".join(a for a, ok in zip(RULES, passes) if ok) or "none") + ".")
    lines.append("")
    lines.append("Reservoir ≠ biology claim about myelin.")
    (H / "RESULTS_myelin_mechanism.md").write_text("\n".join(lines) + "\n")
    (OUT / "hyp_stats.json").write_text(json.dumps(hyp, indent=2, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    means, sems, labs, cols = [], [], [], []
    cmap = {"REAL": "#2a9d8f", "SHUF": "#e76f51", "NODELAY": "#6c757d",
            "RULE_LAYER": "#457b9d", "RULE_DEGREE": "#1d3557", "RULE_EI_LAYER": "#a8dadc", "RULE_FF_DEPTH": "#f4a261"}
    for arm in ALL:
        m, se, _ = mean_sem(pn, arm)
        if m is None: continue
        means.append(m); sems.append(se); labs.append(arm); cols.append(cmap.get(arm, "#333"))
    x = np.arange(len(labs))
    ax.bar(x, means, yerr=sems, color=cols, alpha=0.9, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("NARMA NRMSE (↓)"); ax.set_title(f"Myelin mechanism — {verdict}")
    if m_real is not None:
        ax.axhline(m_real, color="#2a9d8f", ls="--", lw=0.8, label="REAL")
    if m_shuf is not None:
        ax.axhline(m_shuf, color="#e76f51", ls="--", lw=0.8, label="SHUF")
        if np.isfinite(gap):
            ax.axhline(m_shuf - 0.5 * gap, color="#457b9d", ls=":", lw=0.8, label="50% gap")
    ax.legend(fontsize=7)
    ax = axes[1]
    recs = [hyp[a]["recovery"] for a in RULES]
    cols2 = ["#2a9d8f" if hyp[a]["pass"] else "#e76f51" for a in RULES]
    ax.bar(RULES, recs, color=cols2, alpha=0.9)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_ylabel("recovery fraction"); ax.set_title("Rule recovery of REAL−SHUF gap")
    for i, a in enumerate(RULES):
        ax.text(i, recs[i] + 0.02, f"p={hyp[a]['p_bonf']:.3g}", ha="center", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "myelin_mechanism.png", dpi=140)
    plt.close()
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()

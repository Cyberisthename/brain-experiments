"""Analyze ei_cross_species → RESULTS + figs."""
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
SEEDS = list(range(900, 924))
SPECIES = ["MOUSE", "HUMAN", "FINCH"]
ARMS = ["REAL", "ER", "SHUF", "EI_SHUF", "SBM_EI"]


def load_rows():
    rows = []
    if not JF.exists():
        return rows
    for line in open(JF):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or not np.isfinite(r.get("value", float("nan"))):
            continue
        rows.append(r)
    return rows


def pivot(rows, species):
    out = {a: {} for a in ARMS + ["DESIGN_EI_ONLY", "ER4000"]}
    for r in rows:
        if r.get("species") != species:
            continue
        arm = r["arm"]
        if arm in out:
            out[arm][int(r["seed"])] = float(r["value"])
    return out


def paired_greater(a, b, seeds):
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
            p = float(stats.wilcoxon(d, alternative="greater").pvalue)
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
    n_sp = len(SPECIES)
    lines = []
    lines.append("# E/I cross-species results")
    lines.append("")
    lines.append("Prereg: `PREREG_ei_cross_species.md` (locked 2026-10-29 08:11 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Seeds {SEEDS[0]}–{SEEDS[-1]}. Task XOR_DELAY d=5 (1−NRMSE ↑).")
    lines.append("**Skipped ELEGAN:** E/I unavailable (`ei_status=unknown`); see `SOURCES.md`.")
    lines.append("")
    lines.append("## Plain words")
    lines.append("")
    lines.append(
        "We asked whether the fly CX finding that E/I class structure helps delayed XOR "
        "shows up in other animals' wiring modules that have excitatory/inhibitory labels. "
        "For each species we compare the real graph and an E/I-preserving rewire against a "
        "same-size random (ER) graph. C. elegans is skipped because our ingested tables have no E/I."
    )
    lines.append("")
    lines.append("## Per-species means")
    lines.append("")
    hyp = {}
    passes = []
    for sp in SPECIES:
        pv = pivot(rows, sp)
        lines.append(f"### {sp}")
        lines.append("")
        lines.append("| Arm | XOR d=5 |")
        lines.append("|---|---:|")
        for arm in ARMS:
            m, se, n = mean_sem(pv, arm)
            lines.append(f"| `{arm}` | {'—' if m is None else f'{m:.4f}±{se:.4f} (n={n})'} |")
        st_r = paired_greater(pv["REAL"], pv["ER"], SEEDS)
        st_e = paired_greater(pv["EI_SHUF"], pv["ER"], SEEDS)
        p_min = min(st_r["p_raw"], st_e["p_raw"])
        p_bonf = min(1.0, n_sp * p_min)
        ok = p_bonf < 0.05
        passes.append(ok)
        hyp[sp] = {"REAL_vs_ER": st_r, "EI_SHUF_vs_ER": st_e, "p_min": p_min, "p_bonf": p_bonf, "pass": ok}
        lines.append("")
        lines.append(
            f"REAL vs ER: Δ={st_r['mean_d']:.4f}, p_raw={st_r['p_raw']:.4g}; "
            f"EI_SHUF vs ER: Δ={st_e['mean_d']:.4f}, p_raw={st_e['p_raw']:.4g}; "
            f"**p_bonf={p_bonf:.4g}** → {'PASS' if ok else 'fail'}."
        )
        lines.append("")

    # confirmatory
    pvD = pivot(rows, "DESIGN")
    stD = paired_greater(pvD["DESIGN_EI_ONLY"], pvD["ER4000"], SEEDS)
    hyp["DESIGN"] = stD
    lines.append("## Confirmatory H_design (not in species Bonferroni)")
    lines.append("")
    md, sed, nd = mean_sem(pvD, "DESIGN_EI_ONLY")
    me, see, ne = mean_sem(pvD, "ER4000")
    lines.append(f"| `DESIGN_EI_ONLY` | {'—' if md is None else f'{md:.4f}±{sed:.4f}'} |")
    lines.append(f"| `ER4000` | {'—' if me is None else f'{me:.4f}±{see:.4f}'} |")
    lines.append(f"Δ={stD['mean_d']:.4f}, p_raw={stD['p_raw']:.4g} → "
                 f"{'PASS' if stD['p_raw'] < 0.05 else 'fail'} (α=0.05 single test).")
    lines.append("")

    verdict = "SUPPORTED" if any(passes) else "NOT SUPPORTED"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("Passing species: " + (", ".join(s for s, ok in zip(SPECIES, passes) if ok) or "none") + ".")
    lines.append("")
    lines.append("Reservoir ≠ biology claim.")
    (H / "RESULTS_ei_cross_species.md").write_text("\n".join(lines) + "\n")
    (OUT / "hyp_stats.json").write_text(json.dumps(hyp, indent=2, default=float))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x0 = 0
    width = 0.15
    colors = {"REAL": "#2a9d8f", "ER": "#e76f51", "SHUF": "#f4a261", "EI_SHUF": "#457b9d", "SBM_EI": "#1d3557"}
    for i, arm in enumerate(ARMS):
        xs, ys, yes = [], [], []
        for j, sp in enumerate(SPECIES):
            pv = pivot(rows, sp)
            m, se, _ = mean_sem(pv, arm)
            if m is None: continue
            xs.append(j + (i - 2) * width); ys.append(m); yes.append(se)
        ax.bar(xs, ys, width=width, yerr=yes, label=arm, color=colors[arm], alpha=0.9, capsize=2)
    ax.set_xticks(range(len(SPECIES))); ax.set_xticklabels(SPECIES)
    ax.set_ylabel("XOR_DELAY d=5 (1−NRMSE) ↑")
    ax.set_title(f"E/I cross-species — {verdict}")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    labs = ["DESIGN_EI_ONLY", "ER4000"]
    ms, ses = [], []
    for arm in labs:
        m, se, _ = mean_sem(pvD, arm)
        ms.append(0 if m is None else m); ses.append(0 if se is None else se)
    ax.bar(labs, ms, yerr=ses, color=["#2a9d8f", "#e76f51"], alpha=0.9, capsize=3)
    ax.set_ylabel("XOR d=5 ↑"); ax.set_title(f"Confirmatory DESIGN_EI  p={stD['p_raw']:.3g}")
    fig.tight_layout()
    fig.savefig(FIGS / "ei_cross_species.png", dpi=140)
    plt.close()
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()

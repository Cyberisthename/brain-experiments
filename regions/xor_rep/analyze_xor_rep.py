"""Analyze xor_rep. Writes RESULTS_xor_rep.md + figs/xor_rep.png."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
OUT = H / "out"
FIGS = H / "figs"; FIGS.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
SEEDS = list(range(200, 224))
PRIMARY_ARMS = ["FRANKEN", "ER", "SHUF"]
ALL_ARMS = ["FRANKEN", "ER", "SHUF", "DENS", "FLY_CX", "HUMAN", "SPEC"]


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


def pivot(rows, arms, xor_d=5, delay_mode="REALMYELIN", lesion="NONE"):
    """arm -> {seed: value}"""
    out = {a: {} for a in arms}
    for r in rows:
        if r.get("xor_d") != xor_d: continue
        if r.get("delay_mode") != delay_mode: continue
        if r.get("lesion", "NONE") != lesion: continue
        if r["arm"] not in out: continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired(a, b, seeds, alternative="greater"):
    xa, xb, d = [], [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            xa.append(a[s]); xb.append(b[s]); d.append(a[s] - b[s])
    xa, xb, d = np.asarray(xa), np.asarray(xb), np.asarray(d)
    n = len(d)
    if n < 2:
        return dict(n=n, mean_a=float("nan"), mean_b=float("nan"), mean_d=float("nan"),
                    sd_d=float("nan"), d_z=float("nan"), p_raw=1.0, ci_lo=float("nan"), ci_hi=float("nan"))
    # Wilcoxon signed-rank; if all zero -> p=1
    if np.allclose(d, 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d, alternative=alternative).pvalue)
        except Exception:
            p = 1.0
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    dz = float(np.mean(d) / sd) if sd > 1e-15 else float("nan")
    # paired CI on mean diff (t)
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    return dict(
        n=n, mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
        sem_a=float(np.std(xa, ddof=1) / math.sqrt(n)),
        sem_b=float(np.std(xb, ddof=1) / math.sqrt(n)),
        mean_d=float(np.mean(d)), sd_d=sd, d_z=dz, p_raw=p,
        ci_lo=float(np.mean(d) - tcrit * se), ci_hi=float(np.mean(d) + tcrit * se),
    )


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    pv = pivot(rows, ALL_ARMS, 5, "REALMYELIN", "NONE")

    lines = []
    lines.append("# XOR_DELAY confirmatory replication results")
    lines.append("")
    lines.append("Prereg: `PREREG_xor_rep.md` (locked 2026-10-28 15:47 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append(f"Parent exploratory (seeds 0–7, NOT confirmatory): FRANKEN 0.525 vs ER 0.213 vs SHUF −0.159.")
    lines.append("")
    lines.append("## Ridge α (tuned on 90–92 only, XOR d=5 REALMYELIN)")
    lines.append("")
    if alphas.get("alphas"):
        lines.append("| Arm | α |")
        lines.append("|---|---:|")
        for arm in ALL_ARMS:
            if arm in alphas["alphas"]:
                lines.append(f"| `{arm}` | {alphas['alphas'][arm]['alpha']} |")
    lines.append("")
    lines.append("## Primary: XOR_DELAY d=5, REALMYELIN — means")
    lines.append("")
    lines.append("| Arm | mean ± sem | n |")
    lines.append("|---|---:|---:|")
    means = {}
    for arm in ALL_ARMS:
        vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
        if not vals:
            lines.append(f"| `{arm}` | — | 0 |")
            continue
        m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        means[arm] = m
        lines.append(f"| `{arm}` | {m:.4f} ± {se:.4f} | {len(vals)} |")
    lines.append("")
    lines.append("Metric ↑ better (1 − NRMSE).")
    lines.append("")

    c_er = paired(pv["FRANKEN"], pv["ER"], SEEDS, "greater")
    c_sh = paired(pv["FRANKEN"], pv["SHUF"], SEEDS, "greater")
    p_er_b = min(1.0, c_er["p_raw"] * 2)
    p_sh_b = min(1.0, c_sh["p_raw"] * 2)
    supported = (
        c_er["n"] >= 20 and c_sh["n"] >= 20
        and p_er_b < 0.05 and p_sh_b < 0.05
        and c_er["mean_a"] > c_er["mean_b"] and c_sh["mean_a"] > c_sh["mean_b"]
    )
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    lines.append("## Primary hypothesis (Bonferroni ×2, one-sided Wilcoxon)")
    lines.append("")
    lines.append("| Contrast | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| FRANKEN > ER | {c_er['mean_d']:+.4f} | [{c_er['ci_lo']:+.4f}, {c_er['ci_hi']:+.4f}] | "
        f"{c_er['d_z']:.3f} | {c_er['p_raw']:.4g} | {p_er_b:.4g} |"
    )
    lines.append(
        f"| FRANKEN > SHUF | {c_sh['mean_d']:+.4f} | [{c_sh['ci_lo']:+.4f}, {c_sh['ci_hi']:+.4f}] | "
        f"{c_sh['d_z']:.3f} | {c_sh['p_raw']:.4g} | {p_sh_b:.4g} |"
    )
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    # Secondary delay depths
    lines.append("## Secondary: XOR delay depth (REALMYELIN, primary arms)")
    lines.append("")
    lines.append("| d | FRANKEN | ER | SHUF | FRANKEN−ER Δ | p_raw (>) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for d in [2, 3, 5, 8, 10]:
        pvd = pivot(rows, PRIMARY_ARMS, d, "REALMYELIN", "NONE")
        def msem(arm):
            vals = [pvd[arm][s] for s in SEEDS if s in pvd[arm]]
            if not vals: return "—"
            return f"{np.mean(vals):.3f}"
        c = paired(pvd["FRANKEN"], pvd["ER"], SEEDS, "greater")
        lines.append(
            f"| {d} | {msem('FRANKEN')} | {msem('ER')} | {msem('SHUF')} | "
            f"{c['mean_d']:+.3f} | {c['p_raw']:.4g} |"
        )
    lines.append("")

    lines.append("## Secondary: delay mode at d=5 (FRANKEN / ER / SHUF)")
    lines.append("")
    lines.append("| Mode | FRANKEN | ER | SHUF |")
    lines.append("|---|---:|---:|---:|")
    for mode in ["REALMYELIN", "FIXED", "NODELAY"]:
        pvm = pivot(rows, PRIMARY_ARMS, 5, mode, "NONE")
        def msem(arm):
            vals = [pvm[arm][s] for s in SEEDS if s in pvm[arm]]
            if not vals: return "—"
            return f"{np.mean(vals):.4f}±{np.std(vals, ddof=1)/math.sqrt(len(vals)):.4f}"
        lines.append(f"| {mode} | {msem('FRANKEN')} | {msem('ER')} | {msem('SHUF')} |")
    lines.append("")

    # Mechanism if present
    les_arms = ["NONE", "FLY", "HUMAN", "BRIDGES"]
    les_vals = {L: [] for L in les_arms}
    for r in rows:
        if r.get("arm") != "FRANKEN": continue
        if r.get("xor_d") != 5 or r.get("delay_mode") != "REALMYELIN": continue
        if int(r["seed"]) not in range(200, 208): continue
        les_vals[r.get("lesion", "NONE")].append(float(r["value"]))
    if any(len(les_vals[L]) for L in ("FLY", "HUMAN", "BRIDGES")):
        lines.append("## Mechanism probe (FRANKEN lesions, seeds 200–207)")
        lines.append("")
        lines.append("| Condition | mean ± sem | n | Δ vs intact |")
        lines.append("|---|---:|---:|---:|")
        base = np.mean(les_vals["NONE"]) if les_vals["NONE"] else float("nan")
        for L in les_arms:
            v = les_vals[L]
            if not v: continue
            m = float(np.mean(v)); se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
            delta = m - base if np.isfinite(base) else float("nan")
            lines.append(f"| LES_{L} | {m:.4f} ± {se:.4f} | {len(v)} | {delta:+.4f} |")
        lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    if supported:
        lines.append(
            f"**Primary SUPPORTED.** FRANKEN XOR_DELAY d=5 beats ER (p_bonf={p_er_b:.4g}) "
            f"and SHUF (p_bonf={p_sh_b:.4g}) on fresh seeds 200–223."
        )
    else:
        lines.append(
            f"**Primary NOT SUPPORTED.** Confirmatory replication failed. "
            f"FRANKEN mean={means.get('FRANKEN', float('nan')):.4f}, "
            f"ER={means.get('ER', float('nan')):.4f}, "
            f"SHUF={means.get('SHUF', float('nan')):.4f}. "
            f"p_bonf ER={p_er_b:.4g}, SHUF={p_sh_b:.4g}. "
            "The seeds-0–7 exploratory bump does not hold under preregistered fresh seeds + fair α."
        )
    lines.append("")
    lines.append(
        "Limits: reservoir 1−NRMSE on delayed XOR is not biological XOR computation; "
        "SPEC is a spectrum-preserving layer-misalignment control, not a full eigen-null model; "
        "DENS/FLY_CX/HUMAN are secondary."
    )
    lines.append("")

    (H / "RESULTS_xor_rep.md").write_text("\n".join(lines) + "\n")
    print("Wrote RESULTS_xor_rep.md", "verdict=", verdict)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    plot_arms = [a for a in ALL_ARMS if any(s in pv[a] for s in SEEDS)]
    data = [[pv[a][s] for s in SEEDS if s in pv[a]] for a in plot_arms]
    bp = ax.boxplot(data, tick_labels=plot_arms, patch_artist=True, showmeans=True)
    colors = {
        "FRANKEN": "#2ecc71", "ER": "#e74c3c", "SHUF": "#e67e22", "DENS": "#9b59b6",
        "FLY_CX": "#3498db", "HUMAN": "#1abc9c", "SPEC": "#95a5a6",
    }
    for patch, a in zip(bp["boxes"], plot_arms):
        patch.set_facecolor(colors.get(a, "#ddd")); patch.set_alpha(0.7)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("XOR_DELAY score (1−NRMSE) ↑")
    ax.set_title(f"d=5 REALMYELIN — verdict: {verdict}")
    ax.tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    ds = [2, 3, 5, 8, 10]
    for arm, col in [("FRANKEN", "#2ecc71"), ("ER", "#e74c3c"), ("SHUF", "#e67e22")]:
        ys, yse = [], []
        for d in ds:
            pvd = pivot(rows, [arm], d, "REALMYELIN", "NONE")
            vals = [pvd[arm][s] for s in SEEDS if s in pvd[arm]]
            ys.append(np.mean(vals) if vals else np.nan)
            yse.append(np.std(vals, ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax2.errorbar(ds, ys, yerr=yse, marker="o", label=arm, color=col, capsize=3)
    ax2.set_xlabel("XOR delay d")
    ax2.set_ylabel("score ↑")
    ax2.set_title("Score vs delay depth")
    ax2.legend(frameon=False)
    ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "xor_rep.png", dpi=140)
    fig.savefig(OUT / "xor_rep.png", dpi=140)
    print("Wrote figs/xor_rep.png")
    # also dump verdict machine-readable
    (OUT / "verdict.json").write_text(json.dumps(dict(
        verdict=verdict, c_er=c_er, c_sh=c_sh, p_er_bonf=p_er_b, p_sh_bonf=p_sh_b,
        means=means,
    ), indent=2, default=float))


if __name__ == "__main__":
    main()

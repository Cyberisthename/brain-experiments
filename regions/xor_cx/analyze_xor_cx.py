"""Analyze xor_cx → RESULTS_xor_cx.md + figs/xor_cx.png."""
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
SEEDS = list(range(300, 324))
PRIMARY = ["CX", "ER", "SHUF", "DENS"]
ALL = ["CX", "ER", "SHUF", "DENS", "EI_SHUF", "SPEC"]


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


def pivot(rows, arms, xor_d=5):
    out = {a: {} for a in arms}
    for r in rows:
        if r.get("xor_d") != xor_d: continue
        if r["arm"] not in out: continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired(a, b, seeds, alternative="greater"):
    d = []
    xa, xb = [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            xa.append(a[s]); xb.append(b[s]); d.append(a[s] - b[s])
    d = np.asarray(d); xa=np.asarray(xa); xb=np.asarray(xb); n=len(d)
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
                sem_a=float(np.std(xa, ddof=1)/math.sqrt(n)),
                sem_b=float(np.std(xb, ddof=1)/math.sqrt(n)),
                mean_d=float(np.mean(d)), sd_d=sd, d_z=dz, p_raw=p,
                ci_lo=float(np.mean(d) - tcrit * se),
                ci_hi=float(np.mean(d) + tcrit * se))


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    pv = pivot(rows, ALL, 5)
    lines = []
    lines.append("# XOR_CX results — FLY_CX alone vs own matched controls")
    lines.append("")
    lines.append("Prereg: `PREREG_xor_cx.md` (locked 2026-10-28 16:23 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("Parent: xor_rep mechanism → fly_CX carries FRANKEN XOR; this tests CX vs CX-matched random.")
    lines.append("")
    lines.append("## Ridge α (tuned 90–92, XOR d=5 FIXED)")
    lines.append("")
    if alphas.get("alphas"):
        lines.append("| Arm | α |")
        lines.append("|---|---:|")
        for arm in ALL:
            if arm in alphas["alphas"]:
                lines.append(f"| `{arm}` | {alphas['alphas'][arm]['alpha']} |")
    if alphas.get("spectral_audit"):
        lines.append("")
        lines.append("### Spectral audit (tune seeds, top-64 frac \\|λ\\|>0.5)")
        lines.append("")
        lines.append("| Arm | mean frac>0.5 |")
        lines.append("|---|---:|")
        for arm in ALL:
            fr = alphas["spectral_audit"].get(arm, [])
            vals = [x.get("frac_gt_0_5") for x in fr if x.get("frac_gt_0_5") == x.get("frac_gt_0_5")]
            if vals:
                lines.append(f"| `{arm}` | {float(np.mean(vals)):.3f} |")
    lines.append("")
    lines.append("## Primary: XOR_DELAY d=5 FIXED — means")
    lines.append("")
    lines.append("| Arm | mean ± sem | n |")
    lines.append("|---|---:|---:|")
    means = {}
    for arm in ALL:
        vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
        if not vals:
            lines.append(f"| `{arm}` | — | 0 |"); continue
        m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        means[arm] = m
        lines.append(f"| `{arm}` | {m:.4f} ± {se:.4f} | {len(vals)} |")
    lines.append("")
    lines.append("Metric ↑ better (1 − NRMSE).")
    lines.append("")

    contrasts = []
    for ctrl in ("ER", "SHUF", "DENS"):
        c = paired(pv["CX"], pv[ctrl], SEEDS, "greater")
        contrasts.append((ctrl, c))
    p_bonfs = [min(1.0, c["p_raw"] * 3) for _, c in contrasts]
    supported = all(
        c["n"] >= 20 and pb < 0.05 and c["mean_a"] > c["mean_b"]
        for (_, c), pb in zip(contrasts, p_bonfs)
    )
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    lines.append("## Primary hypothesis (Bonferroni ×3, one-sided Wilcoxon)")
    lines.append("")
    lines.append("| Contrast | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (ctrl, c), pb in zip(contrasts, p_bonfs):
        lines.append(
            f"| CX > {ctrl} | {c['mean_d']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
            f"{c['d_z']:.3f} | {c['p_raw']:.4g} | {pb:.4g} |"
        )
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    lines.append("## Secondary: XOR delay depth")
    lines.append("")
    lines.append("| d | CX | ER | SHUF | DENS | CX−ER Δ | p_raw (>) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for d in [2, 3, 5, 8, 10]:
        pvd = pivot(rows, PRIMARY, d)
        def m(arm):
            vals = [pvd[arm][s] for s in SEEDS if s in pvd[arm]]
            return f"{np.mean(vals):.3f}" if vals else "—"
        c = paired(pvd["CX"], pvd["ER"], SEEDS, "greater")
        lines.append(
            f"| {d} | {m('CX')} | {m('ER')} | {m('SHUF')} | {m('DENS')} | "
            f"{c['mean_d']:+.3f} | {c['p_raw']:.4g} |"
        )
    lines.append("")

    # EI / SPEC note
    lines.append("## Secondary: EI_SHUF and SPEC at d=5")
    lines.append("")
    for arm in ("EI_SHUF", "SPEC"):
        if arm in means:
            c = paired(pv["CX"], pv[arm], SEEDS, "greater")
            lines.append(
                f"- **{arm}** mean={means[arm]:.4f}; CX−{arm} Δ={c['mean_d']:+.4f}, "
                f"p_raw={c['p_raw']:.4g}, d_z={c['d_z']:.3f}"
            )
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    if supported:
        lines.append(
            f"**Primary SUPPORTED.** Real FlyWire CX beats ER, SHUF, and DENS on XOR_DELAY d=5 "
            f"(all p_bonf < 0.05) under fresh seeds 300–323 and per-arm α."
        )
    else:
        lines.append(
            f"**Primary NOT SUPPORTED.** CX mean={means.get('CX', float('nan')):.4f}; "
            f"ER={means.get('ER', float('nan')):.4f}; SHUF={means.get('SHUF', float('nan')):.4f}; "
            f"DENS={means.get('DENS', float('nan')):.4f}."
        )
    # spectral interpretation
    if "SPEC" in means and "CX" in means:
        if means["CX"] > means.get("SPEC", -9) + 0.05:
            lines.append(
                " SPEC (spectral-bulk tempered dens) did not fully close the gap → "
                "spectral sparsity alone is unlikely to be the whole story."
            )
        else:
            lines.append(
                " SPEC approached CX → spectral bulk matching may explain much of the edge; "
                "treat biology claims cautiously."
            )
    lines.append("")
    lines.append(
        "Limits: reservoir 1−NRMSE ≠ biological XOR; EI_SHUF preserves E/I edge classes not biology; "
        "SPEC temper is a heuristic spectral-bulk match, not an exact eigen-null model; "
        "FIXED delays (no myelin on fly)."
    )
    lines.append("")

    (H / "RESULTS_xor_cx.md").write_text("\n".join(lines) + "\n")
    print("Wrote RESULTS verdict=", verdict)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    plot_arms = [a for a in ALL if any(s in pv[a] for s in SEEDS)]
    data = [[pv[a][s] for s in SEEDS if s in pv[a]] for a in plot_arms]
    bp = ax.boxplot(data, tick_labels=plot_arms, patch_artist=True, showmeans=True)
    colors = {"CX": "#2ecc71", "ER": "#e74c3c", "SHUF": "#e67e22", "DENS": "#9b59b6",
              "EI_SHUF": "#3498db", "SPEC": "#95a5a6"}
    for patch, a in zip(bp["boxes"], plot_arms):
        patch.set_facecolor(colors.get(a, "#ddd")); patch.set_alpha(0.7)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("XOR_DELAY (1−NRMSE) ↑")
    ax.set_title(f"FLY_CX alone d=5 — {verdict}")
    ax.tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    ds = [2, 3, 5, 8, 10]
    for arm, col in [("CX", "#2ecc71"), ("ER", "#e74c3c"), ("SHUF", "#e67e22"), ("DENS", "#9b59b6")]:
        ys, yse = [], []
        for d in ds:
            pvd = pivot(rows, [arm], d)
            vals = [pvd[arm][s] for s in SEEDS if s in pvd[arm]]
            ys.append(np.mean(vals) if vals else np.nan)
            yse.append(np.std(vals, ddof=1)/math.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax2.errorbar(ds, ys, yerr=yse, marker="o", label=arm, color=col, capsize=3)
    ax2.set_xlabel("XOR delay d"); ax2.set_ylabel("score ↑")
    ax2.set_title("Score vs delay depth"); ax2.legend(frameon=False)
    ax2.axhline(0, color="k", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "xor_cx.png", dpi=140)
    fig.savefig(OUT / "xor_cx.png", dpi=140)
    print("Wrote figs/xor_cx.png")
    (OUT / "verdict.json").write_text(json.dumps(dict(
        verdict=verdict,
        contrasts={ctrl: c for ctrl, c in contrasts},
        p_bonf={ctrl: pb for (ctrl, _), pb in zip(contrasts, p_bonfs)},
        means=means,
    ), indent=2, default=float))


if __name__ == "__main__":
    main()

"""Analyze gated_team → RESULTS + figs/gated_team.png."""
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
SEEDS = list(range(700, 724))
ALL = ["GATED_REAL", "GATED_REAL_noLeak", "GATED_ER", "GATED_EI", "TEAM", "FLY_CX", "HUMAN", "ER4962"]
MARGIN_XOR = 0.03
MARGIN_NARMA = 0.02


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


def pivot_task(rows, task):
    out = {a: {} for a in ALL}
    for r in rows:
        if r.get("task") != task:
            continue
        if r["arm"] not in out:
            continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired(a, b, seeds, alternative="greater", transform=None):
    d, xa, xb = [], [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            va, vb = a[s], b[s]
            if transform:
                va, vb = transform(va, vb)
            xa.append(va); xb.append(vb); d.append(va - vb)
    d = np.asarray(d); xa = np.asarray(xa); xb = np.asarray(xb); n = len(d)
    if n < 2:
        return dict(n=n, mean_a=float("nan"), mean_b=float("nan"), mean_d=float("nan"),
                    sd_d=float("nan"), d_z=float("nan"), p_raw=1.0,
                    ci_lo=float("nan"), ci_hi=float("nan"))
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
                mean_d=float(np.mean(d)), sd_d=sd, d_z=dz, p_raw=p,
                ci_lo=float(np.mean(d) - tcrit * se),
                ci_hi=float(np.mean(d) + tcrit * se))


def composite_maps(pxor, pnarma, zparams):
    mu_x, sd_x = zparams["mu_xor"], max(zparams["sd_xor"], 1e-12)
    mu_n, sd_n = zparams["mu_narma"], max(zparams["sd_narma"], 1e-12)
    out = {a: {} for a in ALL}
    for a in ALL:
        for s in SEEDS:
            if s in pxor[a] and s in pnarma[a]:
                zx = (pxor[a][s] - mu_x) / sd_x
                zn = -(pnarma[a][s] - mu_n) / sd_n
                out[a][s] = 0.5 * (zx + zn)
    return out


def mean_sem(pv, arm):
    vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
    if not vals:
        return None, None, 0
    m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return m, se, len(vals)


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    zparams = alphas.get("composite_z", {"mu_xor": 0, "sd_xor": 1, "mu_narma": 0, "sd_narma": 1})
    hypers = alphas.get("hypers", {})
    pxor = pivot_task(rows, "XOR_DELAY")
    pnarma = pivot_task(rows, "NARMA")
    pmc = pivot_task(rows, "MC")
    ppc = pivot_task(rows, "PC_HARD")
    pcomp = composite_maps(pxor, pnarma, zparams)

    lines = []
    lines.append("# GATED TEAM results — Design 1+8 (experts + neuromod + timescales)")
    lines.append("")
    lines.append("Prereg: `PREREG_gated_team.md` (locked 2026-10-29 07:46 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("GATED_REAL = FLY_CX ⊕ HUMAN, zero bridges, expert ridges + convex gate + neuromod gain + species leaks.")
    lines.append("")
    lines.append("## Locked hypers (tune seeds 90–92)")
    lines.append("")
    lines.append("| Arm | γ_fly | γ_human | leak_fly | leak_human | leak_single |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ALL:
        h = hypers.get(arm, {})
        lines.append(
            f"| `{arm}` | {h.get('gamma_fly','—')} | {h.get('gamma_human','—')} | "
            f"{h.get('leak_fly','—')} | {h.get('leak_human','—')} | {h.get('leak_single','—')} |"
        )
    lines.append("")
    lines.append("## Ridge α (tuned 90–92, per arm × task)")
    lines.append("")
    if alphas.get("alphas"):
        lines.append("| Arm | XOR | NARMA | MC | PC_HARD |")
        lines.append("|---|---:|---:|---:|---:|")
        for arm in ALL:
            if arm not in alphas["alphas"]:
                continue
            aa = alphas["alphas"][arm]
            lines.append(
                f"| `{arm}` | {aa.get('XOR_DELAY',{}).get('alpha','—')} | "
                f"{aa.get('NARMA',{}).get('alpha','—')} | "
                f"{aa.get('MC',{}).get('alpha','—')} | "
                f"{aa.get('PC_HARD',{}).get('alpha','—')} |"
            )
    lines.append("")
    lines.append(
        f"Composite z locked from tune: μ_xor={zparams.get('mu_xor'):.4f}, "
        f"σ_xor={zparams.get('sd_xor'):.4f}, μ_narma={zparams.get('mu_narma'):.4f}, "
        f"σ_narma={zparams.get('sd_narma'):.4f}."
    )
    lines.append("")

    lines.append("## Primary tasks + composite (means)")
    lines.append("")
    lines.append("| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ |")
    lines.append("|---|---:|---:|---:|---:|")
    means = {}
    for arm in ALL:
        ns = [r["n"] for r in rows if r["arm"] == arm]
        N = int(ns[0]) if ns else 0
        mx, sx, _ = mean_sem(pxor, arm)
        mn, sn, _ = mean_sem(pnarma, arm)
        mc, sc, _ = mean_sem(pcomp, arm)
        means[arm] = dict(xor=mx, narma=mn, comp=mc, N=N)
        def fmt(m, s):
            return "—" if m is None else f"{m:.4f}±{s:.4f}"
        lines.append(f"| `{arm}` | {N} | {fmt(mx,sx)} | {fmt(mn,sn)} | {fmt(mc,sc)} |")
    lines.append("")

    # H1x XOR NI: GATED - FLY + margin
    def xor_ni(va, vb):
        return va + MARGIN_XOR, vb

    h1x = paired(pxor["GATED_REAL"], pxor["FLY_CX"], SEEDS, "greater", transform=xor_ni)
    # H1n NARMA NI: HUMAN - GATED + margin  (lower NARMA better)
    def narma_ni(va, vb):
        # paired(HUMAN, GATED) with transform → we want d = HUMAN - GATED + margin
        # paired(a,b) computes a-b; so a=HUMAN, b=GATED, transform adds margin to a
        return va + MARGIN_NARMA, vb

    h1n = paired(pnarma["HUMAN"], pnarma["GATED_REAL"], SEEDS, "greater", transform=narma_ni)
    p_h1 = max(h1x["p_raw"], h1n["p_raw"])
    h1_means_ok = (
        means["GATED_REAL"]["xor"] is not None and means["FLY_CX"]["xor"] is not None
        and means["GATED_REAL"]["xor"] >= means["FLY_CX"]["xor"] - MARGIN_XOR
        and means["GATED_REAL"]["narma"] is not None and means["HUMAN"]["narma"] is not None
        and means["GATED_REAL"]["narma"] <= means["HUMAN"]["narma"] + MARGIN_NARMA
    )

    h2 = paired(pcomp["GATED_REAL"], pcomp["GATED_ER"], SEEDS, "greater")
    h3 = paired(pcomp["GATED_REAL"], pcomp["TEAM"], SEEDS, "greater")

    tests = [
        ("H1 joint specialists (max p of NI)", p_h1, h1_means_ok,
         dict(mean_d=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), d_z=float("nan"),
              note=f"H1x_p={h1x['p_raw']:.4g} H1n_p={h1n['p_raw']:.4g}")),
        ("H2 GATED_REAL>GATED_ER composite", h2["p_raw"],
         means["GATED_REAL"]["comp"] is not None and means["GATED_ER"]["comp"] is not None
         and means["GATED_REAL"]["comp"] > means["GATED_ER"]["comp"], h2),
        ("H3 GATED_REAL>TEAM composite", h3["p_raw"],
         means["GATED_REAL"]["comp"] is not None and means["TEAM"]["comp"] is not None
         and means["GATED_REAL"]["comp"] > means["TEAM"]["comp"], h3),
    ]

    lines.append("## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon)")
    lines.append("")
    lines.append("| Hypothesis | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf | pass |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    passes = []
    detail = {"H1x": h1x, "H1n": h1n, "H2": h2, "H3": h3}
    for name, p_raw, means_ok, extra in tests:
        p_bonf = min(1.0, 3.0 * p_raw)
        ok = bool(means_ok and p_bonf < 0.05)
        passes.append(ok)
        if name.startswith("H1"):
            lines.append(
                f"| {name} | — | — | — | {p_raw:.4g} | {p_bonf:.4g} | {'✓' if ok else '✗'} |"
            )
        else:
            lines.append(
                f"| {name} | {extra['mean_d']:+.4f} | [{extra['ci_lo']:+.4f}, {extra['ci_hi']:+.4f}] | "
                f"{extra['d_z']:.3f} | {p_raw:.4g} | {p_bonf:.4g} | {'✓' if ok else '✗'} |"
            )
    lines.append("")
    lines.append("### H1 components")
    lines.append("")
    lines.append(
        f"- **H1x XOR NI (margin {MARGIN_XOR}):** mean(GATED−FLY)+margin="
        f"{h1x['mean_d']:+.4f}, p={h1x['p_raw']:.4g}, "
        f"XOR GATED={means['GATED_REAL']['xor']}, FLY={means['FLY_CX']['xor']}"
    )
    lines.append(
        f"- **H1n NARMA NI (margin {MARGIN_NARMA}):** mean(HUMAN−GATED)+margin="
        f"{h1n['mean_d']:+.4f}, p={h1n['p_raw']:.4g}, "
        f"NARMA GATED={means['GATED_REAL']['narma']}, HUMAN={means['HUMAN']['narma']}"
    )
    lines.append("")
    verdict = "SUPPORTED" if all(passes) else "NOT SUPPORTED"
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    lines.append("## Secondary tasks")
    lines.append("")
    lines.append("| Arm | MC ↑ | PC_HARD ↑ |")
    lines.append("|---|---:|---:|")
    for arm in ALL:
        mm, sm, _ = mean_sem(pmc, arm)
        mp, sp, _ = mean_sem(ppc, arm)
        def fmt(m, s):
            return "—" if m is None else f"{m:.3f}±{s:.3f}"
        lines.append(f"| `{arm}` | {fmt(mm,sm)} | {fmt(mp,sp)} |")
    lines.append("")

    # secondary contrasts
    sec = []
    for label, A, B, pv in [
        ("GATED_REAL vs GATED_REAL_noLeak composite", "GATED_REAL", "GATED_REAL_noLeak", pcomp),
        ("GATED_REAL vs GATED_EI composite", "GATED_REAL", "GATED_EI", pcomp),
        ("GATED_REAL vs ER4962 composite", "GATED_REAL", "ER4962", pcomp),
        ("GATED_REAL vs FLY_CX composite", "GATED_REAL", "FLY_CX", pcomp),
        ("GATED_REAL vs HUMAN composite", "GATED_REAL", "HUMAN", pcomp),
        ("GATED_REAL vs FLY_CX XOR", "GATED_REAL", "FLY_CX", pxor),
        ("GATED_REAL vs HUMAN NARMA (lower better → use HUMAN-GATED)", "HUMAN", "GATED_REAL", pnarma),
    ]:
        r = paired(pv[A], pv[B], SEEDS, "greater")
        sec.append((label, r))
        lines.append(f"- **{label}**: Δ={r['mean_d']:+.4f}, p_raw={r['p_raw']:.4g}, d_z={r['d_z']:.3f}")
    lines.append("")

    # mean lambda on gated
    lams = {a: {t: [] for t in ["XOR_DELAY", "NARMA", "MC", "PC_HARD"]} for a in ALL if a.startswith("GATED")}
    for r in rows:
        if r["arm"] in lams and r.get("lam") is not None and np.isfinite(r["lam"]):
            lams[r["arm"]][r["task"]].append(r["lam"])
    lines.append("## Mean convex gate λ (fly weight) on eval")
    lines.append("")
    lines.append("| Arm | XOR | NARMA | MC | PC_HARD |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in lams:
        cells = []
        for t in ["XOR_DELAY", "NARMA", "MC", "PC_HARD"]:
            v = lams[arm][t]
            cells.append(f"{np.mean(v):.3f}" if v else "—")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    lines.append(
        f"**Primary {verdict}.** "
        f"GATED_REAL XOR={means['GATED_REAL']['xor']}; FLY_CX={means['FLY_CX']['xor']}; "
        f"GATED_REAL NARMA={means['GATED_REAL']['narma']}; HUMAN={means['HUMAN']['narma']}; "
        f"comp GATED={means['GATED_REAL']['comp']}; GATED_ER={means['GATED_ER']['comp']}; "
        f"TEAM={means['TEAM']['comp']}."
    )
    lines.append(
        " Limits: task cue is experimenter-given; convex gate is linear; zero-bridge tests readout+gain "
        "specialization not axons; reservoir ≠ biology; CPU shared with other jobs."
    )
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    # filled by runner note if present
    rt = OUT / "runtime.txt"
    if rt.exists():
        lines.append(rt.read_text().strip())
    else:
        lines.append("- See `out/run.log` for wall time. WORKERS=2. Figure: `figs/gated_team.png`.")
    lines.append("")

    (H / "RESULTS_gated_team.md").write_text("\n".join(lines) + "\n")
    verdict_obj = dict(
        verdict=verdict, passes=passes,
        H1x=h1x, H1n=h1n, H2=h2, H3=h3, p_H1=p_h1,
        means=means, zparams=zparams, hypers=hypers,
    )
    (OUT / "verdict.json").write_text(json.dumps(verdict_obj, indent=2, default=float))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    order = ALL
    colors = plt.cm.tab10(np.linspace(0, 1, len(order)))

    def box(ax, pv, title, ylabel, higher_better=True):
        data, labels = [], []
        for arm in order:
            vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
            if vals:
                data.append(vals); labels.append(arm)
        if not data:
            ax.set_title(title); return
        try:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        except TypeError:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors[:len(data)]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    box(axes[0], pxor, "XOR_DELAY (↑)", "1−NRMSE")
    box(axes[1], pnarma, "NARMA-10 (↓)", "NRMSE")
    box(axes[2], pcomp, "Composite (↑)", "z-score")
    fig.suptitle(f"gated_team Design 1+8 — {verdict}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "gated_team.png", dpi=140)
    fig.savefig(OUT / "gated_team.png", dpi=140)
    print("Wrote RESULTS_gated_team.md and figs/gated_team.png", verdict, flush=True)


if __name__ == "__main__":
    main()

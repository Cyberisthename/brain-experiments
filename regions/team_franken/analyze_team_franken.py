"""Analyze team_franken → RESULTS + figs/team_franken.png."""
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
SEEDS = list(range(400, 424))
ALL = ["TEAM", "FLY_CX", "HUMAN", "FRANKEN", "TEAM_ER", "TEAM_EI", "FLY_CX_BIG"]
MARGIN = 0.03


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
        if r.get("task") != task: continue
        if r["arm"] not in out: continue
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


def main():
    rows = load_rows()
    alphas = json.loads((OUT / "alpha_choice.json").read_text()) if (OUT / "alpha_choice.json").exists() else {}
    zparams = alphas.get("composite_z", {"mu_xor": 0, "sd_xor": 1, "mu_narma": 0, "sd_narma": 1})
    pxor = pivot_task(rows, "XOR_DELAY")
    pnarma = pivot_task(rows, "NARMA")
    pmc = pivot_task(rows, "MC")
    ppc = pivot_task(rows, "PC_HARD")
    pcomp = composite_maps(pxor, pnarma, zparams)

    lines = []
    lines.append("# TEAM Frankenstein results — division of labor")
    lines.append("")
    lines.append("Prereg: `PREREG_team_franken.md` (locked 2026-10-28 17:56 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("TEAM = FLY_CX ⊕ HUMAN, zero bridges, shared L4 drive, concatenated ridge.")
    lines.append("")
    lines.append("## Ridge α (tuned 90–92, per arm × task)")
    lines.append("")
    if alphas.get("alphas"):
        lines.append("| Arm | XOR | NARMA | MC | PC_HARD |")
        lines.append("|---|---:|---:|---:|---:|")
        for arm in ALL:
            if arm not in alphas["alphas"]: continue
            aa = alphas["alphas"][arm]
            lines.append(
                f"| `{arm}` | {aa.get('XOR_DELAY',{}).get('alpha','—')} | "
                f"{aa.get('NARMA',{}).get('alpha','—')} | "
                f"{aa.get('MC',{}).get('alpha','—')} | "
                f"{aa.get('PC_HARD',{}).get('alpha','—')} |"
            )
    lines.append("")
    lines.append(f"Composite z locked from tune: μ_xor={zparams.get('mu_xor'):.4f}, "
                 f"σ_xor={zparams.get('sd_xor'):.4f}, μ_narma={zparams.get('mu_narma'):.4f}, "
                 f"σ_narma={zparams.get('sd_narma'):.4f}.")
    lines.append("")

    def mean_sem(pv, arm):
        vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
        if not vals: return None, None, 0
        m = float(np.mean(vals)); se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        return m, se, len(vals)

    lines.append("## Primary tasks + composite (means)")
    lines.append("")
    lines.append("| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ |")
    lines.append("|---|---:|---:|---:|---:|")
    means = {}
    for arm in ALL:
        # get N from any row
        ns = [r["n"] for r in rows if r["arm"] == arm]
        N = int(ns[0]) if ns else 0
        mx, sx, nx = mean_sem(pxor, arm)
        mn, sn, nn = mean_sem(pnarma, arm)
        mc, sc, nc = mean_sem(pcomp, arm)
        means[arm] = dict(xor=mx, narma=mn, comp=mc, N=N)
        def fmt(m, s):
            return "—" if m is None else f"{m:.4f}±{s:.4f}"
        lines.append(f"| `{arm}` | {N} | {fmt(mx,sx)} | {fmt(mn,sn)} | {fmt(mc,sc)} |")
    lines.append("")

    # Primary tests
    c_vs_fly = paired(pcomp["TEAM"], pcomp["FLY_CX"], SEEDS, "greater")
    # non-inferiority: TEAM - FLY + margin > 0
    def ni_transform(va, vb):
        return va + MARGIN, vb  # so va' - vb = TEAM - FLY + margin
    c_ni = paired(pxor["TEAM"], pxor["FLY_CX"], SEEDS, "greater",
                  transform=lambda va, vb: (va + MARGIN, vb))
    c_vs_er = paired(pcomp["TEAM"], pcomp["TEAM_ER"], SEEDS, "greater")
    c_vs_big = paired(pcomp["TEAM"], pcomp["FLY_CX_BIG"], SEEDS, "greater")

    tests = [
        ("H1a TEAM>FLY_CX composite", c_vs_fly,
         means["TEAM"]["comp"] is not None and means["FLY_CX"]["comp"] is not None
         and means["TEAM"]["comp"] > means["FLY_CX"]["comp"]),
        ("H1b XOR non-inf (margin 0.03)", c_ni,
         means["TEAM"]["xor"] is not None and means["FLY_CX"]["xor"] is not None
         and means["TEAM"]["xor"] >= means["FLY_CX"]["xor"] - MARGIN),
        ("H2 TEAM>TEAM_ER composite", c_vs_er,
         means["TEAM"]["comp"] is not None and means["TEAM_ER"]["comp"] is not None
         and means["TEAM"]["comp"] > means["TEAM_ER"]["comp"]),
        ("H3 TEAM>FLY_CX_BIG composite", c_vs_big,
         means["TEAM"]["comp"] is not None and means["FLY_CX_BIG"]["comp"] is not None
         and means["TEAM"]["comp"] > means["FLY_CX_BIG"]["comp"]),
    ]
    p_bonfs = [min(1.0, t[1]["p_raw"] * 4) for t in tests]
    supported = all(t[1]["n"] >= 20 and pb < 0.05 and t[2] for t, pb in zip(tests, p_bonfs))
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    lines.append("## Primary hypotheses (Bonferroni ×4, one-sided Wilcoxon)")
    lines.append("")
    lines.append("| Hypothesis | mean Δ | 95% CI Δ | d_z | p_raw | p_bonf | pass |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for (name, c, mean_ok), pb in zip(tests, p_bonfs):
        ok = c["n"] >= 20 and pb < 0.05 and mean_ok
        lines.append(
            f"| {name} | {c['mean_d']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
            f"{c['d_z']:.3f} | {c['p_raw']:.4g} | {pb:.4g} | {'✓' if ok else '✗'} |"
        )
    lines.append("")
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

    lines.append("## Secondary contrasts (raw)")
    lines.append("")
    for label, a, b, pv in [
        ("TEAM vs HUMAN composite", "TEAM", "HUMAN", pcomp),
        ("TEAM vs FRANKEN composite", "TEAM", "FRANKEN", pcomp),
        ("TEAM vs TEAM_EI composite", "TEAM", "TEAM_EI", pcomp),
        ("TEAM vs HUMAN NARMA (lower better → reverse)", "HUMAN", "TEAM", pnarma),  # HUMAN < TEAM means TEAM worse
        ("TEAM vs FLY_CX XOR", "TEAM", "FLY_CX", pxor),
    ]:
        c = paired(pv[a], pv[b], SEEDS, "greater")
        lines.append(f"- **{label}**: Δ={c['mean_d']:+.4f}, p_raw={c['p_raw']:.4g}, d_z={c['d_z']:.3f}")
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    if supported:
        lines.append(
            "**Primary SUPPORTED.** TEAM beats FLY_CX, TEAM_ER, and FLY_CX_BIG on the XOR/−NARMA "
            "composite and is non-inferior on XOR (margin 0.03) under seeds 400–423."
        )
    else:
        lines.append(
            f"**Primary NOT SUPPORTED.** TEAM comp={means['TEAM']['comp']}; "
            f"FLY_CX={means['FLY_CX']['comp']}; TEAM_ER={means['TEAM_ER']['comp']}; "
            f"FLY_CX_BIG={means['FLY_CX_BIG']['comp']}. "
            f"XOR TEAM={means['TEAM']['xor']} vs FLY_CX={means['FLY_CX']['xor']}."
        )
    lines.append(
        " Limits: zero-bridge TEAM is a readout-selection test; FRANKEN is larger (adds mouse); "
        "NARMA ↓ / XOR ↑ composite depends on tune-locked z; reservoir ≠ biology."
    )
    lines.append("")

    (H / "RESULTS_team_franken.md").write_text("\n".join(lines) + "\n")
    print("Wrote RESULTS verdict=", verdict)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    colors = {
        "TEAM": "#2ecc71", "FLY_CX": "#27ae60", "HUMAN": "#3498db", "FRANKEN": "#9b59b6",
        "TEAM_ER": "#e74c3c", "TEAM_EI": "#f39c12", "FLY_CX_BIG": "#95a5a6",
    }

    ax = axes[0]
    data = [[pcomp[a][s] for s in SEEDS if s in pcomp[a]] for a in ALL]
    bp = ax.boxplot(data, tick_labels=ALL, patch_artist=True, showmeans=True)
    for patch, a in zip(bp["boxes"], ALL):
        patch.set_facecolor(colors.get(a, "#ddd")); patch.set_alpha(0.7)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("composite (z XOR + z −NARMA) ↑")
    ax.set_title(f"Composite — {verdict}")
    ax.tick_params(axis="x", rotation=40)

    ax = axes[1]
    data = [[pxor[a][s] for s in SEEDS if s in pxor[a]] for a in ALL]
    bp = ax.boxplot(data, tick_labels=ALL, patch_artist=True, showmeans=True)
    for patch, a in zip(bp["boxes"], ALL):
        patch.set_facecolor(colors.get(a, "#ddd")); patch.set_alpha(0.7)
    ax.set_ylabel("XOR_DELAY d=5 (1−NRMSE) ↑")
    ax.set_title("XOR")
    ax.tick_params(axis="x", rotation=40)

    ax = axes[2]
    data = [[pnarma[a][s] for s in SEEDS if s in pnarma[a]] for a in ALL]
    bp = ax.boxplot(data, tick_labels=ALL, patch_artist=True, showmeans=True)
    for patch, a in zip(bp["boxes"], ALL):
        patch.set_facecolor(colors.get(a, "#ddd")); patch.set_alpha(0.7)
    ax.set_ylabel("NARMA-10 NRMSE ↓")
    ax.set_title("NARMA")
    ax.tick_params(axis="x", rotation=40)

    fig.tight_layout()
    fig.savefig(FIGS / "team_franken.png", dpi=140)
    fig.savefig(OUT / "team_franken.png", dpi=140)
    print("Wrote figs/team_franken.png")

    (OUT / "verdict.json").write_text(json.dumps(dict(
        verdict=verdict,
        tests={name: dict(c, mean_ok=mean_ok, p_bonf=pb)
               for (name, c, mean_ok), pb in zip(tests, p_bonfs)},
        means=means,
        zparams=zparams,
    ), indent=2, default=float))


if __name__ == "__main__":
    main()

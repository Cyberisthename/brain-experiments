"""Analyze designed_brain → RESULTS + figs/designed_brain.png."""
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
SEEDS = list(range(500, 524))
ALL = ["DESIGN", "DESIGN_EI_ONLY", "DESIGN_MYELIN_ONLY", "FLY_CX", "HUMAN", "TEAM_ER", "ER", "DENS", "SHUF"]
MARGIN_XOR, MARGIN_NARMA = 0.05, 0.03


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
        if r.get("task") != task or r["arm"] not in out: continue
        out[r["arm"]][int(r["seed"])] = float(r["value"])
    return out


def paired(a, b, seeds, alternative="greater", transform=None):
    d, xa, xb = [], [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            va, vb = a[s], b[s]
            if transform: va, vb = transform(va, vb)
            xa.append(va); xb.append(vb); d.append(va - vb)
    d = np.asarray(d); xa=np.asarray(xa); xb=np.asarray(xb); n=len(d)
    if n < 2:
        return dict(n=n, mean_a=float("nan"), mean_b=float("nan"), mean_d=float("nan"),
                    sd_d=float("nan"), d_z=float("nan"), p_raw=1.0,
                    ci_lo=float("nan"), ci_hi=float("nan"))
    if np.allclose(d, 0): p = 1.0
    else:
        try: p = float(stats.wilcoxon(d, alternative=alternative).pvalue)
        except Exception: p = 1.0
    sd = float(np.std(d, ddof=1)); dz = float(np.mean(d)/sd) if sd > 1e-15 else float("nan")
    se = sd / math.sqrt(n); tcrit = float(stats.t.ppf(0.975, n-1))
    return dict(n=n, mean_a=float(np.mean(xa)), mean_b=float(np.mean(xb)),
                mean_d=float(np.mean(d)), sd_d=sd, d_z=dz, p_raw=p,
                ci_lo=float(np.mean(d)-tcrit*se), ci_hi=float(np.mean(d)+tcrit*se))


def composite_maps(pxor, pnarma, z):
    mu_x, sd_x = z["mu_xor"], max(z["sd_xor"], 1e-12)
    mu_n, sd_n = z["mu_narma"], max(z["sd_narma"], 1e-12)
    out = {a: {} for a in ALL}
    for a in ALL:
        for s in SEEDS:
            if s in pxor[a] and s in pnarma[a]:
                out[a][s] = 0.5 * ((pxor[a][s]-mu_x)/sd_x + (-(pnarma[a][s]-mu_n)/sd_n))
    return out


def mean_sem(pv, arm):
    vals = [pv[arm][s] for s in SEEDS if s in pv[arm]]
    if not vals: return None, None, 0
    m = float(np.mean(vals)); se = float(np.std(vals, ddof=1)/math.sqrt(len(vals)))
    return m, se, len(vals)


def main():
    rows = load_rows()
    alphas = json.loads((OUT/"alpha_choice.json").read_text()) if (OUT/"alpha_choice.json").exists() else {}
    z = alphas.get("composite_z", {"mu_xor":0,"sd_xor":1,"mu_narma":0,"sd_narma":1})
    meta = alphas.get("design_meta") or (json.loads((OUT/"design_meta.json").read_text()) if (OUT/"design_meta.json").exists() else {})
    pxor, pnarma = pivot(rows,"XOR_DELAY"), pivot(rows,"NARMA")
    pmc, ppc = pivot(rows,"MC"), pivot(rows,"PC_HARD")
    pcomp = composite_maps(pxor, pnarma, z)

    lines = []
    lines.append("# Designed brain (CHIMERA) results")
    lines.append("")
    lines.append("Prereg: `PREREG_designed_brain.md` (locked 2026-10-28 19:37 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("")
    lines.append("## Design in plain words")
    lines.append("")
    lines.append(meta.get("plain_words", "(see design_meta.json)"))
    lines.append("")
    lines.append(
        f"- N={meta.get('N')}, nnz={meta.get('nnz')}, frac_E={meta.get('frac_E')}, "
        f"spectral frac\\|λ\\|>0.5 after temper={meta.get('spectral_frac')}, "
        f"mf_mean={meta.get('mf_mean')}, layers={meta.get('layer_counts')}."
    )
    lines.append("")
    lines.append("## Ridge α (90–92)")
    lines.append("")
    if alphas.get("alphas"):
        lines.append("| Arm | XOR | NARMA | MC | PC_HARD |")
        lines.append("|---|---:|---:|---:|---:|")
        for arm in ALL:
            aa = alphas["alphas"].get(arm, {})
            lines.append(
                f"| `{arm}` | {aa.get('XOR_DELAY',{}).get('alpha','—')} | "
                f"{aa.get('NARMA',{}).get('alpha','—')} | "
                f"{aa.get('MC',{}).get('alpha','—')} | "
                f"{aa.get('PC_HARD',{}).get('alpha','—')} |"
            )
    lines.append("")
    lines.append(f"Composite z (tune): μ_xor={z.get('mu_xor'):.4f}, σ_xor={z.get('sd_xor'):.4f}, "
                 f"μ_narma={z.get('mu_narma'):.4f}, σ_narma={z.get('sd_narma'):.4f}.")
    lines.append("")

    lines.append("## Means")
    lines.append("")
    lines.append("| Arm | N | XOR ↑ | NARMA ↓ | composite ↑ | MC ↑ | PC_HARD ↑ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    means = {}
    for arm in ALL:
        ns = [r["n"] for r in rows if r["arm"]==arm]
        N = int(ns[0]) if ns else 0
        mx,sx,_ = mean_sem(pxor, arm); mn,sn,_ = mean_sem(pnarma, arm)
        mc,sc,_ = mean_sem(pcomp, arm); mm,sm,_ = mean_sem(pmc, arm); mp,sp,_ = mean_sem(ppc, arm)
        means[arm] = dict(xor=mx, narma=mn, comp=mc, mc=mm, pc=mp, N=N)
        def fmt(m,s): return "—" if m is None else f"{m:.4f}±{(s or 0):.4f}"
        lines.append(f"| `{arm}` | {N} | {fmt(mx,sx)} | {fmt(mn,sn)} | {fmt(mc,sc)} | {fmt(mm,sm)} | {fmt(mp,sp)} |")
    lines.append("")

    # Primary tests
    c_h1 = paired(pcomp["DESIGN"], pcomp["ER"], SEEDS, "greater")
    c_h2 = paired(pcomp["DESIGN"], pcomp["TEAM_ER"], SEEDS, "greater")
    c_h3 = paired(pxor["DESIGN"], pxor["FLY_CX"], SEEDS, "greater",
                  transform=lambda va, vb: (va + MARGIN_XOR, vb))
    # H4: HUMAN_narma - DESIGN_narma + margin > 0  (DESIGN not much worse than HUMAN)
    c_h4 = paired(pnarma["HUMAN"], pnarma["DESIGN"], SEEDS, "greater",
                  transform=lambda va, vb: (va + MARGIN_NARMA, vb))
    # Wait: paired(a,b) does a-b. We want HUMAN - DESIGN + margin > 0
    # transform on (va=HUMAN, vb=DESIGN): return (va+margin, vb) → diff = HUMAN+margin - DESIGN. Good.

    tests = [
        ("H1 DESIGN>ER composite", c_h1,
         means["DESIGN"]["comp"] is not None and means["ER"]["comp"] is not None
         and means["DESIGN"]["comp"] > means["ER"]["comp"]),
        ("H2 DESIGN>TEAM_ER composite", c_h2,
         means["DESIGN"]["comp"] is not None and means["TEAM_ER"]["comp"] is not None
         and means["DESIGN"]["comp"] > means["TEAM_ER"]["comp"]),
        ("H3 XOR non-inf vs FLY_CX (m=0.05)", c_h3,
         means["DESIGN"]["xor"] is not None and means["FLY_CX"]["xor"] is not None
         and means["DESIGN"]["xor"] >= means["FLY_CX"]["xor"] - MARGIN_XOR),
        ("H4 NARMA non-inf vs HUMAN (m=0.03)", c_h4,
         means["DESIGN"]["narma"] is not None and means["HUMAN"]["narma"] is not None
         and means["DESIGN"]["narma"] <= means["HUMAN"]["narma"] + MARGIN_NARMA),
    ]
    p_bonfs = [min(1.0, t[1]["p_raw"]*4) for t in tests]
    supported = all(t[1]["n"]>=20 and pb<0.05 and t[2] for t,pb in zip(tests, p_bonfs))
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    lines.append("## Primary hypotheses (Bonferroni ×4)")
    lines.append("")
    lines.append("| Hypothesis | mean Δ | 95% CI | d_z | p_raw | p_bonf | pass |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for (name,c,mean_ok), pb in zip(tests, p_bonfs):
        ok = c["n"]>=20 and pb<0.05 and mean_ok
        lines.append(
            f"| {name} | {c['mean_d']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
            f"{c['d_z']:.3f} | {c['p_raw']:.4g} | {pb:.4g} | {'✓' if ok else '✗'} |"
        )
    lines.append("")
    lines.append(f"**Overall verdict: {verdict}**")
    lines.append("")

    lines.append("## Ablations / secondary (raw)")
    lines.append("")
    for label, a, b, pv in [
        ("DESIGN vs DESIGN_EI_ONLY composite", "DESIGN", "DESIGN_EI_ONLY", pcomp),
        ("DESIGN vs DESIGN_MYELIN_ONLY composite", "DESIGN", "DESIGN_MYELIN_ONLY", pcomp),
        ("DESIGN vs DENS composite", "DESIGN", "DENS", pcomp),
        ("DESIGN vs SHUF composite", "DESIGN", "SHUF", pcomp),
        ("DESIGN vs FLY_CX XOR", "DESIGN", "FLY_CX", pxor),
        ("DESIGN vs HUMAN NARMA (lower better: HUMAN−DESIGN)", "HUMAN", "DESIGN", pnarma),
    ]:
        c = paired(pv[a], pv[b], SEEDS, "greater")
        lines.append(f"- **{label}**: Δ={c['mean_d']:+.4f}, p_raw={c['p_raw']:.4g}, d_z={c['d_z']:.3f}")
    lines.append("")

    lines.append("## Honest verdict")
    lines.append("")
    if supported:
        lines.append("**Primary SUPPORTED.** DESIGN beats ER and TEAM_ER on composite and meets XOR/NARMA non-inferiority gates.")
    else:
        lines.append(
            f"**Primary NOT SUPPORTED.** DESIGN comp={means['DESIGN']['comp']}, ER={means['ER']['comp']}, "
            f"TEAM_ER={means['TEAM_ER']['comp']}; XOR DESIGN={means['DESIGN']['xor']} vs FLY_CX={means['FLY_CX']['xor']}; "
            f"NARMA DESIGN={means['DESIGN']['narma']} vs HUMAN={means['HUMAN']['narma']}."
        )
    lines.append(
        " Limits: synthetic SBM ≠ biology; myelin rule is layer-conditional sampling from H01, not axon geometry; "
        "DESIGN adjacency frozen across seeds (Win/task RNG vary); TEAM_ER is larger (two blocks)."
    )
    lines.append("")

    (H/"RESULTS_designed_brain.md").write_text("\n".join(lines)+"\n")
    print("Wrote RESULTS verdict=", verdict)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    colors = {
        "DESIGN":"#2ecc71","DESIGN_EI_ONLY":"#27ae60","DESIGN_MYELIN_ONLY":"#1abc9c",
        "FLY_CX":"#3498db","HUMAN":"#9b59b6","TEAM_ER":"#e74c3c",
        "ER":"#e67e22","DENS":"#95a5a6","SHUF":"#7f8c8d",
    }
    for ax, pv, ylab, title in [
        (axes[0], pcomp, "composite ↑", f"Composite — {verdict}"),
        (axes[1], pxor, "XOR ↑", "XOR_DELAY d=5"),
        (axes[2], pnarma, "NARMA ↓", "NARMA-10"),
    ]:
        data = [[pv[a][s] for s in SEEDS if s in pv[a]] for a in ALL]
        bp = ax.boxplot(data, tick_labels=ALL, patch_artist=True, showmeans=True)
        for patch,a in zip(bp["boxes"], ALL):
            patch.set_facecolor(colors.get(a,"#ddd")); patch.set_alpha(0.7)
        ax.set_ylabel(ylab); ax.set_title(title); ax.tick_params(axis="x", rotation=55)
        if pv is pcomp: ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS/"designed_brain.png", dpi=140)
    fig.savefig(OUT/"designed_brain.png", dpi=140)
    print("Wrote figs/designed_brain.png")
    (OUT/"verdict.json").write_text(json.dumps(dict(
        verdict=verdict,
        tests={n: dict(c, mean_ok=mo, p_bonf=pb) for (n,c,mo),pb in zip(tests,p_bonfs)},
        means=means, design_meta=meta, zparams=z,
    ), indent=2, default=float))


if __name__ == "__main__":
    main()

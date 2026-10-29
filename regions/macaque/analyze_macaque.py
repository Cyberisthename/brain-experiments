"""Analyze macaque → RESULTS_macaque.md + figs/macaque.png."""
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
SEEDS = list(range(600, 624))
ALL = [
    "MACAQUE", "MACAQUE_ER", "MACAQUE_SHUF", "DENS",
    "FLY_CX", "FRANKEN", "FRANKEN+MACAQUE", "FRANKEN+ER_addon",
]
BONF = 3


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


def paired(a, b, seeds, alternative="greater"):
    d, xa, xb = [], [], []
    for s in seeds:
        if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]):
            xa.append(a[s]); xb.append(b[s]); d.append(a[s] - b[s])
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
    lines.append("# Macaque FLN reservoir results")
    lines.append("")
    lines.append("Prereg: `PREREG_macaque.md` (locked 2026-10-29 07:39 America/New_York, before eval).")
    lines.append(f"Rows kept: {len(rows)}. Eval seeds: {SEEDS[0]}–{SEEDS[-1]} (n={len(SEEDS)}).")
    lines.append("")
    lines.append("## Design in plain words")
    lines.append("")
    lines.append(meta.get("plain_words", "(see design_meta.json)"))
    lines.append("")
    lines.append(
        f"- N={meta.get('N')}, nnz={meta.get('nnz')}, frac_E={meta.get('frac_E')}, "
        f"fln_dens={meta.get('fln_dens')}, sln_proxy={meta.get('sln_proxy')}, "
        f"layers={meta.get('layer_counts')}."
    )
    lines.append("- Primary source: Markov FLNe 29×29 (tract-tracing). OpenNeuro ds001875 MRI (~11.5 GB) not downloaded; TVB dMRI secondary unused for wiring. See `SOURCES.md`.")
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
    lines.append("## Task means (eval seeds)")
    lines.append("")
    lines.append("| Arm | XOR ↑ | NARMA ↓ | MC ↑ | PC_HARD ↑ | composite |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ALL:
        cells = []
        for pv, lower in ((pxor, False), (pnarma, True), (pmc, False), (ppc, False), (pcomp, False)):
            m, se, n = mean_sem(pv, arm)
            cells.append("—" if m is None else f"{m:.3f}±{se:.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")

    hyps = [
        ("H1", "MACAQUE", "MACAQUE_ER", "MACAQUE composite > MACAQUE_ER"),
        ("H2", "MACAQUE", "MACAQUE_SHUF", "MACAQUE composite > MACAQUE_SHUF"),
        ("H3", "FRANKEN+MACAQUE", "FRANKEN+ER_addon", "FRANKEN+MACAQUE composite > FRANKEN+ER_addon"),
    ]
    lines.append("## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon)")
    lines.append("")
    lines.append("| H | Contrast | mean_a | mean_b | Δ | p_raw | p_bonf | pass |")
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")
    passes = []
    hyp_stats = {}
    for hid, a, b, desc in hyps:
        st = paired(pcomp[a], pcomp[b], SEEDS, alternative="greater")
        p_bonf = min(1.0, BONF * st["p_raw"])
        mean_ok = st["mean_a"] > st["mean_b"] if np.isfinite(st["mean_a"]) else False
        ok = bool(p_bonf < 0.05 and mean_ok)
        passes.append(ok)
        hyp_stats[hid] = {**st, "p_bonf": p_bonf, "pass": ok, "desc": desc}
        lines.append(
            f"| {hid} | `{a}` > `{b}` | {st['mean_a']:.4f} | {st['mean_b']:.4f} | "
            f"{st['mean_d']:.4f} | {st['p_raw']:.4g} | {p_bonf:.4g} | {'yes' if ok else 'no'} |"
        )
    verdict = "SUPPORTED" if all(passes) else "NOT SUPPORTED"
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Reservoir performance ≠ biological claim about macaque cortex.")
    lines.append("- SLN was not in the downloadable Markov pack; FF/FB used hierarchy-rank proxy.")
    lines.append("")

    (H / "RESULTS_macaque.md").write_text("\n".join(lines) + "\n")
    (OUT / "hyp_stats.json").write_text(json.dumps(hyp_stats, indent=2, default=float))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # composite bars
    ax = axes[0]
    means, sems, labels = [], [], []
    for arm in ALL:
        m, se, n = mean_sem(pcomp, arm)
        if m is None: continue
        means.append(m); sems.append(se); labels.append(arm)
    x = np.arange(len(labels))
    colors = ["#2a6f97" if "MACAQUE" in L and "ER" not in L and "SHUF" not in L and "FRANKEN" not in L
              else "#e76f51" if "ER" in L or L == "DENS" else "#8ab17d" if "FRANKEN" in L
              else "#264653" for L in labels]
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.9, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("composite z (XOR, −NARMA)"); ax.set_title(f"Macaque — {verdict}")
    ax.axhline(0, color="k", lw=0.6)
    # H p-values
    ax = axes[1]
    names = [h[0] for h in hyps]
    pvals = [hyp_stats[h]["p_bonf"] for h in names]
    deltas = [hyp_stats[h]["mean_d"] for h in names]
    cols = ["#2a9d8f" if hyp_stats[h]["pass"] else "#e76f51" for h in names]
    ax.bar(names, deltas, color=cols, alpha=0.9)
    for i, (nm, pv) in enumerate(zip(names, pvals)):
        ax.text(i, deltas[i] + (0.02 if deltas[i] >= 0 else -0.05), f"p_bonf={pv:.3g}",
                ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Δ composite (a−b)"); ax.set_title("Primary H1–H3")
    fig.tight_layout()
    fig.savefig(FIGS / "macaque.png", dpi=140)
    plt.close()
    print("VERDICT", verdict)
    print("wrote", H / "RESULTS_macaque.md", FIGS / "macaque.png")


if __name__ == "__main__":
    main()

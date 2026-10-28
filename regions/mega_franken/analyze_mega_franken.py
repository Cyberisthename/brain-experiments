"""Analyze mega_franken runs → RESULTS_mega_franken.md + figs/mega_franken_*.png"""
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
JF = OUT / "runs.jsonl"
FIGS = Path(__file__).resolve().parents[2] / "figs"
FIGS.mkdir(exist_ok=True)

TASKS = ["MC", "NARMA", "PC_HARD"]
HIGHER = {"MC": True, "PC_HARD": True, "NARMA": False}
SIGMAS = [0.05, 0.2, 0.5]
STACKS_PRIMARY = [
    ("FRANKEN+FINCH", "FINCH"),
    ("FRANKEN+WORM", "WORM"),
    ("MEGA", "MEGA"),
]
PRIMARY_N = 54  # 3 stacks × 2 ctrls × 3 sigma × 3 tasks
SECONDARY_N = 54


def load_rows():
    rows = []
    for line in JF.open():
        r = json.loads(line)
        if r.get("error") or r.get("value") is None:
            continue
        if isinstance(r["value"], float) and (r["value"] != r["value"]):
            continue
        rows.append(r)
    return rows


def arm_for(stack, ctrl, sigma, bridge):
    if stack == "FRANKEN":
        return f"FRANKEN__s{sigma:g}__{bridge}"
    return f"{stack}_{ctrl}__s{sigma:g}__{bridge}"


def pivot(rows):
    """(arm, task) -> {seed: value}"""
    d = defaultdict(dict)
    for r in rows:
        d[(r["arm"], r["task"])][int(r["seed"])] = float(r["value"])
    return d


def paired(a_vals, b_vals):
    seeds = sorted(set(a_vals) & set(b_vals))
    if len(seeds) < 2:
        return dict(n=len(seeds), delta=float("nan"), p_raw=1.0, xs=[], ys=[])
    xs = np.array([a_vals[s] for s in seeds])
    ys = np.array([b_vals[s] for s in seeds])
    diff = xs - ys
    if np.allclose(diff, 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(xs, ys, alternative="two-sided", zero_method="wilcox")
        except Exception:
            p = 1.0
    return dict(n=len(seeds), delta=float(np.mean(diff)), p_raw=float(p),
                mean_a=float(np.mean(xs)), mean_b=float(np.mean(ys)),
                sem_a=float(np.std(xs, ddof=1) / math.sqrt(len(xs))),
                sem_b=float(np.std(ys, ddof=1) / math.sqrt(len(ys))))


def favor(task, delta):
    if not np.isfinite(delta):
        return "na"
    if HIGHER[task]:
        return "REAL" if delta > 0 else ("CTRL" if delta < 0 else "tie")
    return "REAL" if delta < 0 else ("CTRL" if delta > 0 else "tie")


def family(piv, bridge, bonf_n, label):
    results = []
    for stack, tag in STACKS_PRIMARY:
        for sigma in SIGMAS:
            real_arm = arm_for(stack, "REAL", sigma, bridge)
            for ctrl in ("ER", "SHUF"):
                ctrl_arm = arm_for(stack, ctrl, sigma, bridge)
                for task in TASKS:
                    a = piv.get((real_arm, task), {})
                    b = piv.get((ctrl_arm, task), {})
                    st = paired(a, b)
                    st.update(
                        family=label, stack=stack, tag=tag, sigma=sigma, bridge=bridge,
                        ctrl=ctrl, task=task, real_arm=real_arm, ctrl_arm=ctrl_arm,
                    )
                    st["p_bonf"] = min(1.0, st["p_raw"] * bonf_n)
                    st["favor"] = favor(task, st["delta"])
                    st["sig"] = st["p_bonf"] < 0.05
                    results.append(st)
    return results


def means_table(rows, bridge):
    """stack/ctrl/sigma -> task means"""
    buckets = defaultdict(list)
    for r in rows:
        if r["bridge"] != bridge:
            continue
        key = (r["stack"], r["ctrl"], r["sigma"], r["task"])
        buckets[key].append(r["value"])
    out = {}
    for k, vs in buckets.items():
        out[k] = (float(np.mean(vs)), float(np.std(vs, ddof=1) / math.sqrt(len(vs))), len(vs))
    return out


def make_figure(rows, primary):
    # Panel: for each stack tag, MC mean REAL vs ER vs SHUF across sigma (UNIF)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    colors = {"REAL": "#2ca02c", "ER": "#d62728", "SHUF": "#1f77b4"}
    for ax, (stack, tag) in zip(axes, STACKS_PRIMARY):
        for ctrl in ("REAL", "ER", "SHUF"):
            ys, yerr = [], []
            for sigma in SIGMAS:
                arm = arm_for(stack, ctrl, sigma, "UNIF")
                vals = [r["value"] for r in rows if r["arm"] == arm and r["task"] == "MC"]
                ys.append(np.mean(vals) if vals else np.nan)
                yerr.append((np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0)
            ax.errorbar(SIGMAS, ys, yerr=yerr, marker="o", label=ctrl, color=colors[ctrl], lw=1.8)
        ax.set_title(f"{tag} add-on (UNIF)")
        ax.set_xlabel("bridge σ")
        ax.set_xticks(SIGMAS)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MC (↑ better)")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Mega-Frankenstein: real vs random added blocks vs bridge strength", fontsize=11)
    fig.tight_layout()
    p1 = FIGS / "mega_franken_mc_vs_sigma.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # Second fig: REAL−ER Δ MC across sigma for UNIF and REALISTIC
    fig, ax = plt.subplots(figsize=(7, 3.8))
    markers = {"UNIF": "o", "REALISTIC": "s"}
    for bridge in ("UNIF", "REALISTIC"):
        for stack, tag in STACKS_PRIMARY:
            deltas = []
            for sigma in SIGMAS:
                hit = [p for p in (primary if bridge == "UNIF" else family(pivot(rows), bridge, SECONDARY_N, "sec"))
                       if p["stack"] == stack and p["sigma"] == sigma and p["ctrl"] == "ER" and p["task"] == "MC" and p["bridge"] == bridge]
                # recompute cleanly
                piv = pivot(rows)
                st = paired(piv.get((arm_for(stack, "REAL", sigma, bridge), "MC"), {}),
                            piv.get((arm_for(stack, "ER", sigma, bridge), "MC"), {}))
                deltas.append(st["delta"])
            ax.plot(SIGMAS, deltas, marker=markers[bridge], label=f"{tag}/{bridge}", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("bridge σ")
    ax.set_ylabel("Δ MC (REAL − ER)")
    ax.set_title("Does stronger bridging help real wiring beat ER?")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = FIGS / "mega_franken_delta_mc.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    return p1, p2


def verdict(primary, secondary, rows):
    real_wins = [p for p in primary if p["sig"] and p["favor"] == "REAL"]
    ctrl_wins = [p for p in primary if p["sig"] and p["favor"] == "CTRL"]
    # sigma effect: does |mean Δ| grow?
    lines = []
    lines.append(f"Primary (UNIF) REAL wins after Bonferroni×{PRIMARY_N}: **{len(real_wins)} / {len(primary)}**.")
    lines.append(f"Primary CTRL wins after correction: **{len(ctrl_wins)} / {len(primary)}**.")
    sec_real = [p for p in secondary if p["sig"] and p["favor"] == "REAL"]
    lines.append(f"Secondary (REALISTIC) REAL wins: **{len(sec_real)} / {len(secondary)}**.")
    # H2: any change with sigma? compare mean |Δ| at 0.05 vs 0.5 for MC REAL-ER
    piv = pivot(rows)
    def mean_abs_delta(sigma, bridge="UNIF"):
        ds = []
        for stack, _ in STACKS_PRIMARY:
            st = paired(piv.get((arm_for(stack, "REAL", sigma, bridge), "MC"), {}),
                        piv.get((arm_for(stack, "ER", sigma, bridge), "MC"), {}))
            if np.isfinite(st["delta"]):
                ds.append(abs(st["delta"]))
        return float(np.mean(ds)) if ds else float("nan")
    d05, d5 = mean_abs_delta(0.05), mean_abs_delta(0.5)
    lines.append(f"Mean |Δ MC| (REAL−ER, UNIF) at σ=0.05: {d05:.3f}; at σ=0.5: {d5:.3f}.")
    if len(real_wins) == 0 and len(ctrl_wins) == 0:
        honest = ("**Honest verdict: null.** Adding finch / worm / synthetic-octo sections does not "
                  "beat size-matched ER/SHUF replacements after correction at any tested σ under UNIF "
                  "bridges. Stronger bridges did not unlock a Level-2 composition win on these generic tasks.")
    elif len(real_wins) > 0:
        honest = ("**Honest verdict: partial real-wiring signal.** Some REAL-vs-control contrasts survive "
                  f"Bonferroni ({len(real_wins)}). Inspect which stack/σ/task before claiming Level-2.")
    else:
        honest = ("**Honest verdict: controls win.** Random replacements of added blocks beat real wiring "
                  "on the corrected primary family — stacking more real sections does not help beyond size/density.")
    if abs(d5 - d05) < 0.05:
        honest += " Raising σ barely moved the REAL−ER MC gap."
    else:
        honest += f" Raising σ changed mean |Δ MC| by {d5 - d05:+.3f} (exploratory)."
    return "\n".join(lines), honest, real_wins, ctrl_wins


def main():
    rows = load_rows()
    assert rows, "no rows"
    piv = pivot(rows)
    primary = family(piv, "UNIF", PRIMARY_N, "primary")
    secondary = family(piv, "REALISTIC", SECONDARY_N, "secondary")
    p1, p2 = make_figure(rows, primary)
    summary, honest, real_wins, ctrl_wins = verdict(primary, secondary, rows)

    # means markdown
    def fmt_means(bridge):
        lines = ["| Arm | σ | MC | NARMA | PC_HARD | N |", "|---|---:|---:|---:|---:|---:|"]
        for stack in ["FRANKEN", "FRANKEN+FINCH", "FRANKEN+WORM", "MEGA"]:
            ctrls = ["REAL"] if stack == "FRANKEN" else ["REAL", "ER", "SHUF"]
            for ctrl in ctrls:
                for sigma in SIGMAS:
                    arm = arm_for(stack, ctrl, sigma, bridge)
                    cells = []
                    n = None
                    for task in TASKS:
                        vals = [r["value"] for r in rows if r["arm"] == arm and r["task"] == task]
                        if not vals:
                            cells.append("—")
                        else:
                            cells.append(f"{np.mean(vals):.3f}±{np.std(vals, ddof=1)/math.sqrt(len(vals)):.3f}")
                            n = len(vals)
                    label = "FRANKEN" if stack == "FRANKEN" else f"{stack}_{ctrl}"
                    lines.append(f"| `{label}` | {sigma:g} | {cells[0]} | {cells[1]} | {cells[2]} | {n or 0} |")
        return "\n".join(lines)

    def fmt_contrasts(results, title):
        lines = [f"### {title}", "",
                 "| Stack | σ | Ctrl | Task | Δ (REAL−ctrl) | p_raw | p_bonf | favor | call |",
                 "|---|---:|---|---|---:|---:|---:|---|---|"]
        for r in results:
            call = "REAL*" if (r["sig"] and r["favor"] == "REAL") else (
                "CTRL*" if (r["sig"] and r["favor"] == "CTRL") else "n.s.")
            lines.append(
                f"| {r['tag']} | {r['sigma']:g} | {r['ctrl']} | {r['task']} | "
                f"{r['delta']:+.4f} | {r['p_raw']:.4g} | {r['p_bonf']:.4g} | {r['favor']} | {call} |"
            )
        return "\n".join(lines)

    n_rows = len(rows)
    n_arms = len({r["arm"] for r in rows})
    sizes = {}
    for r in rows:
        sizes.setdefault(r["stack"], r["n"])

    md = f"""# RESULTS: Mega-Frankenstein

**Date:** 2026-10-28  
**Prereg:** `PREREGISTRATION_mega_franken.md`  
**Rows:** {n_rows} across {n_arms} arms (eval seeds 0–7).  
**Delays:** FIXED + real H01 myelin on human-pre (franken_realmyelin).  
**Octopus block:** **SYNTHETIC** (`octo_vl_synthetic.npz`) — not a real cephalopod connectome.  
**Sizes:** {json.dumps(sizes)}  
**Figures:** `{p1.name}`, `{p2.name}`

## Summary

{summary}

{honest}

## Means ± SEM — UNIF bridges

{fmt_means("UNIF")}

## Means ± SEM — REALISTIC bridges

{fmt_means("REALISTIC")}

{fmt_contrasts(primary, f"Primary family (UNIF, Bonferroni ×{PRIMARY_N})")}

{fmt_contrasts(secondary, f"Secondary family (REALISTIC, Bonferroni ×{SECONDARY_N})")}

## Notes

- Primary isolates whether **added** real blocks beat ER/SHUF replacements; mouse/human/fly stay real in every arm.
- MEGA > FRANKEN alone is size-confounded and not a primary claim.
- Alpha per arm tuned on seeds 90–92 (MC), locked in `out/alpha_choice.json`.
"""
    (H / "RESULTS_mega_franken.md").write_text(md)
    (OUT / "analyze_summary.json").write_text(json.dumps(dict(
        n_rows=n_rows, n_arms=n_arms, sizes=sizes,
        n_primary_real_wins=len(real_wins), n_primary_ctrl_wins=len(ctrl_wins),
        honest=honest, figs=[str(p1), str(p2)],
        primary_hits=[{k: r[k] for k in ("tag","sigma","ctrl","task","delta","p_bonf","favor")} for r in real_wins + ctrl_wins],
    ), indent=2))
    print(honest)
    print("wrote", H / "RESULTS_mega_franken.md")
    print("figs", p1, p2)


if __name__ == "__main__":
    main()

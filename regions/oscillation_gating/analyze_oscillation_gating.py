#!/usr/bin/env python3
"""Analyze oscillation_gating runs; write RESULTS + figure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

ARMS = ["COHERENT", "INCOHERENT", "NO_GATE", "HUB", "RAND_MATCH"]
PRIMARY_ARMS = ["COHERENT"]  # compared to INCOHERENT, NO_GATE, RAND_MATCH
CONTRASTS = [
    ("COHERENT", "INCOHERENT"),
    ("COHERENT", "NO_GATE"),
    ("COHERENT", "RAND_MATCH"),
]
PRIMARY = [("XOR", "greater"), ("CHAR", "less")]
ALPHA = 0.05
N_PRIMARY = 6

def load_runs():
    rows = []
    p = OUT / "runs.jsonl"
    if not p.exists():
        raise SystemExit("missing out/runs.jsonl")
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def paired_wilcoxon_p(a, b, alternative="greater"):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    try:
        from scipy.stats import wilcoxon
        try:
            return float(wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)
        except TypeError:
            p2 = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            diff = a - b
            if alternative == "greater":
                return p2 / 2 if diff.mean() > 0 else 1 - p2 / 2
            if alternative == "less":
                return p2 / 2 if diff.mean() < 0 else 1 - p2 / 2
            return p2
        except ValueError:
            # all zeros
            return 1.0
    except Exception:
        diff = a - b
        nz = diff[diff != 0]
        if len(nz) == 0:
            return 1.0
        from math import comb
        k = int((nz > 0).sum()) if alternative != "less" else int((nz < 0).sum())
        p = sum(comb(len(nz), i) for i in range(k, len(nz) + 1)) / (2 ** len(nz))
        return float(p)

def main():
    rows = load_runs()
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["task"]), []).append(r)

    lines = []
    lines.append("# RESULTS — oscillation_gating (FAST PILOT)\n")
    lines.append("Prereg: `PREREG_oscillation_gating.md` (frozen 2026-10-29 13:57:30 EDT).\n")
    lines.append("Primary (Bonferroni α=0.05/6): COHERENT vs {INCOHERENT, NO_GATE, RAND_MATCH} on XOR↑ and CHAR↓. Seeds 1100–1105 (n=6 pilot).\n")

    lines.append("\n## Summary (mean ± sd over seeds)\n")
    lines.append("| arm | XOR acc ↑ | CHAR CE ↓ | n_params | nnz |\n|---|---|---|---|---|")
    table = {}
    for arm in ARMS:
        def stats(task):
            xs = [r["metric"] for r in by.get((arm, task), [])]
            if not xs:
                return float("nan"), float("nan"), []
            return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, xs
        xm, xs, xv = stats("XOR")
        cm, cs, cv = stats("CHAR")
        npar = by.get((arm, "XOR"), [{}])[0].get("n_params", "?")
        nnz = by.get((arm, "XOR"), [{}])[0].get("nnz", "?")
        lines.append(f"| {arm} | {xm:.3f}±{xs:.3f} | {cm:.3f}±{cs:.3f} | {npar} | {nnz} |")
        table[arm] = dict(XOR=(xm, xs, xv), CHAR=(cm, cs, cv))

    lines.append("\n## Primary tests (paired Wilcoxon, Bonferroni)\n")
    lines.append("| comparison | task | direction | p raw | p adj | pass? |\n|---|---|---|---|---|---|")

    def seed_metric(arm, task):
        rs = by.get((arm, task), [])
        return {r["seed"]: r["metric"] for r in rs}

    verdicts = []
    ctc_ok = True
    overall_ok = True
    for a, b in CONTRASTS:
        for task, alt in PRIMARY:
            sa, sb = seed_metric(a, task), seed_metric(b, task)
            seeds = sorted(set(sa) & set(sb))
            aa = [sa[s] for s in seeds]
            bb = [sb[s] for s in seeds]
            if len(aa) < 3:
                p = 1.0
            else:
                p = paired_wilcoxon_p(aa, bb, alternative=alt)
            padj = min(1.0, p * N_PRIMARY)
            ok = padj < ALPHA
            verdicts.append(ok)
            if a == "COHERENT" and b == "INCOHERENT" and not ok:
                ctc_ok = False
            if not ok:
                overall_ok = False
            lines.append(
                f"| {a} vs {b} | {task} | {alt} | {p:.4g} | {padj:.4g} | {'YES' if ok else 'no'} |"
            )

    n_pass = sum(verdicts)
    if ctc_ok and all(
        # recompute ctc pair passes
        True for _ in [0]
    ):
        # ctc_ok already tracks COHERENT vs INCOHERENT both tasks
        pass
    # Fix ctc_ok properly
    ctc_ok = True
    for task, alt in PRIMARY:
        sa, sb = seed_metric("COHERENT", task), seed_metric("INCOHERENT", task)
        seeds = sorted(set(sa) & set(sb))
        aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
        p = paired_wilcoxon_p(aa, bb, alternative=alt) if len(aa) >= 3 else 1.0
        if min(1.0, p * N_PRIMARY) >= ALPHA:
            ctc_ok = False

    if overall_ok:
        verd = "SUPPORTED (pilot) — COHERENT beats INCOHERENT, NO_GATE, and RAND_MATCH on BOTH primary endpoints after Bonferroni."
    elif ctc_ok:
        verd = "PARTIAL (pilot) — CTC contrast supported (COHERENT > INCOHERENT on both), but not all primary contrasts pass."
    else:
        verd = "NOT SUPPORTED (pilot) — COHERENT does not beat INCOHERENT on both primary endpoints after Bonferroni."
    lines.append(f"\n**Primary verdict:** {verd} ({n_pass}/{len(verdicts)} comparisons significant).\n")

    # secondary HUB
    lines.append("\n## Secondary: HUB (descriptive)\n")
    for other in ("COHERENT", "RAND_MATCH"):
        for task, alt in PRIMARY:
            # HUB vs other: for XOR higher better means we report HUB vs other with greater if asking whether HUB wins
            sa, sb = seed_metric("HUB", task), seed_metric(other, task)
            seeds = sorted(set(sa) & set(sb))
            aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
            if len(aa) < 3:
                continue
            p = paired_wilcoxon_p(aa, bb, alternative=alt)
            lines.append(f"- HUB vs {other} on {task} ({alt}): raw p={p:.4g} (uncorrected)\n")

    lines.append("\n## Learning curves\nSee `figs/oscillation_gating.png` (mean metric vs epoch ± SEM).\n")
    lines.append("\n## Honesty notes\n")
    lines.append("- Fast pilot: n=6 seeds (1100–1105), XOR+CHAR only, N=128, epochs≈25.\n")
    lines.append("- Hyperparams tuned on seeds 90–91 only (COHERENT + INCOHERENT).\n")
    lines.append("- Gate mean-normalized so COHERENT vs INCOHERENT differ in timing, not mean gain.\n")
    lines.append("- Null allowed; low power — provisional.\n")

    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS_oscillation_gating.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    print("wrote RESULTS_oscillation_gating.md")
    print("VERDICT:", verd)
    print("CTC_BEATS_INCOHERENT:", ctc_ok)
    print("N_PASS:", n_pass)

    # means table for parent
    print("MEANS_TABLE")
    for arm in ARMS:
        xm, xs, _ = table[arm]["XOR"]
        cm, cs, _ = table[arm]["CHAR"]
        print(f"  {arm}: XOR={xm:.3f}±{xs:.3f} CHAR={cm:.3f}±{cs:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("no matplotlib", e)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = dict(
        COHERENT="#1b9e77", INCOHERENT="#d95f02", NO_GATE="#7570b3",
        HUB="#e7298a", RAND_MATCH="#666666",
    )
    for ax, task, ylabel in zip(axes, ["XOR", "CHAR"], ["XOR accuracy ↑", "CHAR CE ↓"]):
        for arm in ARMS:
            rs = by.get((arm, task), [])
            if not rs or not rs[0].get("curve"):
                continue
            curves = [r["curve"] for r in rs if r.get("curve")]
            if not curves:
                continue
            L = min(len(c) for c in curves)
            M = np.array([[c[i]["metric"] for i in range(L)] for c in curves], float)
            mean = M.mean(0); sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(L)
            ep = np.arange(L)
            ax.plot(ep, mean, label=arm, color=colors.get(arm, None), lw=1.8)
            ax.fill_between(ep, mean - sem, mean + sem, color=colors.get(arm, "gray"), alpha=0.15)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(task); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("oscillation_gating fast pilot", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "oscillation_gating.png", dpi=140)
    print("wrote figs/oscillation_gating.png")

if __name__ == "__main__":
    main()

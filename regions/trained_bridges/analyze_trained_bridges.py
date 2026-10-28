"""Analyze trained_bridges; write RESULTS + figure. Prereg stats."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
OUT = H / "out"
FIGS = H / "figs"
FIGS.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"

TASKS = ["MC", "NARMA", "PC_HARD", "XOR_DELAY"]
WIRINGS = ["FRANKEN", "ER", "SHUF"]
HIGHER_BETTER = {"MC": True, "NARMA": False, "PC_HARD": True, "XOR_DELAY": True}
PRIMARY = ["PC_HARD", "XOR_DELAY"]
SECONDARY = ["MC", "NARMA"]


def wilcoxon_signed(diffs):
    """Two-sided Wilcoxon signed-rank; all-zero -> p=1. No scipy required fallback."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 1.0, 0
    if np.allclose(d, 0):
        return 1.0, len(d)
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zeros
        res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.pvalue), int(len(d))
    except Exception:
        # sign-flip permutation p (exact for n<=8-ish)
        n = len(d)
        obs = abs(d.mean())
        # 2^n enum if n<=12
        if n <= 12:
            count = 0
            total = 1 << n
            for mask in range(total):
                s = 0.0
                for i in range(n):
                    s += d[i] if (mask >> i) & 1 else -d[i]
                if abs(s / n) >= obs - 1e-15:
                    count += 1
            return count / total, n
        return float("nan"), n


def load_rows():
    rows = []
    if not JF.exists():
        return rows
    for line in open(JF):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or r.get("value") is None:
            continue
        if not np.isfinite(r.get("value", float("nan"))):
            continue
        rows.append(r)
    return rows


def index(rows):
    """(wiring, cond, task, seed) -> value"""
    m = {}
    meta = {}
    for r in rows:
        w, c, t, s = r["wiring"], r["cond"], r["task"], int(r["seed"])
        m[(w, c, t, s)] = float(r["value"])
        meta[(w, c, t, s)] = r
    return m, meta


def gain(m, wiring, task, seed):
    """Positive = training helped (direction-normalized)."""
    tr = m.get((wiring, "ES", task, seed))
    fx = m.get((wiring, "FIXED", task, seed))
    if tr is None or fx is None:
        return float("nan")
    if HIGHER_BETTER[task]:
        return tr - fx
    return fx - tr  # NARMA


def main():
    rows = load_rows()
    m, meta = index(rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    lines = []
    lines.append("# Trained bridges results")
    lines.append("")
    lines.append("Prereg: `PREREGISTRATION_trained_bridges.md`.")
    lines.append(f"Rows kept: {len(rows)}. Seeds: {seeds}.")
    lines.append("")
    # means table
    lines.append("## Mean ± sem by arm × task")
    lines.append("")
    lines.append("| Arm | " + " | ".join(TASKS) + " |")
    lines.append("|---| " + " | ".join(["---:"] * len(TASKS)) + " |")
    mean_tbl = {}
    for w in WIRINGS:
        for c in ("FIXED", "ES"):
            arm = f"{w}_{c}"
            cells = []
            mean_tbl[arm] = {}
            for t in TASKS:
                vals = [m[(w, c, t, s)] for s in seeds if (w, c, t, s) in m]
                if not vals:
                    cells.append("—")
                    mean_tbl[arm][t] = None
                else:
                    mu, sem = float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                    mean_tbl[arm][t] = (mu, sem, len(vals))
                    cells.append(f"{mu:.4f}±{sem:.4f}")
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Direction: MC / PC_HARD / XOR_DELAY ↑ better; NARMA ↓ better.")
    lines.append("")

    # Gains
    lines.append("## Training gains (direction-normalized: ⊕ = ES helped)")
    lines.append("")
    lines.append("| Wiring | " + " | ".join(TASKS) + " |")
    lines.append("|---| " + " | ".join(["---:"] * len(TASKS)) + " |")
    gains = {w: {t: [] for t in TASKS} for w in WIRINGS}
    for w in WIRINGS:
        cells = []
        for t in TASKS:
            gs = [gain(m, w, t, s) for s in seeds]
            gs_f = [g for g in gs if np.isfinite(g)]
            gains[w][t] = gs_f
            if not gs_f:
                cells.append("—")
            else:
                mu = float(np.mean(gs_f))
                sem = float(np.std(gs_f, ddof=1) / math.sqrt(len(gs_f))) if len(gs_f) > 1 else 0.0
                cells.append(f"{mu:+.4f}±{sem:.4f}")
        lines.append(f"| `{w}` | " + " | ".join(cells) + " |")
    lines.append("")

    # Primary / secondary contrasts
    def contrast_block(title, task_list, bonf_n):
        out = [f"## {title} (Bonferroni ×{bonf_n})", ""]
        out.append("| Contrast | Task | mean Δgain | p_raw | p_bonf | sig |")
        out.append("|---|---|---:|---:|---:|---|")
        results = []
        for ctrl in ("ER", "SHUF"):
            for t in task_list:
                d = []
                for s in seeds:
                    gR = gain(m, "FRANKEN", t, s)
                    gC = gain(m, ctrl, t, s)
                    if np.isfinite(gR) and np.isfinite(gC):
                        d.append(gR - gC)
                p, n = wilcoxon_signed(d)
                mu = float(np.mean(d)) if d else float("nan")
                results.append((f"gain_FRANKEN − gain_{ctrl}", t, mu, p, n, d))
        for name, t, mu, p, n, d in results:
            pb = min(1.0, p * bonf_n) if np.isfinite(p) else float("nan")
            sig = "YES" if (np.isfinite(pb) and pb < 0.05) else "no"
            out.append(f"| {name} | {t} | {mu:+.4f} | {p:.4g} | {pb:.4g} | {sig} |")
        out.append("")
        return out, results

    prim_lines, prim = contrast_block("Primary family — held-out tasks", PRIMARY, 4)
    sec_lines, sec = contrast_block("Secondary family — train tasks", SECONDARY, 4)
    lines.extend(prim_lines)
    lines.extend(sec_lines)

    # Exploratory FIXED baseline
    lines.append("## Exploratory: FIXED FRANKEN vs FIXED ER / SHUF (raw p)")
    lines.append("")
    lines.append("| Contrast | Task | mean Δ | p_raw |")
    lines.append("|---|---|---:|---:|")
    for ctrl in ("ER", "SHUF"):
        for t in TASKS:
            d = []
            for s in seeds:
                a = m.get(("FRANKEN", "FIXED", t, s))
                b = m.get((ctrl, "FIXED", t, s))
                if a is None or b is None:
                    continue
                if HIGHER_BETTER[t]:
                    d.append(a - b)
                else:
                    d.append(b - a)  # ⊕ = FRANKEN better
            p, n = wilcoxon_signed(d)
            mu = float(np.mean(d)) if d else float("nan")
            lines.append(f"| FIXED FRANKEN − FIXED {ctrl} (⊕ better) | {t} | {mu:+.4f} | {p:.4g} |")
    lines.append("")

    # ES train curves summary
    lines.append("## ES training (from cached logs on ES rows)")
    lines.append("")
    f_first, f_last, n_br, rhos = [], [], [], []
    seen = set()
    for r in rows:
        if r.get("cond") != "ES":
            continue
        key = (r["wiring"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        if r.get("train_f_first") is not None:
            f_first.append(r["train_f_first"])
            f_last.append(r["train_f_last"])
        if r.get("n_bridge") is not None:
            n_br.append(r["n_bridge"])
        if r.get("final_rho") is not None:
            rhos.append(r["final_rho"])
    if f_first:
        lines.append(f"- Bridge edges: {int(np.mean(n_br)) if n_br else '?'} (mean)")
        lines.append(f"- Fitness first→last gen (mean): {np.mean(f_first):+.4f} → {np.mean(f_last):+.4f}")
        lines.append(f"- Final ρ (mean): {np.mean(rhos):.4f}" if rhos else "")
    else:
        lines.append("- No ES train logs in rows.")
    lines.append("")

    # Verdict
    lines.append("## Honest verdict")
    lines.append("")
    prim_sig = [x for x in prim if np.isfinite(x[3]) and min(1.0, x[3] * 4) < 0.05]
    sec_sig = [x for x in sec if np.isfinite(x[3]) and min(1.0, x[3] * 4) < 0.05]
    if prim_sig:
        lines.append(
            f"**Primary:** {len(prim_sig)}/4 gain-contrasts survive Bonferroni. "
            "Real blocks benefit more from bridge training than random on held-out tasks — "
            "a Level-2-style composition signal *if* absolute gains are also meaningful."
        )
    else:
        lines.append(
            "**Primary:** no gain-contrast survives Bonferroni on PC_HARD / XOR_DELAY. "
            "Training bridges does **not** unlock a real-wiring advantage over ER/SHUF on "
            "held-out tasks under this budget (H3 / Level-1 continues)."
        )
    if sec_sig:
        lines.append(f"**Secondary:** {len(sec_sig)}/4 train-task contrasts significant after correction.")
    else:
        lines.append("**Secondary:** no train-task gain-contrast survives Bonferroni.")
    # Did ES help anyone?
    help_lines = []
    for w in WIRINGS:
        for t in TASKS:
            gs = gains[w][t]
            if gs and np.mean(gs) > 0:
                help_lines.append(f"{w}/{t}: {np.mean(gs):+.4f}")
    if help_lines:
        lines.append("Raw mean gains ⊕: " + "; ".join(help_lines[:12]))
    else:
        lines.append("Mean training gains were ≤0 for all wiring×task (ES did not help on average).")
    lines.append("")
    lines.append("Limits: antithetic ES G=10 (20 fitness rollouts), short fitness horizon, "
                 "bridges-only, frozen blocks. Not biological learning.")
    lines.append("")
    lines.append("No GitHub push.")

    text = "\n".join(lines) + "\n"
    (H / "RESULTS_trained_bridges.md").write_text(text)
    print(text)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Left: gains by wiring for primary tasks
        ax = axes[0]
        x = np.arange(len(PRIMARY))
        width = 0.25
        for i, w in enumerate(WIRINGS):
            mus, sems = [], []
            for t in PRIMARY:
                gs = gains[w][t]
                mus.append(np.mean(gs) if gs else 0)
                sems.append(np.std(gs, ddof=1) / math.sqrt(len(gs)) if len(gs) > 1 else 0)
            ax.bar(x + (i - 1) * width, mus, width, yerr=sems, label=w, capsize=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(PRIMARY)
        ax.set_ylabel("gain (⊕ = ES helped)")
        ax.set_title("Held-out training gains")
        ax.legend(fontsize=8)
        # Right: mean scores FIXED vs ES for FRANKEN on all tasks
        ax = axes[1]
        x = np.arange(len(TASKS))
        for i, c in enumerate(("FIXED", "ES")):
            mus, sems = [], []
            for t in TASKS:
                info = mean_tbl.get(f"FRANKEN_{c}", {}).get(t)
                if info is None:
                    mus.append(0); sems.append(0)
                else:
                    mus.append(info[0]); sems.append(info[1])
            ax.bar(x + (i - 0.5) * 0.35, mus, 0.35, yerr=sems, label=c, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS, rotation=15)
        ax.set_title("FRANKEN absolute scores")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "trained_bridges_gains.png", dpi=140)
        fig.savefig(OUT / "trained_bridges_gains.png", dpi=140)
        plt.close()
        print("Wrote", FIGS / "trained_bridges_gains.png")
    except Exception as e:
        print("Figure failed:", e)

    (OUT / "analyze.done").write_text("ok\n")


if __name__ == "__main__":
    main()

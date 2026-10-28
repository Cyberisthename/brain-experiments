"""Analyze C. elegans multi-worm results; write RESULTS + figure."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

H = Path(__file__).resolve().parent
OUT = H / "out"
FIGS = Path("/workspace/flybrain/figs")
FIGS.mkdir(parents=True, exist_ok=True)
JF = OUT / "runs.jsonl"
MANIFEST = json.loads((H / "data" / "manifest.json").read_text())
META = {m["id"]: m for m in MANIFEST}
WORMS_WIT = [m["id"] for m in MANIFEST if m["id"].startswith("W")]
WORMS_ALL = [m["id"] for m in MANIFEST]
WIRINGS_CTRL = ["ER", "SHUF", "DENS"]
DELAYS = ["NOD", "DEL"]
TASKS = ["MC", "NARMA", "PC_HARD"]
HIGHER = {"MC": True, "PC_HARD": True, "NARMA": False, "CHEMO": True}


def load_rows():
    rows = []
    for line in open(JF):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        if r.get("value") is None or (isinstance(r.get("value"), float) and math.isnan(r["value"])):
            # keep chemo skips marked
            if r.get("skip"):
                rows.append(r)
            continue
        rows.append(r)
    return rows


def mannwhitney(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def wilcoxon_signed(xs):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    if len(xs) < 3:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        # zero_method wilcox; if all zero, nan
        if np.allclose(xs, 0):
            return 1.0
        return float(wilcoxon(xs).pvalue)
    except Exception:
        return float("nan")


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x[m], y[m])
        return float(r.correlation), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def real_better(task, real_med, ctrl_med):
    if not (np.isfinite(real_med) and np.isfinite(ctrl_med)):
        return None
    if HIGHER[task]:
        return real_med > ctrl_med
    return real_med < ctrl_med


def effect(task, real_med, ctrl_med):
    """Signed effect: positive => REAL better."""
    if not (np.isfinite(real_med) and np.isfinite(ctrl_med)):
        return float("nan")
    if HIGHER[task]:
        return float(real_med - ctrl_med)
    return float(ctrl_med - real_med)


def main():
    rows = load_rows()
    main_rows = [r for r in rows if r.get("test") == "main"]
    chemo_rows = [r for r in rows if r.get("test") == "chemo"]
    # index: (worm, wiring, delay, task) -> values
    vals = defaultdict(list)
    mc_n = defaultdict(list)
    for r in main_rows:
        key = (r["worm"], r["wiring"], r["delay_mode"], r["task"])
        vals[key].append(r["value"])
        if r["task"] == "MC" and r.get("mc_over_n") is not None:
            mc_n[key].append(r["mc_over_n"])

    lines = []
    lines.append("# Results: C. elegans individual-variation connectomes")
    lines.append("")
    lines.append("Prereg: `PREREGISTRATION_celegans.md`. Sources: `SOURCES.md`.")
    lines.append("")
    lines.append("## Worm graphs")
    lines.append("")
    lines.append("| Worm | Stage | Sex | N | nnz | source |")
    lines.append("|---|---|---|---:|---:|---|")
    for m in MANIFEST:
        lines.append(f"| {m['id']} | {m['stage']} | {m['sex']} | {m['N']} | {m['nnz']} | {m['source']} |")
    lines.append("")
    lines.append("E/I: unknown → unsigned weights. Witvliet gap junctions incomplete (paper note).")
    lines.append("")

    # Per-worm table
    lines.append("## Q1 — Per-worm REAL vs controls (median over 8 seeds)")
    lines.append("")
    lines.append("MC ↑, NARMA ↓, PC_HARD ↑. Effect = signed so **positive ⇒ REAL better**.")
    lines.append("")

    # Collect meta effects
    # key (task, delay, ctrl) -> list of (worm, effect, real_med, ctrl_med, p_raw)
    meta = defaultdict(list)
    per_worm_tables = []

    for worm in WORMS_ALL:
        stage = META[worm]["stage"]
        block = [f"### {worm} ({stage}, N={META[worm]['N']})", ""]
        block.append("| Task | Delay | REAL | ER | SHUF | DENS | REAL−bestCTRL effect |")
        block.append("|---|---|---:|---:|---:|---:|---:|")
        for task in TASKS:
            for delay in DELAYS:
                rkey = (worm, "REAL", delay, task)
                rv = vals.get(rkey, [])
                rmed = float(np.nanmedian(rv)) if rv else float("nan")
                cells = [f"{rmed:.3f}" if np.isfinite(rmed) else "nan"]
                effects = []
                for ctrl in WIRINGS_CTRL:
                    cv = vals.get((worm, ctrl, delay, task), [])
                    cmed = float(np.nanmedian(cv)) if cv else float("nan")
                    cells.append(f"{cmed:.3f}" if np.isfinite(cmed) else "nan")
                    _, p = mannwhitney(rv, cv)
                    eff = effect(task, rmed, cmed)
                    meta[(task, delay, ctrl)].append(
                        dict(worm=worm, effect=eff, real=rmed, ctrl=cmed, p=p,
                             stage_ord=META[worm]["stage_ord"], wit=worm.startswith("W"))
                    )
                    effects.append(eff)
                # show mean effect vs three controls
                me = float(np.nanmean(effects)) if effects else float("nan")
                block.append(
                    f"| {task} | {delay} | " + " | ".join(cells) + f" | {me:+.3f} |"
                )
                if task == "MC":
                    rmn = mc_n.get(rkey, [])
                    if rmn:
                        block.append(
                            f"| MC/N | {delay} | {float(np.nanmedian(rmn)):.4f} |  |  |  |  |"
                        )
        per_worm_tables.append("\n".join(block))

    lines.extend(["", "### Summary table (Witvliet + Cook)", ""])
    # compact: for each worm x delay, MC REAL vs ER effect
    lines.append("| Worm | Stage | Delay | MC_REAL | MC_ER | MC_effect | NARMA_eff(vs ER) | PC_eff(vs ER) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for worm in WORMS_ALL:
        for delay in DELAYS:
            rowbits = [worm, META[worm]["stage"], delay]
            for task, ctrl in [("MC", "ER"), ("MC", "ER"), ("MC", "ER"), ("NARMA", "ER"), ("PC_HARD", "ER")]:
                pass
            r_mc = float(np.nanmedian(vals.get((worm, "REAL", delay, "MC"), [np.nan])))
            e_mc = float(np.nanmedian(vals.get((worm, "ER", delay, "MC"), [np.nan])))
            eff_mc = effect("MC", r_mc, e_mc)
            r_na = float(np.nanmedian(vals.get((worm, "REAL", delay, "NARMA"), [np.nan])))
            e_na = float(np.nanmedian(vals.get((worm, "ER", delay, "NARMA"), [np.nan])))
            eff_na = effect("NARMA", r_na, e_na)
            r_pc = float(np.nanmedian(vals.get((worm, "REAL", delay, "PC_HARD"), [np.nan])))
            e_pc = float(np.nanmedian(vals.get((worm, "ER", delay, "PC_HARD"), [np.nan])))
            eff_pc = effect("PC_HARD", r_pc, e_pc)
            lines.append(
                f"| {worm} | {META[worm]['stage']} | {delay} | {r_mc:.2f} | {e_mc:.2f} | {eff_mc:+.2f} | {eff_na:+.3f} | {eff_pc:+.3f} |"
            )

    lines.append("")
    lines.append("### Per-worm detail")
    lines.append("")
    lines.extend(per_worm_tables)

    # Meta-analysis
    lines.append("")
    lines.append("## Q1 meta — cross-worm consistency (Witvliet W1–W8 as replicates)")
    lines.append("")
    lines.append("Bonferroni over 18 primary meta-contrasts (3 tasks × 2 delays × 3 controls).")
    lines.append("")
    lines.append("| Task | Delay | Ctrl | n_worms | n_REAL_better | mean effect | Wilcoxon p | p_Bonf | consistent? |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    n_family = 18
    verdicts = []
    for task in TASKS:
        for delay in DELAYS:
            for ctrl in WIRINGS_CTRL:
                ents = [e for e in meta[(task, delay, ctrl)] if e["wit"]]
                effs = [e["effect"] for e in ents]
                n_better = sum(1 for e in effs if np.isfinite(e) and e > 0)
                n = sum(1 for e in effs if np.isfinite(e))
                mean_e = float(np.nanmean(effs)) if n else float("nan")
                p = wilcoxon_signed(effs)
                p_b = min(1.0, p * n_family) if np.isfinite(p) else float("nan")
                consistent = (n_better >= 6) and np.isfinite(p_b) and (p_b < 0.05)
                verdicts.append(consistent)
                lines.append(
                    f"| {task} | {delay} | {ctrl} | {n} | {n_better} | {mean_e:+.3f} | {p:.4g} | {p_b:.4g} | {'YES' if consistent else 'no'} |"
                )
    n_yes = sum(verdicts)
    lines.append("")
    lines.append(f"**Consistency verdict:** {n_yes}/18 primary meta-contrasts meet the prereg bar (≥6/8 same sign + p_Bonf<0.05).")
    if n_yes == 0:
        lines.append("No wiring-specific edge is reproducible across individuals under this bar.")
    lines.append("")

    # Q2 development
    lines.append("## Q2 — Development (Witvliet effect vs stage)")
    lines.append("")
    lines.append("| Task | Delay | Ctrl | Spearman ρ (effect vs stage) | p |")
    lines.append("|---|---|---|---:|---:|")
    for task in TASKS:
        for delay in DELAYS:
            for ctrl in WIRINGS_CTRL:
                ents = [e for e in meta[(task, delay, ctrl)] if e["wit"]]
                rho, p = spearman([e["stage_ord"] for e in ents], [e["effect"] for e in ents])
                lines.append(f"| {task} | {delay} | {ctrl} | {rho:+.3f} | {p:.4g} |")
    lines.append("")
    lines.append("Exploratory (raw p). No claim without a clear monotonic pattern.")
    lines.append("")

    # Q3 chemo
    lines.append("## Q3 — Chemotaxis-like bonus")
    lines.append("")
    chemo_vals = defaultdict(list)
    for r in chemo_rows:
        if r.get("skip"):
            continue
        chemo_vals[(r["worm"], r["delay_mode"], r["io_mode"])].append(r["value"])
    lines.append("| Worm | Delay | REAL_IO R² | RAND_IO R² | Δ | p_raw |")
    lines.append("|---|---|---:|---:|---:|---:|")
    chemo_wins = 0
    chemo_tests = 0
    for worm in WORMS_ALL:
        for delay in DELAYS:
            a = chemo_vals.get((worm, delay, "REAL"), [])
            b = chemo_vals.get((worm, delay, "RAND"), [])
            am = float(np.nanmedian(a)) if a else float("nan")
            bm = float(np.nanmedian(b)) if b else float("nan")
            _, p = mannwhitney(a, b)
            dlt = am - bm if np.isfinite(am) and np.isfinite(bm) else float("nan")
            if np.isfinite(dlt):
                chemo_tests += 1
                if dlt > 0 and np.isfinite(p) and p < 0.05:
                    chemo_wins += 1
            lines.append(
                f"| {worm} | {delay} | {am:.3f} | {bm:.3f} | {dlt:+.3f} | {p:.4g} |"
            )
    lines.append("")
    lines.append(f"Exploratory: {chemo_wins}/{chemo_tests} worm×delay cells with REAL_IO>RAND_IO and raw p<0.05.")
    lines.append("")

    lines.append("## Honest takeaway")
    lines.append("")
    lines.append(
        "This experiment asks whether *C. elegans* real wiring beats fair random controls "
        "**consistently across individuals**. See the consistency verdict above. "
        "Small N (~165–300) and unsigned weights limit absolute MC; MC/N is reported. "
        "Delays (FIXED) are expected to help both REAL and controls (prior project lesson)."
    )
    lines.append("")

    (H / "RESULTS_celegans.md").write_text("\n".join(lines))
    print("Wrote RESULTS_celegans.md")

    # Figure: effect per worm across development for MC/NARMA/PC × NOD/DEL vs ER
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib missing", e)
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    stage_map = {1: "L1", 2: "L2", 3: "L3", 4: "adult"}
    colors = {"ER": "#d62728", "SHUF": "#ff7f0e", "DENS": "#2ca02c"}
    for col, task in enumerate(TASKS):
        for row, delay in enumerate(DELAYS):
            ax = axes[row, col]
            for ctrl in WIRINGS_CTRL:
                ents = [e for e in meta[(task, delay, ctrl)] if e["wit"]]
                # group by worm order W1..W8
                xs, ys = [], []
                for e in ents:
                    # jitter by ctrl
                    wid = e["worm"]
                    xi = int(wid[1:]) + {"ER": -0.15, "SHUF": 0.0, "DENS": 0.15}[ctrl]
                    xs.append(xi)
                    ys.append(e["effect"])
                ax.scatter(xs, ys, s=36, alpha=0.85, label=ctrl, c=colors[ctrl])
            ax.axhline(0, color="k", lw=0.8, alpha=0.5)
            ax.set_title(f"{task} / {delay}")
            if col == 0:
                ax.set_ylabel("effect (REAL better > 0)")
            if row == 1:
                ax.set_xlabel("Witvliet worm # (1–4 L1, 5 L2, 6 L3, 7–8 adult)")
            ax.set_xticks(range(1, 9))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("C. elegans: REAL−control effect across individuals / development", y=1.02)
    fig.tight_layout()
    outp = FIGS / "celegans_effect_by_worm.png"
    fig.savefig(outp, dpi=140, bbox_inches="tight")
    print("Wrote", outp)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze SECONDARY FLY followup vs NO_GATE / MYELIN_ONLY / CALLOSUM.

Writes RESULTS_FLY_FOLLOWUP.md — does not overwrite main RESULTS.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from analyze_oscillation_gating import (
    load_runs, paired_wilcoxon_p, seed_metric, stats_for, OUT, HERE, FIGS,
)

COMPARE = ["FLY", "NO_GATE", "MYELIN_ONLY", "CALLOSUM"]
DIRS = [("XOR", "greater"), ("CHAR", "less"), ("NARMA", "less")]
NOTES = {
    "FLY": "EI_SBM Dale; delay=1; no gate",
    "NO_GATE": "modular; delay=2 inter; no gate",
    "MYELIN_ONLY": "modular; H01 RULE_FF_DEPTH delays",
    "CALLOSUM": "hemi split + homotopic bridge; split XOR",
}


def main():
    fly_path = OUT / "fly_followup_runs.jsonl"
    if not fly_path.exists():
        raise SystemExit("missing out/fly_followup_runs.jsonl — run --followup-fly first")

    main_rows = load_runs()
    fly_rows = [json.loads(l) for l in fly_path.read_text().splitlines() if l.strip()]
    by = {}
    for r in main_rows + fly_rows:
        by.setdefault((r["arm"], r["task"]), []).append(r)

    meta = {}
    mp = OUT / "fly_followup_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())

    lines = []
    lines.append("# RESULTS — SECONDARY FLY FOLLOWUP (separate from primary CTC eval)\n")
    lines.append(
        "Labeled secondary run. Same oscillation_gating harness, seeds 1100–1111, "
        "epochs/hparams as main. FLY = fly CX / EI_SBM wiring from `bio_wired_net` "
        "(FRAC_E≈0.592, EI block dens scaled to match modular nnz), Dale softplus×esign, "
        "delay=1, no oscillation gate.\n"
    )
    lines.append(
        "**Not part of the primary preregistered CTC analysis.** Compared fairly against "
        "NO_GATE (modular), MYELIN_ONLY, and CALLOSUM from the main eval.\n"
    )
    if meta:
        lines.append(
            f"Meta: fly_nnz={meta.get('fly_nnz')} modular_nnz={meta.get('modular_nnz')} "
            f"dens_scale={meta.get('dens_scale')} frac_E={meta.get('frac_E')}\n"
        )

    lines.append("\n## Summary (mean ± sd)\n")
    lines.append("| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | n_params | nnz | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    table = {}
    for arm in COMPARE:
        xm, xs, _ = stats_for(by, arm, "XOR")
        cm, cs, _ = stats_for(by, arm, "CHAR")
        nm, ns, _ = stats_for(by, arm, "NARMA")
        rs = by.get((arm, "XOR"), [{}])
        npar = rs[0].get("n_params", "?") if rs else "?"
        nnz = rs[0].get("nnz", "?") if rs else "?"
        lines.append(
            f"| {arm} | {xm:.3f}±{xs:.3f} | {cm:.3f}±{cs:.3f} | {nm:.3f}±{ns:.3f} | "
            f"{npar} | {nnz} | {NOTES.get(arm, '')} |"
        )
        table[arm] = dict(XOR=(xm, xs), CHAR=(cm, cs), NARMA=(nm, ns))

    lines.append("\n## Pairwise vs FLY (paired Wilcoxon, uncorrected raw p)\n")
    for b in ("NO_GATE", "MYELIN_ONLY", "CALLOSUM"):
        for task, alt in DIRS:
            sa, sb = seed_metric(by, "FLY", task), seed_metric(by, b, task)
            seeds = sorted(set(sa) & set(sb))
            if len(seeds) < 3:
                lines.append(f"- FLY vs {b} on {task}: insufficient paired seeds\n")
                continue
            aa = [sa[s] for s in seeds]
            bb = [sb[s] for s in seeds]
            p = paired_wilcoxon_p(aa, bb, alternative=alt)
            ma, mb = float(np.mean(aa)), float(np.mean(bb))
            better = (ma > mb) if alt == "greater" else (ma < mb)
            lines.append(
                f"- FLY vs {b} on {task} (FLY {alt}): {ma:.3f} vs {mb:.3f}, "
                f"raw p={p:.4g}, FLY_better={better}\n"
            )

    def m(arm, task):
        return stats_for(by, arm, task)[0]

    lines.append("\n## Plain comparison\n")
    lines.append(
        f"- XOR: FLY={m('FLY','XOR'):.3f}, NO_GATE={m('NO_GATE','XOR'):.3f}, "
        f"MYELIN_ONLY={m('MYELIN_ONLY','XOR'):.3f}, CALLOSUM={m('CALLOSUM','XOR'):.3f}\n"
    )
    lines.append(
        f"- CHAR: FLY={m('FLY','CHAR'):.3f}, NO_GATE={m('NO_GATE','CHAR'):.3f}, "
        f"MYELIN_ONLY={m('MYELIN_ONLY','CHAR'):.3f}, CALLOSUM={m('CALLOSUM','CHAR'):.3f}\n"
    )
    lines.append(
        f"- NARMA: FLY={m('FLY','NARMA'):.3f}, NO_GATE={m('NO_GATE','NARMA'):.3f}, "
        f"MYELIN_ONLY={m('MYELIN_ONLY','NARMA'):.3f}, CALLOSUM={m('CALLOSUM','NARMA'):.3f}\n"
    )
    lines.append(
        "\nNote: CALLOSUM uses split-input XOR (different task formulation) — "
        "CHAR/NARMA are the cleaner cross-architecture compares; XOR vs CALLOSUM is "
        "descriptive only.\n"
    )
    lines.append("\nNull allowed. Secondary / exploratory — not Bonferroni-corrected.\n")

    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS_FLY_FOLLOWUP.md").write_text(text)
    (OUT / "fly_followup_verdict.json").write_text(json.dumps({
        "label": "SECONDARY_FLY_FOLLOWUP",
        "means": {arm: {t: table[arm][t][0] for t in ("XOR", "CHAR", "NARMA")} for arm in table},
        "meta": meta,
    }, indent=2))
    print("wrote RESULTS_FLY_FOLLOWUP.md")
    for arm in COMPARE:
        if arm in table:
            xm, xs = table[arm]["XOR"]
            cm, cs = table[arm]["CHAR"]
            nm, ns = table[arm]["NARMA"]
            print(f"  {arm}: XOR={xm:.3f}±{xs:.3f} CHAR={cm:.3f}±{cs:.3f} NARMA={nm:.3f}±{ns:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {
            "FLY": "#e41a1c", "NO_GATE": "#7570b3",
            "MYELIN_ONLY": "#a6761d", "CALLOSUM": "#e6ab02",
        }
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, task, ylabel in zip(
            axes, ["XOR", "CHAR", "NARMA"],
            ["XOR accuracy ↑", "CHAR CE ↓", "NARMA NRMSE ↓"],
        ):
            for arm in COMPARE:
                rs = by.get((arm, task), [])
                curves = [r["curve"] for r in rs if r.get("curve")]
                if not curves:
                    continue
                L = min(len(c) for c in curves)
                M = np.array([[c[i]["metric"] for i in range(L)] for c in curves], float)
                mean_c = M.mean(0)
                sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(L)
                ep = np.arange(L)
                ax.plot(ep, mean_c, label=arm, color=colors.get(arm), lw=1.8)
                ax.fill_between(ep, mean_c - sem, mean_c + sem, color=colors.get(arm, "gray"), alpha=0.12)
            ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(task); ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle("SECONDARY FLY followup vs NO_GATE / MYELIN_ONLY / CALLOSUM", fontsize=11)
        fig.tight_layout()
        FIGS.mkdir(exist_ok=True)
        fig.savefig(FIGS / "oscillation_gating_fly_followup.png", dpi=140)
        print("wrote figs/oscillation_gating_fly_followup.png")
    except Exception as e:
        print("fly fig skip", e)


if __name__ == "__main__":
    main()

"""Write RESULTS_hybrid.md from out/runs.jsonl. Bonferroni on prereg contrasts."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

H = Path(__file__).resolve().parent
OUT = H / "out"
SRC = json.loads((H / "data" / "SOURCE.json").read_text())
d = pd.read_json(OUT / "runs.jsonl", lines=True)
lines = []
lines.append("# Fly–mouse hybrid results\n\n")
lines.append("Prereg: `PREREGISTRATION_hybrid.md`.\n\n")
lines.append("## Data source (plain)\n\n")
lines.append(f"- **Source:** {SRC.get('source')}\n")
lines.append(f"- **Underlying:** {SRC.get('underlying')}\n")
lines.append(f"- **Datastack / materialization:** {SRC.get('datastack')} / v{SRC.get('materialization')}\n")
lines.append(f"- **Download date:** {SRC.get('download_date')}\n")
lines.append(f"- **CAVE token required:** {SRC.get('cave_token_required')} "
             f"(public CAVE needs a free token; used Zenodo static export instead)\n")
lines.append(f"- **Proofread module:** N={SRC.get('n_proofread_module')} neurons, "
             f"{SRC.get('n_edges_condensed')} condensed edges, "
             f"E={SRC.get('n_E')} I={SRC.get('n_I')}\n")
lines.append(f"- **Criteria:** {SRC.get('criteria')}\n")
lines.append(f"- **Weights:** {SRC.get('weight')}\n")
lines.append(f"- **Layers:** {SRC.get('layer_counts')}\n\n")

comp = d[d.task != "maze"].copy() if len(d) else d
maze = d[d.task == "maze"].copy() if len(d) else d
if len(comp):
    lines.append("## Compress tasks (mean over seeds)\n\n")
    g = comp.groupby(["task", "arm"]).value.agg(["mean", "std", "count"]).reset_index()
    lines.append(g.to_markdown(index=False) + "\n\n")

    def mw_row(a_arm, b_arm, task="MC", col="value"):
        sub = comp[comp.task == task] if task else comp
        a = sub[sub.arm == a_arm][col].dropna().to_numpy()
        b = sub[sub.arm == b_arm][col].dropna().to_numpy()
        if len(a) and len(b):
            u = mannwhitneyu(a, b, alternative="two-sided")
            return a.mean() - b.mean(), u.pvalue, len(a), len(b)
        return None, None, len(a), len(b)

    lines.append("### MC contrasts (legacy Bonferroni ×3)\n\n")
    lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
    for other in ("ER", "SHUF", "FLY"):
        delta, p, na, nb = mw_row("HYBRID", other)
        if p is not None:
            lines.append(f"| HYBRID vs {other} | {delta:+.4f} | {p:.4g} | {min(1.0, p*3):.4g} | {na}/{nb} |\n")
        else:
            lines.append(f"| HYBRID vs {other} | — | — | — | {na}/{nb} |\n")
    lines.append("\n### MC layered contrasts (Bonferroni ×2)\n\n")
    lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
    for a_arm, b_arm, k in (("HYBRID_LAYERED", "RANDOM_LAYERED", 2),
                            ("HYBRID_LAYERED", "FLY", 2)):
        delta, p, na, nb = mw_row(a_arm, b_arm)
        if p is not None:
            lines.append(f"| {a_arm} vs {b_arm} | {delta:+.4f} | {p:.4g} | {min(1.0, p*k):.4g} | {na}/{nb} |\n")
        else:
            lines.append(f"| {a_arm} vs {b_arm} | — | — | — | {na}/{nb} |\n")
    lines.append("\n### MC Hebb contrast (Bonferroni ×1)\n\n")
    lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
    delta, p, na, nb = mw_row("HYBRID_LAYERED+HEBB", "RANDOM_LAYERED+HEBB")
    if p is not None:
        lines.append(f"| HYBRID_LAYERED+HEBB vs RANDOM_LAYERED+HEBB | {delta:+.4f} | {p:.4g} | {p:.4g} | {na}/{nb} |\n")
    else:
        lines.append(f"| HYBRID_LAYERED+HEBB vs RANDOM_LAYERED+HEBB | — | — | — | {na}/{nb} |\n")
    lines.append("\n")

if len(maze):
    lines.append("## Maze (v2 learner; mouse z for non-FLY)\n\n")
    g = maze.groupby("arm").agg(success=("success", "mean"), threat=("threat_hit_loom", "mean"),
                                steps=("steps_to_food", "mean"), sec=("sec", "mean")).reset_index()
    lines.append(g.to_markdown(index=False) + "\n\n")

lines.append("## Predictions (prereg)\n\n")
lines.append("- H1 HYBRID ≥ ER on MC — see legacy MC contrasts.\n")
lines.append("- H2 HYBRID ≥ SHUF on MC — see legacy MC contrasts.\n")
lines.append("- H3 Maze exploratory.\n")
lines.append("- H4 HYBRID_LAYERED beats RANDOM_LAYERED on MC only if hierarchy matters.\n")
lines.append("- H5 Hebb applied equally; mouse margin under Hebb still about wiring.\n")
lines.append("- H6 Primary layered news contrast: HYBRID_LAYERED vs RANDOM_LAYERED on MC.\n\n")
lines.append("Raw: `out/runs.jsonl`. Module: `data/mouse_module.npz`. Source: `data/SOURCE.json`.\n")
(H / "RESULTS_hybrid.md").write_text("".join(lines))
print("wrote RESULTS_hybrid.md")

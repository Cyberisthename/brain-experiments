"""Write RESULTS_hybrid_human.md from out/runs.jsonl. Bonferroni on prereg contrasts."""
import json
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

H = Path(__file__).resolve().parent
OUT = H / "out"
SRC = json.loads((H / "data" / "SOURCE.json").read_text())
d = pd.read_json(OUT / "runs.jsonl", lines=True) if (OUT / "runs.jsonl").exists() else pd.DataFrame()
lines = []
lines.append("# Fly–human hybrid results\n\n")
lines.append("Prereg: `PREREGISTRATION_hybrid_human.md`.\n\n")
lines.append("## Data / caveat\n\n")
ms = SRC.get("module_stats", {})
rel = SRC.get("reliability_vs_microns", {})
lines.append(f"- **Source:** H01 via {SRC.get('derived_edge_list', {}).get('repo')} + official somas.csv\n")
lines.append(f"- **Module:** N={ms.get('n_neurons')} nnz={ms.get('n_edges_nnz')} "
             f"E={ms.get('n_E')} I={ms.get('n_I')} layers={ms.get('layer_counts')}\n")
lines.append(f"- **Proofread overlap:** {ms.get('n_proofread_104_overlap')} "
             f"(public H01 proofread ≈104; this module is **auto-segmented**)\n")
lines.append(f"- **Vs MICrONS:** {rel.get('verdict')}\n")
lines.append(f"- **Slab:** {rel.get('slab_caveat')}\n\n")

if len(d) == 0:
    lines.append("No runs yet.\n")
    (H / "RESULTS_hybrid_human.md").write_text("".join(lines))
    raise SystemExit(0)

comp = d.copy()
lines.append("## Compress tasks (mean over seeds)\n\n")
g = comp.groupby(["task", "arm"]).value.agg(["mean", "std", "count"]).reset_index()
lines.append(g.to_markdown(index=False) + "\n\n")

def mw_row(a_arm, b_arm, task="MC"):
    sub = comp[comp.task == task]
    a = sub[sub.arm == a_arm].value.dropna().to_numpy()
    b = sub[sub.arm == b_arm].value.dropna().to_numpy()
    if len(a) and len(b):
        u = mannwhitneyu(a, b, alternative="two-sided")
        return a.mean() - b.mean(), u.pvalue, len(a), len(b)
    return None, None, len(a), len(b)

lines.append("### MC contrasts (legacy Bonferroni ×3)\n\n")
lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
for other in ("ER", "SHUF", "FLY"):
    delta, p, na, nb = mw_row("HUMAN_HYBRID", other)
    if p is not None:
        lines.append(f"| HUMAN_HYBRID vs {other} | {delta:+.4f} | {p:.4g} | {min(1.0, p*3):.4g} | {na}/{nb} |\n")
    else:
        lines.append(f"| HUMAN_HYBRID vs {other} | — | — | — | {na}/{nb} |\n")

lines.append("\n### MC layered contrasts (Bonferroni ×2)\n\n")
lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
for a_arm, b_arm, k in (("HUMAN_LAYERED", "RANDOM_LAYERED", 2), ("HUMAN_LAYERED", "FLY", 2)):
    delta, p, na, nb = mw_row(a_arm, b_arm)
    if p is not None:
        lines.append(f"| {a_arm} vs {b_arm} | {delta:+.4f} | {p:.4g} | {min(1.0, p*k):.4g} | {na}/{nb} |\n")
    else:
        lines.append(f"| {a_arm} vs {b_arm} | — | — | — | {na}/{nb} |\n")

lines.append("\n### MC Hebb contrast (Bonferroni ×1)\n\n")
lines.append("| contrast | Δ mean | p_raw | p_bonf | n |\n|---|---:|---:|---:|---:|\n")
delta, p, na, nb = mw_row("HUMAN_LAYERED+HEBB", "RANDOM_LAYERED+HEBB")
if p is not None:
    lines.append(f"| HUMAN_LAYERED+HEBB vs RANDOM_LAYERED+HEBB | {delta:+.4f} | {p:.4g} | {p:.4g} | {na}/{nb} |\n")
else:
    lines.append(f"| HUMAN_LAYERED+HEBB vs RANDOM_LAYERED+HEBB | — | — | — | {na}/{nb} |\n")

lines.append("\n## Predictions (prereg)\n\n")
lines.append("- H1 HUMAN_HYBRID ≥ ER on MC.\n")
lines.append("- H2 HUMAN_HYBRID ≥ SHUF on MC.\n")
lines.append("- H3 HUMAN_LAYERED vs RANDOM_LAYERED (hierarchy).\n")
lines.append("- H4 equal-Hebb contrast.\n")
lines.append("- H5 expect noisier margins than mouse (auto-seg).\n")
(H / "RESULTS_hybrid_human.md").write_text("".join(lines))
print("wrote RESULTS_hybrid_human.md", len(d), "rows")

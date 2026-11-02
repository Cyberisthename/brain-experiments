#!/usr/bin/env python3
"""Analyze callosum_scaled runs.jsonl → RESULTS.md (prereg contrasts)."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = OUT / "runs.jsonl"

ARMS = [
    "REAL_SC", "DEGREE_SHUFFLE", "WEIGHT_SHUFFLE",
    "SPARSE_BRIDGE", "NO_BRIDGE", "DENSE", "AGCC",
]
TASKS = ["XOR_HARD", "CHAR", "NARMA", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"]
HIGHER = {"XOR_HARD", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"}

# Primary family from PREREG (applied at largest completed N for TRANSFER)
PRIMARY_TEMPLATES = [
    ("TRANSFER", "REAL_SC", "DEGREE_SHUFFLE"),
    ("SPLIT_BIND", "REAL_SC", "DEGREE_SHUFFLE"),
    ("DELAYED_MATCH", "REAL_SC", "DEGREE_SHUFFLE"),
    ("TRANSFER", "REAL_SC", "NO_BRIDGE"),
    ("NARMA", "REAL_SC", "DEGREE_SHUFFLE"),
    ("TRANSFER", "REAL_SC", "WEIGHT_SHUFFLE"),
]


def wilcoxon_paired(xa, xb, higher_better):
    from scipy.stats import wilcoxon
    d = np.asarray(xa, float) - np.asarray(xb, float)
    if not higher_better:
        d = -d
    if np.allclose(d, 0):
        return 1.0, float(np.mean(d))
    try:
        res = wilcoxon(d, alternative="greater", zero_method="wilcox")
        return float(res.pvalue), float(np.mean(d))
    except Exception:
        return 1.0, float(np.mean(d))


def main():
    rows = []
    if RUNS.exists():
        for line in RUNS.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by = defaultdict(list)
    n_labels = sorted({int(r.get("n_label", -1)) for r in rows})
    for r in rows:
        by[(r["arm"], r["task"], int(r.get("n_label", -1)))].append(r)

    # largest N with complete TRANSFER for REAL_SC and DEGREE_SHUFFLE (8 seeds)
    primary_N = None
    for n in sorted(n_labels, reverse=True):
        ra = {r["seed"] for r in by[("REAL_SC", "TRANSFER", n)]}
        rb = {r["seed"] for r in by[("DEGREE_SHUFFLE", "TRANSFER", n)]}
        if len(ra & rb) >= 8:
            primary_N = n
            break
        if len(ra & rb) >= 3 and primary_N is None:
            primary_N = n  # incomplete fallback for reporting

    lines = []
    lines.append("# callosum_scaled RESULTS")
    lines.append("")
    lines.append(f"Runs loaded: {len(rows)}")
    lines.append(f"N labels present: {n_labels}")
    lines.append(f"Primary claim N (largest with TRANSFER paired seeds): **{primary_N}**")
    lines.append("")

    for n in n_labels:
        lines.append(f"## Means at N_label={n}")
        lines.append("")
        lines.append("| arm | XOR_HARD ↑ | CHAR ↓ | NARMA ↓ | TRANSFER ↑ | SPLIT_BIND ↑ | DELAYED_MATCH ↑ | n |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            cells = []
            ns = []
            for task in TASKS:
                vals = [r["metric"] for r in by[(arm, task, n)] if np.isfinite(r["metric"])]
                ns.append(len(vals))
                cells.append(f"{np.mean(vals):.4f}" if vals else "—")
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {min(ns) if ns else 0} |")
        lines.append("")

    m = len(PRIMARY_TEMPLATES)
    alpha = 0.05 / m if m else 0.05
    lines.append(f"## Primary contrasts at N={primary_N} (Bonferroni α={alpha:.4f}, m={m})")
    lines.append("")
    lines.append("| contrast | task | mean_a | mean_b | pairedΔ(a better+) | p | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    any_supported = False
    transfer_map_supported = False
    if primary_N is None:
        lines.append("| — | — | — | — | — | — | INCOMPLETE |")
    else:
        for task, a, b in PRIMARY_TEMPLATES:
            ra = {r["seed"]: r["metric"] for r in by[(a, task, primary_N)] if np.isfinite(r["metric"])}
            rb = {r["seed"]: r["metric"] for r in by[(b, task, primary_N)] if np.isfinite(r["metric"])}
            seeds = sorted(set(ra) & set(rb))
            if len(seeds) < 3:
                lines.append(f"| {a} vs {b} | {task} | — | — | — | — | INCOMPLETE (n={len(seeds)}) |")
                continue
            xa = [ra[s] for s in seeds]; xb = [rb[s] for s in seeds]
            hb = task in HIGHER
            p, d = wilcoxon_paired(xa, xb, hb)
            verdict = "SUPPORTED" if p < alpha else "NOT SUPPORTED"
            if verdict == "SUPPORTED":
                any_supported = True
                if task == "TRANSFER" and b == "DEGREE_SHUFFLE":
                    transfer_map_supported = True
            lines.append(
                f"| {a} vs {b} | {task} | {np.mean(xa):.4f} | {np.mean(xb):.4f} | {d:.4f} | {p:.4g} | {verdict} |"
            )
    lines.append("")
    if transfer_map_supported:
        overall = "SUPPORTED (primary claim: REAL_SC > DEGREE_SHUFFLE on TRANSFER)"
    elif any_supported:
        overall = "PARTIAL (some primary contrasts passed; map claim on TRANSFER did not)"
    else:
        overall = "NOT SUPPORTED (no primary contrast passed)"
    lines.append(f"**Overall primary:** {overall}")
    lines.append("")
    lines.append("Data: Griffa et al. 2019 Zenodo 2872624 + Caminiti/FA delay priors (local only).")
    lines.append("Prereg: PREREG_callosum_scaled.md")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (OUT / "verdict.json").write_text(json.dumps(dict(
        primary_N=primary_N, overall=overall, n_runs=len(rows),
        transfer_map_supported=transfer_map_supported,
    ), indent=2))
    print(text)


if __name__ == "__main__":
    main()

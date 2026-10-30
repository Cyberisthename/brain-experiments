#!/usr/bin/env python3
"""Analyze human_wholebrain runs.jsonl → RESULTS.md (prereg contrasts)."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = OUT / "runs.jsonl"

ARMS = [
    "HUMAN_SC", "HUMAN_SC_MYELIN", "DEGREE_PRESERVING_SHUFFLE",
    "WEIGHT_SHUFFLE", "ER_RANDOM", "MYELIN_ONLY", "INDIVIDUAL",
]
TASKS = ["XOR_HARD", "XOR2BIT", "CHAR", "NARMA"]
HIGHER = {"XOR_HARD", "XOR2BIT"}

# Primary contrasts from PREREG (task, arm_a, arm_b, a_should_beat_b)
PRIMARY = [
    ("XOR_HARD", "HUMAN_SC_MYELIN", "DEGREE_PRESERVING_SHUFFLE"),
    ("NARMA", "HUMAN_SC_MYELIN", "DEGREE_PRESERVING_SHUFFLE"),
    ("XOR_HARD", "HUMAN_SC", "ER_RANDOM"),
    ("XOR_HARD", "HUMAN_SC", "WEIGHT_SHUFFLE"),
    ("XOR_HARD", "HUMAN_SC_MYELIN", "MYELIN_ONLY"),
    ("NARMA", "HUMAN_SC_MYELIN", "DEGREE_PRESERVING_SHUFFLE"),  # listed once in set; keep unique below
]
# unique primary tests
_seen = set(); PRIMARY_U = []
for t, a, b in PRIMARY:
    k = (t, a, b)
    if k not in _seen:
        _seen.add(k); PRIMARY_U.append(k)


def wilcoxon_paired(xa, xb, higher_better):
    from scipy.stats import wilcoxon
    d = np.asarray(xa, float) - np.asarray(xb, float)
    if not higher_better:
        d = -d  # positive d means a better
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
    for r in rows:
        by[(r["arm"], r["task"])].append(r)

    lines = []
    lines.append("# human_wholebrain RESULTS")
    lines.append("")
    lines.append(f"Runs loaded: {len(rows)}")
    lines.append("")
    lines.append("## Means (metric)")
    lines.append("")
    lines.append("| arm | XOR_HARD ↑ | XOR2BIT ↑ | CHAR ↓ | NARMA ↓ | n/seed |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        cells = []
        ns = []
        for task in TASKS:
            vals = [r["metric"] for r in by[(arm, task)]]
            ns.append(len(vals))
            if vals:
                cells.append(f"{np.mean(vals):.4f}")
            else:
                cells.append("—")
        lines.append(f"| {arm} | " + " | ".join(cells) + f" | {min(ns) if ns else 0} |")
    lines.append("")

    m = len(PRIMARY_U)
    alpha = 0.05 / m if m else 0.05
    lines.append(f"## Primary contrasts (Bonferroni α={alpha:.4f}, m={m})")
    lines.append("")
    lines.append("| contrast | task | mean_a | mean_b | pairedΔ(a better+) | p | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    any_supported = False
    for task, a, b in PRIMARY_U:
        ra = {r["seed"]: r["metric"] for r in by[(a, task)]}
        rb = {r["seed"]: r["metric"] for r in by[(b, task)]}
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
        lines.append(
            f"| {a} vs {b} | {task} | {np.mean(xa):.4f} | {np.mean(xb):.4f} | {d:.4f} | {p:.4g} | {verdict} |"
        )
    lines.append("")
    lines.append(f"**Overall primary:** {'at least one SUPPORTED' if any_supported else 'NOT SUPPORTED (no primary contrast passed)'}")
    lines.append("")

    # INDIVIDUAL spread
    lines.append("## INDIVIDUAL spread (secondary)")
    lines.append("")
    for task in TASKS:
        vals = [r["metric"] for r in by[("INDIVIDUAL", task)]]
        if vals:
            lines.append(f"- {task}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")
    lines.append("")
    lines.append("Data: Griffa et al. 2019 Zenodo 2872624 + Lausanne consensus fiber lengths (local only).")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

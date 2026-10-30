#!/usr/bin/env python3
"""Analyze realistic_callosum runs → RESULTS + verdict.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ARMS = [
    "REAL_CALLOSUM", "REAL_CALLOSUM_SHUFFLED", "INDIVIDUAL_CALLOSUM",
    "CALLOSUM", "RAND_BRIDGE", "NO_BRIDGE", "DENSE_BRIDGE",
]
TASKS = ["XOR", "CHAR", "NARMA"]
SEEDS = list(range(2100, 2112))

# Primary family (Bonferroni ×6) from PREREG
PRIMARY = [
    ("REAL_CALLOSUM", "REAL_CALLOSUM_SHUFFLED", "XOR", "greater"),
    ("REAL_CALLOSUM", "CALLOSUM", "XOR", "greater"),
    ("REAL_CALLOSUM", "RAND_BRIDGE", "XOR", "greater"),
    ("REAL_CALLOSUM", "NO_BRIDGE", "XOR", "greater"),
    ("REAL_CALLOSUM", "REAL_CALLOSUM_SHUFFLED", "NARMA", "less"),
    ("REAL_CALLOSUM", "NO_BRIDGE", "NARMA", "less"),
]


def load_runs():
    rows = []
    p = OUT / "runs.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def metric_map(rows):
    m = {}
    for r in rows:
        m[(r["arm"], r["task"], r["seed"])] = float(r["metric"])
    return m


def paired(a, b, task, direction, mmap):
    xa, xb = [], []
    for s in SEEDS:
        ka, kb = (a, task, s), (b, task, s)
        if ka in mmap and kb in mmap:
            xa.append(mmap[ka]); xb.append(mmap[kb])
    xa, xb = np.asarray(xa), np.asarray(xb)
    if len(xa) < 3:
        return dict(n=len(xa), mean_a=float("nan"), mean_b=float("nan"),
                    p_raw=1.0, pass_=False)
    diff = xa - xb
    if np.allclose(diff, 0):
        p = 1.0
    else:
        # wilcoxon alternative
        alt = direction  # greater / less
        try:
            res = wilcoxon(xa, xb, alternative=alt, zero_method="wilcox")
            p = float(res.pvalue)
        except ValueError:
            p = 1.0
    return dict(n=len(xa), mean_a=float(xa.mean()), mean_b=float(xb.mean()),
                sd_a=float(xa.std(ddof=1)), sd_b=float(xb.std(ddof=1)),
                p_raw=p, pass_=False)


def main():
    rows = load_runs()
    mmap = metric_map(rows)
    lines = []
    lines.append("# RESULTS — realistic_callosum\n")
    lines.append("Prereg: `PREREG_realistic_callosum.md` (frozen 2026-10-30 09:51:38 EDT).\n")
    lines.append("Left-only readout; delayed hemispheric XOR (d=5). Seeds 2100–2111 (n=12).\n")
    lines.append("Primary family: 6 paired Wilcoxon, Bonferroni α=0.05/6.\n\n")

    # summary table
    lines.append("## Summary (mean ± sd)\n\n")
    lines.append("| arm | XOR ↑ | CHAR ↓ | NARMA ↓ | n_bridge (typ) |\n|---|---|---|---|---|\n")
    bridge_typ = {}
    for r in rows:
        bridge_typ.setdefault(r["arm"], r.get("n_bridge", -1))
    for arm in ARMS:
        cells = []
        for task in TASKS:
            vals = [mmap[(arm, task, s)] for s in SEEDS if (arm, task, s) in mmap]
            if vals:
                cells.append(f"{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}")
            else:
                cells.append("—")
        lines.append(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {bridge_typ.get(arm, '—')} |\n")

    # primary tests
    lines.append("\n## Primary tests (Bonferroni ×6)\n\n")
    lines.append("| comparison | task | dir | mean_a | mean_b | p raw | p adj | pass? |\n|---|---|---|---|---|---|---|---|\n")
    alpha = 0.05 / 6
    results = []
    n_pass = 0
    for a, b, task, direction in PRIMARY:
        r = paired(a, b, task, direction, mmap)
        p_adj = min(1.0, r["p_raw"] * 6)
        ok = r["p_raw"] < alpha and (
            (direction == "greater" and r["mean_a"] > r["mean_b"]) or
            (direction == "less" and r["mean_a"] < r["mean_b"])
        )
        # use adjusted threshold equivalently
        ok = (p_adj < 0.05) and (
            (direction == "greater" and r["mean_a"] > r["mean_b"]) or
            (direction == "less" and r["mean_a"] < r["mean_b"])
        )
        r["p_adj"] = p_adj; r["pass"] = ok
        r["cmp"] = f"{a} vs {b}"; r["task"] = task; r["dir"] = direction
        results.append(r)
        if ok:
            n_pass += 1
        lines.append(
            f"| {a} vs {b} | {task} | {direction} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
            f"{r['p_raw']:.4g} | {p_adj:.4g} | {'YES' if ok else 'no'} |\n"
        )

    h1 = results[0]["pass"]
    h4 = results[3]["pass"]
    supported = h1 and h4
    lines.append(
        f"\n**Primary verdict:** {'SUPPORTED' if supported else 'NOT SUPPORTED'} — "
        f"REAL beats SHUFFLED on XOR: {h1}; REAL beats NO_BRIDGE on XOR: {h4}. "
        f"({n_pass}/6 family tests significant after Bonferroni).\n"
    )

    # secondary CHAR
    lines.append("\n## Secondary (raw p)\n\n")
    for b in ("REAL_CALLOSUM_SHUFFLED", "CALLOSUM", "RAND_BRIDGE", "NO_BRIDGE", "DENSE_BRIDGE"):
        r = paired("REAL_CALLOSUM", b, "CHAR", "less", mmap)
        lines.append(
            f"- REAL vs {b} CHAR (less): {r['mean_a']:.3f} vs {r['mean_b']:.3f}, raw p={r['p_raw']:.4g}\n"
        )
    # individual spread
    ind = [mmap[("INDIVIDUAL_CALLOSUM", "XOR", s)] for s in SEEDS if ("INDIVIDUAL_CALLOSUM", "XOR", s) in mmap]
    real = [mmap[("REAL_CALLOSUM", "XOR", s)] for s in SEEDS if ("REAL_CALLOSUM", "XOR", s) in mmap]
    if ind and real:
        lines.append(
            f"\n- INDIVIDUAL XOR mean±sd={np.mean(ind):.3f}±{np.std(ind, ddof=1):.3f}; "
            f"REAL group={np.mean(real):.3f}±{np.std(real, ddof=1):.3f}\n"
        )
    nob = [mmap[("NO_BRIDGE", "XOR", s)] for s in SEEDS if ("NO_BRIDGE", "XOR", s) in mmap]
    if nob:
        lines.append(f"- NO_BRIDGE XOR (bridge-required check): {np.mean(nob):.3f}±{np.std(nob, ddof=1):.3f} (expect near chance ~0.5)\n")

    lines.append("\n## Honesty\n\nNull allowed. Numbers from `out/runs.jsonl` only. Connectome data not committed.\n")

    text = "".join(lines)
    (HERE / "RESULTS_realistic_callosum.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)
    verdict = dict(
        supported=supported, n_pass_family=n_pass, n_family=6,
        h1_real_gt_shuf_xor=h1, h4_real_gt_nobridge_xor=h4,
        tests=[{k: v for k, v in r.items() if k != "pass_"} for r in results],
    )
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(text)
    print("verdict", verdict["supported"], flush=True)


if __name__ == "__main__":
    main()

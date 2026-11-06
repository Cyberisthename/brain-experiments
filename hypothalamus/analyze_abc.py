#!/usr/bin/env python3
"""Summarize hypothalamus A/B/C jsonl with planned contrasts + Bonferroni."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JSONL = HERE / "results" / "abc_runs.jsonl"
HIGHER = {"XOR_HARD", "TRANSFER"}

# Locked in HPARAMS_FROZEN.md
PLANNED = [
    ("A", "HOME_O_ERR", "RAND_MOD", "XOR_HARD"),
    ("A", "HOME_O_ERR", "BASE", "XOR_HARD"),
    ("A", "HOME_O_ERR", "RAND_MOD", "NARMA"),
    ("A", "HOME_O_ERR", "BASE", "NARMA"),
    ("B", "RIVAL", "SINGLE", "XOR_HARD"),
    ("B", "RIVAL", "SHUF_RIVAL", "XOR_HARD"),
    ("B", "RIVAL", "SINGLE", "TRANSFER"),
    ("B", "RIVAL", "SHUF_RIVAL", "TRANSFER"),
    ("C", "VOL", "SYN", "NARMA"),
    ("C", "VOL", "BASE", "NARMA"),
    ("C", "VOL", "SYN", "XOR_HARD"),
    ("C", "VOL", "BASE", "XOR_HARD"),
]
ALPHA = 0.05 / len(PLANNED)


def wilcoxon_signed(x, y):
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(x) - np.asarray(y)
        if np.allclose(d, 0):
            return 0.0, 1.0
        res = wilcoxon(x, y, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def effect(task, a_vals, b_vals):
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    if task in HIGHER:
        return float(np.mean(a - b))
    return float(np.mean(b - a))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=str(JSONL))
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()
    jsonl = Path(args.jsonl)
    rows = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("metric") is not None and not r.get("poisoned"):
            rows.append(r)
    by = {(r["feature"], r["task"], r["seed"]): r["metric"] for r in rows}
    features = sorted({r["feature"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    means = []
    lines = []
    lines.append("# Hypothalamus A/B/C results\n")
    lines.append("**Time zone:** America/New_York.\n")
    lines.append(f"Rows={len(rows)} features={features} tasks={tasks} seeds={seeds}\n")
    lines.append(f"Bonferroni α={ALPHA:.6f} over {len(PLANNED)} planned contrasts.\n")
    lines.append("Wilcoxon n=8 floor≈0.0078 > α → confirmatory bar unreachable; mark exploratory.\n")
    lines.append("\n## Means ± std\n")
    lines.append("| Arm | XOR_HARD ↑ | NARMA ↓ | CHAR ↓ | TRANSFER ↑ |")
    lines.append("|---|---:|---:|---:|---:|")

    for feat in ["BASE", "HOME_O_ERR", "RAND_MOD", "RIVAL", "SINGLE", "SHUF_RIVAL", "VOL", "SYN"]:
        if feat not in features:
            continue
        cells = []
        for task in ["XOR_HARD", "NARMA", "CHAR", "TRANSFER"]:
            vals = [by[(feat, task, s)] for s in seeds if (feat, task, s) in by]
            if not vals:
                cells.append("—")
                continue
            m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cells.append(f"{m:.4f}±{sd:.4f}")
            means.append(dict(feature=feat, task=task, n=len(vals), mean=m, std=sd))
        lines.append(f"| {feat} | " + " | ".join(cells) + " |")

    lines.append("\n## Planned contrasts\n")
    lines.append("| Test | A vs B | Task | effect (+better for A) | p | survives α? | verdict |")
    lines.append("|---|---|---|---:|---:|---|---|")
    contrasts = []
    for test, fa, fb, task in PLANNED:
        paired_a, paired_b = [], []
        for s in seeds:
            if (fa, task, s) in by and (fb, task, s) in by:
                paired_a.append(by[(fa, task, s)])
                paired_b.append(by[(fb, task, s)])
        if len(paired_a) < 4:
            lines.append(f"| {test} | {fa} vs {fb} | {task} | n/a | n/a | no | incomplete |")
            continue
        eff = effect(task, paired_a, paired_b)
        stat, p = wilcoxon_signed(paired_a, paired_b)
        survives = bool(p < ALPHA)
        # hypothesis direction: fa should beat fb
        beats = eff > 0
        if survives and beats:
            verd = "SUPPORT (exploratory)" if p >= 0.0025 else "SUPPORT"
        elif beats and p <= 0.007812 + 1e-12:
            verd = "EXPLORATORY_HIT (floor)"
        elif beats:
            verd = "direction_ok_ns"
        else:
            verd = "FAIL_direction"
        contrasts.append(
            dict(
                test=test, a=fa, b=fb, task=task, n=len(paired_a),
                effect=eff, p=p, survives_alpha=survives, verdict=verd,
            )
        )
        lines.append(
            f"| {test} | {fa} vs {fb} | {task} | {eff:+.4f} | {p:.4g} | "
            f"{'yes' if survives else 'no'} | {verd} |"
        )

    # Hypothesis-level verdicts
    lines.append("\n## Hypothesis verdicts (prereg)\n")
    # A: HOME beats RAND and BASE on XOR and/or NARMA
    a_hits = [c for c in contrasts if c["test"] == "A" and c["effect"] > 0 and c["p"] <= 0.007812 + 1e-9]
    a_fail_rand = any(c["a"] == "HOME_O_ERR" and c["b"] == "RAND_MOD" and c["effect"] <= 0 for c in contrasts if c["test"] == "A")
    a_both_le_base = all(
        c["effect"] <= 0
        for c in contrasts
        if c["test"] == "A" and c["b"] == "BASE"
    )
    if a_fail_rand or a_both_le_base:
        a_verdict = "FALSIFIED (HOME ≤ RAND or both ≤ BASE on planned tasks)"
    elif a_hits:
        a_verdict = "EXPLORATORY SUPPORT on ≥1 planned contrast; not confirmatory"
    else:
        a_verdict = "NOT SUPPORTED (no planned exploratory hit)"

    b_rival_vs = [c for c in contrasts if c["test"] == "B"]
    b_ok = all(c["effect"] > 0 for c in b_rival_vs)
    b_hits = [c for c in b_rival_vs if c["effect"] > 0 and c["p"] <= 0.007812 + 1e-9]
    if any(c["effect"] <= 0 for c in b_rival_vs):
        b_verdict = "FALSIFIED (RIVAL ≤ SINGLE or ≤ SHUF on a planned task)"
    elif b_hits:
        b_verdict = "EXPLORATORY SUPPORT; not confirmatory"
    else:
        b_verdict = "NOT SUPPORTED"

    c_contrasts = [c for c in contrasts if c["test"] == "C"]
    c_vol_syn_narma = next((c for c in c_contrasts if c["b"] == "SYN" and c["task"] == "NARMA"), None)
    c_vol_syn_xor = next((c for c in c_contrasts if c["b"] == "SYN" and c["task"] == "XOR_HARD"), None)
    if c_vol_syn_narma and c_vol_syn_narma["effect"] <= 0:
        c_verdict = "FALSIFIED (SYN ≥ VOL on NARMA)"
    elif c_vol_syn_xor and c_vol_syn_xor["effect"] < 0:
        c_verdict = "FALSIFIED (VOL < SYN on XOR; prereg asked ≥)"
    elif all(c["effect"] <= 0 for c in c_contrasts if c["b"] == "BASE"):
        c_verdict = "FALSIFIED (neither VOL nor controls beat BASE pattern / VOL ≤ BASE)"
    elif c_vol_syn_narma and c_vol_syn_narma["effect"] > 0:
        c_verdict = "EXPLORATORY SUPPORT on VOL>SYN NARMA; check BASE; not confirmatory"
    else:
        c_verdict = "NOT SUPPORTED"

    lines.append(f"- **Test A:** {a_verdict}")
    lines.append(f"- **Test B:** {b_verdict}")
    lines.append(f"- **Test C:** {c_verdict}")
    lines.append("\n## Honesty")
    lines.append("- Motif abstraction only; not a biological hypothalamus.")
    lines.append("- No QPU. Callosum_scaled left untouched.")
    lines.append("- STP ablation raw jsonl preserved under engine_v3/out/.\n")

    out_md = Path(args.out_md) if args.out_md else (HERE / "RESULTS.md")
    out_json = Path(args.out_json) if args.out_json else (HERE / "results" / "abc_summary.json")
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(
        json.dumps(
            dict(means=means, contrasts=contrasts, alpha=ALPHA,
                 verdicts=dict(A=a_verdict, B=b_verdict, C=c_verdict)),
            indent=2,
        )
    )
    print("\n".join(lines))
    print("wrote", out_md, out_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize ablation jsonl → means + Wilcoxon vs BASE."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JSONL = HERE / "out" / "ablation_runs.jsonl"
HIGHER = {"XOR_HARD", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"}


def wilcoxon_signed(x, y):
    """Two-sided Wilcoxon signed-rank; returns (stat, p) or (nan, nan)."""
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(x) - np.asarray(y)
        if np.allclose(d, 0):
            return 0.0, 1.0
        res = wilcoxon(x, y, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def main(jsonl: Path | None = None, out_summary: Path | None = None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default=None)
    ap.add_argument("--out", type=str, default=None, help="summary json path")
    args, _unknown = ap.parse_known_args()
    jsonl = Path(args.jsonl) if args.jsonl else (jsonl or JSONL)
    out_summary = Path(args.out) if args.out else (out_summary or (jsonl.parent / "ablation_summary.json"))
    rows = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("metric") is not None and not r.get("poisoned"):
                    rows.append(r)
    by = defaultdict(dict)  # (feature,task,seed)->metric
    for r in rows:
        by[(r["feature"], r["task"], r["seed"])] = r["metric"]

    features = sorted({r["feature"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    print(f"n_rows={len(rows)} features={features} tasks={tasks} seeds={seeds}")

    summary = []
    for task in tasks:
        base = {s: by.get(("BASE", task, s)) for s in seeds}
        base = {s: v for s, v in base.items() if v is not None}
        for feat in features:
            vals = [by.get((feat, task, s)) for s in seeds]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            paired_f, paired_b = [], []
            for s in seeds:
                a = by.get((feat, task, s))
                b = by.get(("BASE", task, s))
                if a is not None and b is not None:
                    paired_f.append(a)
                    paired_b.append(b)
            stat, p = (float("nan"), float("nan"))
            if feat != "BASE" and len(paired_f) >= 4:
                stat, p = wilcoxon_signed(paired_f, paired_b)
            # effect: + = better
            if paired_f and paired_b:
                if task in HIGHER:
                    eff = float(np.mean(np.asarray(paired_f) - np.asarray(paired_b)))
                else:
                    eff = float(np.mean(np.asarray(paired_b) - np.asarray(paired_f)))
            else:
                eff = float("nan")
            summary.append(dict(
                task=task, feature=feat, n=len(vals), mean=mean, std=std,
                effect_vs_base=eff, wilcoxon_stat=stat, p=p,
            ))
            print(
                f"{task:10s} {feat:10s} n={len(vals)} mean={mean:.4f}±{std:.4f} "
                f"eff={eff:+.4f} p={p:.4g}",
                flush=True,
            )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    print("wrote", out_summary)


if __name__ == "__main__":
    main()

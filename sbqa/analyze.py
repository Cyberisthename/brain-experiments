"""Summarize SBQA jsonl runs -> summary.json + RESULTS.md."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent


def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return {"n": int(len(a)), "p": 1.0, "stat": 0.0, "mean_a": float(np.mean(a)) if len(a) else None,
                "mean_b": float(np.mean(b)) if len(b) else None, "wins_a": int(np.sum(a < b))}
    try:
        stat, p = wilcoxon(a, b, alternative="less")  # a better (lower)
    except Exception:
        stat, p = 0.0, 1.0
    return {"n": int(len(a)), "p": float(p), "stat": float(stat),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
            "wins_a": int(np.sum(a < b))}


def pivot(rows, metric="best_score"):
    # method -> seed -> score (keep best if dup)
    d = defaultdict(dict)
    for r in rows:
        s, m = r["seed"], r["method"]
        sc = r[metric]
        if s not in d[m] or sc < d[m][s]:
            d[m][s] = sc
    return d


def compare_family(disc_rows):
    d = pivot(disc_rows)
    seeds = sorted(set().union(*[set(v) for v in d.values()])) if d else []
    out = {}
    if "sbqa" not in d:
        return out
    for other in ("sa", "random", "greedy"):
        if other not in d:
            continue
        a = [d["sbqa"][s] for s in seeds if s in d["sbqa"] and s in d[other]]
        b = [d[other][s] for s in seeds if s in d["sbqa"] and s in d[other]]
        out[f"sbqa_vs_{other}"] = paired(a, b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir or (HERE / "out"))
    disc = load_jsonl(out / "discrete_runs.jsonl")
    cont = load_jsonl(out / "continuous_runs.jsonl")
    abl = load_jsonl(out / "ablation_runs.jsonl")

    summary = {
        "note": "SBQA is classical / quantum-inspired; not real qubits.",
        "discrete": {
            "n_rows": len(disc),
            "by_method_mean": {},
            "tests": compare_family([r for r in disc if r.get("task") == "narma"]),
        },
        "continuous": {"n_rows": len(cont), "by_method_mean": {}, "tests": {}},
        "ablation": {"n_rows": len(abl), "by_method_mean": {}},
    }
    for label, rows, key in (
        ("discrete", [r for r in disc if r.get("task") == "narma"], "discrete"),
        ("continuous", [r for r in cont if r.get("task") == "narma"], "continuous"),
        ("ablation", abl, "ablation"),
    ):
        d = pivot(rows)
        summary[key]["by_method_mean"] = {
            m: {"mean": float(np.mean(list(v.values()))), "std": float(np.std(list(v.values()))), "n": len(v)}
            for m, v in d.items()
        }
    # continuous tests: SBQA vs Adam (expect Adam wins / ties — higher p for sbqa<adam)
    cd = pivot([r for r in cont if r.get("task") == "narma"])
    if "sbqa" in cd and "adam" in cd:
        seeds = sorted(set(cd["sbqa"]) & set(cd["adam"]))
        a = [cd["sbqa"][s] for s in seeds]
        b = [cd["adam"][s] for s in seeds]
        summary["continuous"]["tests"]["sbqa_vs_adam"] = paired(a, b)

    # ablations pairwise vs sbqa
    ad = pivot(abl)
    if "sbqa" in ad:
        for other in ("sbqa_metropolis", "sbqa_nomask"):
            if other in ad:
                seeds = sorted(set(ad["sbqa"]) & set(ad[other]))
                summary["ablation"][f"sbqa_vs_{other}"] = paired(
                    [ad["sbqa"][s] for s in seeds], [ad[other][s] for s in seeds]
                )

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # RESULTS.md
    lines = []
    lines.append("# RESULTS — SBQA (Simulated Bio-Quantum Annealing)")
    lines.append("")
    lines.append("**Classical / quantum-inspired only. Not real qubits.**")
    lines.append("")
    lines.append(f"Prereg: `PREREG_SBQA.md`. Outputs in `{out}`.")
    lines.append("")
    lines.append("## Discrete wiring search (NARMA NRMSE ↓)")
    lines.append("")
    lines.append("| Method | mean ± std (n) |")
    lines.append("|--------|----------------|")
    for m, st in summary["discrete"]["by_method_mean"].items():
        lines.append(f"| {m} | {st['mean']:.4f} ± {st['std']:.4f} (n={st['n']}) |")
    lines.append("")
    lines.append("### Primary tests (Wilcoxon, SBQA better = lower)")
    lines.append("")
    for k, t in summary["discrete"]["tests"].items():
        lines.append(f"- **{k}**: mean {t['mean_a']:.4f} vs {t['mean_b']:.4f}, wins {t['wins_a']}/{t['n']}, p={t['p']:.4g}")
    lines.append("")
    lines.append("## Continuous weight training (NARMA NRMSE ↓) — negative control")
    lines.append("")
    lines.append("| Method | mean ± std (n) |")
    lines.append("|--------|----------------|")
    for m, st in summary["continuous"]["by_method_mean"].items():
        lines.append(f"| {m} | {st['mean']:.4f} ± {st['std']:.4f} (n={st['n']}) |")
    lines.append("")
    for k, t in summary["continuous"]["tests"].items():
        lines.append(f"- **{k}**: mean {t['mean_a']:.4f} vs {t['mean_b']:.4f}, wins {t['wins_a']}/{t['n']}, p={t['p']:.4g}")
    lines.append("")
    lines.append("## Ablation")
    lines.append("")
    for m, st in summary["ablation"]["by_method_mean"].items():
        lines.append(f"- {m}: {st['mean']:.4f} ± {st['std']:.4f} (n={st['n']})")
    lines.append("")
    # verdicts
    lines.append("## Verdicts")
    lines.append("")
    tests = summary["discrete"]["tests"]
    ok_sa = tests.get("sbqa_vs_sa", {}).get("p", 1) < 0.05 / 4
    ok_rand = tests.get("sbqa_vs_random", {}).get("p", 1) < 0.05 / 4
    if disc:
        if ok_sa and ok_rand:
            lines.append("- **H1 (discrete SBQA beats SA & RANDOM): SUPPORTED** (Bonferroni α=0.0125).")
        else:
            lines.append("- **H1 (discrete SBQA beats SA & RANDOM): NOT SUPPORTED** at Bonferroni-corrected α.")
    else:
        lines.append("- **H1:** no discrete rows yet.")
    ct = summary["continuous"]["tests"].get("sbqa_vs_adam")
    if ct:
        # negative control: we predict Adam ≤ SBQA is false for "SBQA beats Adam"
        if ct["p"] < 0.05 / 4:
            lines.append("- **H2 (Adam ≥ SBQA on continuous): FAILED prediction** — SBQA unexpectedly beat Adam.")
        else:
            lines.append("- **H2 (Adam ≥ SBQA on continuous): CONSISTENT** — SBQA did not significantly beat Adam (smooth-landscape prior).")
    else:
        lines.append("- **H2:** no continuous rows yet.")
    lines.append("- **H3 (ablation):** see means above; no strong claim without full n=8.")
    lines.append("")
    lines.append("Honest note: smoke / partial runs are under-powered; trust full Kaggle n=8.")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    # also copy into out
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print("wrote", out / "summary.json", "and RESULTS.md")


if __name__ == "__main__":
    main()

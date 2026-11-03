"""Summarize DTM/MPS/SKF jsonl -> summary.json + RESULTS.md."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pivot(rows, metric="best_score"):
    d = defaultdict(dict)
    for r in rows:
        s, m = r["seed"], r["method"]
        sc = r[metric]
        if s not in d[m] or sc < d[m][s]:
            d[m][s] = sc
    return d


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a, b):
        return {"n": int(len(a)), "p": 1.0, "mean_a": float(np.mean(a)) if len(a) else None,
                "mean_b": float(np.mean(b)) if len(b) else None, "wins_a": int(np.sum(a < b)) if len(a) else 0}
    try:
        stat, p = wilcoxon(a, b, alternative="less")
    except Exception:
        stat, p = 0.0, 1.0
    return {"n": int(len(a)), "p": float(p), "stat": float(stat),
            "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "wins_a": int(np.sum(a < b))}


def means(d):
    return {m: {"mean": float(np.mean(list(v.values()))), "std": float(np.std(list(v.values()))), "n": len(v)}
            for m, v in d.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir or (HERE / "out"))
    disc = [r for r in load_jsonl(out / "discrete_runs.jsonl") if r.get("task") == "narma"]
    cont = [r for r in load_jsonl(out / "continuous_runs.jsonl") if r.get("task") == "narma"]
    pd, pc = pivot(disc), pivot(cont)

    primary = {}
    if "dtm" in pd:
        for other in ("sa", "random", "greedy", "set", "sbqa", "rigl", "long_random",
                      "dtm_mps_warp", "dtm_wave", "dtm_both"):
            if other not in pd:
                continue
            seeds = sorted(set(pd["dtm"]) & set(pd[other]))
            primary[f"dtm_vs_{other}"] = paired([pd["dtm"][s] for s in seeds], [pd[other][s] for s in seeds])
    
    if "skf" in pd:
        for other in ("pso", "skf_k1", "dtm", "random", "set"):
            if other in pd:
                seeds = sorted(set(pd["skf"]) & set(pd[other]))
                primary[f"skf_vs_{other}"] = paired(
                    [pd["skf"][s] for s in seeds], [pd[other][s] for s in seeds])

    if "dtm_both" in pd:
        for other in ("long_random", "dtm"):
            if other in pd:
                seeds = sorted(set(pd["dtm_both"]) & set(pd[other]))
                primary[f"dtm_both_vs_{other}"] = paired(
                    [pd["dtm_both"][s] for s in seeds], [pd[other][s] for s in seeds])

    # efficiency
    eff = defaultdict(list)
    nnz = {}
    for r in disc:
        eff[r["method"]].append((r.get("seconds"), r.get("peak_rss_mb")))
        if r["method"].startswith("dtm") or r["method"] in ("long_random", "skf", "skf_k1"):
            if r.get("nnz_init") is not None:
                nnz.setdefault(r["method"], []).append((r["nnz_init"], r["nnz_final"]))

    summary = {
        "note": "DTM/MPS/SKF classical biology-inspired grow/prune. Not quantum. Not backprop.",
        "discrete_means": means(pd),
        "continuous_means": means(pc),
        "tests": primary,
        "efficiency_mean_seconds": {m: float(np.mean([s for s, _ in v if s is not None])) for m, v in eff.items()},
        "efficiency_mean_rss_mb": {m: float(np.mean([r for _, r in v if r is not None])) for m, v in eff.items()},
        "nnz_change": {m: {"mean_init": float(np.mean([a for a, b in v])),
                           "mean_final": float(np.mean([b for a, b in v])),
                           "n": len(v)} for m, v in nnz.items()},
    }
    if "dtm" in pc and "adam" in pc:
        seeds = sorted(set(pc["dtm"]) & set(pc["adam"]))
        summary["continuous_tests"] = {
            "dtm_vs_adam": paired([pc["dtm"][s] for s in seeds], [pc["adam"][s] for s in seeds])
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# RESULTS — DTM (Neuro-Vines) + MPS + SKF",
        "",
        "**Classical biology-inspired grow/prune + MPS + Spatial Kinetic Folding (SKF). Not quantum.**",
        "",
        f"Prereg: `PREREG_DTM.md`. Out: `{out}`.",
        "",
        "## Discrete NARMA (NRMSE ↓)",
        "",
        "| Method | mean ± std (n) | mean seconds | mean peak RSS MB |",
        "|--------|----------------|--------------|------------------|",
    ]
    for m, st in summary["discrete_means"].items():
        sec = summary["efficiency_mean_seconds"].get(m, float("nan"))
        rss = summary["efficiency_mean_rss_mb"].get(m, float("nan"))
        lines.append(f"| {m} | {st['mean']:.4f} ± {st['std']:.4f} (n={st['n']}) | {sec:.3f} | {rss:.1f} |")
    lines += ["", "### Tests (Wilcoxon, first method better = lower)", ""]
    for k, t in summary["tests"].items():
        lines.append(f"- **{k}**: {t['mean_a']:.4f} vs {t['mean_b']:.4f}, wins {t['wins_a']}/{t['n']}, p={t['p']:.4g}")
    lines += ["", "## Continuous NARMA", ""]
    for m, st in summary["continuous_means"].items():
        lines.append(f"- {m}: {st['mean']:.4f} ± {st['std']:.4f} (n={st['n']})")
    lines += ["", "## NNZ traces (DTM family)", ""]
    for m, st in summary["nnz_change"].items():
        lines.append(f"- {m}: nnz {st['mean_init']:.0f} → {st['mean_final']:.0f}")
    lines += ["", "## Verdicts", ""]
    # Bonferroni m=9
    alpha = 0.05 / 9
    t2 = summary["tests"].get("dtm_vs_random", {})
    t4 = summary["tests"].get("dtm_vs_set", {})
    t6 = summary["tests"].get("dtm_both_vs_long_random", {})
    if disc:
        ok = (t2.get("p", 1) < alpha) and (t4.get("p", 1) < alpha)
        lines.append(f"- **H1 (DTM < RANDOM & SET):** {'SUPPORTED' if ok else 'NOT SUPPORTED'} (α={alpha:.4f}).")
        ok6 = t6.get("p", 1) < alpha
        lines.append(f"- **MPS success (DTM_BOTH < LONG_RANDOM):** {'SUPPORTED' if ok6 else 'NOT SUPPORTED'}.")
        t8 = summary["tests"].get("skf_vs_pso", {})
        ok8 = t8.get("p", 1) < alpha
        lines.append(f"- **SKF success (SKF < PSO):** {'SUPPORTED' if ok8 else 'NOT SUPPORTED'}.")
    else:
        lines.append("- No discrete rows yet.")
    lines.append("- Efficiency/nnz: descriptive — see table & NNZ section.")
    lines.append("- Smoke / n=1 is underpowered; trust full Kaggle n=8.")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (out / "RESULTS.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

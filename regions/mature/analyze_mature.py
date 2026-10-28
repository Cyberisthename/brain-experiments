"""Analyze mature runs. Bonferroni over 144 planned ΔΔ contrasts."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

H = Path(__file__).resolve().parent
OUT = H / "out"
JF = OUT / "runs.jsonl"
AGES = [0, 1000, 5000, 20000, 50000]
ARMS = ["FLY_CX", "FRANKEN", "ER", "SHUF", "DENS_MATCHED"]
CONDS = ["none", "hebb", "hebb_prune", "full"]
METRICS = {
    "MC": "higher",
    "NARMA": "lower",  # improvement = NARMA0 - NARMAa
    "PC_HARD": "higher",
    "LES20_RET": "higher",
}
CONTROLS = ["ER", "SHUF", "DENS_MATCHED", "FLY_CX"]
N_CONTRASTS = 4 * 3 * 3 * 4  # controls × ages{5k,20k,50k} × conds{hebb,hebb_prune,full} × metrics = 144


def load():
    rows = []
    if not JF.exists():
        return rows
    for line in JF.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error") or r.get("age", -1) < 0:
            continue
        rows.append(r)
    return rows


def index(rows):
    # (arm, seed, condition, age) -> row
    d = {}
    for r in rows:
        d[(r["arm"], r["seed"], r["condition"], r["age"])] = r
    return d


def improvement(r0, ra, metric):
    v0, va = r0.get(metric), ra.get(metric)
    if v0 is None or va is None:
        return float("nan")
    if not (np.isfinite(v0) and np.isfinite(va)):
        return float("nan")
    if METRICS[metric] == "higher":
        return float(va - v0)
    return float(v0 - va)  # NARMA: lower error = positive improvement


def summarize(rows):
    idx = index(rows)
    lines = []
    lines.append("# RESULTS: Developmental maturation\n")
    lines.append(f"**Date:** analysis run after mature.done\n")
    lines.append(f"**Prereg:** PREREGISTRATION_mature.md\n")
    lines.append(f"**Bonferroni family size:** {N_CONTRASTS}\n")
    hp = {}
    if (OUT / "tune.json").exists():
        hp = json.loads((OUT / "tune.json").read_text())
        lines.append(f"**Locked HP (tune seeds 90–92):** `{json.dumps(hp)}`\n")

    # absolute means at age 0 and 50k
    lines.append("\n## Absolute scores (mean ± sd over seeds with data)\n")
    for age in (0, 50000):
        lines.append(f"\n### Age {age}\n")
        lines.append("| arm | cond | MC ↑ | NARMA ↓ | PC_HARD ↑ | LES20_RET ↑ | n |\n|---|---|---|---|---|---|---|\n")
        for arm in ARMS:
            for cond in CONDS:
                vals = {m: [] for m in METRICS}
                for seed in range(8):
                    r = idx.get((arm, seed, cond, age))
                    if not r:
                        continue
                    for m in METRICS:
                        v = r.get(m)
                        if v is not None and np.isfinite(v):
                            vals[m].append(v)
                n = max((len(vals[m]) for m in METRICS), default=0)
                if n == 0:
                    continue
                def fmt(m):
                    a = vals[m]
                    if not a:
                        return "—"
                    return f"{np.mean(a):.3f} ± {np.std(a, ddof=1) if len(a)>1 else 0:.3f}"
                lines.append(f"| {arm} | {cond} | {fmt('MC')} | {fmt('NARMA')} | {fmt('PC_HARD')} | {fmt('LES20_RET')} | {n} |\n")

    # H3 sanity: none flat?
    lines.append("\n## H3 sanity — `none` flat across ages?\n")
    none_flat = True
    for arm in ARMS:
        for seed in range(8):
            r0 = idx.get((arm, seed, "none", 0))
            if not r0:
                continue
            for age in AGES[1:]:
                ra = idx.get((arm, seed, "none", age))
                if not ra:
                    continue
                for m in METRICS:
                    if abs((r0.get(m) or 0) - (ra.get(m) or 0)) > 1e-9:
                        none_flat = False
    lines.append(f"none scores identical across ages: **{none_flat}**\n")

    # Δ and ΔΔ
    lines.append("\n## Improvement Δ (age − 0) and maturation×wiring ΔΔ\n")
    lines.append("ΔΔ = mean_Δ(FRANKEN) − mean_Δ(control). Positive = real gains more from maturation.\n")
    lines.append(f"Mann–Whitney U on per-seed Δ; p_bonf = min(1, p×{N_CONTRASTS}).\n")

    survivors = []
    all_contrasts = []
    for cond in ("hebb", "hebb_prune", "full"):
        for age in (5000, 20000, 50000):
            for metric in METRICS:
                # per-seed Δ for FRANKEN and controls
                frank_d = []
                for seed in range(8):
                    r0 = idx.get(("FRANKEN", seed, cond, 0))
                    ra = idx.get(("FRANKEN", seed, cond, age))
                    if r0 and ra:
                        frank_d.append(improvement(r0, ra, metric))
                for ctrl in CONTROLS:
                    ctrl_d = []
                    for seed in range(8):
                        r0 = idx.get((ctrl, seed, cond, 0))
                        ra = idx.get((ctrl, seed, cond, age))
                        if r0 and ra:
                            ctrl_d.append(improvement(r0, ra, metric))
                    frank_d_f = [x for x in frank_d if np.isfinite(x)]
                    ctrl_d_f = [x for x in ctrl_d if np.isfinite(x)]
                    if len(frank_d_f) < 3 or len(ctrl_d_f) < 3:
                        p = float("nan"); u = float("nan")
                    else:
                        u, p = stats.mannwhitneyu(frank_d_f, ctrl_d_f, alternative="two-sided")
                    dd = float(np.mean(frank_d_f) - np.mean(ctrl_d_f)) if frank_d_f and ctrl_d_f else float("nan")
                    p_bonf = min(1.0, p * N_CONTRASTS) if np.isfinite(p) else float("nan")
                    rec = dict(cond=cond, age=age, metric=metric, control=ctrl,
                               d_franken=float(np.mean(frank_d_f)) if frank_d_f else None,
                               d_control=float(np.mean(ctrl_d_f)) if ctrl_d_f else None,
                               dd=dd, p=p, p_bonf=p_bonf,
                               n_f=len(frank_d_f), n_c=len(ctrl_d_f))
                    all_contrasts.append(rec)
                    if np.isfinite(p_bonf) and p_bonf < 0.05:
                        survivors.append(rec)

    lines.append("\n### Survivors (p_bonf < 0.05)\n")
    if not survivors:
        lines.append("**None.** No maturation×wiring contrast survived Bonferroni.\n")
    else:
        lines.append("| cond | age | metric | vs | Δ(FRANKEN) | Δ(ctrl) | ΔΔ | p | p_bonf |\n|---|---|---|---|---|---|---|---|---|\n")
        for s in survivors:
            lines.append(
                f"| {s['cond']} | {s['age']} | {s['metric']} | {s['control']} | "
                f"{s['d_franken']:.4f} | {s['d_control']:.4f} | {s['dd']:.4f} | "
                f"{s['p']:.2e} | {s['p_bonf']:.3f} |\n"
            )

    # top |ΔΔ| regardless of significance
    lines.append("\n### Largest |ΔΔ| (exploratory, not corrected)\n")
    ranked = sorted(all_contrasts, key=lambda r: abs(r["dd"]) if np.isfinite(r["dd"]) else -1, reverse=True)[:12]
    lines.append("| cond | age | metric | vs | ΔΔ | p_raw | p_bonf |\n|---|---|---|---|---|---|---|\n")
    for s in ranked:
        if not np.isfinite(s["dd"]):
            continue
        lines.append(
            f"| {s['cond']} | {s['age']} | {s['metric']} | {s['control']} | "
            f"{s['dd']:.4f} | {s['p']:.2e} | {s['p_bonf']:.3f} |\n"
        )

    # Hypothesis verdicts
    lines.append("\n## Hypothesis verdicts\n")
    def any_pos_surv(metric, conds=None):
        conds = conds or ("full", "hebb_prune")
        return [s for s in survivors if s["metric"] == metric and s["cond"] in conds
                and s["control"] == "ER" and s["dd"] > 0]

    h1_les = any_pos_surv("LES20_RET")
    h1_pc = any_pos_surv("PC_HARD")
    h1 = bool(h1_les or h1_pc)
    lines.append(f"- **H1** (full/hebb_prune: FRANKEN gains more than ER on LES20_RET or PC_HARD): "
                 f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'} "
                 f"(LES survivors={len(h1_les)}, PC survivors={len(h1_pc)})\n")

    # H2: hebb alone ΔΔ MC ≤ 0 or null
    hebb_mc = [c for c in all_contrasts if c["cond"]=="hebb" and c["metric"]=="MC" and c["control"]=="ER"]
    h2_ok = all((not np.isfinite(c["p_bonf"])) or c["p_bonf"] >= 0.05 or c["dd"] <= 0 for c in hebb_mc)
    lines.append(f"- **H2** (hebb alone: no positive significant ΔΔ on MC vs ER): "
                 f"{'SUPPORTED' if h2_ok else 'REJECTED'}\n")
    lines.append(f"- **H3** (none flat): {'SUPPORTED' if none_flat else 'FAILED — implementation bug'}\n")

    dens = [s for s in survivors if s["control"] == "DENS_MATCHED" and s["dd"] > 0]
    er_only = [s for s in survivors if s["control"] == "ER" and s["dd"] > 0]
    lines.append(f"- **H5** note: positive survivors vs DENS_MATCHED={len(dens)}; vs ER only={len(er_only)}. "
                 f"{'Topology-beyond-density claim weak/absent.' if not dens else 'Some dens-matched survivors.'}\n")

    lines.append("\n## Honest bottom line\n")
    if not survivors:
        lines.append(
            "Maturation under this protocol did **not** produce a Bonferroni-surviving "
            "maturation×wiring interaction favoring real FRANKEN topology over random controls "
            "on the preregistered family. Absolute task gaps (ER often strong on MC) are not "
            "rescued by developmental exposure here. Treat as a null on H1 unless a pre-registered "
            "narrower family is justified in a follow-up.\n"
        )
    else:
        lines.append(
            f"{len(survivors)} contrasts survived. Inspect direction and which controls; "
            f"only wins vs DENS_MATCHED support topology beyond density+weights.\n"
        )

    # deviations
    lines.append("\n## Deviations from prereg\n")
    lines.append(
        "- Delay ADAPT updates applied every 100 maturation steps (not every step) for compute; "
        "rule identical to myelin ADAPT (shorten above-median coactivity, lengthen bottom 20%).\n"
        "- After prune under `full`, delays re-sampled (edge identity lost); delay *structure learning* "
        "is therefore partial across prune events.\n"
        "- Tune grid: η ∈ {3e-5,1e-4,3e-4} on hebb@5k only; prune_frac locked 0.02.\n"
        "- `full` age-0 already uses random delays, so absolute MC is not comparable to non-delay conditions; "
        "Δ within `full` remains valid.\n"
    )

    text = "".join(lines)
    (H / "RESULTS_mature.md").write_text(text)
    (OUT / "analyze_contrasts.json").write_text(json.dumps(all_contrasts, indent=2, default=float))
    (OUT / "analyze_survivors.json").write_text(json.dumps(survivors, indent=2, default=float))
    print(text)
    return survivors, all_contrasts


if __name__ == "__main__":
    summarize(load())

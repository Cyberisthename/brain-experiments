"""Analyze fewshot_delay: Wilcoxon + Bonferroni; write RESULTS + learning-curve fig."""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

H = Path(__file__).resolve().parent
OUT = H / "out"
FIGS = Path("/workspace/flybrain/figs")
FIGS.mkdir(exist_ok=True)

SEEDS = list(range(8))
N_FOCUS = 100
CTRLS = ["ER_DEL", "SHUF_DEL", "DENS_DEL"]
CTRLS_NOD = ["ER_NOD", "SHUF_NOD", "DENS_NOD"]
TASKS = ["MC", "NARMA", "PC_HARD"]


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.open():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def wilcox(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2:
        return 1.0
    d = xs - ys
    if np.allclose(d, 0):
        return 1.0
    try:
        _, p = wilcoxon(xs, ys, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except ValueError:
        return 1.0


few = [r for r in load_jsonl(OUT / "fewshot.jsonl") if not r.get("error")]
noise = [r for r in load_jsonl(OUT / "noise.jsonl") if not r.get("error")]
mech = [r for r in few if r.get("test") == "mechanism"]
few = [r for r in few if r.get("test") == "fewshot"]

# (arm, task, n_train) -> {seed: value}
by = defaultdict(dict)
for r in few:
    v = r.get("value")
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        continue
    by[(r["arm"], r["task"], int(r["n_train"]))][int(r["seed"])] = float(v)

# mechanism
by_pr = defaultdict(dict)
for r in mech:
    if r.get("PR") is None or not np.isfinite(r["PR"]):
        continue
    by_pr[r["arm"]][int(r["seed"])] = float(r["PR"])

# noise mean retention
by_nret = defaultdict(dict)  # (arm, task) -> {seed: mean_retention}
by_auc = defaultdict(dict)
for r in noise:
    if r.get("test") == "noise_auc" and r.get("channel") == "input":
        mr = r.get("mean_retention")
        if mr is not None and np.isfinite(mr):
            by_nret[(r["arm"], r["task"])][int(r["seed"])] = float(mr)
        auc = r.get("auc")
        if auc is not None and np.isfinite(auc):
            by_auc[(r["arm"], r["task"])][int(r["seed"])] = float(auc)

alpha_choice = {}
if (OUT / "alpha_choice.json").exists():
    alpha_choice = json.loads((OUT / "alpha_choice.json").read_text())


def paired_vals(arm_a, arm_b, task, n_train):
    a = by.get((arm_a, task, n_train), {})
    b = by.get((arm_b, task, n_train), {})
    xs, ys = [], []
    for s in SEEDS:
        if s in a and s in b:
            xs.append(a[s]); ys.append(b[s])
    return xs, ys


def mean_sd(arm, task, n_train):
    d = by.get((arm, task, n_train), {})
    vals = [d[s] for s in SEEDS if s in d]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


lines = []
def w(s=""):
    lines.append(s)

w("# Few-shot + matched delays — results")
w()
w("Prereg: `PREREGISTRATION_fewshot_delay.md`.")
w(f"Eval seeds 0–7. Focus n_train={N_FOCUS}.")
w()

w("## Locked ridge α (tune seeds 90–92, MC @ n_train=100)")
w()
w("| arm | α |")
w("|---|---:|")
for arm, info in sorted((alpha_choice.get("alphas") or {}).items()):
    w(f"| {arm} | {info.get('alpha')} |")
w()

w("## Learning curve means (MC ↑)")
w()
# header
ns = [25, 50, 100, 200, 500, 1000, 3000]
w("| arm | " + " | ".join(str(n) for n in ns) + " |")
w("|---|" + "---:|" * len(ns))
for delay in ("NOD", "DEL"):
    for wiring in ("FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"):
        arm = f"{wiring}_{delay}"
        cells = []
        for n in ns:
            m, s, k = mean_sd(arm, "MC", n)
            cells.append(f"{m:.2f}" if m is not None else "—")
        w(f"| {arm} | " + " | ".join(cells) + " |")
w()

w("## Focus table — n_train=100")
w()
w("| arm | MC ↑ | NARMA ↓ | PC_HARD ↑ | PR (mech) |")
w("|---|---:|---:|---:|---:|")
focus_rows = {}
for delay in ("NOD", "DEL"):
    for wiring in ("FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"):
        arm = f"{wiring}_{delay}"
        cells = []
        entry = {}
        for task in TASKS:
            # PC_HARD may use n_train=100 or clipped
            nt = N_FOCUS
            if task == "PC_HARD":
                # prefer 100 if present else max available <=600
                avail = [k[2] for k in by if k[0] == arm and k[1] == task]
                if N_FOCUS in avail:
                    nt = N_FOCUS
                elif avail:
                    nt = max(avail)
                else:
                    nt = N_FOCUS
            m, s, k = mean_sd(arm, task, nt)
            entry[task] = (m, s, k, nt)
            cells.append(f"{m:.3f}±{s:.3f}" if m is not None else "—")
        prs = [by_pr[arm][s] for s in SEEDS if s in by_pr.get(arm, {})]
        pr_m = float(np.mean(prs)) if prs else float("nan")
        entry["PR"] = pr_m
        focus_rows[arm] = entry
        pr_s = f"{pr_m:.1f}" if np.isfinite(pr_m) else "—"
        w(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {pr_s} |")
w()

def family_table(title, real_arm, ctrls, n_train, m_bonf, task_list=TASKS):
    w(f"## {title} (Bonferroni ×{m_bonf})")
    w()
    w("| real | ctrl | task | real | ctrl | Δ | p_raw | p_bonf | sig | favor |")
    w("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    results = []
    for ctrl in ctrls:
        for task in task_list:
            nt = n_train
            if task == "PC_HARD":
                avail = [k[2] for k in by if k[0] == real_arm and k[1] == task]
                if n_train in avail:
                    nt = n_train
                elif avail:
                    nt = max([a for a in avail if a <= 600], default=max(avail))
            xs, ys = paired_vals(real_arm, ctrl, task, nt)
            if len(xs) < 2:
                w(f"| {real_arm} | {ctrl} | {task} | — | — | — | — | — | no | — |")
                continue
            p = wilcox(xs, ys)
            pb = min(1.0, p * m_bonf)
            da, db = float(np.mean(xs)), float(np.mean(ys))
            delta = da - db
            if task == "NARMA":
                favor = real_arm if delta < 0 else ctrl
            else:
                favor = real_arm if delta > 0 else ctrl
            sig = "YES" if pb < 0.05 else "no"
            results.append(dict(real=real_arm, ctrl=ctrl, task=task, mean_a=da, mean_b=db,
                                delta=delta, p_raw=p, p_bonf=pb, sig=sig, favor=favor))
            w(f"| {real_arm} | {ctrl} | {task} | {da:.3f} | {db:.3f} | {delta:+.3f} | {p:.4g} | {pb:.4g} | {sig} | {favor} |")
    w()
    return results


primary = family_table("PRIMARY — FRANKEN_DEL vs delay-matched controls @ n=100",
                       "FRANKEN_DEL", CTRLS, N_FOCUS, 9)
secondary = family_table("SECONDARY — FRANKEN_NOD vs NOD controls @ n=100 (replicate)",
                         "FRANKEN_NOD", CTRLS_NOD, N_FOCUS, 9)

# Delay-survival: gap_DEL - gap_NOD for MC
w("## Delay-survival family — MC @ n=100 (Bonferroni ×3)")
w()
w("Per-seed gap = FRANKEN − ctrl. Test whether gap_DEL − gap_NOD differs from 0 (Wilcoxon).")
w()
w("| ctrl | mean gap_NOD | mean gap_DEL | mean(gap_DEL−gap_NOD) | p_raw | p_bonf | sig | interpretation |")
w("|---|---:|---:|---:|---:|---:|---|---|")
surv = []
for ctrl_base in ("ER", "SHUF", "DENS"):
    xs_n, ys_n = paired_vals("FRANKEN_NOD", f"{ctrl_base}_NOD", "MC", N_FOCUS)
    xs_d, ys_d = paired_vals("FRANKEN_DEL", f"{ctrl_base}_DEL", "MC", N_FOCUS)
    # align seeds
    # rebuild from by
    gaps_n, gaps_d = [], []
    for s in SEEDS:
        an = by.get(("FRANKEN_NOD", "MC", N_FOCUS), {}).get(s)
        bn = by.get((f"{ctrl_base}_NOD", "MC", N_FOCUS), {}).get(s)
        ad = by.get(("FRANKEN_DEL", "MC", N_FOCUS), {}).get(s)
        bd = by.get((f"{ctrl_base}_DEL", "MC", N_FOCUS), {}).get(s)
        if None not in (an, bn, ad, bd):
            gaps_n.append(an - bn)
            gaps_d.append(ad - bd)
    if len(gaps_n) < 2:
        w(f"| {ctrl_base} | — | — | — | — | — | no | — |")
        continue
    gaps_n, gaps_d = np.asarray(gaps_n), np.asarray(gaps_d)
    diff = gaps_d - gaps_n
    p = wilcox(gaps_d, gaps_n)  # paired on gaps
    pb = min(1.0, p * 3)
    sig = "YES" if pb < 0.05 else "no"
    # positive mean(diff) means DEL gap > NOD gap (advantage grew); negative = advantage shrank
    interp = "advantage grew under delays" if diff.mean() > 0 else "advantage shrank / flipped under delays"
    surv.append(dict(ctrl=ctrl_base, gap_nod=float(gaps_n.mean()), gap_del=float(gaps_d.mean()),
                     ddiff=float(diff.mean()), p_raw=p, p_bonf=pb, sig=sig))
    w(f"| {ctrl_base} | {gaps_n.mean():+.3f} | {gaps_d.mean():+.3f} | {diff.mean():+.3f} | {p:.4g} | {pb:.4g} | {sig} | {interp} |")
w()

# Noise family
w("## Noise family — mean input-noise MC retention σ>0 (Bonferroni ×6)")
w()
w("| real | ctrl | real_ret | ctrl_ret | Δ | p_raw | p_bonf | sig | favor |")
w("|---|---|---:|---:|---:|---:|---:|---|---|")
noise_res = []
for real in ("FRANKEN_DEL", "FLY_CX_DEL"):
    for ctrl in CTRLS:
        a = by_nret.get((real, "MC"), {})
        b = by_nret.get((ctrl, "MC"), {})
        xs, ys = [], []
        for s in SEEDS:
            if s in a and s in b:
                xs.append(a[s]); ys.append(b[s])
        if len(xs) < 2:
            w(f"| {real} | {ctrl} | — | — | — | — | — | no | — |")
            continue
        p = wilcox(xs, ys)
        pb = min(1.0, p * 6)
        da, db = float(np.mean(xs)), float(np.mean(ys))
        delta = da - db
        favor = real if delta > 0 else ctrl
        sig = "YES" if pb < 0.05 else "no"
        note = " (size-confounded)" if real.startswith("FLY") else ""
        noise_res.append(dict(real=real, ctrl=ctrl, mean_a=da, mean_b=db, delta=delta,
                              p_raw=p, p_bonf=pb, sig=sig, favor=favor))
        w(f"| {real}{note} | {ctrl} | {da:.3f} | {db:.3f} | {delta:+.3f} | {p:.4g} | {pb:.4g} | {sig} | {favor} |")
w()

# Mechanism exploratory
w("## Mechanism — participation ratio vs MC@100 (exploratory)")
w()
w("| arm | mean PR | MC@100 |")
w("|---|---:|---:|")
pr_pts, mc_pts, labels = [], [], []
for delay in ("NOD", "DEL"):
    for wiring in ("FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"):
        arm = f"{wiring}_{delay}"
        prs = [by_pr[arm][s] for s in SEEDS if s in by_pr.get(arm, {})]
        m, s, k = mean_sd(arm, "MC", N_FOCUS)
        if prs and m is not None:
            w(f"| {arm} | {np.mean(prs):.1f} | {m:.3f} |")
            pr_pts.append(float(np.mean(prs)))
            mc_pts.append(m)
            labels.append(arm)
w()
if len(pr_pts) >= 3:
    corr = float(np.corrcoef(pr_pts, mc_pts)[0, 1])
    w(f"Pearson r(PR, MC@100) across arms = **{corr:.3f}** (H3 predicted negative: lower PR → better few-shot).")
else:
    corr = float("nan")
    w("Not enough arms for correlation.")
w()

# Verdict
w("## Honest verdict")
w()
n_pri_sig_favor = sum(1 for r in primary if r["sig"] == "YES" and r["favor"] == "FRANKEN_DEL")
n_pri_sig_against = sum(1 for r in primary if r["sig"] == "YES" and r["favor"] != "FRANKEN_DEL")
n_sec_sig_favor = sum(1 for r in secondary if r["sig"] == "YES" and r["favor"] == "FRANKEN_NOD")
# P2: primary MC vs controls
mc_primary = [r for r in primary if r["task"] == "MC"]
mc_wins = [r for r in mc_primary if r["sig"] == "YES" and r["favor"] == "FRANKEN_DEL"]
mc_losses = [r for r in mc_primary if r["favor"] != "FRANKEN_DEL"]

# means for headline
def m100(arm):
    m, _, _ = mean_sd(arm, "MC", 100)
    return m

w(f"- **NOD replicate (P1):** FRANKEN_NOD MC@100={m100('FRANKEN_NOD')}; "
  f"ER_NOD={m100('ER_NOD')}; SHUF_NOD={m100('SHUF_NOD')}; DENS_NOD={m100('DENS_NOD')}. "
  f"Secondary Bonf wins favoring FRANKEN_NOD: {n_sec_sig_favor}/9.")
w(f"- **DEL primary (P2):** FRANKEN_DEL MC@100={m100('FRANKEN_DEL')}; "
  f"ER_DEL={m100('ER_DEL')}; SHUF_DEL={m100('SHUF_DEL')}; DENS_DEL={m100('DENS_DEL')}. "
  f"Primary Bonf wins favoring FRANKEN_DEL: {n_pri_sig_favor}/9 "
  f"(against: {n_pri_sig_against}). MC-only wins: {len(mc_wins)}/3.")
gap_shrunk = sum(1 for s in surv if s["ddiff"] < 0 and s["sig"] == "YES")
w(f"- **Delay-survival:** significant shrink/flip of FRANKEN−ctrl gap under delays: {gap_shrunk}/3 controls.")
n_noise_win = sum(1 for r in noise_res if r["real"] == "FRANKEN_DEL" and r["sig"] == "YES" and r["favor"] == "FRANKEN_DEL")
w(f"- **Noise (P3):** FRANKEN_DEL Bonf wins on mean retention vs DEL controls: {n_noise_win}/3.")
w(f"- **Mechanism (H3):** r(PR, MC@100)={corr:.3f}.")
w()

# Decide P2
p2_pass = len(mc_wins) >= 1
if p2_pass:
    verdict = ("**P2 supported (partial/full):** real FRANKEN wiring keeps a few-shot MC edge "
               "under matched delays against at least one delay-matched random control after Bonferroni.")
elif n_pri_sig_against > 0 and n_pri_sig_favor == 0:
    verdict = ("**P2 rejected:** under matched delays, random wiring matches or beats FRANKEN at "
               "n_train=100 after correction. The earlier few-shot advantage did **not** survive delays "
               "(or was α-protocol-sensitive). Delays dominate few-shot capacity here.")
else:
    verdict = ("**P2 inconclusive:** no Bonferroni-significant FRANKEN_DEL win on MC@100 vs delay-matched "
               "controls; raw means may still lean one way — see tables. Do not claim a breakthrough.")
w(verdict)
w()
w("Honest one-liner goes in chat for the user; this file is the archival record.")
w()

# Full NARMA / PC curves briefly
w("## NARMA learning curve means (↓)")
w()
w("| arm | " + " | ".join(str(n) for n in ns) + " |")
w("|---|" + "---:|" * len(ns))
for delay in ("NOD", "DEL"):
    for wiring in ("FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"):
        arm = f"{wiring}_{delay}"
        cells = []
        for n in ns:
            m, s, k = mean_sd(arm, "NARMA", n)
            cells.append(f"{m:.3f}" if m is not None else "—")
        w(f"| {arm} | " + " | ".join(cells) + " |")
w()

text = "\n".join(lines) + "\n"
(OUT / "analyze.txt").write_text(text)
(H / "RESULTS_fewshot_delay.md").write_text(text)
print(text)

# Save machine-readable summary
summary = dict(
    primary=primary, secondary=secondary, survival=surv, noise=noise_res,
    focus={k: {kk: (vv if not isinstance(vv, tuple) else {"mean": vv[0], "sd": vv[1], "n": vv[2], "nt": vv[3]})
               for kk, vv in v.items()} for k, v in focus_rows.items()},
    corr_PR_MC100=corr, verdict=verdict, p2_pass=p2_pass,
)
(OUT / "analyze.json").write_text(json.dumps(summary, indent=2, default=float))

# Figure: learning curves MC NOD vs DEL
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib missing:", e)
    raise SystemExit(0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
colors = {
    "FLY_CX": "#1b9e77",
    "FRANKEN": "#d95f02",
    "ER": "#7570b3",
    "SHUF": "#e7298a",
    "DENS": "#66a61e",
}
for ax, delay, title in zip(axes, ("NOD", "DEL"), ("No delays", "Matched delays")):
    for wiring in ("FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"):
        arm = f"{wiring}_{delay}"
        ms, ss = [], []
        for n in ns:
            m, s, k = mean_sd(arm, "MC", n)
            ms.append(m if m is not None else np.nan)
            ss.append(s if s is not None else np.nan)
        ms, ss = np.asarray(ms, float), np.asarray(ss, float)
        ax.plot(ns, ms, "-o", label=wiring, color=colors[wiring], markersize=4)
        ax.fill_between(ns, ms - ss, ms + ss, color=colors[wiring], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_title(title)
    ax.axvline(100, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Memory capacity (MC)")
axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle("Few-shot MC learning curves: real vs random (±1 sd, 8 seeds)", fontsize=11)
fig.tight_layout()
fig_path = FIGS / "fewshot_delay_mc_curves.png"
fig.savefig(fig_path, dpi=140)
print("wrote", fig_path)

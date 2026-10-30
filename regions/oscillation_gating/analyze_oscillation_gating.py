#!/usr/bin/env python3
"""Analyze full oscillation_gating runs; write RESULTS + figure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

CTC_ARMS = ["COHERENT", "INCOHERENT", "NO_GATE", "HUB", "RAND_MATCH"]
MYELIN_ARMS = ["MYELIN_ONLY", "MYELIN_COHERENT"]
CALLOSUM_ARMS = ["CALLOSUM", "RAND_BRIDGE", "NO_BRIDGE", "DENSE_BRIDGE"]
ARMS = CTC_ARMS + MYELIN_ARMS + CALLOSUM_ARMS

CONTRASTS = [
    ("COHERENT", "INCOHERENT"),
    ("COHERENT", "NO_GATE"),
    ("COHERENT", "RAND_MATCH"),
]
PRIMARY = [("XOR", "greater"), ("CHAR", "less"), ("NARMA", "less")]
ALPHA = 0.05
N_PRIMARY = 9  # 3 contrasts × 3 tasks


def load_runs():
    rows = []
    p = OUT / "runs.jsonl"
    if not p.exists():
        raise SystemExit("missing out/runs.jsonl")
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_wilcoxon_p(a, b, alternative="greater"):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    try:
        from scipy.stats import wilcoxon
        try:
            return float(wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)
        except TypeError:
            p2 = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            diff = a - b
            if alternative == "greater":
                return p2 / 2 if diff.mean() > 0 else 1 - p2 / 2
            if alternative == "less":
                return p2 / 2 if diff.mean() < 0 else 1 - p2 / 2
            return p2
        except ValueError:
            return 1.0
    except Exception:
        diff = a - b
        nz = diff[diff != 0]
        if len(nz) == 0:
            return 1.0
        from math import comb
        k = int((nz > 0).sum()) if alternative != "less" else int((nz < 0).sum())
        p = sum(comb(len(nz), i) for i in range(k, len(nz) + 1)) / (2 ** len(nz))
        return float(p)


def seed_metric(by, arm, task):
    rs = by.get((arm, task), [])
    return {r["seed"]: r["metric"] for r in rs}


def stats_for(by, arm, task):
    xs = [r["metric"] for r in by.get((arm, task), [])]
    if not xs:
        return float("nan"), float("nan"), []
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, xs


def main():
    rows = load_runs()
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["task"]), []).append(r)

    hp = {}
    hp_path = OUT / "hparams.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())

    lines = []
    lines.append("# RESULTS — oscillation_gating (FULL PROTOCOL)\n")
    lines.append("Prereg: `PREREG_oscillation_gating.md` (frozen 2026-10-30 07:34:00 EDT).\n")
    lines.append(
        "Primary (Bonferroni α=0.05/9): COHERENT vs {INCOHERENT, NO_GATE, RAND_MATCH} "
        "on XOR↑, CHAR↓, NARMA↓. Seeds 1100–1111 (n=12).\n"
    )
    if hp:
        lines.append(f"Hparams: `{json.dumps(hp)}`\n")

    # ---- summary tables ----
    lines.append("\n## Summary — CTC + myelin arms (mean ± sd)\n")
    lines.append("| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | n_params | nnz |")
    lines.append("|---|---|---|---|---|---|")
    table = {}
    for arm in CTC_ARMS + MYELIN_ARMS:
        xm, xs, xv = stats_for(by, arm, "XOR")
        cm, cs, cv = stats_for(by, arm, "CHAR")
        nm, ns, nv = stats_for(by, arm, "NARMA")
        npar = by.get((arm, "XOR"), [{}])[0].get("n_params", "?")
        nnz = by.get((arm, "XOR"), [{}])[0].get("nnz", "?")
        lines.append(
            f"| {arm} | {xm:.3f}±{xs:.3f} | {cm:.3f}±{cs:.3f} | {nm:.3f}±{ns:.3f} | {npar} | {nnz} |"
        )
        table[arm] = dict(XOR=(xm, xs, xv), CHAR=(cm, cs, cv), NARMA=(nm, ns, nv))

    lines.append("\n## Summary — callosum arms (split-input XOR; mean ± sd)\n")
    lines.append("| arm | XOR acc ↑ | CHAR CE ↓ | NARMA NRMSE ↓ | nnz | n_bridge |")
    lines.append("|---|---|---|---|---|---|")
    for arm in CALLOSUM_ARMS:
        xm, xs, xv = stats_for(by, arm, "XOR")
        cm, cs, cv = stats_for(by, arm, "CHAR")
        nm, ns, nv = stats_for(by, arm, "NARMA")
        nnz = by.get((arm, "XOR"), [{}])[0].get("nnz", "?")
        nb = by.get((arm, "XOR"), [{}])[0].get("n_bridge", "?")
        lines.append(
            f"| {arm} | {xm:.3f}±{xs:.3f} | {cm:.3f}±{cs:.3f} | {nm:.3f}±{ns:.3f} | {nnz} | {nb} |"
        )
        table[arm] = dict(XOR=(xm, xs, xv), CHAR=(cm, cs, cv), NARMA=(nm, ns, nv))

    # ---- primary tests ----
    lines.append("\n## Primary tests (paired Wilcoxon, Bonferroni ×9)\n")
    lines.append("| comparison | task | direction | mean_a | mean_b | p raw | p adj | pass? |")
    lines.append("|---|---|---|---|---|---|---|---|")

    verdicts = []
    for a, b in CONTRASTS:
        for task, alt in PRIMARY:
            sa, sb = seed_metric(by, a, task), seed_metric(by, b, task)
            seeds = sorted(set(sa) & set(sb))
            aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
            p = paired_wilcoxon_p(aa, bb, alternative=alt) if len(aa) >= 3 else 1.0
            padj = min(1.0, p * N_PRIMARY)
            ok = padj < ALPHA
            verdicts.append(ok)
            ma = float(np.mean(aa)) if aa else float("nan")
            mb = float(np.mean(bb)) if bb else float("nan")
            lines.append(
                f"| {a} vs {b} | {task} | {alt} | {ma:.3f} | {mb:.3f} | {p:.4g} | {padj:.4g} | "
                f"{'YES' if ok else 'no'} |"
            )

    # CTC contrast
    ctc_ok = True
    for task, alt in PRIMARY:
        sa, sb = seed_metric(by, "COHERENT", task), seed_metric(by, "INCOHERENT", task)
        seeds = sorted(set(sa) & set(sb))
        aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
        p = paired_wilcoxon_p(aa, bb, alternative=alt) if len(aa) >= 3 else 1.0
        if min(1.0, p * N_PRIMARY) >= ALPHA:
            ctc_ok = False

    overall_ok = all(verdicts)
    n_pass = sum(verdicts)
    if overall_ok:
        verd = (
            "SUPPORTED — COHERENT beats INCOHERENT, NO_GATE, and RAND_MATCH on ALL three "
            "primary endpoints after Bonferroni."
        )
    elif ctc_ok:
        verd = (
            "PARTIAL — CTC contrast supported (COHERENT beats INCOHERENT on all three tasks), "
            "but not all primary contrasts pass."
        )
    else:
        verd = (
            "NOT SUPPORTED — COHERENT does not beat INCOHERENT on all three primary endpoints "
            "after Bonferroni."
        )
    lines.append(f"\n**Primary CTC verdict:** {verd} ({n_pass}/{len(verdicts)} comparisons significant).\n")

    # ---- secondary myelin ----
    lines.append("\n## Secondary: myelin (uncorrected raw p)\n")
    myel_contrasts = [
        ("MYELIN_ONLY", "NO_GATE"),
        ("MYELIN_COHERENT", "MYELIN_ONLY"),
        ("MYELIN_COHERENT", "COHERENT"),
        ("MYELIN_COHERENT", "NO_GATE"),
    ]
    myel_dirs = [("XOR", "greater"), ("NARMA", "less"), ("CHAR", "less")]
    myelin_sync_beats = True
    for a, b in myel_contrasts:
        for task, alt in myel_dirs:
            sa, sb = seed_metric(by, a, task), seed_metric(by, b, task)
            seeds = sorted(set(sa) & set(sb))
            if len(seeds) < 3:
                continue
            aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
            p = paired_wilcoxon_p(aa, bb, alternative=alt)
            ma, mb = float(np.mean(aa)), float(np.mean(bb))
            lines.append(
                f"- {a} vs {b} on {task} ({alt}): {ma:.3f} vs {mb:.3f}, raw p={p:.4g}\n"
            )
            if a == "MYELIN_COHERENT" and b in ("MYELIN_ONLY", "COHERENT") and task in ("XOR", "NARMA"):
                if p >= 0.05:
                    myelin_sync_beats = False

    def mean(arm, task):
        m, _, _ = stats_for(by, arm, task)
        return m
    lines.append("\n**Myelin plain answer:** ")
    mo_n, mc_n, ng_n, co_n = mean("MYELIN_ONLY", "NARMA"), mean("MYELIN_COHERENT", "NARMA"), mean("NO_GATE", "NARMA"), mean("COHERENT", "NARMA")
    mo_x, mc_x, ng_x, co_x = mean("MYELIN_ONLY", "XOR"), mean("MYELIN_COHERENT", "XOR"), mean("NO_GATE", "XOR"), mean("COHERENT", "XOR")
    lines.append(
        f"MYELIN_ONLY NARMA={mo_n:.3f} XOR={mo_x:.3f}; MYELIN_COHERENT NARMA={mc_n:.3f} XOR={mc_x:.3f}; "
        f"NO_GATE NARMA={ng_n:.3f} XOR={ng_x:.3f}; COHERENT NARMA={co_n:.3f} XOR={co_x:.3f}. "
    )
    if myelin_sync_beats:
        lines.append("Myelin+sync beats both alone on XOR/NARMA at raw p<0.05 (see tests).\n")
    else:
        lines.append(
            "Myelin+sync does **not** clearly beat both myelin-alone and sync-alone on XOR/NARMA "
            "(raw p≥0.05 on at least one required contrast).\n"
        )

    # ---- secondary callosum ----
    lines.append("\n## Secondary: corpus callosum (uncorrected raw p; split-input XOR)\n")
    call_contrasts = [
        ("CALLOSUM", "RAND_BRIDGE"),
        ("CALLOSUM", "NO_BRIDGE"),
        ("CALLOSUM", "DENSE_BRIDGE"),
    ]
    callosum_beats_rand = False
    callosum_beats_none = False
    for a, b in call_contrasts:
        for task, alt in [("XOR", "greater"), ("NARMA", "less"), ("CHAR", "less")]:
            sa, sb = seed_metric(by, a, task), seed_metric(by, b, task)
            seeds = sorted(set(sa) & set(sb))
            if len(seeds) < 3:
                continue
            aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
            p = paired_wilcoxon_p(aa, bb, alternative=alt)
            ma, mb = float(np.mean(aa)), float(np.mean(bb))
            lines.append(
                f"- {a} vs {b} on {task} ({alt}): {ma:.3f} vs {mb:.3f}, raw p={p:.4g}\n"
            )
            if task == "XOR" and a == "CALLOSUM" and b == "RAND_BRIDGE" and p < 0.05 and ma > mb:
                callosum_beats_rand = True
            if task == "XOR" and a == "CALLOSUM" and b == "NO_BRIDGE" and p < 0.05 and ma > mb:
                callosum_beats_none = True

    lines.append("\n**Callosum plain answer:** ")
    for arm in CALLOSUM_ARMS:
        lines.append(f"{arm} XOR={mean(arm,'XOR'):.3f}; ")
    if callosum_beats_rand and callosum_beats_none:
        lines.append("CALLOSUM beats both random bridge and no bridge on split XOR (raw p<0.05).\n")
    elif callosum_beats_none and not callosum_beats_rand:
        lines.append("CALLOSUM beats no-bridge but not random bridge on split XOR.\n")
    elif callosum_beats_rand and not callosum_beats_none:
        lines.append("CALLOSUM beats random bridge but not no-bridge on split XOR (unexpected).\n")
    else:
        lines.append("CALLOSUM does **not** clearly beat random bridge and no bridge on split XOR.\n")

    # ---- HUB descriptive ----
    lines.append("\n## Secondary: HUB (descriptive)\n")
    for other in ("COHERENT", "RAND_MATCH"):
        for task, alt in PRIMARY:
            sa, sb = seed_metric(by, "HUB", task), seed_metric(by, other, task)
            seeds = sorted(set(sa) & set(sb))
            if len(seeds) < 3:
                continue
            aa = [sa[s] for s in seeds]; bb = [sb[s] for s in seeds]
            p = paired_wilcoxon_p(aa, bb, alternative=alt)
            lines.append(f"- HUB vs {other} on {task} ({alt}): raw p={p:.4g}\n")

    lines.append("\n## Learning curves\nSee `figs/oscillation_gating.png`.\n")
    lines.append("\n## Honesty notes\n")
    lines.append("- Full protocol: n=12 seeds (1100–1111), XOR+CHAR+NARMA, N=160, epochs≈45 × 20 steps.\n")
    lines.append("- Fast pilot (2026-10-29) was undertrained (XOR≈chance); baselines here required XOR≥0.70 on tune seeds.\n")
    lines.append("- Hyperparams tuned on seeds 90–91 only (NO_GATE + RAND_MATCH floor + grid).\n")
    lines.append("- Gate mean-normalized so COHERENT vs INCOHERENT differ in timing, not mean gain.\n")
    lines.append("- Myelin delays from real H01 myelin_frac→{1..5}, RULE_FF_DEPTH assignment on inter edges.\n")
    lines.append("- Callosum DENSE_BRIDGE has higher nnz (not edge-matched); others edge-matched to CALLOSUM.\n")
    lines.append("- Null allowed.\n")

    lines.append("\n## Headline answers (for report-back)\n")
    lines.append(f"1. **Coherence beats incoherence?** {'YES' if ctc_ok else 'NO'} (primary CTC).\n")
    lines.append(
        f"2. **Myelin+sync beats either alone?** {'YES (raw)' if myelin_sync_beats else 'NO / unclear (raw)'}.\n"
    )
    lines.append(
        f"3. **Callosum beats random + no bridge?** "
        f"{'YES (raw XOR)' if (callosum_beats_rand and callosum_beats_none) else 'NO / partial (raw XOR)'}.\n"
    )
    lines.append(
        "\n## Secondary FLY followup\n"
        "See `RESULTS_FLY_FOLLOWUP.md` (separate labeled secondary run of FLY EI_SBM "
        "vs NO_GATE / MYELIN_ONLY / CALLOSUM; not part of primary prereg).\n"
    )

    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS_oscillation_gating.md").write_text(text)
    (HERE / "RESULTS.md").write_text(text)

    verdict = dict(
        primary_ctc=ctc_ok,
        primary_overall=overall_ok,
        n_pass=n_pass,
        n_tests=len(verdicts),
        myelin_sync_beats_either_alone_raw=myelin_sync_beats,
        callosum_beats_rand_xor_raw=callosum_beats_rand,
        callosum_beats_nobridge_xor_raw=callosum_beats_none,
        verdict_text=verd,
        means={arm: {t: table[arm][t][0] for t in ("XOR", "CHAR", "NARMA")} for arm in table},
    )
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("wrote RESULTS_oscillation_gating.md")
    print("VERDICT:", verd)
    print("CTC_BEATS_INCOHERENT:", ctc_ok)
    print("MYELIN_SYNC_BEATS:", myelin_sync_beats)
    print("CALLOSUM_BEATS_RAND:", callosum_beats_rand, "BEATS_NONE:", callosum_beats_none)
    print("N_PASS:", n_pass)
    print("MEANS_TABLE")
    for arm in ARMS:
        if arm not in table:
            continue
        xm, xs, _ = table[arm]["XOR"]
        cm, cs, _ = table[arm]["CHAR"]
        nm, ns, _ = table[arm]["NARMA"]
        print(f"  {arm}: XOR={xm:.3f}±{xs:.3f} CHAR={cm:.3f}±{cs:.3f} NARMA={nm:.3f}±{ns:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("no matplotlib", e)
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    colors = {
        "COHERENT": "#1b9e77", "INCOHERENT": "#d95f02", "NO_GATE": "#7570b3",
        "HUB": "#e7298a", "RAND_MATCH": "#666666",
        "MYELIN_ONLY": "#a6761d", "MYELIN_COHERENT": "#66a61e",
        "CALLOSUM": "#e6ab02", "RAND_BRIDGE": "#a6cee3", "NO_BRIDGE": "#1f78b4",
        "DENSE_BRIDGE": "#b2df8a",
    }
    plot_arms = CTC_ARMS + MYELIN_ARMS
    for ax, task, ylabel in zip(
        axes, ["XOR", "CHAR", "NARMA"], ["XOR accuracy ↑", "CHAR CE ↓", "NARMA NRMSE ↓"]
    ):
        for arm in plot_arms:
            rs = by.get((arm, task), [])
            curves = [r["curve"] for r in rs if r.get("curve")]
            if not curves:
                continue
            L = min(len(c) for c in curves)
            M = np.array([[c[i]["metric"] for i in range(L)] for c in curves], float)
            mean_c = M.mean(0)
            sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(L)
            ep = np.arange(L)
            ax.plot(ep, mean_c, label=arm, color=colors.get(arm), lw=1.6)
            ax.fill_between(ep, mean_c - sem, mean_c + sem, color=colors.get(arm, "gray"), alpha=0.12)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(task); ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("oscillation_gating FULL (CTC + myelin)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "oscillation_gating.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(6, 3.5))
    for arm in CALLOSUM_ARMS:
        rs = by.get((arm, "XOR"), [])
        curves = [r["curve"] for r in rs if r.get("curve")]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        M = np.array([[c[i]["metric"] for i in range(L)] for c in curves], float)
        mean_c = M.mean(0)
        sem = M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(L)
        ep = np.arange(L)
        ax.plot(ep, mean_c, label=arm, color=colors.get(arm), lw=1.8)
        ax.fill_between(ep, mean_c - sem, mean_c + sem, color=colors.get(arm, "gray"), alpha=0.15)
    ax.set_xlabel("epoch"); ax.set_ylabel("split XOR accuracy ↑")
    ax.set_title("corpus callosum arms"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGS / "oscillation_gating_callosum.png", dpi=140)
    print("wrote figs/oscillation_gating.png and callosum fig")


if __name__ == "__main__":
    main()

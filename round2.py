"""Round 2: tuning sweep, tuned test, cell-type knockouts, structure. See PREREGISTRATION_2.md."""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json, itertools, numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.stats import mannwhitneyu, wilcoxon
from multiprocessing import Pool
import flybrain_reservoir as F

HERE = F.HERE; OUT = os.path.join(HERE, "out2"); os.makedirs(OUT, exist_ok=True)
LEAKS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]; RHOS = [0.5, 0.7, 0.9, 0.95, 0.99, 1.1]
TUNE_SEEDS = [900, 901]; TEST_SEEDS = list(range(12))

def unit(N, pre, post, w):
    W = sp.csr_matrix((w, (post, pre)), shape=(N, N))
    r = abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=10000, tol=1e-6)[0])
    return (W / r).toarray()

def run(W, Win, b, U, leak):
    x = np.zeros(W.shape[0]); X = np.empty((len(U), W.shape[0]))
    for t, u in enumerate(U):
        x = (1 - leak) * x + leak * np.tanh(W @ x + Win[:, 0] * u + b); X[t] = x
    return X

def inp(N, rng, d=1): return rng.uniform(-1, 1, (N, d)) * 0.5, rng.uniform(-.1, .1, N)

def mc(W, leak, rng, val):
    wash, tr, te, D = (200, 1500, 500, 100) if val else (200, 2000, 1000, 100)
    u = rng.uniform(-1, 1, wash + tr + te); Win, b = inp(W.shape[0], rng); X = run(W, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6); P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))

def narma(W, leak, rng, val):
    wash, tr, te = (200, 1500, 500) if val else (200, 2000, 1000)
    T = wash + tr + te; u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    Win, b = inp(W.shape[0], rng); X = run(W, Win, b, u, leak)
    B = F.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6); p = F.pred(X[wash + tr:], B)
    v = float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:]))
    return v if np.isfinite(v) else 1e9

def t0(W, leak, rng, val, word):
    n = 6; ntr, nte = (512, 256) if val else (512, 512)
    Xin = rng.standard_normal((ntr + nte, n)) / np.sqrt(n); Y = np.tanh(Xin @ F.bench.weights_from_word(word))
    Win, b = inp(W.shape[0], rng, n); S = np.zeros((len(Xin), W.shape[0])); drive = Xin @ Win.T + b
    for _ in range(10): S = (1 - leak) * S + leak * np.tanh(S @ W.T + drive)
    B = F.ridge(S[:ntr], Y[:ntr], 1e-4); return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2))

G = {}
def graph(cond, s):
    key = (cond, s)
    if key not in G:
        G.clear(); G[key] = np.load(os.path.join(OUT, f"g_{cond}_{s}.npy"), mmap_mode="r")[:]
    return G[key]

WORD = F.bench.make_regression_target(F.bench.RNG_SEED)[3]
def job_sweep(a):
    cond, s, leak, rho = a; W = graph(cond, gs(cond, s)) * rho
    return dict(cond=cond, seed=s, leak=leak, rho=rho, MC=mc(W, leak, np.random.default_rng(s), True),
                NARMA=narma(W, leak, np.random.default_rng(s + 1), True), T0=t0(W, leak, np.random.default_rng(s + 2), True, WORD))

def job_test(a):
    cond, s, task, leak, rho = a; W = graph(cond, gs(cond, s)) * rho; g = np.random.default_rng(10 * s + {"MC": 0, "NARMA": 1, "T0": 2}[task])
    v = {"MC": lambda: mc(W, leak, g, False), "NARMA": lambda: narma(W, leak, g, False), "T0": lambda: t0(W, leak, g, False, WORD)}[task]()
    return dict(cond=cond, seed=s, task=task, value=v)

KO = {}
def job_ko(a):
    grp, idx, s, leak, rho = a
    if "W" not in KO: KO["W"] = F.load_cx()
    N, pre, post, w = KO["W"]; keep = ~(np.isin(pre, idx) | np.isin(post, idx))
    W = unit(N, pre[keep], post[keep], w[keep]) * rho
    return dict(group=grp, seed=s, MC=mc(W, leak, np.random.default_rng(5000 + s), False))

def build_graphs():
    N, pre, post, w = F.load_cx(); E = len(pre)
    np.save(os.path.join(OUT, "g_REAL_0.npy"), unit(N, pre, post, w))
    for s in TUNE_SEEDS + TEST_SEEDS:
        g = np.random.default_rng(1000 + s)
        np.save(os.path.join(OUT, f"g_ER_{s}.npy"), unit(N, *F.er(N, E, w, g)))
        dp, dq, dw, _ = F.deg(N, pre, post, w, g); np.save(os.path.join(OUT, f"g_DEG_{s}.npy"), unit(N, dp, dq, dw))

def gs(cond, s): return 0 if cond == "REAL" else s

def groups():
    ann = pd.read_csv(os.path.join(HERE, "data/ann.tsv"), sep="\t", low_memory=False, usecols=["root_id", "cell_class", "cell_type"])
    ann = ann[ann.cell_class == "CX"]
    ids = np.array(sorted(set(ann.root_id))); pos = {r: i for i, r in enumerate(ids)}
    fam = ann.cell_type.fillna("NA").str.replace(r"[_\d].*", "", regex=True).str.replace(r"[a-z]+$", "", regex=True)
    fam = fam.where(~fam.str.startswith("vDelta"), "vDelta").where(~fam.str.startswith("hDelta"), "hDelta").replace({"D": "Delta7"})
    out = {}
    for f, sub in ann.groupby(fam.values):
        if len(sub) >= 20 and f != "NA": out[f] = np.array([pos[r] for r in sub.root_id])
    return out, len(ids)

def structure():
    N, pre, post, w = F.load_cx(); E = len(pre); g = np.random.default_rng(1)
    graphs = {"REAL": (pre, post, w), "ER": F.er(N, E, w, g), "DEG": F.deg(N, pre, post, w, g)[:3]}
    rows = {}; eigd = {}
    for c, (p, q, ww) in graphs.items():
        A = sp.csr_matrix((np.ones(len(p)), (p, q)), shape=(N, N)); A.data[:] = 1
        recip = A.multiply(A.T).sum() / A.sum()
        Ad = A.toarray().astype(np.float32); A2 = Ad @ Ad
        c2 = float(np.trace(A2)) / 2; c3 = float(np.einsum("ij,ji->", A2, Ad)) / 3
        Wd = np.zeros((N, N)); np.add.at(Wd, (q, p), ww)
        ev = np.linalg.eigvals(Wd); ev = ev / np.abs(ev).max(); eigd[c] = ev
        inE = np.clip(Wd, 0, None).sum(1); inI = -np.clip(Wd, None, 0).sum(1)
        bal = (inE - inI) / np.maximum(inE + inI, 1)
        rows[c] = dict(reciprocity=float(recip), two_cycles=c2, three_cycles=c3,
                       EI_balance_mean=float(bal.mean()), EI_balance_std=float(bal.std()),
                       frac_eig_abs_gt_0_5=float(np.mean(np.abs(ev) > 0.5)), eig_abs_mean=float(np.abs(ev).mean()),
                       frac_real_eigs=float(np.mean(np.abs(ev.imag) < 1e-9)))
    return rows, eigd

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not os.path.exists(os.path.join(OUT, "g_DEG_11.npy")): build_graphs()
    jobs = [(c, s, l, r) for c in ["REAL", "ER", "DEG"] for s in TUNE_SEEDS for l in LEAKS for r in RHOS]
    jobs.sort(key=lambda j: (j[0], j[1]))
    if not os.path.exists(os.path.join(OUT, "sweep.csv")):
        with Pool(7) as p: sw = pd.DataFrame(p.map(job_sweep, jobs, chunksize=36))
        sw.to_csv(os.path.join(OUT, "sweep.csv"), index=False)
    sw = pd.read_csv(os.path.join(OUT, "sweep.csv"))
    # REAL uses one graph -> its two tuning rows per cell differ only by input seed; dedupe by mean over rows
    m = sw.groupby(["cond", "leak", "rho"])[["MC", "NARMA", "T0"]].mean().reset_index()
    best = {}
    for c in ["REAL", "ER", "DEG"]:
        mm = m[m.cond == c]
        best[c] = {"MC": mm.loc[mm.MC.idxmax(), ["leak", "rho"]].tolist(),
                   "NARMA": mm.loc[mm.NARMA.idxmin(), ["leak", "rho"]].tolist(),
                   "T0": mm.loc[mm.T0.idxmin(), ["leak", "rho"]].tolist()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5)); vmax = m.MC.max()
    for a_, c in zip(ax, ["REAL", "ER", "DEG"]):
        H = m[m.cond == c].pivot(index="leak", columns="rho", values="MC")
        im = a_.imshow(H.values, vmin=0, vmax=vmax, origin="lower", cmap="viridis")
        a_.set_xticks(range(len(RHOS)), RHOS); a_.set_yticks(range(len(LEAKS)), LEAKS)
        a_.set_xlabel("spectral radius"); a_.set_ylabel("leak"); a_.set_title(f"{c} validation MC")
        for i in range(len(LEAKS)):
            for j in range(len(RHOS)): a_.text(j, i, f"{H.values[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w")
    plt.colorbar(im, ax=ax); plt.savefig(os.path.join(OUT, "mc_heatmaps.png"), dpi=120); plt.close()
    tj = [(c, s, t, *best[c][t]) for c in ["REAL", "ER", "DEG"] for t in ["MC", "NARMA", "T0"] for s in TEST_SEEDS]
    tj.sort(key=lambda j: (j[0], j[1]))
    with Pool(7) as p: te = pd.DataFrame(p.map(job_test, tj))
    te.to_csv(os.path.join(OUT, "tuned_test.csv"), index=False)
    tuned = {}
    for t in ["MC", "NARMA", "T0"]:
        d = {c: te[(te.cond == c) & (te.task == t)].value.values for c in ["REAL", "ER", "DEG"]}
        tuned[t] = {c: [float(v.mean()), float(v.std()), best[c][t]] for c, v in d.items()}
        for c in ["ER", "DEG"]: tuned[t][f"p_REAL_vs_{c}"] = float(mannwhitneyu(d["REAL"], d[c]).pvalue)
    res = {"best_params": best, "tuned_test": tuned}
    print(json.dumps(res, indent=1), flush=True)
    win = all(tuned["MC"]["REAL"][0] > tuned["MC"][c][0] and tuned["MC"][f"p_REAL_vs_{c}"] < 0.05 for c in ["ER", "DEG"])
    res["mc_survived"] = win
    if win:
        grp, N = groups(); leak, rho = best["REAL"]["MC"]; rng = np.random.default_rng(77)
        kj = [("NONE", np.array([], int), s, leak, rho) for s in TEST_SEEDS]
        for gname, idx in grp.items():
            for s in TEST_SEEDS:
                kj.append((gname, idx, s, leak, rho)); kj.append(("RAND_" + gname, rng.choice(N, len(idx), replace=False), s, leak, rho))
        with Pool(6) as p: ko = pd.DataFrame(p.map(job_ko, kj))
        ko.to_csv(os.path.join(OUT, "knockout.csv"), index=False)
        rows = []; nb = len(grp); base = ko[ko.group == "NONE"].MC.mean()
        for gname, idx in grp.items():
            a = ko[ko.group == gname].sort_values("seed").MC.values; r = ko[ko.group == "RAND_" + gname].sort_values("seed").MC.values
            p = float(wilcoxon(r, a).pvalue)
            rows.append(dict(group=gname, n=len(idx), MC_KO=a.mean(), MC_KO_std=a.std(), MC_randKO=r.mean(),
                             effect=float((r - a).mean()), effect_std=float((r - a).std()), p=p, p_bonf=min(1, p * nb)))
        kt = pd.DataFrame(rows).sort_values("effect", ascending=False); kt.to_csv(os.path.join(OUT, "knockout_ranked.csv"), index=False)
        res["knockout_baseline_MC"] = float(base); res["knockout"] = kt.to_dict("records")
        plt.figure(figsize=(10, 4.5)); col = ["C3" if p < 0.05 else "gray" for p in kt.p_bonf]
        plt.bar(kt.group, kt.effect, yerr=kt.effect_std, color=col); plt.axhline(0, c="k", lw=.8)
        plt.ylabel("MC(random KO) − MC(type KO)\n(>0 = type matters more than random)"); plt.xticks(rotation=60)
        plt.title("Cell-type knockouts (red = Bonferroni p<0.05)"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "knockout.png"), dpi=120); plt.close()
    st, eigd = structure(); res["structure"] = st
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a_, (c, ev) in zip(ax, eigd.items()):
        a_.scatter(ev.real, ev.imag, s=1); a_.set_aspect("equal"); a_.set_title(f"{c} eigenvalues (scaled to radius 1)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "eigenspectra.png"), dpi=110); plt.close()
    json.dump(res, open(os.path.join(OUT, "round2_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

if __name__ == "__main__":
    main()

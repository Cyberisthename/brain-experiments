"""Fly central-complex connectome vs random wiring as a reservoir. See PREREGISTRATION.md.
Run: python3 flybrain_reservoir.py   (needs data/ as described in README.md)"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CHATBOT_REPO", "/workspace/chatbot-site-src")
SEEDS = list(range(12))
LEAK, RHO = 0.5, 0.95

# ---- user's benchmark functions (imported, not copied) ----
spec = importlib.util.spec_from_file_location("bench", os.path.join(REPO, "scripts", "braid_vs_gd_benchmark.py"))
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

def load_cx():
    ann = pd.read_csv(os.path.join(HERE, "data/ann.tsv"), sep="\t", low_memory=False, usecols=["root_id", "cell_class"])
    ids = np.array(sorted(set(ann.root_id[ann.cell_class == "CX"])))
    d = pd.read_parquet(os.path.join(HERE, "data/Drosophila_brain_model/Connectivity_783.parquet"),
                        columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"])
    d = d[d.Presynaptic_ID.isin(ids) & d.Postsynaptic_ID.isin(ids)]
    idx = {r: i for i, r in enumerate(ids)}
    pre = d.Presynaptic_ID.map(idx).values; post = d.Postsynaptic_ID.map(idx).values
    w = d["Excitatory x Connectivity"].values.astype(float)
    return len(ids), pre, post, w

def to_matrix(N, pre, post, w):
    # row = postsynaptic, col = presynaptic (x_post += W[post,pre] x_pre)
    W = sp.csr_matrix((w, (post, pre)), shape=(N, N))
    r = abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-6)[0])
    return (W * (RHO / r)).toarray()

def er(N, E, w, rng):
    keys = set()
    while len(keys) < E:
        k = rng.integers(0, N * N, size=2 * (E - len(keys)))
        k = k[(k // N) != (k % N)]
        for x in k:
            keys.add(int(x))
            if len(keys) == E: break
    k = np.array(list(keys)); return k // N, k % N, rng.permutation(w)

def deg(N, pre, post, w, rng):
    post = rng.permutation(post)
    for _ in range(200):
        key = pre.astype(np.int64) * N + post
        _, first = np.unique(key, return_index=True)
        bad = np.ones(len(key), bool); bad[first] = False; bad |= pre == post
        if not bad.any(): break
        b = np.where(bad)[0]; others = rng.integers(0, len(post), len(b))
        post[b], post[others] = post[others].copy(), post[b].copy()
    key = pre.astype(np.int64) * N + post
    _, keep = np.unique(key, return_index=True); keep = keep[pre[keep] != post[keep]]
    return pre[keep], post[keep], w[keep], len(pre) - len(keep)

def run_seq(W, Win, b, U):
    x = np.zeros(W.shape[0]); X = np.empty((len(U), W.shape[0]))
    for t, u in enumerate(U):
        x = (1 - LEAK) * x + LEAK * np.tanh(W @ x + Win @ np.atleast_1d(u) + b); X[t] = x
    return X

def ridge(X, Y, lam):
    X = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
def pred(X, B): return np.hstack([X, np.ones((len(X), 1))]) @ B

def narma(W, rng, wash=200, tr=2000, te=1000):
    T = wash + tr + te; u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    Win = rng.uniform(-1, 1, (W.shape[0], 1)) * 0.5; b = rng.uniform(-.1, .1, W.shape[0])
    X = run_seq(W, Win, b, u)
    B = ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6); p = pred(X[wash + tr:], B)
    return float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:]))

def memcap(W, rng, wash=200, tr=2000, te=1000, D=100):
    T = wash + tr + te; u = rng.uniform(-1, 1, T)
    Win = rng.uniform(-1, 1, (W.shape[0], 1)) * 0.5; b = rng.uniform(-.1, .1, W.shape[0])
    X = run_seq(W, Win, b, u)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6); P = pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.corrcoef(P[:, k], Yt[:, k])[0, 1] ** 2 for k in range(D)))

def static_feats(W, Xin, rng, steps=10):
    Win = rng.uniform(-1, 1, (W.shape[0], Xin.shape[1])) * 0.5; b = rng.uniform(-.1, .1, W.shape[0])
    S = np.zeros((len(Xin), W.shape[0])); drive = Xin @ Win.T + b
    for _ in range(steps): S = (1 - LEAK) * S + LEAK * np.tanh(S @ W.T + drive)
    return S

def teacher_task(W, rng, words):
    n = bench.N_STRANDS
    Xin = rng.standard_normal((1024, n)) / np.sqrt(n)
    Ys = [np.tanh(Xin @ bench.weights_from_word(w)) for w in words]
    F = static_feats(W, Xin, rng); tr, te = slice(0, 512), slice(512, 1024)
    Ymean = np.mean(Ys, 0)   # minimiser of mean-over-teachers MSE
    B = ridge(F[tr], Ymean[tr], 1e-4); P = pred(F[te], B)
    return float(np.mean([np.mean((P - Y[te]) ** 2) for Y in Ys]))

def main():
    N, pre, post, w = load_cx(); E = len(pre)
    print(f"CX: N={N} E={E}")
    t0_word = bench.make_regression_target(bench.RNG_SEED)[3]
    t1 = bench.T1(); adapter_words = t1.teacher_words
    print("adapter tags used:", t1.tags_used)
    Wreal = to_matrix(N, pre, post, w)
    res = []; dropped = []
    for s in SEEDS:
        g = np.random.default_rng(1000 + s)
        mats = {"REAL": Wreal, "ER": to_matrix(N, *er(N, E, w, g))}
        dp, dq, dw, nd = deg(N, pre, post, w, g); dropped.append(nd)
        mats["DEG"] = to_matrix(N, dp, dq, dw)
        rand_words = [bench.random_word(np.random.RandomState(50000 + 3 * s + k)) for k in range(3)]
        for c, W in mats.items():
            r = {"seed": s, "cond": c}
            r["NARMA10_NRMSE"] = narma(W, np.random.default_rng(s))
            r["MemoryCapacity"] = memcap(W, np.random.default_rng(s + 100))
            r["T0_MSE"] = teacher_task(W, np.random.default_rng(s + 200), [t0_word])
            r["T1_MSE_adapters"] = teacher_task(W, np.random.default_rng(s + 300), adapter_words)
            r["T1_MSE_noadapters"] = teacher_task(W, np.random.default_rng(s + 300), rand_words)
            res.append(r); print(r, flush=True)
    df = pd.DataFrame(res); os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "out/results.csv"), index=False)
    metrics = [m for m in df.columns if m not in ("seed", "cond")]
    summ = {"N": N, "E": E, "deg_shuffle_dropped_edges": dropped, "adapter_tags": t1.tags_used, "stats": {}}
    for m in metrics:
        st = {c: [float(df[df.cond == c][m].mean()), float(df[df.cond == c][m].std())] for c in ["REAL", "ER", "DEG"]}
        for c in ["ER", "DEG"]:
            st[f"p_REAL_vs_{c}"] = float(mannwhitneyu(df[df.cond == "REAL"][m], df[df.cond == c][m]).pvalue)
        summ["stats"][m] = st
    for c in ["REAL", "ER", "DEG"]:
        a = df[df.cond == c]
        summ["stats"][f"adapters_vs_none_{c}_p"] = float(mannwhitneyu(a.T1_MSE_adapters, a.T1_MSE_noadapters).pvalue)
    json.dump(summ, open(os.path.join(HERE, "out/summary.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a_, m in zip(ax, ["NARMA10_NRMSE", "MemoryCapacity", "T0_MSE"]):
        data = [df[df.cond == c][m] for c in ["REAL", "ER", "DEG"]]
        a_.boxplot(data, tick_labels=["fly CX", "Erdős–Rényi", "degree-shuffle"]); a_.set_title(m)
    for i, c in enumerate(["REAL", "ER", "DEG"]):
        a = df[df.cond == c]
        ax[3].scatter(np.full(len(a), i - .1), a.T1_MSE_adapters, c="C0", label="adapters" if i == 0 else None)
        ax[3].scatter(np.full(len(a), i + .1), a.T1_MSE_noadapters, c="C1", label="no adapters" if i == 0 else None)
    ax[3].set_xticks([0, 1, 2], ["fly CX", "ER", "deg-shuffle"]); ax[3].set_title("T1 MSE"); ax[3].legend()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "out/comparison.png"), dpi=120)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()

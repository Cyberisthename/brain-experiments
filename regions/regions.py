"""Grow CX with real regions vs random neurons. See PREREGISTRATION_regions.md."""
import os, sys
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain")
import time, json, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.stats import mannwhitneyu
from multiprocessing import Pool
import flybrain_reservoir as F
HERE = "/workspace/flybrain"; OUT = os.path.join(HERE, "regions/out"); G = os.path.join(OUT, "graphs"); os.makedirs(G, exist_ok=True)
LEAKS = [0.2, 0.5, 0.9]; RHOS = [0.7, 0.95, 1.1]; TUNE = [900, 901]; TEST = list(range(8)); NR = 2875
STEPS = [("S0 CX", ["CX"], None), ("S1 +mushroom body", ["Kenyon_Cell", "MBON", "DAN"], None),
         ("S2 +LH/olfactory PN/TuBu", ["ALPN", "LHLN", "LHCENT", "TuBu"], None),
         ("S3 +rest of central brain", None, "central"), ("S4 +visual projection", None, "visual_projection")]
NSTEP = int(os.environ.get("NSTEP", 5)); CONDS = ["REAL", "ADD", "SHUF"]

def build():
    ann = pd.read_csv(os.path.join(HERE, "data/ann.tsv"), sep="\t", low_memory=False, usecols=["root_id", "super_class", "cell_class"])
    d = pd.read_parquet(os.path.join(HERE, "data/Drosophila_brain_model/Connectivity_783.parquet"),
                        columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"])
    cx = set(ann.root_id[ann.cell_class == "CX"]); inc = set(); info = []
    for k, (name, cls, sup) in enumerate(STEPS[:NSTEP]):
        inc |= set(ann.root_id[ann.cell_class.isin(cls)] if cls else ann.root_id[ann.super_class == sup])
        ids = np.array(sorted(inc)); idx = pd.Series(np.arange(len(ids)), index=ids)
        e = d[d.Presynaptic_ID.isin(inc) & d.Postsynaptic_ID.isin(inc)]
        pre = idx[e.Presynaptic_ID].values; post = idx[e.Postsynaptic_ID].values; w = e["Excitatory x Connectivity"].values.astype(float)
        N = len(ids); iscx = np.isin(ids, list(cx)); inner = iscx[pre] & iscx[post]
        info.append(dict(step=name, neurons=N, added_neurons=N - len(cx), edges=len(pre), edges_touching_added=int((~inner).sum())))
        save(f"{k}_REAL_0", N, pre, post, w)
        for s in TUNE + TEST:
            g = np.random.default_rng(1000 + s + 100 * k)
            q = post.copy(); o = ~inner; q[o] = g.permutation(q[o])
            key = pre.astype(np.int64) * N + q; _, keep = np.unique(key, return_index=True); keep = keep[pre[keep] != q[keep]]
            save(f"{k}_ADD_{s}", N, pre[keep], q[keep], w[keep]); info[-1].setdefault("add_dropped", []).append(len(pre) - len(keep))
            dp, dq, dw, dr = F.deg(N, pre, post, w, g); save(f"{k}_SHUF_{s}", N, dp, dq, dw); info[-1].setdefault("shuf_dropped", []).append(dr)
        print(info[-1], flush=True)
    json.dump(info, open(os.path.join(OUT, "steps.json"), "w"), indent=1, default=int)

def save(name, N, pre, post, w):
    W = sp.csr_matrix((w, (post, pre)), shape=(N, N))
    r = abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-6)[0])
    sp.save_npz(os.path.join(G, name + ".npz"), (W / r).tocsr())

C = {}
def graph(k, c, s):
    key = f"{k}_{c}_{0 if c == 'REAL' else s}"
    if key not in C: C.clear(); C[key] = sp.load_npz(os.path.join(G, key + ".npz")).tocsr()
    return C[key]

def run(W, Win, b, U, leak, ro):
    x = np.zeros(W.shape[0]); X = np.empty((len(U), len(ro)))
    for t, u in enumerate(U):
        x = (1 - leak) * x + leak * np.tanh(W @ x + Win[:, 0] * u + b); X[t] = x[ro]
    return X
def inp(N, rng, d=1): return rng.uniform(-1, 1, (N, d)) * 0.5, rng.uniform(-.1, .1, N)
def ro_idx(N, s): return np.arange(N) if N <= NR else np.sort(np.random.default_rng(7000 + s).choice(N, NR, replace=False))

def mc(W, leak, rng, val, ro):
    wash, tr, te, D = (200, 1500, 500, 100) if val else (200, 2000, 1000, 100)
    u = rng.uniform(-1, 1, wash + tr + te); Win, b = inp(W.shape[0], rng); X = run(W, Win, b, u, leak, ro)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6); P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
def narma(W, leak, rng, val, ro):
    wash, tr, te = (200, 1500, 500) if val else (200, 2000, 1000)
    T = wash + tr + te; u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(10, T - 1): y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    Win, b = inp(W.shape[0], rng); X = run(W, Win, b, u, leak, ro)
    B = F.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6); p = F.pred(X[wash + tr:], B)
    v = float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:])); return v if np.isfinite(v) else 1e9
WORD = F.bench.make_regression_target(F.bench.RNG_SEED)[3]
def t0(W, leak, rng, val, ro):
    n = 6; ntr, nte = (512, 256) if val else (512, 512)
    Xin = rng.standard_normal((ntr + nte, n)) / np.sqrt(n); Y = np.tanh(Xin @ F.bench.weights_from_word(WORD))
    Win, b = inp(W.shape[0], rng, n); S = np.zeros((len(Xin), W.shape[0])); drive = Xin @ Win.T + b; WT = W.T.tocsr()
    for _ in range(10): S = (1 - leak) * S + leak * np.tanh((WT.T @ S.T).T + drive)
    S = S[:, ro]; B = F.ridge(S[:ntr], Y[:ntr], 1e-4); return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2))
TASK = {"MC": mc, "NARMA": narma, "T0": t0}; OFF = {"MC": 0, "NARMA": 1, "T0": 2}

def job(a):
    k, c, s, task, leak, rho, val = a; W = graph(k, c, s) * rho
    g = np.random.default_rng(s + OFF[task] if val else 10 * s + OFF[task])
    t = time.time(); v = TASK[task](W, leak, g, val, ro_idx(W.shape[0], s))
    return dict(step=k, cond=c, seed=s, task=task, leak=leak, rho=rho, value=v, sec=time.time() - t)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(OUT, "steps.json")): build()
    P = Pool(4)
    sf = os.path.join(OUT, "sweep.csv")
    if not os.path.exists(sf):
        J = sorted([(k, c, s, t, l, r, True) for k in range(NSTEP) for c in CONDS for s in TUNE for t in TASK for l in LEAKS for r in RHOS], key=lambda j: j[:3])
        pd.DataFrame(P.map(job, J, chunksize=9)).to_csv(sf, index=False)
    sw = pd.read_csv(sf); m = sw.groupby(["step", "cond", "task", "leak", "rho"]).value.mean().reset_index()
    best = {}
    for (k, c, t), mm in m.groupby(["step", "cond", "task"]):
        i = mm.value.idxmax() if t == "MC" else mm.value.idxmin(); best[(k, c, t)] = (mm.loc[i, "leak"], mm.loc[i, "rho"])
    J = sorted([(k, c, s, t, *best[(k, c, t)], False) for k in range(NSTEP) for c in CONDS for t in TASK for s in TEST], key=lambda j: j[:3])
    pd.DataFrame(P.map(job, J, chunksize=3)).to_csv(os.path.join(OUT, "test.csv"), index=False)
    json.dump({f"{k}|{c}|{t}": v for (k, c, t), v in best.items()}, open(os.path.join(OUT, "best.json"), "w"), indent=1, default=float)

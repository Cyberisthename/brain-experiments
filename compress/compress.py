"""Compression test. See PREREGISTRATION_compress.md. Resumable: out/jobs.jsonl"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigs, svds
from scipy.cluster.vq import kmeans2
from multiprocessing import Pool
import flybrain_reservoir as F
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); JF = os.path.join(OUT, "jobs.jsonl")
RG = "/workspace/flybrain/regions/out/graphs"; REAL = sp.load_npz(f"{RG}/0_REAL_0.npz").tocsr(); N = REAL.shape[0]
LEAKS = [0.2, 0.5, 0.9]; RHOS = [0.7, 0.95, 1.1]; TUNE = [900, 901]; TEST = list(range(8))
WIRES = ["REAL", "ER", "DEG"]; METH = [("FULL", 1.0)] + [(m, f) for m in ["RANDP", "MAGP", "LOWRANK", "CONSOL"] for f in [0.5, 0.25, 0.1]]

def radius(W):
    if sp.issparse(W): return abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-6)[0])
    return np.max(np.abs(np.linalg.eigvals(W)))
def base(w, s):
    if w == "REAL": return REAL
    if w == "DEG": return sp.load_npz(f"{RG}/0_SHUF_{s}.npz").tocsr()
    g = np.random.default_rng(3000 + s); c = REAL.tocoo(); E = c.nnz
    k = np.unique(g.integers(0, N * N, int(E * 1.1))); k = g.permutation(k[(k // N) != (k % N)])[:E]
    W = sp.csr_matrix((g.permutation(c.data), (k // N, k % N)), shape=(N, N)); return W / radius(W)
def compress(W, m, f, s):
    g = np.random.default_rng(4000 + s); k = int(round(f * N))
    if m == "FULL": return W
    if m in ("RANDP", "MAGP"):
        A = abs(W); keep = g.choice(N, k, replace=False) if m == "RANDP" else np.argsort(-(np.ravel(A.sum(0)) + np.ravel(A.sum(1))))[:k]
        keep = np.sort(keep); Wc = W[keep][:, keep].tocsr()
    elif m == "LOWRANK":
        U, S, Vt = np.linalg.svd(W.toarray()); Wc = (Vt[:k] @ U[:, :k]) * S[:k]
    else:
        U, S, Vt = svds(W, k=32); X = np.hstack([U * S, Vt.T * S]); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        _, lab = kmeans2(X, k, seed=int(4000 + s), minit="++"); _, lab = np.unique(lab, return_inverse=True); m_ = lab.max() + 1
        P = sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, m_)); sz = np.ravel(P.sum(0))
        Wc = (sp.diags(1 / sz) @ P.T @ W @ P).tocsr()
        if Wc.nnz > 0.3 * m_ * m_: Wc = Wc.toarray()
    r = radius(Wc); return Wc / r if r > 0 else Wc

def inp(n, rng, d=1): return rng.uniform(-1, 1, (n, d)) * 0.5, rng.uniform(-.1, .1, n)
def run(W, Win, b, U, leak):
    x = np.zeros(W.shape[0]); X = np.empty((len(U), W.shape[0])); t = time.perf_counter()
    for i, u in enumerate(U): x = (1 - leak) * x + leak * np.tanh(W @ x + Win[:, 0] * u + b); X[i] = x
    return X, time.perf_counter() - t
def mc(W, leak, rng, val):
    wash, tr, te, D = (200, 1500, 500, 100) if val else (200, 2000, 1000, 100)
    u = rng.uniform(-1, 1, wash + tr + te); Win, b = inp(W.shape[0], rng); X, t = run(W, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1); B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D))), t
def narma(W, leak, rng, val):
    wash, tr, te = (200, 1500, 500) if val else (200, 2000, 1000); T = wash + tr + te
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for i in range(10, T - 1): y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = inp(W.shape[0], rng); X, t = run(W, Win, b, u, leak)
    B = F.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6); p = F.pred(X[wash + tr:], B)
    v = float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:])); return (v if np.isfinite(v) else 1e9), t
WORD = F.bench.make_regression_target(F.bench.RNG_SEED)[3]
def t0(W, leak, rng, val):
    n = 6; ntr, nte = (512, 256) if val else (512, 512)
    Xin = rng.standard_normal((ntr + nte, n)) / np.sqrt(n); Y = np.tanh(Xin @ F.bench.weights_from_word(WORD))
    Win, b = inp(W.shape[0], rng, n); S = np.zeros((len(Xin), W.shape[0])); d = Xin @ Win.T + b; t = time.perf_counter()
    for _ in range(10): S = (1 - leak) * S + leak * np.tanh((W @ S.T).T + d)
    t = time.perf_counter() - t; B = F.ridge(S[:ntr], Y[:ntr], 1e-4); return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2)), t
TASK = {"MC": mc, "NARMA": narma, "T0": t0}; OFF = {"MC": 0, "NARMA": 1, "T0": 2}

_C = {}
def get(w, m, f, s):
    key = (w, m, f, s if w != "REAL" or m in ("RANDP", "CONSOL") else 0)
    if key not in _C: _C.clear(); _C[key] = compress(base(w, s), m, f, s)
    return _C[key]
def job(a):
    w, m, f, s, task, leak, rho, val = a; W = get(w, m, f, s) * rho
    g = np.random.default_rng(s + OFF[task] if val else 10 * s + OFF[task]); v, t = TASK[task](W, leak, g, val)
    return dict(wire=w, meth=m, f=f, seed=s, task=task, leak=leak, rho=rho, val=val, value=v, sim_sec=t, n=W.shape[0])
def done():
    d = {}
    if os.path.exists(JF):
        for l in open(JF): r = json.loads(l); d[(r["wire"], r["meth"], r["f"], r["seed"], r["task"], r["leak"], r["rho"], r["val"])] = r
    return d
def runall(P, J):
    have = done(); J = [j for j in J if tuple(j) not in have]
    with open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1): fh.write(json.dumps(r) + "\n"); fh.flush()
def best():
    sw = pd.DataFrame([r for r in done().values() if r["val"]]); m = sw.groupby(["wire", "meth", "f", "task", "leak", "rho"]).value.mean().reset_index(); b = {}
    for k, mm in m.groupby(["wire", "meth", "f", "task"]):
        i = mm.value.idxmax() if k[3] == "MC" else mm.value.idxmin(); b[k] = (float(mm.loc[i, "leak"]), float(mm.loc[i, "rho"]))
    return b
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for m, f in METH:
            t = time.time(); W = get("REAL", m, f, 0); print(m, f, W.shape, round(time.time() - t, 1), mc(W, .5, np.random.default_rng(0), True), flush=True)
        sys.exit()
    P = Pool(int(os.environ.get("WORKERS", 3))); key = lambda j: (j[0], j[1], j[2], j[3])
    runall(P, sorted([(w, m, f, s, t, l, r, True) for w in WIRES for m, f in METH for s in TUNE for t in TASK for l in LEAKS for r in RHOS], key=key))
    b = best(); json.dump({"|".join(map(str, k)): v for k, v in b.items()}, open(os.path.join(OUT, "best.json"), "w"), indent=1)
    runall(P, sorted([(w, m, f, s, t, *b[(w, m, f, t)], False) for w in WIRES for m, f in METH for s in TEST for t in TASK], key=key))
    print("all done", flush=True)

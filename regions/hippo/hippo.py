"""Synthetic hippocampus on fly CX. See PREREGISTRATION_hippo.md. Resumable: out/jobs.jsonl."""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain")
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigs
from multiprocessing import Pool
import flybrain_reservoir as F
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); JF = os.path.join(OUT, "jobs.jsonl")
CXW = sp.load_npz("/workspace/flybrain/regions/out/graphs/0_REAL_0.npz").tocsr(); NC = CXW.shape[0]
nEC, nDG, nCA3, nCA1 = 200, 2000, 400, 400; NM = nEC + nDG + nCA3 + nCA1; KDG = 40
EC, DG, CA3, CA1 = [np.arange(a, a + n) for a, n in zip(np.cumsum([0, nEC, nDG, nCA3]), [nEC, nDG, nCA3, nCA1])]
LEAKS = [0.2, 0.5, 0.9]; RHOS = [0.7, 0.95, 1.1]; TUNE = [900, 901]; TEST = list(range(8)); CONDS = ["CX", "RAND", "SHUF", "HIPPO"]

def proj(g, src, dst, k, scale=1.0):  # each dst gets k random src inputs
    pre = np.concatenate([g.choice(src, k, replace=False) for _ in dst]); post = np.repeat(dst, k)
    return pre, post, scale * g.lognormal(0, 0.5, len(pre))

def build(cond, s):
    g = np.random.default_rng(5000 + s)
    if cond == "CX": return CXW, np.arange(0), np.arange(NC)
    parts = [proj(g, EC, DG, 20), proj(g, DG, CA3, 10, 4.0), proj(g, EC, CA3, 20), proj(g, CA3, CA3[:, None].ravel(), 40),
             proj(g, CA3, CA1, 40), proj(g, EC, CA1, 20), proj(g, CA1, EC, 10)]
    pre, post, w = [np.concatenate(x) for x in zip(*parts)]
    k = pre != post; pre, post, w = pre[k], post[k], w[k]
    dg, outn = DG, CA1
    if cond == "SHUF": pre, post, w, _ = F.deg(NM, pre, post, w, g)
    if cond == "RAND":
        E = len(pre); key = np.unique(g.integers(0, NM, (int(E * 1.05), 2)) @ np.array([NM, 1]))
        key = g.permutation(key[(key // NM) != (key % NM)])[:E]; pre, post = key // NM, key % NM; w = g.permutation(w)[:len(pre)]
        perm = g.permutation(NM); dg, outn = np.sort(perm[:nDG]), np.sort(perm[nDG:nDG + nCA1])
    M = sp.csr_matrix((w, (post, pre)), shape=(NM, NM)); M = M / abs(eigs(M, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-6)[0])
    # CX -> EC (50 per EC cell) and CA1 -> CX (30 targets per CA1 cell); RAND uses random module cells of same counts
    ecin = EC if cond != "RAND" else np.sort(g.permutation(np.setdiff1d(np.arange(NM), dg))[:nEC])
    a_pre = np.concatenate([g.choice(NC, 50, replace=False) for _ in ecin]); a_post = np.repeat(ecin, 50)
    b_pre = np.repeat(outn, 30); b_post = np.concatenate([g.choice(NC, 30, replace=False) for _ in outn])
    sc = 1 / np.sqrt(50)
    A = sp.csr_matrix((sc * g.lognormal(0, .5, len(a_pre)), (a_post, a_pre)), shape=(NM, NC))
    B = sp.csr_matrix((sc * g.choice([-1, 1], len(b_pre)) * g.lognormal(0, .5, len(b_pre)), (b_post, b_pre)), shape=(NC, NM))
    W = sp.bmat([[CXW, B], [A, M]]).tocsr()
    return W, NC + dg, np.concatenate([np.arange(NC), NC + outn])

def kwta(h, dg):  # h: (N,) or (B,N)
    if len(dg) == 0: return h
    sub = h[..., dg]; thr = -np.partition(-sub, KDG - 1, axis=-1)[..., KDG - 1:KDG]
    h[..., dg] = np.where(sub >= thr, sub, 0.0); return h
def inp(N, rng, d=1):
    Win = np.zeros((N, d)); Win[:NC] = rng.uniform(-1, 1, (NC, d)) * 0.5; b = np.zeros(N); b[:NC] = rng.uniform(-.1, .1, NC); return Win, b
def run(W, dg, Win, b, U, leak, ro):
    x = np.zeros(W.shape[0]); X = np.empty((len(U), len(ro)))
    for t, u in enumerate(U):
        x = (1 - leak) * x + leak * kwta(np.tanh(W @ x + Win @ np.atleast_1d(u) + b), dg); X[t] = x[ro]
    return X
def batch(W, dg, D, leak, steps):  # D: list of (B,N) drives per step
    S = np.zeros_like(D[0]); WT = W.T.tocsr()
    for d in D: S = (1 - leak) * S + leak * kwta(np.tanh(S @ WT + d), dg)  # S@W.T
    return S

def mc(G, leak, rng, val):
    W, dg, ro = G; wash, tr, te, D = (200, 1500, 500, 100) if val else (200, 2000, 1000, 100)
    u = rng.uniform(-1, 1, wash + tr + te); Win, b = inp(W.shape[0], rng); X = run(W, dg, Win, b, u, leak, ro)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1); B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
def narma(G, leak, rng, val):
    W, dg, ro = G; wash, tr, te = (200, 1500, 500) if val else (200, 2000, 1000); T = wash + tr + te
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(10, T - 1): y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    Win, b = inp(W.shape[0], rng); X = run(W, dg, Win, b, u, leak, ro)
    B = F.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6); p = F.pred(X[wash + tr:], B)
    v = float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:])); return v if np.isfinite(v) else 1e9
WORD = F.bench.make_regression_target(F.bench.RNG_SEED)[3]
def t0(G, leak, rng, val):
    W, dg, ro = G; n = 6; ntr, nte = (512, 256) if val else (512, 512)
    Xin = rng.standard_normal((ntr + nte, n)) / np.sqrt(n); Y = np.tanh(Xin @ F.bench.weights_from_word(WORD))
    Win, b = inp(W.shape[0], rng, n); S = batch(W, dg, [Xin @ Win.T + b] * 10, leak, 10)[:, ro]
    B = F.ridge(S[:ntr], Y[:ntr], 1e-4); return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2))
def pc(G, leak, rng, val):
    W, dg, ro = G; P, d = 20, 20; ntr, nte = (800, 200) if val else (800, 400)
    pats = rng.choice([-1.0, 1.0], (P, d)); lab = rng.integers(0, P, ntr + nte)
    cue = pats[lab] * (rng.random((ntr + nte, d)) > 0.5) + 0.5 * rng.standard_normal((ntr + nte, d))
    Win, b = inp(W.shape[0], rng, d); on = cue @ Win.T + b; off = np.zeros_like(on) + b
    S = batch(W, dg, [on] * 5 + [off] * 5, leak, 10)[:, ro]
    B = F.ridge(S[:ntr], pats[lab[:ntr]], 1e-4); out = F.pred(S[ntr:], B)
    return float(np.mean(np.argmax(out @ pats.T, 1) == lab[ntr:]))
TASK = {"MC": mc, "NARMA": narma, "T0": t0, "PC": pc}; OFF = {"MC": 0, "NARMA": 1, "T0": 2, "PC": 3}; HIGH = {"MC", "PC"}

_C = {}
def job(a):
    c, s, task, leak, rho, val = a
    if (c, s) not in _C: _C.clear(); _C[(c, s)] = build(c, s)
    W, dg, ro = _C[(c, s)]; g = np.random.default_rng(s + OFF[task] if val else 10 * s + OFF[task]); t = time.time()
    v = TASK[task]((W * rho, dg, ro), leak, g, val)
    return dict(cond=c, seed=s, task=task, leak=leak, rho=rho, val=val, value=v, sec=time.time() - t)
def done():
    d = {}
    if os.path.exists(JF):
        for l in open(JF): r = json.loads(l); d[(r["cond"], r["seed"], r["task"], r["leak"], r["rho"], r["val"])] = r
    return d
def runall(P, J):
    have = done(); J = [j for j in J if tuple(j) not in have]
    with open(JF, "a") as f:
        for r in P.imap_unordered(job, J, chunksize=1): f.write(json.dumps(r) + "\n"); f.flush()
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for c in CONDS:
            W, dg, ro = build(c, 0); t = time.time(); print(c, W.shape, W.nnz, len(dg), len(ro), {k: round(TASK[k]((W, dg, ro), .5, np.random.default_rng(0), True), 4) for k in TASK}, round(time.time() - t, 1), flush=True)
        sys.exit()
    os.makedirs(OUT, exist_ok=True); P = Pool(int(os.environ.get("WORKERS", 3)))
    runall(P, sorted([(c, s, t, l, r, True) for c in CONDS for s in TUNE for t in TASK for l in LEAKS for r in RHOS], key=lambda j: (j[0], j[1])))
    sw = pd.DataFrame(list(done().values())); sw = sw[sw.val]; m = sw.groupby(["cond", "task", "leak", "rho"]).value.mean().reset_index(); best = {}
    for (c, t), mm in m.groupby(["cond", "task"]):
        i = mm.value.idxmax() if t in HIGH else mm.value.idxmin(); best[(c, t)] = (float(mm.loc[i, "leak"]), float(mm.loc[i, "rho"]))
    json.dump({f"{c}|{t}": v for (c, t), v in best.items()}, open(os.path.join(OUT, "best.json"), "w"), indent=1)
    runall(P, sorted([(c, s, t, *best[(c, t)], False) for c in CONDS for t in TASK for s in TEST], key=lambda j: (j[0], j[1])))
    print("all done", flush=True)

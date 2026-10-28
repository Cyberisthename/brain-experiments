"""Round 3: train weight magnitudes on fixed masks. See PREREGISTRATION_3.md."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr
from multiprocessing import Pool
import flybrain_reservoir as F
torch.set_num_threads(1)
HERE = F.HERE; OUT = os.path.join(HERE, "out3"); os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(8)); STEPS = int(os.environ.get("STEPS", 60)); LEAK = 0.5; RHO = 0.95
GRID = [(l, m) for l in (0.3, 0.5, 0.7) for m in (0.8, 1.0, 1.2)]
WORD = F.bench.make_regression_target(F.bench.RNG_SEED)[3]

def mask(cond, s):
    N, pre, post, w = F.load_cx()
    if cond == "ER": pre, post, w = F.er(N, len(pre), w, np.random.default_rng(1000 + s))
    if cond == "DEG": pre, post, w, _ = F.deg(N, pre, post, w, np.random.default_rng(1000 + s))
    W = sp.csr_matrix((w, (post, pre)), shape=(N, N))
    r = abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=10000, tol=1e-6)[0])
    W = W.tocoo(); return N, W.row, W.col, W.data * RHO / r

def narma_seq(rng, T):
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(10, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y[:, None]

def delay_seq(rng, T, D=30):
    u = rng.uniform(-1, 1, T); return u, np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)

def inputs(N, s, d):
    g = np.random.default_rng(s); return g.uniform(-1, 1, (N, d)) * 0.5, g.uniform(-.1, .1, N)

def train(task, N, row, col, w0, s):
    d_in = 6 if task == "T0" else 1; Win_np, b_np = inputs(N, s, d_in)
    Win = torch.tensor(Win_np, dtype=torch.float32); b = torch.tensor(b_np, dtype=torch.float32)
    g = np.random.default_rng(3000 + s); torch.manual_seed(s)
    sign = torch.tensor(np.sign(w0), dtype=torch.float32)
    theta = torch.tensor(np.log(np.expm1(np.maximum(np.abs(w0), 1e-8))), dtype=torch.float32, requires_grad=True)
    idx = torch.tensor(np.vstack([row, col]), dtype=torch.long)
    R = torch.zeros(N + 1, {"NARMA": 1, "DELAY": 30, "T0": 6}[task], requires_grad=True)
    opt = torch.optim.Adam([{"params": [theta], "lr": 0.01}, {"params": [R], "lr": 1e-4}]); losses = []
    for it in range(STEPS):
        W = torch.sparse_coo_tensor(idx, sign * torch.nn.functional.softplus(theta), (N, N))
        if task == "T0":
            X = g.standard_normal((64, 6)) / np.sqrt(6); Y = torch.tensor(np.tanh(X @ F.bench.weights_from_word(WORD)), dtype=torch.float32)
            drive = torch.tensor(X, dtype=torch.float32) @ Win.T + b; x = torch.zeros(64, N)
            for _ in range(10): x = (1 - LEAK) * x + LEAK * torch.tanh(torch.sparse.mm(W, x.T).T + drive)
            loss = ((torch.cat([x, torch.ones(64, 1)], 1) @ R - Y) ** 2).mean()
        else:
            seqs = [(narma_seq if task == "NARMA" else delay_seq)(g, 60) for _ in range(16)]
            U = torch.tensor(np.stack([q[0] for q in seqs], 1), dtype=torch.float32)
            Y = torch.tensor(np.stack([q[1] for q in seqs], 1), dtype=torch.float32)
            x = torch.zeros(16, N); loss = 0
            for t in range(60):
                x = (1 - LEAK) * x + LEAK * torch.tanh(torch.sparse.mm(W, x.T).T + U[t][:, None] * Win[:, 0] + b)
                if t >= 20: loss = loss + ((torch.cat([x, torch.ones(16, 1)], 1) @ R - Y[t]) ** 2).mean()
            loss = loss / 40
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    return (sign * torch.nn.functional.softplus(theta)).detach().numpy().astype(float), losses

def evaluate(task, N, row, col, w, s, leak, mult):
    W = sp.csr_matrix((w * mult, (row, col)), shape=(N, N)).toarray()
    if task == "T0":
        Win, b = inputs(N, s, 6); ge = np.random.default_rng(8000 + s)
        X = ge.standard_normal((1024, 6)) / np.sqrt(6); Y = np.tanh(X @ F.bench.weights_from_word(WORD))
        S = np.zeros((1024, N)); drive = X @ Win.T + b
        for _ in range(10): S = (1 - leak) * S + leak * np.tanh(S @ W.T + drive)
        B = F.ridge(S[:512], Y[:512], 1e-4); return float(np.mean((F.pred(S[512:], B) - Y[512:]) ** 2))
    Win, b = inputs(N, s, 1); ge = np.random.default_rng(7000 + s)
    if task == "NARMA": u, Y = narma_seq(ge, 3200)
    else: u = ge.uniform(-1, 1, 3200); Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
    x = np.zeros(N); X = np.empty((3200, N))
    for t in range(3200): x = (1 - leak) * x + leak * np.tanh(W @ x + Win[:, 0] * u[t] + b); X[t] = x
    B = F.ridge(X[200:2200], Y[200:2200], 1e-6); P = F.pred(X[2200:], B); Yt = Y[2200:]
    if task == "NARMA":
        v = float(np.sqrt(np.mean((P - Yt) ** 2)) / np.std(Yt)); return v if np.isfinite(v) else 1e3
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(100)))

def job(a):
    cond, task, s = a; fp = os.path.join(OUT, "jobs", f"{cond}_{task}_{s}.json")
    if os.path.exists(fp): return pd.read_json(fp, typ="series").to_dict()
    N, row, col, w0 = mask(cond, s)
    w, losses = train(task, N, row, col, w0, s)
    out = dict(cond=cond, task=task, seed=s, loss_first=losses[0], loss_last=float(np.mean(losses[-10:])),
               rel_L2=float(np.linalg.norm(w - w0) / np.linalg.norm(w0)), spearman=float(spearmanr(np.abs(w0), np.abs(w)).correlation))
    for tag, ww in (("untrained", w0), ("trained", w)):
        vals = [evaluate(task, N, row, col, ww, s, l, m) for l, m in GRID]
        out[f"{tag}_test"] = vals[GRID.index((0.5, 1.0))]; out[f"{tag}_gridmean"] = float(np.mean(vals))
        out[f"{tag}_worst"] = float(min(vals) if task == "DELAY" else max(vals)); out[f"{tag}_grid"] = vals
    print({k: v for k, v in out.items() if not k.endswith("_grid")}, flush=True)
    os.makedirs(os.path.dirname(fp), exist_ok=True); pd.Series(out).to_json(fp)
    return out

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        import time; t = time.time(); job(("REAL", sys.argv[2], 0)); print("sec", time.time() - t); sys.exit()
    jobs = [(c, t, s) for s in SEEDS for t in ["T0", "DELAY", "NARMA"] for c in ["REAL", "ER", "DEG"]]
    with Pool(int(os.environ.get("WORKERS", 4))) as p: rows = p.map(job, jobs, chunksize=1)
    pd.DataFrame(rows).to_json(os.path.join(OUT, "round3_raw.json"), orient="records", indent=1)

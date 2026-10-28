"""Adapter arm. See PREREGISTRATION_compress.md. Resumable: out/adapt.jsonl"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, compress as C
from multiprocessing import Pool
torch.set_num_threads(1); JF = os.path.join(C.OUT, "adapt.jsonl"); R = 8; STEPS = 100
BITS = np.zeros(32); BITS[[1, 5, 9, 14, 17, 22, 26, 30]] = 1  # 8-on code (y16|z8|x8 layout)

class Op:
    def __init__(s, W, U, V, g): s.W, s.U, s.V, s.g = W, U, V, g; s.shape = W.shape
    def __matmul__(s, x):
        y = s.W @ x
        if s.U is None: return y
        h = np.tanh(s.V.T @ x); h = h * (s.g if x.ndim == 1 else s.g[:, None]); return y + s.U @ h

def train(arm, Wc, task, leak, fixed, s):
    m = Wc.shape[0]; tg = torch.Generator().manual_seed(77 + s); W = torch.tensor(np.asarray(Wc.todense() if hasattr(Wc, "todense") else Wc), dtype=torch.float32)
    Win1, b1 = [torch.tensor(a, dtype=torch.float32) for a in fixed[1]]; Win6, b6 = [torch.tensor(a, dtype=torch.float32) for a in fixed[6]]
    if arm == "ADAPT":
        g = torch.tensor(BITS, dtype=torch.float32); U = torch.zeros(m, 32, requires_grad=True); V = (torch.randn(m, 32, generator=tg) / np.sqrt(m)).requires_grad_()
        ps = [U, V]; rec = lambda x: x @ W.T + (torch.tanh(x @ V) * g) @ U.T
    else:
        A = torch.zeros(m, R, requires_grad=True); B = (torch.randn(m, R, generator=tg) / np.sqrt(m)).requires_grad_(); ps = [A, B]
        rec = lambda x: x @ W.T + (x @ B) @ A.T
    out_d = {"MC": 30, "NARMA": 1, "T0": 1}[task]; head = torch.nn.Linear(m, out_d); opt = torch.optim.Adam(ps + list(head.parameters()), lr=0.01)
    rng = np.random.default_rng(55 + s); wt = C.F.bench.weights_from_word(C.WORD)
    for it in range(STEPS):
        if task == "T0":
            Xin = rng.standard_normal((64, 6)) / np.sqrt(6); Y = torch.tensor(np.tanh(Xin @ wt), dtype=torch.float32).reshape(64, -1)[:, :1]
            d = torch.tensor(Xin, dtype=torch.float32) @ Win6.T + b6; x = torch.zeros(64, m)
            for _ in range(10): x = (1 - leak) * x + leak * torch.tanh(rec(x) + d)
            Yfull = torch.tensor(np.tanh(Xin @ wt), dtype=torch.float32)
            if it == 0: head = torch.nn.Linear(m, Yfull.shape[1]); opt = torch.optim.Adam(ps + list(head.parameters()), lr=0.01)
            loss = ((head(x) - Yfull) ** 2).mean()
        else:
            T = 60; u = rng.uniform(-1, 1, (16, T + 30)) if task == "MC" else rng.uniform(0, .5, (16, T + 30))
            if task == "MC": Y = np.stack([u[:, 30 - k:30 - k + T] for k in range(1, 31)], -1)
            else:
                y = np.zeros_like(u)
                for i in range(10, u.shape[1] - 1): y[:, i + 1] = 0.3 * y[:, i] + 0.05 * y[:, i] * y[:, i - 9:i + 1].sum(1) + 1.5 * u[:, i - 9] * u[:, i] + 0.1
                Y = y[:, 30:, None]
            u = torch.tensor(u[:, 30:], dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.float32); x = torch.zeros(16, m); outs = []
            for t in range(T):
                x = (1 - leak) * x + leak * torch.tanh(rec(x) + u[:, t:t + 1] * Win1[:, 0] + b1)
                if t >= 20: outs.append(head(x))
            loss = ((torch.stack(outs, 1) - Y[:, 20:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if arm == "ADAPT": return Op(np.asarray(W), U.detach().numpy().astype(float), V.detach().numpy().astype(float), BITS), 2 * m * int(BITS.sum())
    return np.asarray(W) + A.detach().numpy() @ B.detach().numpy().T, 2 * m * R

def job(a):
    arm, meth, s, task, leak, rho = a
    Wc = C.get("REAL", meth, 0.25, s) * rho; m = Wc.shape[0]; fr = np.random.default_rng(9000 + s)
    fixed = {1: C.inp(m, fr, 1), 6: C.inp(m, fr, 6)}; C.inp = lambda n, rng, d=1: fixed[d]
    t = time.time(); params = 0
    if arm != "NONE": Wc, params = train(arm, Wc, task, leak, fixed, s)
    tt = time.time() - t; v, sim = C.TASK[task](Wc, leak, np.random.default_rng(10 * s + C.OFF[task]), False)
    return dict(arm=arm, meth=meth, seed=s, task=task, value=v, sim_sec=sim, train_sec=tt, params=params, m=m)
if __name__ == "__main__":
    b = C.best(); have = set()
    if os.path.exists(JF): have = {(r["arm"], r["meth"], r["seed"], r["task"]) for r in map(json.loads, open(JF))}
    J = [(a, me, s, t, *b[("REAL", me, 0.25, t)]) for me in ["CONSOL", "MAGP"] for s in C.TEST for t in C.TASK for a in ["NONE", "ADAPT", "LORA"] if (a, me, s, t) not in have]
    if len(sys.argv) > 1: print(job(J[int(sys.argv[1])])); sys.exit()
    with Pool(int(os.environ.get("WORKERS", 3))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1): fh.write(json.dumps(r) + "\n"); fh.flush()
    print("all done")

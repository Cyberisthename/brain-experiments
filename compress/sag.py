"""Sparse Adapter Gateway. See PREREGISTRATION_sag.md. Resumable: out/sag.jsonl"""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
from scipy.sparse.linalg import svds
from scipy.cluster.vq import kmeans2
import compress as C
torch.set_num_threads(1)
JF = os.path.join(C.OUT, "sag.jsonl"); STEPS = 100; G = 500; KTOP = 10; CLIP_W = 2.0; GRAD_CLIP = 1.0
ann = pd.read_csv("/workspace/flybrain/data/ann.tsv", sep="\t", low_memory=False, usecols=["root_id", "cell_class", "cell_type"])
CXID = np.array(sorted(set(ann.root_id[ann.cell_class == "CX"]))); assert len(CXID) == C.N
CT = ann.drop_duplicates("root_id").set_index("root_id").cell_type.astype(str).reindex(CXID).values
EPG_FULL = np.where(np.isin(CT, ["EPG", "PEG", "EPGt"]))[0]
ARMS = ["NONE", "ADAPT", "LORA", "LORA15", "SAG", "SAG_RAND", "SAG_DENSE"]
METHS = ["CONSOL", "MAGP"]

def compress_map(meth, s):
    """Return (Wc unscaled, local target indices for EPG, local target count for RAND)."""
    g = np.random.default_rng(4000 + s); W = C.REAL; N = C.N; k = int(round(0.25 * N))
    if meth == "MAGP":
        A = abs(W); keep = np.sort(np.argsort(-(np.ravel(A.sum(0)) + np.ravel(A.sum(1))))[:k])
        Wc = W[keep][:, keep].tocsr(); tgt = np.where(np.isin(keep, EPG_FULL))[0]
        return Wc / C.radius(Wc), tgt, len(tgt)
    U, S, Vt = svds(W, k=32); X = np.hstack([U * S, Vt.T * S]); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    _, lab = kmeans2(X, k, seed=int(4000 + s), minit="++"); _, lab = np.unique(lab, return_inverse=True)
    P = sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, lab.max() + 1)); sz = np.ravel(P.sum(0))
    Wc = (sp.diags(1 / sz) @ P.T @ W @ P).tocsr()
    if Wc.nnz > 0.3 * Wc.shape[0] ** 2: Wc = Wc.toarray()
    tgt = np.unique(lab[EPG_FULL]); return Wc / C.radius(Wc if sp.issparse(Wc) else sp.csr_matrix(Wc)), tgt, len(tgt)

class SparseOp:
    """W dense-or-CSR plus sparse Vin (m,G), U (m,G) writing into target rows; optional top-K."""
    def __init__(s, W, Vin, Urows, Udata, topk):
        s.W = W if sp.issparse(W) else np.asarray(W); s.Vin = Vin.tocsr(); s.Urows = Urows; s.Udata = Udata; s.topk = topk
        s.shape = s.W.shape if hasattr(s.W, "shape") else np.shape(s.W); s.G = Vin.shape[1]
    def __matmul__(s, x):
        y = np.asarray(s.W @ x, dtype=float).copy()
        if x.ndim == 1:
            h = s.Vin.T @ x
            if s.topk:
                idx = np.argpartition(-np.abs(h), s.topk - 1)[:s.topk]
                np.add.at(y, s.Urows[idx], s.Udata[idx] * np.tanh(h[idx]))
            else:
                np.add.at(y, s.Urows, s.Udata * np.tanh(h))
            return y
        h = s.Vin.T @ x
        if s.topk:
            idx = np.argpartition(-np.abs(h), s.topk - 1, axis=0)[:s.topk]
            for b in range(h.shape[1]):
                i = idx[:, b]; y[:, b] = y[:, b]; np.add.at(y[:, b], s.Urows[i], s.Udata[i] * np.tanh(h[i, b]))
        else:
            for j in range(s.G): y[s.Urows[j]] += s.Udata[j] * np.tanh(h[j])
        return y

class DenseAdapt:
    def __init__(s, W, U, V, g): s.W, s.U, s.V, s.g = np.asarray(W), U, V, g; s.shape = s.W.shape
    def __matmul__(s, x):
        y = s.W @ x
        if x.ndim == 1: h = np.tanh(s.V.T @ x) * s.g; return y + s.U @ h
        h = np.tanh(s.V.T @ x) * s.g[:, None]; return y + s.U @ h

def make_sag_weights(m, tgt, g, dense_act=False):
    """Vin csr (m,G) 2-in; Urows/Udata length G. Returns structures + param count."""
    rows, cols, data = [], [], []
    for j in range(G):
        pre = g.choice(m, 2, replace=False)
        for p in pre: rows.append(p); cols.append(j); data.append(g.normal() / np.sqrt(2))
    Vin = sp.csr_matrix((data, (rows, cols)), shape=(m, G))
    Urows = g.choice(tgt, G, replace=True); Udata = g.normal(size=G) / np.sqrt(1)
    return Vin, Urows, Udata, Vin.nnz + G

def train(arm, Wc, tgt, n_tgt, task, leak, fixed, s):
    m = Wc.shape[0]; tg = torch.Generator().manual_seed(77 + s)
    Wnp = np.asarray(Wc.todense() if sp.issparse(Wc) else Wc); W = torch.tensor(Wnp, dtype=torch.float32)
    Win1, b1 = [torch.tensor(a, dtype=torch.float32) for a in fixed[1]]; Win6, b6 = [torch.tensor(a, dtype=torch.float32) for a in fixed[6]]
    gnp = np.random.default_rng(88 + s); ps = []; meta = {}
    if arm == "ADAPT":
        bits = torch.zeros(32); bits[[1, 5, 9, 14, 17, 22, 26, 30]] = 1
        U = torch.zeros(m, 32, requires_grad=True); V = (torch.randn(m, 32, generator=tg) / np.sqrt(m)).requires_grad_()
        ps = [U, V]; rec = lambda x: x @ W.T + (torch.tanh(x @ V) * bits) @ U.T; nparam = 2 * m * 8
    elif arm in ("LORA", "LORA15"):
        R = 8 if arm == "LORA" else 1
        A = torch.zeros(m, R, requires_grad=True); B = (torch.randn(m, R, generator=tg) / np.sqrt(m)).requires_grad_()
        ps = [A, B]; rec = lambda x: x @ W.T + (x @ B) @ A.T; nparam = 2 * m * R
    elif arm.startswith("SAG"):
        targets = tgt if arm == "SAG" else gnp.choice(m, n_tgt, replace=False)
        Vin_sp, Urows, Udata, nparam = make_sag_weights(m, targets, gnp)
        # torch params: Vin values and Udata; topology fixed
        Vin_t = torch.tensor(Vin_sp.toarray(), dtype=torch.float32, requires_grad=True)  # train dense then sparsify by mask
        mask = torch.tensor((Vin_sp.toarray() != 0).astype(np.float32))
        U_idx = torch.tensor(Urows, dtype=torch.long); U_val = torch.tensor(Udata, dtype=torch.float32, requires_grad=True)
        ps = [Vin_t, U_val]; topk = arm != "SAG_DENSE"; meta.update(mask=mask, U_idx=U_idx, topk=topk, targets=targets)
        def rec(x):
            h = x @ (Vin_t * mask)  # (B,G)
            if topk:
                # hard top-K on |h|, straight-through: forward mask, backward full
                with torch.no_grad():
                    idx = torch.topk(h.abs(), KTOP, dim=1).indices; msk = torch.zeros_like(h); msk.scatter_(1, idx, 1.0)
                h = torch.tanh(h) * msk
            else: h = torch.tanh(h)
            add = torch.zeros(x.shape[0], m, device=x.device)
            add.index_add_(1, U_idx, h * U_val)  # broadcast U_val over batch via h*(U_val)
            # h is (B,G), U_val (G,) -> h*U_val; index_add along dim1 with U_idx (G,) — need (B,G) sources
            # fix: for each gateway, add h[:,j]*U_val[j] to column U_idx[j]
            return x @ W.T + add
        # fix rec properly
        def rec(x):
            h = x @ (Vin_t * mask)
            if topk:
                with torch.no_grad():
                    idx = torch.topk(h.abs(), KTOP, dim=1).indices; msk = torch.zeros_like(h); msk.scatter_(1, idx, 1.0)
                h = torch.tanh(h) * msk
            else: h = torch.tanh(h)
            add = torch.zeros(x.shape[0], m)
            src = h * U_val  # (B,G)
            add = add.index_add(1, U_idx, src)
            return x @ W.T + add
    else:
        raise ValueError(arm)
    out_d = {"MC": 30, "NARMA": 1, "T0": 6}[task]
    head = torch.nn.Linear(m, out_d); opt = torch.optim.Adam(list(ps) + list(head.parameters()), lr=0.01)
    rng = np.random.default_rng(55 + s); wt = C.F.bench.weights_from_word(C.WORD)
    for it in range(STEPS):
        if task == "T0":
            Xin = rng.standard_normal((64, 6)) / np.sqrt(6); Yfull = torch.tensor(np.tanh(Xin @ wt), dtype=torch.float32)
            d = torch.tensor(Xin, dtype=torch.float32) @ Win6.T + b6; x = torch.zeros(64, m)
            for _ in range(10): x = (1 - leak) * x + leak * torch.tanh(rec(x) + d)
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
        if not torch.isfinite(loss): return None, nparam  # signal blow-up before step
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ps) + list(head.parameters()), GRAD_CLIP); opt.step()
        with torch.no_grad():
            for p in ps: p.clamp_(-CLIP_W, CLIP_W)
    # build eval operator
    if arm == "ADAPT":
        return DenseAdapt(Wnp, U.detach().numpy(), V.detach().numpy(), bits.numpy()), nparam
    if arm in ("LORA", "LORA15"):
        return Wnp + A.detach().numpy() @ B.detach().numpy().T, nparam
    # SAG: sparse eval
    Vin_f = (Vin_t.detach().numpy() * mask.numpy()); Vin_sp = sp.csr_matrix(Vin_f)
    return SparseOp(Wnp, Vin_sp, U_idx.numpy(), U_val.detach().numpy(), topk=arm != "SAG_DENSE"), nparam

def job(a):
    arm, meth, s, task, leak, rho = a; t0 = time.time()
    Wc0, tgt, n_tgt = compress_map(meth, s); Wc = (Wc0 * rho) if not sp.issparse(Wc0) else Wc0 * rho
    m = Wc.shape[0] if hasattr(Wc, "shape") else np.shape(Wc)[0]
    fr = np.random.default_rng(9000 + s); fixed = {1: C.inp(m, fr, 1), 6: C.inp(m, fr, 6)}
    # patch inp for eval so Win/b match training
    _inp = C.inp
    C.inp = lambda n, rng, d=1: fixed[d]
    params = 0; train_sec = 0.0
    try:
        if arm == "NONE":
            W_eval = Wc
        else:
            tt = time.time(); out = train(arm, Wc, tgt, n_tgt, task, leak, fixed, s); train_sec = time.time() - tt
            if out[0] is None:
                C.inp = _inp; return dict(arm=arm, meth=meth, seed=s, task=task, value=1e9, sim_sec=float("nan"), train_sec=train_sec, params=out[1], m=m, n_tgt=int(n_tgt), blowup=True)
            W_eval, params = out
        v, sim = C.TASK[task](W_eval, leak, np.random.default_rng(10 * s + C.OFF[task]), False)
        if not np.isfinite(v): v = 1e9
    finally:
        C.inp = _inp
    return dict(arm=arm, meth=meth, seed=s, task=task, value=float(v), sim_sec=float(sim), train_sec=train_sec, params=int(params), m=m, n_tgt=int(n_tgt), blowup=bool(v >= 1e8), sec=time.time() - t0)

def done():
    return {(r["arm"], r["meth"], r["seed"], r["task"]) for r in map(json.loads, open(JF))} if os.path.exists(JF) else set()

if __name__ == "__main__":
    b = C.best()
    J = [(a, me, s, t, *b[("REAL", me, 0.25, t)]) for me in METHS for s in C.TEST for t in C.TASK for a in ARMS]
    if len(sys.argv) > 1:
        # smoke: one job by index, or arm name
        if sys.argv[1].isdigit(): print(job(J[int(sys.argv[1])]))
        else:
            for me in METHS:
                j = next(x for x in J if x[0] == sys.argv[1] and x[1] == me and x[2] == 0 and x[3] == "MC")
                print(job(j))
        sys.exit()
    have = done(); J = [j for j in J if (j[0], j[1], j[2], j[3]) not in have]
    with Pool(int(os.environ.get("WORKERS", 7))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1): fh.write(json.dumps(r) + "\n"); fh.flush(); print(r["arm"], r["meth"], r["seed"], r["task"], round(r["value"], 4), round(r["sim_sec"], 3), flush=True)
    print("all done", flush=True)

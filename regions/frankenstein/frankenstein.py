"""Fly CX + mouse + human frankenstein graft. Prereg: PREREGISTRATION_frankenstein.md.
No octopus (no public neuron edges). Block ρ=1, bridges σ=0.05, global ρ=0.9.
"""
from __future__ import annotations
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/compress")
import numpy as np, scipy.sparse as sp
from multiprocessing import Pool
from pathlib import Path
import compress as C

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]  # BEST_OF only; never test seeds
ARMS = [
    "FLY_CX", "FRANKEN", "ER", "SHUF", "DENS_MATCHED", "BEST_OF",
    "FRANKEN+HEBB", "DENS_MATCHED+HEBB",
]
TASKS = ["MC", "T0"]
RHO = 0.9
HEBB_ETA = 1e-4
BRIDGE_SIGMA = 0.05
BLOCKS = ("mouse", "human", "fly_cx")  # order in block-diag


def load_mod(name):
    fname = {"mouse": "mouse_module.npz", "human": "human_module.npz", "fly_cx": "fly_cx_module.npz"}[name]
    z = np.load(DATA / fname, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    return W, z["sign"].astype(np.int8), z["layer"].astype(str), z["cell_type"].astype(str)


def scale(W, rho=RHO):
    r = float(C.radius(W))
    return W * (rho / (r + 1e-12))


def er_match(W, sign, g):
    N, nnz = W.shape[0], W.nnz
    rows = g.integers(0, N, nnz); cols = g.integers(0, N, nnz)
    m = rows != cols; rows, cols = rows[m], cols[m]
    data = np.abs(g.standard_normal(len(rows))) * sign[rows].astype(float)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def dens_match(W, sign, g):
    """Same N+nnz; |w| bootstrapped from real |W.data| (fair density+magnitude control)."""
    N, nnz = W.shape[0], W.nnz
    abs_pool = np.abs(W.data) if W.nnz else np.array([1.0])
    rows = g.integers(0, N, nnz); cols = g.integers(0, N, nnz)
    m = rows != cols; rows, cols = rows[m], cols[m]
    mags = g.choice(abs_pool, size=len(rows), replace=True)
    data = mags * sign[rows].astype(float)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def degree_shuffle(W, sign, g):
    W = W.tocoo(); N = W.shape[0]
    out_s, in_s = W.row.copy(), W.col.copy()
    g.shuffle(in_s)
    for _ in range(8):
        sl = out_s == in_s
        if not sl.any(): break
        idx = np.where(sl)[0]; g.shuffle(in_s[idx])
    data = np.abs(W.data) * sign[out_s].astype(float)
    return sp.csr_matrix((data, (out_s, in_s)), shape=(N, N))


def layer_sets(layer, N):
    L4 = np.where(layer == "L4")[0]
    L5 = np.where(layer == "L5")[0]
    L23 = np.where(layer == "L2/3")[0]
    fb = False
    if len(L4) == 0:
        L4 = L23 if len(L23) else np.arange(min(200, N)); fb = True
    if len(L5) == 0:
        L5 = L23 if len(L23) else np.arange(N // 2, N)
    return L4, L5, fb


def make_block(name, kind, seed):
    """kind in {real, er, dens, shuf}. Returns Wm (ρ=1), sign, layer, meta."""
    W, sign, layer, ct = load_mod(name)
    g = np.random.default_rng(seed + 13 + hash(name) % 1000)
    if kind == "er":
        W = er_match(W, sign, g)
    elif kind == "dens":
        W = dens_match(W, sign, g)
    elif kind == "shuf":
        W = degree_shuffle(W, sign, g)
    # else real
    return scale(W, 1.0), sign, layer, ct


def graft_blocks(blocks, seed, g=None):
    """blocks: list of (name, Wm, layer). Hub = fly_cx. Bridges mouse/human L4/L5 ↔ fly targets."""
    if g is None:
        g = np.random.default_rng(seed + 7)
    names = [b[0] for b in blocks]
    Ws = [b[1] for b in blocks]
    layers = [b[2] for b in blocks]
    sizes = [W.shape[0] for W in Ws]
    N = sum(sizes)
    # offsets
    off = np.cumsum([0] + sizes)
    # block diag
    # build lil then csr
    W = sp.lil_matrix((N, N))
    for i, Wi in enumerate(Ws):
        a, b = off[i], off[i + 1]
        W[a:b, a:b] = Wi
    # fly_cx index
    try:
        ifx = names.index("fly_cx")
    except ValueError:
        ifx = len(names) - 1
    fly_off = off[ifx]
    fly_layer = layers[ifx]
    fly_L4, fly_L5, _ = layer_sets(fly_layer, sizes[ifx])
    tgt = np.unique(np.concatenate([fly_L4, fly_L5])) + fly_off
    if len(tgt) == 0:
        tgt = np.arange(fly_off, fly_off + sizes[ifx])
    bridge_info = {}
    for i, name in enumerate(names):
        if name == "fly_cx":
            continue
        L4, L5, fb = layer_sets(layers[i], sizes[i])
        L4g = L4 + off[i]; L5g = L5 + off[i]
        n_out = min(3000, max(100, len(L5) * 4))
        n_in = min(3000, max(100, len(L4) * 4))
        # L5 → fly tgt
        br_from = g.choice(L5g, n_out, replace=True)
        br_to = g.choice(tgt, n_out, replace=True)
        for r, c, d in zip(br_to, br_from, g.standard_normal(n_out) * BRIDGE_SIGMA):
            W[r, c] += d
        # fly tgt → L4
        a_from = g.choice(tgt, n_in, replace=True)
        a_to = g.choice(L4g, n_in, replace=True)
        for r, c, d in zip(a_to, a_from, g.standard_normal(n_in) * BRIDGE_SIGMA):
            W[r, c] += d
        bridge_info[name] = dict(n_in=n_in, n_out=n_out, L4_n=len(L4), L5_n=len(L5),
                                 L4_fallback=fb, offset=int(off[i]))
    W = scale(W.tocsr(), RHO)
    # drive = union of all L4s
    drive = []
    for i, layer in enumerate(layers):
        L4, _, _ = layer_sets(layer, sizes[i])
        drive.append(L4 + off[i])
    drive = np.unique(np.concatenate(drive)) if drive else np.arange(min(200, N))
    info = dict(names=names, sizes=sizes, offsets=[int(x) for x in off], N=N,
                drive=drive, bridge=bridge_info, fly_offset=int(fly_off),
                cx_layer_proxy=True)
    return W, info


def kinds_for_arm(arm):
    base = arm.replace("+HEBB", "")
    if base == "FLY_CX":
        return None
    if base == "FRANKEN":
        return {n: "real" for n in BLOCKS}
    if base == "ER":
        return {n: "er" for n in BLOCKS}
    if base == "SHUF":
        return {n: "shuf" for n in BLOCKS}
    if base == "DENS_MATCHED":
        return {n: "dens" for n in BLOCKS}
    if base == "BEST_OF":
        return "BEST_OF"
    raise KeyError(arm)


def tune_best_of():
    """Per-block pick real vs dens on TUNE_SEEDS mean MC (block-alone, L4-driven). Cache."""
    cache = OUT / "best_of_choices.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import flybrain_reservoir as F
    choices = {}
    for name in BLOCKS:
        scores = {"real": [], "dens": []}
        for kind in ("real", "dens"):
            for ts in TUNE_SEEDS:
                Wm, sign, layer, ct = make_block(name, kind, ts)
                L4, _, _ = layer_sets(layer, Wm.shape[0])
                rng = np.random.default_rng(ts)
                # light MC on block alone
                wash, tr, te, D = 100, 600, 300, 50
                T = wash + tr + te
                u = rng.uniform(-1, 1, T)
                Win = np.zeros((Wm.shape[0], 1)); Win[L4] = rng.uniform(-0.5, 0.5, (len(L4), 1))
                b = rng.uniform(-0.1, 0.1, Wm.shape[0])
                x = np.zeros(Wm.shape[0]); X = np.empty((T, Wm.shape[0]))
                for i in range(T):
                    x = 0.8 * x + 0.2 * np.tanh(Wm @ x + Win[:, 0] * u[i] + b)
                    X[i] = x
                Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
                B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
                P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
                val = float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
                scores[kind].append(val)
        mean_r = float(np.mean(scores["real"]))
        mean_d = float(np.mean(scores["dens"]))
        choices[name] = {"pick": "real" if mean_r >= mean_d else "dens",
                         "tune_real": mean_r, "tune_dens": mean_d,
                         "tune_seeds": TUNE_SEEDS}
    cache.write_text(json.dumps(choices, indent=2))
    return choices


def build_arm(arm, seed, g=None):
    if g is None:
        g = np.random.default_rng(seed + 7)
    base = arm.replace("+HEBB", "")
    if base == "FLY_CX":
        Wm, sign, layer, ct = make_block("fly_cx", "real", seed)
        W = scale(Wm, RHO)
        L4, L5, _ = layer_sets(layer, Wm.shape[0])
        return W, dict(names=["fly_cx"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                       N=Wm.shape[0], drive=L4, bridge={}, fly_offset=0,
                       kinds={"fly_cx": "real"}, hebb=arm.endswith("+HEBB"))
    kinds = kinds_for_arm(arm)
    if kinds == "BEST_OF":
        choices = tune_best_of()
        kinds = {n: choices[n]["pick"] for n in BLOCKS}
    blocks = []
    for name in BLOCKS:
        Wm, sign, layer, ct = make_block(name, kinds[name], seed)
        blocks.append((name, Wm, layer))
    W, info = graft_blocks(blocks, seed, g)
    info["kinds"] = kinds
    info["hebb"] = arm.endswith("+HEBB")
    # stash mouse+human offsets for hebb
    info["plastic"] = [n for n in ("mouse", "human") if n in info["names"]]
    return W, info


def masked_win(n, d, in_idx, rng, scale=0.5):
    Win = np.zeros((n, d))
    idx = np.asarray(in_idx)
    if len(idx) == 0:
        Win[:] = rng.uniform(-scale, scale, (n, d))
    else:
        Win[idx] = rng.uniform(-scale, scale, (len(idx), d))
    b = rng.uniform(-0.1, 0.1, n)
    return Win, b


def oja_update_csr(Wm, h_pre, h_post, eta):
    if eta <= 0 or Wm.nnz == 0:
        return
    rows = np.repeat(np.arange(Wm.shape[0]), np.diff(Wm.indptr))
    cols = Wm.indices
    data = Wm.data
    data += eta * h_post[rows] * (h_pre[cols] - h_post[rows] * data)


def hebb_warmup(W, info, Win, b, U, leak, until):
    """Oja on mouse+human diagonal blocks only."""
    W = W.tolil()
    for name in info.get("plastic", []):
        i = info["names"].index(name)
        a, c = info["offsets"][i], info["offsets"][i + 1]
        Wm = W[a:c, a:c].tocsr()
        h = np.zeros(c - a)
        for i_u, u in enumerate(U):
            if i_u >= until:
                break
            drive = Win[a:c, 0] * u if np.ndim(u) == 0 else Win[a:c] @ np.asarray(u, float)
            h_pre = h.copy()
            h = (1 - leak) * h + leak * np.tanh(Wm.dot(h) + drive + b[a:c])
            oja_update_csr(Wm, h_pre, h, HEBB_ETA)
        W[a:c, a:c] = Wm
    return W.tocsr()


def run_reservoir(W, Win, b, U, leak):
    n = W.shape[0]
    X = np.empty((len(U), n))
    x = np.zeros(n)
    for i, u in enumerate(U):
        drive = Win[:, 0] * u if np.ndim(u) == 0 else Win @ np.asarray(u, float)
        x = (1 - leak) * x + leak * np.tanh(W @ x + drive + b)
        X[i] = x
    return X


def mc_task(W, leak, rng, in_idx, hebb=False, info=None):
    import flybrain_reservoir as F
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    if hebb and info:
        W = hebb_warmup(W, info, Win, b, u, leak, wash + tr)
    X = run_reservoir(W, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def t0_task(W, leak, rng, in_idx, hebb=False, info=None):
    import flybrain_reservoir as F
    n_in, ntr, nte = 6, 512, 256
    Xin = rng.standard_normal((ntr + nte, n_in)) / np.sqrt(n_in)
    Y = np.tanh(Xin @ F.bench.weights_from_word(F.bench.make_regression_target(F.bench.RNG_SEED)[3]))
    Win, b = masked_win(W.shape[0], n_in, in_idx, rng)
    if hebb and info:
        W = hebb_warmup(W, info, Win, b, Xin[:ntr, 0], leak, ntr)
    S = np.zeros((len(Xin), W.shape[0]))
    d = Xin @ Win.T + b
    for _ in range(10):
        S = (1 - leak) * S + leak * np.tanh((W @ S.T).T + d)
    B = F.ridge(S[:ntr], Y[:ntr], 1e-4)
    return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2))


def run_one(arm, seed, task):
    g = np.random.default_rng(seed + {"MC": 0, "T0": 1}[task])
    t0 = time.time()
    W, info = build_arm(arm, seed, g)
    hebb = arm.endswith("+HEBB")
    leak = 0.2
    if task == "MC":
        val = mc_task(W, leak, g, info["drive"], hebb=hebb, info=info)
    else:
        val = t0_task(W, leak, g, info["drive"], hebb=hebb, info=info)
    return dict(arm=arm, seed=seed, task=task, value=float(val), n=int(W.shape[0]),
                sec=time.time() - t0, kinds=info.get("kinds"), hebb=hebb)


def _job(args):
    return run_one(*args)


def main():
    jobs = [(a, s, t) for a in ARMS for s in SEEDS for t in TASKS]
    # resume
    done = set()
    if JF.exists():
        for line in JF.open():
            try:
                r = json.loads(line); done.add((r["arm"], r["seed"], r["task"]))
            except Exception:
                pass
    jobs = [j for j in jobs if j not in done]
    print(f"frankenstein jobs {len(jobs)} (skip {len(done)})", flush=True)
    workers = int(os.environ.get("WORKERS", "2"))
    with JF.open("a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = run_one(*j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r["arm"], r["seed"], r["task"], round(r["value"], 3), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r["arm"], r["seed"], r["task"], round(r["value"], 3), flush=True)
    OUT.joinpath("frankenstein.done").write_text("ok\n")


if __name__ == "__main__":
    main()

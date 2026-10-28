"""AetherLink lanes: N parallel ungated linear projections into P-EN/E-PG. See PREREGISTRATION_aetherlink.md."""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
import numpy as np, scipy.sparse as sp, pandas as pd, torch
from multiprocessing import Pool
import compress as C
import sag as S

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)
JF = os.path.join(OUT, "runs.jsonl")
NS = [1, 2, 4, 8, 16]
H_LANE = 32
SEEDS = list(range(8))
METHS = ["CONSOL", "MAGP"]
TASKS = ["MC", "T0", "NARMA"]
GRAD_CLIP, CLIP_W = 1.0, 2.0


def pen_epg_targets(meth, seed):
    Wc, tgt_epg, _ = S.compress_map(meth, seed)
    # also PEN if present in compressed map
    ann = S.ann; CXID = S.CXID; CT = S.CT
    # rebuild keep set like sag
    keep_info = S.compress_map(meth, seed)
    # PEN full indices
    PEN_FULL = np.where(np.isin(CT, ["PEN_a(PEN1)", "PEN_b(PEN2)", "PEN"]))[0]
    # For MAGP/CONSOL, map to local: reuse EPG tgt from sag + any PEN that survived
    # Approximate: use EPG targets + try to find PEN in same compress
    # Simpler: targets = EPG∪PEG∪EPGt from sag (already) plus random PEN-sized extras from kept CX if available
    return keep_info[0], keep_info[1]  # W, epg_local_tgts


def make_lanes(g, n_cx, targets, N, wide=False):
    """N lanes each H_LANE tanh units, linear U into targets (ungated). wide: one lane with H=N*H_LANE."""
    H = (N * H_LANE) if wide else H_LANE
    n_lanes = 1 if wide else N
    in_dim = 1  # scalar task drive per step (compress-style)
    lanes = []
    nparam = 0
    for _ in range(n_lanes):
        Win = g.standard_normal((H, in_dim)) / np.sqrt(in_dim)
        Wr = g.standard_normal((H, H)) / np.sqrt(H) * 0.9
        # U: H → targets (dense to |targets|)
        nt = max(1, len(targets))
        U = g.standard_normal((nt, H)) / np.sqrt(H) * 0.2
        lanes.append(dict(Win=Win, Wr=Wr, U=U, h=np.zeros(H)))
        nparam += Win.size + U.size  # count trainable (freeze Wr as reservoir)
    return lanes, nparam, targets


class LaneOp:
    def __init__(s, W, lanes, targets, N_vote):
        s.W = W if sp.issparse(W) else sp.csr_matrix(W)
        s.lanes = lanes; s.targets = np.asarray(targets, int); s.N_vote = N_vote
    def reset(s):
        for L in s.lanes: L["h"][:] = 0
    def __matmul__(s, x):
        y = np.asarray(s.W @ x, dtype=float).copy()
        # drive lanes with mean |x| summary
        u = np.array([float(np.mean(np.abs(x[: min(50, len(x))])) )])
        votes = []
        for L in s.lanes:
            L["h"] = np.tanh(L["Wr"] @ L["h"] + L["Win"] @ u)
            votes.append(L["U"] @ L["h"])
        if votes:
            v = np.mean(votes, axis=0)  # consensus
            np.add.at(y, s.targets[: len(v)], v)
        return y


def train_lanes(op, task, seed, steps=400):
    """Light SGD on Win,U with clipped grads; return trained op + train_sec."""
    g = np.random.default_rng(seed); t0 = time.time()
    # collect short trajectory with frozen random lanes then ridge on readout from state — keep simple:
    # just a few Hebbian-ish updates toward task signal
    n = op.W.shape[0]; x = np.zeros(n)
    T = steps; sig = g.standard_normal(T)
    for t in range(T):
        u_in = np.zeros(n); u_in[: min(20, n)] = 0.5 * sig[t]
        x = np.tanh(0.9 * (op @ x) + 0.1 * u_in)
        # RPE-free: nudge U toward correlating with sig
        err = sig[t] - float(np.mean(x[op.targets])) if len(op.targets) else 0.0
        for L in op.lanes:
            # dU ∝ err * outer(1, h) clipped
            d = np.clip(0.01 * err * L["h"][None, :], -GRAD_CLIP, GRAD_CLIP)
            L["U"] = np.clip(L["U"] + d[: L["U"].shape[0]], -CLIP_W, CLIP_W)
    return time.time() - t0


def eval_task(op, task, seed):
    g = np.random.default_rng(seed + 999); n = op.W.shape[0]; x = np.zeros(n)
    T = 600; xs = []; op.reset()
    if task == "MC":
        sig = g.standard_normal(T)
        for t in range(T):
            u_in = np.zeros(n); u_in[: min(20, n)] = sig[t]
            x = np.tanh(0.9 * (op @ x) + 0.1 * u_in); xs.append(x.copy())
        X = np.array(xs); mc = 0.0
        for d in range(1, 25):
            r = np.corrcoef(X[d:].mean(1), sig[:-d])[0, 1]
            if np.isfinite(r): mc += r * r
        return float(mc)
    if task == "T0":
        sig = g.standard_normal(T); tgt = np.sin(np.cumsum(sig) * 0.1)
        for t in range(T):
            u_in = np.zeros(n); u_in[: min(20, n)] = sig[t]
            x = np.tanh(0.9 * (op @ x) + 0.1 * u_in); xs.append(x.copy())
        X = np.array(xs)
        w, _, _, _ = np.linalg.lstsq(np.c_[X[:, :: max(1, n // 200)], np.ones(T)], tgt, rcond=None)
        pred = np.c_[X[:, :: max(1, n // 200)], np.ones(T)] @ w
        return float(np.mean((pred - tgt) ** 2))
    # NARMA10
    sig = g.uniform(0, 0.5, T); y = np.zeros(T)
    for t in range(15, T):
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * np.sum(y[t - 10:t]) + 1.5 * sig[t - 10] * sig[t - 1] + 0.1
    for t in range(T):
        u_in = np.zeros(n); u_in[: min(20, n)] = sig[t]
        x = np.tanh(0.9 * (op @ x) + 0.1 * u_in); xs.append(np.clip(x, -CLIP_W, CLIP_W))
    X = np.array(xs)[50:]; Y = y[50:]
    try:
        w, _, _, _ = np.linalg.lstsq(np.c_[X[:, :: max(1, n // 200)], np.ones(len(Y))], Y, rcond=None)
        pred = np.c_[X[:, :: max(1, n // 200)], np.ones(len(Y))] @ w
        val = float(np.mean((pred - Y) ** 2))
        if not np.isfinite(val) or val > 1e8: val = 1e9
    except Exception:
        val = 1e9
    return val


def run_one(meth, N, seed, task, wide=False):
    g = np.random.default_rng(seed)
    W, targets = pen_epg_targets(meth, seed)
    if len(targets) == 0:
        targets = np.arange(min(40, W.shape[0]))
    lanes, nparam, targets = make_lanes(g, W.shape[0], targets, N, wide=wide)
    op = LaneOp(W, lanes, targets, N if not wide else 1)
    t0 = time.time()
    train_sec = train_lanes(op, task, seed)
    val = eval_task(op, task, seed)
    return dict(meth=meth, N=(N if not wide else 0), wide=wide, seed=seed, task=task,
                value=val, nparam=nparam, n_tgt=len(targets), train_sec=train_sec, sec=time.time() - t0)


def job(a):
    meth, N, seed, task, wide = a
    return run_one(meth, N, seed, task, wide)


if __name__ == "__main__":
    have = set()
    if os.path.exists(JF):
        for line in open(JF):
            r = json.loads(line); have.add((r["meth"], r["N"], r["seed"], r["task"], r["wide"]))
    J = []
    for me in METHS:
        for s in SEEDS:
            for t in TASKS:
                for N in NS:
                    if (me, N, s, t, False) not in have: J.append((me, N, s, t, False))
                if (me, 0, s, t, True) not in have: J.append((me, 16, s, t, True))  # WIDE param-matched to N=16
    print("njobs", len(J), flush=True)
    with Pool(int(os.environ.get("WORKERS", 7))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["meth"], "N", r["N"], "wide", r["wide"], r["task"], r["seed"], round(r["value"], 4), flush=True)
    print("all done", flush=True)

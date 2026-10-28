"""Zebra finch Area X song-sequence reservoir. Prereg: PREREGISTRATION_finch.md.

Fairness: drive set is treated as pure input ports (incoming W edges zeroed) in
EVERY arm, matching HVC axon-only biology. Controls get a random same-size set.
"""
from __future__ import annotations
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
sys.path.insert(0, "/workspace/flybrain/regions/frankenstein")
sys.path.insert(0, "/workspace/flybrain/regions/compass")
import numpy as np, scipy.sparse as sp
from multiprocessing import Pool
from pathlib import Path
import frankenstein as F
import fast_kernels as K
from compass import DelayReservoirFast

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
SEEDS = list(range(8)); TUNE_SEEDS = [90, 91, 92]
LEAK, DMAX, RHO = 0.2, 5, 0.9
WASH, TR, TE = 200, 1200, 400
P, KDIM, NSYL, WPULSE = 80, 6, 4, 4
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
WIRINGS = ["REAL", "ER", "SHUF", "DENS"]; DELAYS = ["NOD", "DEL"]
ALL_ARMS = [f"{w}_{d}" for w in WIRINGS for d in DELAYS]
TASKS = ["SEQ", "TEMPO", "ABLATE"]

z = np.load(H / "data" / "areax_module.npz", allow_pickle=True)
W0 = sp.csr_matrix((z["W_data"].astype(np.float64), z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
SIGN = z["sign"].astype(np.int8)
HVC_IDX = np.where(z["cell_type"].astype(str) == "HVC")[0].astype(int)
N = int(W0.shape[0])


def scale(W):
    return W * (RHO / (float(K.radius_cached(W)) + 1e-12))


def zero_in_edges(W, ports):
    """Make ports pure input units: remove columns feeding into them (W[ports,:]=0).
    Convention: W is stored as contributions to postsynaptic = W @ pre (row=post in CSR from our build).
    In frankenstein/compress, reservoir does W @ x with row=post, col=pre. Zero rows of ports.
    """
    W = W.tolil()
    for i in ports:
        W.rows[i] = []; W.data[i] = []
    return W.tocsr()


def build(arm, seed):
    wiring, dmode = arm.split("_")
    g = np.random.default_rng(seed + 7)
    if wiring == "REAL":
        W = W0.copy(); ports = HVC_IDX.copy()
    elif wiring == "ER":
        W = F.er_match(W0, SIGN, g); ports = np.random.default_rng(seed + 4242).choice(N, len(HVC_IDX), replace=False)
    elif wiring == "SHUF":
        W = F.degree_shuffle(W0, SIGN, g); ports = np.random.default_rng(seed + 4242).choice(N, len(HVC_IDX), replace=False)
    elif wiring == "DENS":
        W = F.dens_match(W0, SIGN, g); ports = np.random.default_rng(seed + 4242).choice(N, len(HVC_IDX), replace=False)
    else:
        raise KeyError(wiring)
    W = zero_in_edges(W, ports)
    W = scale(W)
    Wcoo = W.tocoo()
    if dmode == "NOD":
        delay = np.ones(Wcoo.nnz, dtype=np.int32)
    else:
        delay = np.random.default_rng(seed + 909).integers(1, DMAX + 1, Wcoo.nnz).astype(np.int32)
    return W, ports.astype(int), delay, dmode, Wcoo


def song(T, period, phase0=0):
    t = (np.arange(T) + phase0) % period
    ph = t / float(period)
    Y = np.zeros((T, KDIM), dtype=np.float64)
    for k in range(KDIM):
        c = (k + 0.5) / KDIM
        dist = np.minimum(np.abs(ph - c), 1.0 - np.abs(ph - c))
        Y[:, k] = np.where(dist < 0.25, 0.5 * (1.0 + np.cos(np.pi * dist / 0.25)), 0.0)
    syl_len = period / NSYL
    onset = np.zeros((T, NSYL), dtype=np.float64)
    for s in range(NSYL):
        st = s * syl_len
        m = (t >= st) & (t < st + WPULSE)
        onset[m, s] = 1.0
    return Y, onset


def rollout(W, Wcoo, delay, dmode, Win, b, U):
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if dmode == "NOD":
        return K.run_reservoir_csr(W, Win, b, U, LEAK)
    # Match delay order to this Wcoo
    dr = DelayReservoirFast.__new__(DelayReservoirFast)
    dr.n = Wcoo.shape[0]
    dr.dmax = int(DMAX)
    dr.row = Wcoo.row.copy(); dr.col = Wcoo.col.copy()
    dr.data = Wcoo.data.astype(np.float64).copy()
    dr.delay = delay.astype(np.int32).copy()
    dr._rebuild_buckets()
    dr.buf = np.zeros((dr.dmax + 1, dr.n), dtype=np.float64)
    dr.t = 0
    X = np.empty((len(U), dr.n), dtype=np.float64)
    x = np.zeros(dr.n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = (Win @ U[i] + b) if multi else (Win[:, 0] * U[i] + b)
        y = dr.step(drive)
        x = (1.0 - LEAK) * x + LEAK * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x
    return X


def nrmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)) / (np.std(y) + 1e-12))


def features(arm, seed, task):
    W, ports, delay, dmode, Wcoo = build(arm, seed)
    rng = np.random.default_rng(seed + {"SEQ": 11, "TEMPO": 22, "ABLATE": 33}[task])
    blocks = np.array_split(rng.permutation(ports), NSYL)
    Win = np.zeros((N, NSYL), dtype=np.float64)
    for s, bl in enumerate(blocks):
        if len(bl):
            Win[bl, s] = rng.uniform(0.5, 1.0, len(bl))
    b = rng.uniform(-0.1, 0.1, N)
    Ytr, Otr = song(WASH + TR, P)
    if task == "TEMPO":
        Yte, Ote = song(TE, max(NSYL * WPULSE + 4, int(round(P / 1.5))))
    else:
        Yte, Ote = song(TE, P, phase0=(WASH + TR) % P)
    U = np.vstack([Otr, Ote])
    U = U + 0.02 * rng.standard_normal(U.shape)
    if task == "ABLATE":
        # train intact, then continue from train end state with lesioned Win for test
        Xtr = rollout(W, Wcoo, delay, dmode, Win, b, U[: WASH + TR])
        Win_te = Win.copy()
        kill = np.random.default_rng(seed + 555).choice(len(ports), len(ports) // 2, replace=False)
        Win_te[ports[kill], :] = 0.0
        # re-roll full trajectory with lesioned Win; use train from intact, test from lesioned run
        Xfull = rollout(W, Wcoo, delay, dmode, Win_te, b, U)
        X = np.vstack([Xtr, Xfull[WASH + TR:]])
    else:
        X = rollout(W, Wcoo, delay, dmode, Win, b, U)
    Y = np.vstack([Ytr, Yte])
    return X, Y


def score_all(X, Y):
    Xtr, Ytr = X[WASH: WASH + TR], Y[WASH: WASH + TR]
    Xte, Yte = X[WASH + TR:], Y[WASH + TR:]
    return {a: nrmse(K.pred(Xte, K.ridge(Xtr, Ytr, a)), Yte) for a in ALPHA_GRID}


def tune_job(args):
    arm, seed = args
    X, Y = features(arm, seed, "SEQ")
    return arm, seed, score_all(X, Y)


def eval_job(args):
    arm, seed, task, alpha = args
    t0 = time.time()
    X, Y = features(arm, seed, task)
    val = score_all(X, Y)[alpha]
    return dict(arm=arm, seed=int(seed), task=task, value=float(val),
                alpha=float(alpha), n=N, sec=float(time.time() - t0),
                delay_mode=arm.split("_")[1], wiring=arm.split("_")[0])


def main():
    workers = int(os.environ.get("WORKERS", "2"))
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        alphas = json.loads(cache.read_text())["alphas"]
        print("loaded alphas", alphas, flush=True)
    else:
        jobs = [(a, s) for a in ALL_ARMS for s in TUNE_SEEDS]
        acc = {a: {al: [] for al in ALPHA_GRID} for a in ALL_ARMS}
        with Pool(workers) as pool:
            for arm, seed, sc in pool.imap_unordered(tune_job, jobs):
                for al, v in sc.items():
                    acc[arm][al].append(v)
                print("tune", arm, seed, {k: round(v, 4) for k, v in sc.items()}, flush=True)
        alphas = {a: min(ALPHA_GRID, key=lambda al: float(np.mean(acc[a][al]))) for a in ALL_ARMS}
        cache.write_text(json.dumps(dict(
            alphas=alphas, grid=ALPHA_GRID, tune_seeds=TUNE_SEEDS,
            criterion="argmin mean SEQ NRMSE",
            locked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            fairness="zero in-edges onto drive ports in all arms",
        ), indent=2))
        print("LOCKED", alphas, flush=True)

    done = set()
    if JF.exists():
        for line in JF.read_text().splitlines():
            r = json.loads(line)
            done.add((r["arm"], int(r["seed"]), r["task"]))
    jobs = [(a, s, t, alphas[a]) for t in TASKS for s in SEEDS for a in ALL_ARMS
            if (a, s, t) not in done]
    print("eval jobs", len(jobs), "workers", workers, flush=True)
    with Pool(workers) as pool, open(JF, "a") as f:
        for r in pool.imap_unordered(eval_job, jobs):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["arm"], r["seed"], r["task"], round(r["value"], 4), round(r["sec"], 1), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

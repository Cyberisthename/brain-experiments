"""C. elegans multi-individual reservoirs. Prereg: PREREGISTRATION_celegans.md."""
from __future__ import annotations
import os, sys, json, time, math
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from pathlib import Path
import fast_kernels as K

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
WIRINGS = ["REAL", "ER", "SHUF", "DENS"]
DELAYS = ["NOD", "DEL"]
TASKS = ["MC", "NARMA", "PC_HARD"]
SENSORY = ["ASEL", "ASER", "AWCL", "AWCR", "ASGL", "ASGR"]
READOUT = ["AVAL", "AVAR", "AVBL", "AVBR", "AIYL", "AIYR", "AIZL", "AIZR"]

MANIFEST = json.loads((DATA / "manifest.json").read_text())
WORMS = [m["id"] for m in MANIFEST]
META = {m["id"]: m for m in MANIFEST}


def arm_name(worm, wiring, delay):
    return f"{worm}_{wiring}_{delay}"


def parse_arm(arm):
    # W7_REAL_DEL / COOK_H_ER_NOD
    parts = arm.rsplit("_", 2)
    if len(parts) != 3:
        raise KeyError(arm)
    return parts[0], parts[1], parts[2]


ALL_ARMS = [arm_name(w, wi, d) for w in WORMS for wi in WIRINGS for d in DELAYS]


def load_worm(worm_id):
    z = np.load(DATA / f"{worm_id}.npz", allow_pickle=True)
    shape = tuple(int(x) for x in z["W_shape"])
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=shape)
    names = np.asarray(z["names"]).astype(str)
    sign = z["sign"].astype(np.int8)
    return W, names, sign


def scale(W, rho=RHO):
    r = float(K.radius_cached(W))
    return W * (rho / (r + 1e-12))


def er_match(W, sign, g):
    N, nnz = W.shape[0], W.nnz
    rows = g.integers(0, N, nnz); cols = g.integers(0, N, nnz)
    m = rows != cols; rows, cols = rows[m], cols[m]
    data = np.abs(g.standard_normal(len(rows))) * sign[rows].astype(float)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def dens_match(W, sign, g):
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
        if not sl.any():
            break
        idx = np.where(sl)[0]
        g.shuffle(in_s[idx])
    data = np.abs(W.data) * sign[out_s].astype(float)
    return sp.csr_matrix((data, (out_s, in_s)), shape=(N, N))


_CACHE = {}


def get_real(worm_id):
    if worm_id not in _CACHE:
        _CACHE[worm_id] = load_worm(worm_id)
    return _CACHE[worm_id]


def build_wiring(worm_id, wiring, seed):
    W0, names, sign = get_real(worm_id)
    g = np.random.default_rng(seed + 13 + hash(worm_id) % 1000 + hash(wiring) % 97)
    if wiring == "REAL":
        W = W0.copy()
    elif wiring == "ER":
        W = er_match(W0, sign, g)
    elif wiring == "DENS":
        W = dens_match(W0, sign, g)
    elif wiring == "SHUF":
        W = degree_shuffle(W0, sign, g)
    else:
        raise KeyError(wiring)
    W = scale(W, RHO)
    # default drive: all neurons (small N)
    drive = np.arange(W.shape[0])
    return W, names, drive


class DelayReservoirFast:
    def __init__(self, W, delay, dmax):
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.row = W.row.copy()
        self.col = W.col.copy()
        self.data = W.data.astype(np.float64).copy()
        self.delay = delay.astype(np.int32).copy()
        self._rebuild_buckets()
        self.buf = np.zeros((self.dmax + 1, self.n), dtype=np.float64)
        self.t = 0

    def _rebuild_buckets(self):
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = self.delay == d
            if m.any():
                Wd = sp.csr_matrix(
                    (self.data[m], (self.row[m], self.col[m])), shape=(self.n, self.n)
                )
            else:
                Wd = None
            self.buckets.append(Wd)

    def step(self, x_in):
        self.t += 1
        slot = self.t % (self.dmax + 1)
        h = np.zeros(self.n, dtype=np.float64)
        for di, Wd in enumerate(self.buckets):
            d = di + 1
            if Wd is None:
                continue
            src_slot = (self.t - d) % (self.dmax + 1)
            h += Wd @ self.buf[src_slot]
        y = np.tanh(h + x_in)
        self.buf[slot] = y
        return y


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak):
    dr = DelayReservoirFast(W, delay, dmax)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W.shape[0]
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        y = dr.step(drive)
        x = (1.0 - leak) * x + leak * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x
    return X


def delays_for(W, delay_mode, seed):
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    if delay_mode == "NOD":
        return delay, DMAX
    g = np.random.default_rng(seed + 909)
    delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
    return delay, DMAX


def masked_win(n, d, in_idx, rng):
    Win = np.zeros((n, d), dtype=np.float64)
    b = rng.uniform(-0.1, 0.1, n)
    in_idx = np.asarray(in_idx, dtype=int)
    if len(in_idx) == 0:
        in_idx = np.arange(n)
    for k in range(d):
        Win[in_idx, k] = rng.uniform(-0.5, 0.5, len(in_idx))
    return Win, b


def rollout(W, delay, dmax, delay_mode, Win, b, U, leak):
    if delay_mode == "NOD":
        return K.run_reservoir_csr(W, Win, b, U, leak)
    return run_delay_reservoir(W, delay, dmax, Win, b, U, leak)


def mc_task(W, delay, dmax, delay_mode, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = rollout(W, delay, dmax, delay_mode, Win, b, u, leak)
    D = 100
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], alpha)
    P = K.pred(X[WASH + TR:], B)
    Yt = Y[WASH + TR:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_task(W, delay, dmax, delay_mode, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    if not np.all(np.isfinite(y)):
        return float("nan")
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = rollout(W, delay, dmax, delay_mode, Win, b, u, leak)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B)
    yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def pc_hard_task(W, delay, dmax, delay_mode, leak, rng, in_idx, alpha):
    n = W.shape[0]
    T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = np.asarray(in_idx)
    rng2 = np.random.default_rng(int(rng.integers(1e9)))
    k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
    drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
    Win, b = masked_win(n, 2, drive_idx, rng)
    wash, tr = 200, 800
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = rollout(W, delay, dmax, delay_mode, Win, b, u_drive, leak)
    B = K.ridge(X[wash:tr], u_clean[wash:tr], alpha)
    pred = K.pred(X[tr:], B)
    yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


def name_indices(names, wanted):
    name_to_i = {n: i for i, n in enumerate(names)}
    return np.array([name_to_i[w] for w in wanted if w in name_to_i], dtype=int)


def chemo_task(W, names, delay, dmax, delay_mode, leak, rng, alpha, io_mode, seed):
    """Concentration → sensory; readout from command neurons. R² of delayed c(t)."""
    sens = name_indices(names, SENSORY)
    cmd = name_indices(names, READOUT)
    n = W.shape[0]
    if len(sens) < 2 or len(cmd) < 2:
        return float("nan"), dict(n_sens=int(len(sens)), n_cmd=int(len(cmd)), skip=True)
    if io_mode == "RAND":
        g2 = np.random.default_rng(seed + 4242)
        sens = g2.choice(n, size=len(sens), replace=False)
        cmd = g2.choice(n, size=len(cmd), replace=False)
        # ensure disjoint if possible
        cmd = np.array([c for c in cmd if c not in set(sens.tolist())] or cmd.tolist())
        if len(cmd) < 2:
            cmd = g2.choice([i for i in range(n) if i not in set(sens.tolist())], size=2, replace=False)
    T = WASH + TR + TE
    # slow concentration: OU-like
    c = np.zeros(T)
    x = 0.0
    for t in range(T):
        x = 0.95 * x + 0.05 * rng.standard_normal()
        c[t] = np.tanh(x) * 0.5 + 0.5 * np.sin(2 * np.pi * t / 200.0)
    target = np.roll(c, 5)
    Win, b = masked_win(n, 1, sens, rng)
    X = rollout(W, delay, dmax, delay_mode, Win, b, c, leak)
    Xr = X[:, cmd]
    B = K.ridge(Xr[WASH:WASH + TR], target[WASH:WASH + TR], alpha)
    p = K.pred(Xr[WASH + TR:], B)
    yt = target[WASH + TR:]
    ss = ((yt - yt.mean()) ** 2).sum() + 1e-12
    r2 = float(1 - ((yt - p) ** 2).sum() / ss)
    return r2, dict(n_sens=int(len(sens)), n_cmd=int(len(cmd)), skip=False)


def default_alpha(arm):
    # fallback before tune
    return 1e-6


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("TUNE alphas", len(ALL_ARMS), "arms seeds", TUNE_SEEDS, flush=True)
    choice = {
        "alphas": {},
        "grid": ALPHA_GRID,
        "criterion": "mean MC @ full train",
        "tune_seeds": TUNE_SEEDS,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for arm in ALL_ARMS:
        worm, wiring, delay_mode = parse_arm(arm)
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            W, names, drive = build_wiring(worm, wiring, seed)
            delay, dmax = delays_for(W, delay_mode, seed)
            rng = np.random.default_rng(seed + 0)
            # evaluate all alphas on one rollout
            T = WASH + TR + TE
            u = rng.uniform(-1, 1, T)
            Win, b = masked_win(W.shape[0], 1, drive, rng)
            X = rollout(W, delay, dmax, delay_mode, Win, b, u, LEAK)
            D = 100
            Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
            for a in ALPHA_GRID:
                B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], a)
                P = K.pred(X[WASH + TR:], B)
                Yt = Y[WASH + TR:]
                val = float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
                scores[a].append(val)
            print(f"tune {arm} seed={seed}", flush=True)
        means = {a: float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[a] if np.isfinite(means[a]) else -1e99, -a))
        choice["alphas"][arm] = {"alpha": best, "means": means}
        print(f"LOCK alpha {arm} -> {best}", flush=True)
        # checkpoint
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, task, alpha):
    t0 = time.time()
    worm, wiring, delay_mode = parse_arm(arm)
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4, "CHEMO": 6}[task] + 17)
    W, names, drive = build_wiring(worm, wiring, seed)
    delay, dmax = delays_for(W, delay_mode, seed)
    meta = META[worm]
    if task == "MC":
        val = mc_task(W, delay, dmax, delay_mode, LEAK, g, drive, alpha)
    elif task == "NARMA":
        val = narma_task(W, delay, dmax, delay_mode, LEAK, g, drive, alpha)
    elif task == "PC_HARD":
        val = pc_hard_task(W, delay, dmax, delay_mode, LEAK, g, drive, alpha)
    else:
        raise KeyError(task)
    n = int(W.shape[0])
    out = dict(
        test="main",
        arm=arm, worm=worm, wiring=wiring, delay_mode=delay_mode,
        seed=seed, task=task, value=float(val) if val == val else float("nan"),
        n=n, mc_over_n=(float(val) / n if task == "MC" and val == val else None),
        alpha=alpha, stage=meta["stage"], stage_ord=meta["stage_ord"],
        sex=meta["sex"], source=meta["source"],
        delay_mean=float(delay.mean()), sec=time.time() - t0,
    )
    return out


def run_chemo_one(worm, delay_mode, io_mode, seed, alpha):
    t0 = time.time()
    arm = f"{worm}_CHEMO_{io_mode}_{delay_mode}"
    g = np.random.default_rng(seed + 6 + 17)
    W, names, _ = build_wiring(worm, "REAL", seed)
    delay, dmax = delays_for(W, delay_mode, seed)
    val, info = chemo_task(W, names, delay, dmax, delay_mode, LEAK, g, alpha, io_mode, seed)
    meta = META[worm]
    return dict(
        test="chemo",
        arm=arm, worm=worm, wiring="REAL", delay_mode=delay_mode,
        io_mode=io_mode, seed=seed, task="CHEMO",
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=alpha,
        stage=meta["stage"], stage_ord=meta["stage_ord"],
        sex=meta["sex"], source=meta["source"],
        sec=time.time() - t0, **info,
    )


def _job(args):
    try:
        kind = args[0]
        if kind == "main":
            return run_one(*args[1:])
        return run_chemo_one(*args[1:])
    except Exception as e:
        import traceback
        return dict(test=args[0], error=str(e)[:400], traceback=traceback.format_exc()[-600:],
                    args=str(args)[:200], value=float("nan"))


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "tune":
        tune_alphas()
        return
    alphas = tune_alphas()
    # main jobs
    jobs = []
    for arm in ALL_ARMS:
        a = alphas["alphas"][arm]["alpha"]
        for seed in SEEDS:
            for task in TASKS:
                jobs.append(("main", arm, seed, task, a))
    # chemo: use REAL_NOD / REAL_DEL alphas for that worm
    for worm in WORMS:
        for delay_mode in DELAYS:
            base = arm_name(worm, "REAL", delay_mode)
            a = alphas["alphas"][base]["alpha"]
            for io_mode in ("REAL", "RAND"):
                for seed in SEEDS:
                    jobs.append(("chemo", worm, delay_mode, io_mode, seed, a))

    done = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                if r.get("test") == "main":
                    done.add(("main", r["arm"], r["seed"], r["task"]))
                elif r.get("test") == "chemo":
                    done.add(("chemo", r["worm"], r["delay_mode"], r["io_mode"], r["seed"]))
            except Exception:
                pass
    filt = []
    for j in jobs:
        if j[0] == "main":
            key = ("main", j[1], j[2], j[3])
        else:
            key = ("chemo", j[1], j[2], j[3], j[4])
        if key not in done:
            filt.append(j)
    jobs = filt
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"celegans jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = _job(j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r.get("test"), r.get("arm"), r.get("seed"), r.get("task"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("test"), r.get("arm"), r.get("seed"), r.get("task"), r.get("value"),
                          round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
    (OUT / "celegans.done").write_text("ok\n")
    print("celegans done", flush=True)


if __name__ == "__main__":
    main()

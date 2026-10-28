"""Real H01 myelin → delays on human-pre edges. Prereg: PREREGISTRATION_myelin_real.md."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
H = Path(__file__).resolve().parent
_ROOT = H.parents[1]
sys.path.insert(0, str(_ROOT / "compress"))
sys.path.insert(0, str(_ROOT / "regions" / "empn"))
sys.path.insert(0, str(_ROOT / "regions" / "hybrid_human"))
sys.path.insert(0, str(_ROOT / "regions" / "frankenstein"))
sys.path.insert(0, str(_ROOT))
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import fast_kernels as K

OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
MYELIN = _ROOT / "regions" / "hybrid_human" / "data" / "human_myelin.npz"
SEEDS = list(range(8))
ARMS = ["NODELAY", "REAL_MYELIN", "SHUF_MYELIN", "ADAPT"]
NETS = ["HUMAN_LAYERED", "FRANKEN"]
TASKS = ["MC", "NARMA", "PC_HARD"]
LEAK = 0.2
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5


def load_myelin():
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf_filled = mf.copy()
    mf_filled[~np.isfinite(mf_filled)] = med
    return mf, mf_filled, med, z


def myelin_to_delay(mf, dmax=DMAX):
    mf = np.clip(mf, 0.0, 1.0)
    return (1 + np.rint((1.0 - mf) * (dmax - 1))).astype(np.int32)


def human_index_range(net, info):
    """Return (start, end) for human neurons in full W, and local myelin length check."""
    if net == "HUMAN_LAYERED":
        Nm = int(info["Nm"])
        return 0, Nm
    # FRANKEN: blocks mouse, human, fly_cx
    names = info["names"]
    offs = info["offsets"]
    i = names.index("human")
    return int(offs[i]), int(offs[i + 1])


def build_net(net, seed, g):
    if net == "HUMAN_LAYERED":
        import hybrid_human as HH
        W, info = HH.build_arm("HUMAN_LAYERED", seed, g)
        return W, info
    import frankenstein as F
    # use cached radius like fast runner
    _orig = F.scale
    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale
    try:
        W, info = F.build_arm("FRANKEN", seed, g)
    finally:
        F.scale = _orig
    return W, info


def delays_for_arm(arm, W, net, info, seed, mf_filled):
    """Per-edge int delays; only human-pre edges get myelin-based delays."""
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    h0, h1 = human_index_range(net, info)
    n_h = h1 - h0
    assert n_h == len(mf_filled), (n_h, len(mf_filled), net)
    g = np.random.default_rng(seed + 909)
    # SciPy reservoir W[post, pre]: COO.row=post, COO.col=pre
    pre = Wcoo.col
    if arm == "NODELAY":
        return delay, DMAX
    if arm == "ADAPT":
        pre_h = (pre >= h0) & (pre < h1)
        delay[pre_h] = g.integers(1, DMAX + 1, int(pre_h.sum())).astype(np.int32)
        return delay, DMAX
    mf = mf_filled.copy()
    if arm == "SHUF_MYELIN":
        g.shuffle(mf)
    d_neuron = myelin_to_delay(mf)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    return delay, DMAX


class DelayReservoirFast:
    """Bucketed CSR delays; optional ADAPT shorten/lengthen on human-pre edges."""
    def __init__(self, W, delay, dmax, adapt=False, g=None):
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.adapt = adapt
        self.g = g if g is not None else np.random.default_rng(0)
        self.row = W.row.copy()
        self.col = W.col.copy()
        self.data = W.data.astype(np.float64).copy()
        self.delay = delay.astype(np.int32).copy()
        self._rebuild_buckets()
        self.buf = np.zeros((self.dmax + 1, self.n), dtype=np.float64)
        self.t = 0
        self.pre_trace = np.zeros(self.n)
        self.post_trace = np.zeros(self.n)

    def _rebuild_buckets(self):
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = self.delay == d
            if m.any():
                # Keep W[post, pre] = (row, col) so y = Wd @ x_pre
                Wd = sp.csr_matrix(
                    (self.data[m], (self.row[m], self.col[m])),
                    shape=(self.n, self.n),
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
        if self.adapt:
            self.pre_trace = 0.9 * self.pre_trace + 0.1 * (self.buf[(self.t - 1) % (self.dmax + 1)] ** 2)
            self.post_trace = 0.9 * self.post_trace + 0.1 * (y ** 2)
            co = self.pre_trace[self.col] * self.post_trace[self.row]
            med = np.median(co)
            shorten = co > med
            self.delay[shorten] = np.maximum(1, self.delay[shorten] - 1)
            lengthen = co < np.percentile(co, 20)
            self.delay[lengthen] = np.minimum(self.dmax, self.delay[lengthen] + 1)
            # rebuild every 50 steps to keep cost down
            if self.t % 50 == 0:
                self._rebuild_buckets()
        return y


def masked_win(n, d, in_idx, rng):
    Win = np.zeros((n, d))
    b = rng.uniform(-0.1, 0.1, n)
    idx = np.asarray(in_idx)
    if len(idx) == 0:
        idx = np.arange(n)
    Win[idx] = rng.uniform(-0.5, 0.5, (len(idx), d))
    return Win, b


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak, adapt=False, seed=0):
    """Leak-integrated delay reservoir. U (T,) or (T,d)."""
    g = np.random.default_rng(seed + 11)
    dr = DelayReservoirFast(W, delay, dmax, adapt=adapt, g=g)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W.shape[0]
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        # DelayReservoirFast.step returns tanh(h+drive); blend with leak like ESN
        y = dr.step(drive)
        # overwrite: we want leak on top of delayed tanh — re-do lightly
        # Actually myelin.py used pure tanh without leak. Prereg says leak=0.2 like fast.
        # Compose: x = (1-leak)*x + leak*tanh(delayed + drive)
        # But dr.step already wrote tanh into buf. Rebuild with leak:
        # Simpler approach matching spirit: use dr's tanh output as candidate, leak-integrate.
        x = (1.0 - leak) * x + leak * y
        # keep buf consistent with leaked state for next delays
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x
    return X


def mc_task(W, delay, dmax, leak, rng, in_idx, adapt, seed):
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=adapt, seed=seed)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = K.pred(X[wash + tr:], B)
    Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_task(W, delay, dmax, leak, rng, in_idx, adapt, seed):
    wash, tr, te = 200, 1500, 500
    T = wash + tr + te
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    if not np.all(np.isfinite(y)):
        return float("nan"), {"narma_target_nonfinite": True}
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=adapt, seed=seed)
    if not np.all(np.isfinite(X)):
        return float("nan"), {"narma_state_nonfinite": True}
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan"), {"narma_pred_nonfinite": True}
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return (v if np.isfinite(v) else float("nan")), {}


def pc_hard_task(W, delay, dmax, leak, rng, in_idx, adapt, seed):
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
    X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak, adapt=adapt, seed=seed)
    Wout = np.linalg.lstsq(X[wash:tr], u_clean[wash:tr], rcond=None)[0]
    pred = X[tr:] @ Wout
    yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


def drive_idx_for(net, info):
    if net == "HUMAN_LAYERED":
        # L4 gateway (same as hybrid_human layered convention)
        if "L4" in info and len(info["L4"]):
            return np.asarray(info["L4"])
        return np.arange(info["Nm"])
    return info["drive"]


_MF_CACHE = None

def _mf():
    global _MF_CACHE
    if _MF_CACHE is None:
        _MF_CACHE = load_myelin()
    return _MF_CACHE


def run_one(net, arm, seed, task):
    t0 = time.time()
    mf_raw, mf_filled, med, _ = _mf()
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
    W, info = build_net(net, seed, g)
    # spectral radius already scaled in builders
    delay, dmax = delays_for_arm(arm, W, net, info, seed, mf_filled)
    adapt = arm == "ADAPT"
    in_idx = drive_idx_for(net, info)
    flags = {"myelin_median_fill": med, "dmax": dmax,
             "n_human": int(human_index_range(net, info)[1] - human_index_range(net, info)[0]),
             "delay_mean_human_pre": float(delay.mean()),
             "delay_hist": {str(i): int((delay == i).sum()) for i in range(1, dmax + 1)}}
    if task == "MC":
        val = mc_task(W, delay, dmax, LEAK, g, in_idx, adapt, seed)
        flags_extra = {}
    elif task == "NARMA":
        val, flags_extra = narma_task(W, delay, dmax, LEAK, g, in_idx, adapt, seed)
    else:
        val = pc_hard_task(W, delay, dmax, LEAK, g, in_idx, adapt, seed)
        flags_extra = {"pc_hard_cue": PC_HARD_CUE, "pc_hard_noise": PC_HARD_NOISE}
    flags.update(flags_extra)
    out = dict(net=net, arm=arm, seed=seed, task=task,
               value=(None if (isinstance(val, float) and val != val) else float(val)),
               n=int(W.shape[0]), sec=time.time() - t0, source="myelin_real",
               flags=flags)
    if isinstance(val, float) and val != val:
        out["value"] = float("nan")
        out["narma_nan"] = True
    return out


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        net, arm, seed, task = args
        return dict(net=net, arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-600:],
                    source="myelin_real")


def main():
    # coverage gate
    src = _ROOT / "regions" / "hybrid_human" / "data" / "SOURCE_myelin.json"
    if not MYELIN.exists() or not src.exists():
        print("MISSING myelin artifacts — run extract first", flush=True)
        sys.exit(2)
    cov = json.loads(src.read_text())["coverage"]
    if cov["n_with_axon_annotation"] < 100:
        print("GATE FAIL: axon coverage too low", cov, flush=True)
        (OUT / "STOPPED_low_coverage.json").write_text(json.dumps(cov, indent=2))
        sys.exit(3)
    jobs = [(n, a, s, t) for n in NETS for a in ARMS for s in SEEDS for t in TASKS]
    done = set()
    if JF.exists():
        for line in open(JF):
            r = json.loads(line)
            done.add((r["net"], r["arm"], r["seed"], r["task"]))
    jobs = [j for j in jobs if j not in done]
    print(f"myelin_real jobs {len(jobs)} skip={len(done)} cov_axon={cov['n_with_axon_annotation']}", flush=True)
    workers = int(os.environ.get("WORKERS", "2"))
    with Pool(workers) as P:
        for r in P.imap_unordered(_job, jobs):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r.get("net"), r.get("arm"), r.get("task"), r.get("seed"),
                  r.get("value"), f"{r.get('sec', 0):.1f}s", r.get("error", ""), flush=True)
    (OUT / "myelin_real.done").write_text("ok\n")


if __name__ == "__main__":
    main()

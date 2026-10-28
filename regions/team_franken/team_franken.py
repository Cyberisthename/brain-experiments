"""TEAM Frankenstein: FLY_CX ⊕ HUMAN division of labor. Prereg: PREREG_team_franken.md."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "compress"))
sys.path.insert(0, str(_ROOT / "regions" / "frankenstein"))
sys.path.insert(0, str(_ROOT))
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import frankenstein as F
import fast_kernels as K

try:
    import fastcore2 as FC2  # noqa: F401
    _HAS_FC2 = True
except Exception:
    _HAS_FC2 = False

H = _HERE
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
MYELIN = DATA / "human_myelin.npz"

SEEDS = list(range(400, 424))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
XOR_D = 5
BRIDGE_SIGMA_TEAM = 0.0  # zero bridges (division of labor)

ALL_ARMS = ["TEAM", "FLY_CX", "HUMAN", "FRANKEN", "TEAM_ER", "TEAM_EI", "FLY_CX_BIG"]
PRIMARY_TASKS = ["XOR_DELAY", "NARMA"]
SECONDARY_TASKS = ["MC", "PC_HARD"]
ALL_TASKS = PRIMARY_TASKS + SECONDARY_TASKS

# ---------------------------------------------------------------------------
# Myelin / delays
# ---------------------------------------------------------------------------
_MF = None


def load_myelin():
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf_filled = mf.copy()
    mf_filled[~np.isfinite(mf_filled)] = med
    return mf_filled, med


def mf_filled():
    global _MF
    if _MF is None:
        _MF = load_myelin()
    return _MF


def human_range(info):
    if "human" not in info["names"]:
        return None
    i = info["names"].index("human")
    return int(info["offsets"][i]), int(info["offsets"][i + 1])


def delays_for(W, info, seed, mf, shuffle_myelin=False):
    """REALMYELIN on human-range; FIXED elsewhere. Optional shuffle of mf for TEAM_ER."""
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    g = np.random.default_rng(seed + 909)
    pre = Wcoo.col
    hr = human_range(info)
    if hr is None:
        delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
        return delay, DMAX
    h0, h1 = hr
    assert (h1 - h0) == len(mf), (h1 - h0, len(mf))
    mf_use = mf.copy()
    if shuffle_myelin:
        g2 = np.random.default_rng(seed + 4242)
        g2.shuffle(mf_use)
    mf_use = np.clip(mf_use, 0.0, 1.0)
    d_neuron = (1 + np.rint((1.0 - mf_use) * (DMAX - 1))).astype(np.int32)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    other = ~mask
    delay[other] = g.integers(1, DMAX + 1, int(other.sum())).astype(np.int32)
    return delay, DMAX


# ---------------------------------------------------------------------------
# Delay reservoir (bit-exact with xor_rep / trained_bridges)
# ---------------------------------------------------------------------------
class DelayReservoirFast:
    def __init__(self, W, delay, dmax):
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.row = W.row.copy(); self.col = W.col.copy()
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
                Wd = sp.csr_matrix((self.data[m], (self.row[m], self.col[m])), shape=(self.n, self.n))
            else:
                Wd = None
            self.buckets.append(Wd)

    def step(self, x_in):
        self.t += 1
        slot = self.t % (self.dmax + 1)
        h = np.zeros(self.n, dtype=np.float64)
        for di, Wd in enumerate(self.buckets):
            if Wd is None: continue
            src = (self.t - (di + 1)) % (self.dmax + 1)
            h += Wd @ self.buf[src]
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


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------
def ei_class_shuffle(W, sign, g):
    Wc = W.tocoo()
    N = W.shape[0]
    row, col, data = Wc.row.copy(), Wc.col.copy(), np.abs(Wc.data).copy()
    s_pre = sign[row]
    s_post = sign[col]
    new_col = col.copy()
    for s_pre_v in (-1, 1):
        for s_post_v in (-1, 1):
            m = (s_pre == s_pre_v) & (s_post == s_post_v)
            idx = np.where(m)[0]
            if len(idx) < 2:
                continue
            cands = np.where(sign == s_post_v)[0]
            if len(cands) == 0:
                continue
            posts = col[idx].copy()
            g.shuffle(posts)
            for _ in range(8):
                bad = row[idx] == posts
                if not bad.any():
                    break
                posts[bad] = g.choice(cands, size=int(bad.sum()), replace=True)
            new_col[idx] = posts
    out_data = data * sign[row].astype(float)
    return sp.csr_matrix((out_data, (row, new_col)), shape=(N, N))


def graft_zero_bridge(blocks, seed, g=None):
    """Block-diag only; drive = union of L4s. No inter-block edges."""
    if g is None:
        g = np.random.default_rng(seed + 7)
    names = [b[0] for b in blocks]
    Ws = [b[1] for b in blocks]
    layers = [b[2] for b in blocks]
    sizes = [W.shape[0] for W in Ws]
    N = sum(sizes)
    off = np.cumsum([0] + sizes)
    W = sp.lil_matrix((N, N))
    for i, Wi in enumerate(Ws):
        a, b = off[i], off[i + 1]
        W[a:b, a:b] = Wi
    W = F.scale(W.tocsr(), RHO)
    drive = []
    for i, layer in enumerate(layers):
        L4, _, _ = F.layer_sets(layer, sizes[i])
        drive.append(L4 + off[i])
    drive = np.unique(np.concatenate(drive)) if drive else np.arange(min(200, N))
    fly_off = int(off[names.index("fly_cx")]) if "fly_cx" in names else 0
    info = dict(names=names, sizes=sizes, offsets=[int(x) for x in off], N=N,
                drive=drive, bridge={}, fly_offset=fly_off, bridge_sigma=BRIDGE_SIGMA_TEAM)
    return W, info


def build_wiring(arm, seed):
    g = np.random.default_rng(seed + 7)
    shuffle_myelin = False

    if arm == "FLY_CX":
        W, info = F.build_arm("FLY_CX", seed, g)
        return W.tocsr(), info, shuffle_myelin

    if arm == "HUMAN":
        Wm, sign, layer, ct = F.make_block("human", "real", seed)
        W = F.scale(Wm, RHO)
        L4, _, _ = F.layer_sets(layer, Wm.shape[0])
        info = dict(names=["human"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                    N=Wm.shape[0], drive=L4, bridge={}, fly_offset=0, kinds={"human": "real"})
        return W.tocsr(), info, shuffle_myelin

    if arm == "FRANKEN":
        W, info = F.build_arm("FRANKEN", seed, g)
        return W.tocsr(), info, shuffle_myelin

    if arm == "TEAM":
        Wf, sf, lf, _ = F.make_block("fly_cx", "real", seed)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "real", "human": "real"}
        return W.tocsr(), info, shuffle_myelin

    if arm == "TEAM_ER":
        Wf0, sf, lf, _ = F.load_mod("fly_cx")
        Wh0, sh, lh, _ = F.load_mod("human")
        g1 = np.random.default_rng(seed + 13 + hash("fly_cx") % 1000)
        g2 = np.random.default_rng(seed + 13 + hash("human") % 1000)
        Wf = F.scale(F.er_match(Wf0, sf, g1), 1.0)
        Wh = F.scale(F.er_match(Wh0, sh, g2), 1.0)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "er", "human": "er"}
        info["names"] = ["fly_cx", "human"]  # keep human name so myelin range applies
        return W.tocsr(), info, True  # shuffled myelin

    if arm == "TEAM_EI":
        Wf0, sf, lf, _ = F.load_mod("fly_cx")
        g1 = np.random.default_rng(seed + 13 + hash("fly_cx") % 1000)
        Wf = F.scale(ei_class_shuffle(Wf0, sf, g1), 1.0)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "ei_shuf", "human": "real"}
        return W.tocsr(), info, shuffle_myelin

    if arm == "FLY_CX_BIG":
        Wf, sf, lf, _ = F.make_block("fly_cx", "real", seed)
        Wh0, sh, lh, _ = F.load_mod("human")
        g2 = np.random.default_rng(seed + 13 + hash("human") % 1000)
        # dens_match ER addon sized like human
        Wh = F.scale(F.dens_match(Wh0, sh, g2), 1.0)
        # name addon "addon" so REALMYELIN does NOT apply (FIXED everywhere)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("addon", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "real", "addon": "dens"}
        return W.tocsr(), info, shuffle_myelin

    raise KeyError(arm)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def score_xor(W, delay, dmax, leak, rng, in_idx, alpha, d=XOR_D):
    T = WASH + TR + TE
    a = rng.integers(0, 2, T).astype(np.float64)
    b = rng.integers(0, 2, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(d, T):
        y[t] = 1.0 if (int(a[t - d]) ^ int(b[t - d])) else -1.0
    U = np.stack([2 * a - 1, 2 * b - 1], 1)
    Win, bias = masked_win(W.shape[0], 2, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B)
    yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan")


def score_narma(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    if not np.all(np.isfinite(y)):
        return float("nan")
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B)
    yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def score_mc(W, delay, dmax, leak, rng, in_idx, alpha, D=100):
    T = WASH + TR + TE
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], alpha)
    P = K.pred(X[WASH + TR:], B)
    Yt = Y[WASH + TR:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def score_pc_hard(W, delay, dmax, leak, rng, in_idx, alpha):
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
    X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak)
    B = K.ridge(X[wash:tr], u_clean[wash:tr], alpha)
    pred = K.pred(X[tr:], B)
    yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


TASK_SEED_OFF = {"MC": 0, "NARMA": 2, "PC_HARD": 4, "XOR_DELAY": 6}
TASK_FN = {
    "XOR_DELAY": score_xor,
    "NARMA": score_narma,
    "MC": score_mc,
    "PC_HARD": score_pc_hard,
}


def run_one(arm, seed, task, alpha):
    t0 = time.time()
    W, info, shuf_m = build_wiring(arm, seed)
    mf, med = mf_filled()
    delay, dmax = delays_for(W, info, seed, mf, shuffle_myelin=shuf_m)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    fn = TASK_FN[task]
    val = fn(W, delay, dmax, LEAK, rng, info["drive"], alpha)
    return dict(
        arm=arm, seed=int(seed), task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        delay_mean=float(np.mean(delay)), myelin_median=med,
        shuffle_myelin=bool(shuf_m), bridge_sigma=float(info.get("bridge_sigma", 0.05 if arm == "FRANKEN" else 0.0)),
        sec=time.time() - t0, has_fastcore2=_HAS_FC2, source="team_franken",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="team_franken")


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------
def _score_from_XY(task, X, y, alpha):
    if X is None or not np.all(np.isfinite(X)):
        return float("nan")
    if task == "MC":
        B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
        P = K.pred(X[WASH + TR:], B); Yt = y[WASH + TR:]
        D = Yt.shape[1]
        return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
    if task == "PC_HARD":
        wash, tr = 200, 800
        B = K.ridge(X[wash:tr], y[wash:tr], alpha)
        pred = K.pred(X[tr:], B); yt = y[tr:]
        return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))
    # XOR / NARMA scalar y
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B)
    yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse):
        return float("nan")
    if task == "XOR_DELAY":
        return float(1.0 - nrmse)
    return nrmse  # NARMA


def _make_states(arm, seed, task):
    W, info, shuf_m = build_wiring(arm, seed)
    mf, med = mf_filled()
    delay, dmax = delays_for(W, info, seed, mf, shuffle_myelin=shuf_m)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    if task == "XOR_DELAY":
        T = WASH + TR + TE
        a = rng.integers(0, 2, T).astype(np.float64)
        b = rng.integers(0, 2, T).astype(np.float64)
        y = np.zeros(T)
        for t in range(XOR_D, T):
            y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
        U = np.stack([2 * a - 1, 2 * b - 1], 1)
        Win, bias = masked_win(W.shape[0], 2, info["drive"], rng)
        X = run_delay_reservoir(W, delay, dmax, Win, bias, U, LEAK)
        return X, y, W.shape[0]
    if task == "NARMA":
        T = WASH + TR + TE
        u = rng.uniform(0, 0.5, T)
        y = np.zeros(T)
        for i in range(10, T - 1):
            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
        Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
        X = run_delay_reservoir(W, delay, dmax, Win, b, u, LEAK)
        return X, y, W.shape[0]
    if task == "MC":
        T = WASH + TR + TE
        u = rng.uniform(-1, 1, T)
        Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
        X = run_delay_reservoir(W, delay, dmax, Win, b, u, LEAK)
        Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
        return X, Y, W.shape[0]
    if task == "PC_HARD":
        n = W.shape[0]
        T = 1200
        theta = np.linspace(0, 8 * np.pi, T)
        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
        idx = np.asarray(info["drive"])
        rng2 = np.random.default_rng(int(rng.integers(1e9)))
        k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
        Win, b = masked_win(n, 2, drive_idx, rng)
        u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
        X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, LEAK)
        return X, u_clean, n
    raise KeyError(task)


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"TUNE arms={ALL_ARMS} tasks={ALL_TASKS} seeds={TUNE_SEEDS} fc2={_HAS_FC2}", flush=True)
    choice = {
        "alphas": {}, "grid": ALPHA_GRID,
        "tune_seeds": TUNE_SEEDS, "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_team_franken.md",
        "composite_z": {},
        "tune_scores": {},
    }
    # collect tune scores for z-lock
    xor_vals, narma_vals = [], []
    for arm in ALL_ARMS:
        choice["alphas"][arm] = {}
        choice["tune_scores"][arm] = {}
        for task in ALL_TASKS:
            scores = {a: [] for a in ALPHA_GRID}
            raw_best = []
            for seed in TUNE_SEEDS:
                t0 = time.time()
                X, y, n = _make_states(arm, seed, task)
                for alpha in ALPHA_GRID:
                    scores[alpha].append(_score_from_XY(task, X, y, alpha))
                # provisional best for logging
                if task == "NARMA":
                    best_a = min(ALPHA_GRID, key=lambda a: (scores[a][-1] if np.isfinite(scores[a][-1]) else 1e99, a))
                else:
                    best_a = max(ALPHA_GRID, key=lambda a: (scores[a][-1] if np.isfinite(scores[a][-1]) else -1e99, -a))
                print(f"tune {arm} {task} seed={seed} n={n} a~{best_a} v={scores[best_a][-1]:.4f} {time.time()-t0:.1f}s", flush=True)
            means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
            if task == "NARMA":
                best = min(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else 1e99, a))
            else:
                best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
            choice["alphas"][arm][task] = {"alpha": best, "means": means}
            # store per-seed scores at locked alpha for composite z
            locked_scores = [scores[best][i] for i in range(len(TUNE_SEEDS))]
            choice["tune_scores"][arm][task] = locked_scores
            if task == "XOR_DELAY":
                xor_vals.extend([v for v in locked_scores if np.isfinite(v)])
            if task == "NARMA":
                narma_vals.extend([v for v in locked_scores if np.isfinite(v)])
            print(f"LOCK alpha {arm} {task} -> {best}", flush=True)
            cache.write_text(json.dumps(choice, indent=2))
    # lock composite z from tune pool
    xor_vals = np.asarray(xor_vals, float)
    narma_vals = np.asarray(narma_vals, float)
    choice["composite_z"] = {
        "mu_xor": float(np.mean(xor_vals)) if len(xor_vals) else 0.0,
        "sd_xor": float(np.std(xor_vals, ddof=1)) if len(xor_vals) > 1 else 1.0,
        "mu_narma": float(np.mean(narma_vals)) if len(narma_vals) else 0.0,
        "sd_narma": float(np.std(narma_vals, ddof=1)) if len(narma_vals) > 1 else 1.0,
        "n_xor": int(len(xor_vals)), "n_narma": int(len(narma_vals)),
        "source": "tune seeds × all arms, locked before eval",
    }
    cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    print("composite_z", choice["composite_z"], flush=True)
    return choice


def _load_done():
    done = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if r.get("error"): continue
                done.add((r["arm"], r["seed"], r["task"]))
            except Exception:
                pass
    return done


def smoke():
    out = []
    for arm in ("TEAM", "FLY_CX", "HUMAN", "FLY_CX_BIG"):
        for task in ("XOR_DELAY", "NARMA"):
            r = run_one(arm, 400, task, 1e-6)
            out.append(r)
            print("SMOKE", {k: r.get(k) for k in ("arm", "task", "value", "n", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")


def build_jobs(alphas, phase="all"):
    jobs = []
    tasks = ALL_TASKS if phase == "all" else (PRIMARY_TASKS if phase == "primary" else SECONDARY_TASKS)
    for arm in ALL_ARMS:
        for task in tasks:
            a = alphas["alphas"][arm][task]["alpha"]
            for seed in SEEDS:
                jobs.append((arm, seed, task, a))
    return jobs


def run_jobs(jobs, label="run"):
    done = _load_done()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "1"))
    print(f"{label} jobs={len(jobs)} skip={len(done)} workers={workers} fc2={_HAS_FC2}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = _job(j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"),
                          round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "tune":
        tune_alphas(); return
    alphas = tune_alphas()
    phase = os.environ.get("PHASE", "all")
    if phase == "primary":
        run_jobs(build_jobs(alphas, "primary"), "primary")
        (OUT / "primary.done").write_text("ok\n"); return
    if phase == "secondary":
        run_jobs(build_jobs(alphas, "secondary"), "secondary")
        (OUT / "secondary.done").write_text("ok\n"); return
    run_jobs(build_jobs(alphas, "primary"), "primary")
    (OUT / "primary.done").write_text("ok\n")
    run_jobs(build_jobs(alphas, "secondary"), "secondary")
    (OUT / "secondary.done").write_text("ok\n")
    (OUT / "team_franken.done").write_text("ok\n")
    print("team_franken done", flush=True)


if __name__ == "__main__":
    main()

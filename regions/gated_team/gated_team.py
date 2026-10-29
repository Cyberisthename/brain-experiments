"""GATED TEAM: Design 1+8 experts + neuromod + species leaks. Prereg: PREREG_gated_team.md."""
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

SEEDS = list(range(700, 724))
TUNE_SEEDS = [90, 91, 92]
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
GAMMA_GRID = [0.0, 0.5, 1.0, 2.0]
LEAK_FLY_GRID = [0.25, 0.35, 0.5]
LEAK_HUM_GRID = [0.08, 0.12, 0.2]
LEAK_SINGLE_GRID = [0.12, 0.2, 0.35]
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
XOR_D = 5
BRIDGE_SIGMA = 0.0

GATED_ARMS = ["GATED_REAL", "GATED_REAL_noLeak", "GATED_ER", "GATED_EI"]
ALONE_ARMS = ["FLY_CX", "HUMAN", "ER4962", "TEAM"]
ALL_ARMS = GATED_ARMS + ALONE_ARMS
PRIMARY_TASKS = ["XOR_DELAY", "NARMA"]
SECONDARY_TASKS = ["MC", "PC_HARD"]
ALL_TASKS = PRIMARY_TASKS + SECONDARY_TASKS

# task cue (m_fly, m_human)
TASK_CUE = {
    "XOR_DELAY": (1.0, 0.0),
    "NARMA": (0.0, 1.0),
    "MC": (0.5, 0.5),
    "PC_HARD": (0.5, 0.5),
}

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


def fly_range(info):
    if "fly_cx" not in info["names"]:
        return None
    i = info["names"].index("fly_cx")
    return int(info["offsets"][i]), int(info["offsets"][i + 1])


def delays_for(W, info, seed, mf, shuffle_myelin=False):
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
            if Wd is None:
                continue
            src = (self.t - (di + 1)) % (self.dmax + 1)
            h += Wd @ self.buf[src]
        y = np.tanh(h + x_in)
        self.buf[slot] = y
        return y


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak):
    """leak: scalar or length-N vector."""
    dr = DelayReservoirFast(W, delay, dmax)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W.shape[0]
    leak_arr = np.asarray(leak, dtype=np.float64)
    if leak_arr.ndim == 0:
        leak_arr = np.full(n, float(leak_arr))
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        y = dr.step(drive)
        x = (1.0 - leak_arr) * x + leak_arr * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x
    return X


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


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
        a, b_ = off[i], off[i + 1]
        W[a:b_, a:b_] = Wi
    W = F.scale(W.tocsr(), RHO)
    drive = []
    for i, layer in enumerate(layers):
        L4, _, _ = F.layer_sets(layer, sizes[i])
        drive.append(L4 + off[i])
    drive = np.unique(np.concatenate(drive)) if drive else np.arange(min(200, N))
    fly_off = int(off[names.index("fly_cx")]) if "fly_cx" in names else 0
    info = dict(names=names, sizes=sizes, offsets=[int(x) for x in off], N=N,
                drive=drive, bridge={}, fly_offset=fly_off, bridge_sigma=BRIDGE_SIGMA)
    return W, info


def scale_module_gains(W, Win, info, g_fly, g_human):
    """Multiply fly/human recurrent blocks and input rows by neuromod gains."""
    W = W.tocsr().copy()
    Win = np.asarray(Win, dtype=np.float64).copy()
    fr = fly_range(info)
    hr = human_range(info)
    if fr is not None:
        a, b_ = fr
        # scale edges with pre-synaptic in fly block (both ends in block already for zero-bridge)
        W = W.tolil()
        W[a:b_, a:b_] = W[a:b_, a:b_] * float(g_fly)
        W = W.tocsr()
        Win[a:b_] *= float(g_fly)
    if hr is not None:
        a, b_ = hr
        W = W.tolil()
        W[a:b_, a:b_] = W[a:b_, a:b_] * float(g_human)
        W = W.tocsr()
        Win[a:b_] *= float(g_human)
    return W, Win


def leak_vector(info, leak_fly, leak_human, leak_default=0.2):
    n = int(info["N"])
    leak = np.full(n, float(leak_default))
    fr = fly_range(info)
    hr = human_range(info)
    if fr is not None:
        a, b_ = fr
        leak[a:b_] = float(leak_fly)
    if hr is not None:
        a, b_ = hr
        leak[a:b_] = float(leak_human)
    return leak


def build_wiring(arm, seed):
    g = np.random.default_rng(seed + 7)
    shuffle_myelin = False
    gated = arm in GATED_ARMS

    if arm == "FLY_CX":
        W, info = F.build_arm("FLY_CX", seed, g)
        return W.tocsr(), info, shuffle_myelin, False

    if arm == "HUMAN":
        Wm, sign, layer, ct = F.make_block("human", "real", seed)
        W = F.scale(Wm, RHO)
        L4, _, _ = F.layer_sets(layer, Wm.shape[0])
        info = dict(names=["human"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                    N=Wm.shape[0], drive=L4, bridge={}, fly_offset=0, kinds={"human": "real"})
        return W.tocsr(), info, shuffle_myelin, False

    if arm == "TEAM":
        Wf, sf, lf, _ = F.make_block("fly_cx", "real", seed)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "real", "human": "real"}
        return W.tocsr(), info, shuffle_myelin, False

    if arm == "ER4962":
        Wf, sf, lf, _ = F.make_block("fly_cx", "real", seed)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W0, info0 = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        sign = np.concatenate([sf, sh])
        g2 = np.random.default_rng(seed + 99)
        Wer = F.scale(F.er_match(W0, sign, g2), RHO)
        # monolithic: rename so myelin human-range does NOT apply; FIXED delays
        N = Wer.shape[0]
        n_drive = len(info0["drive"])
        drive = g2.choice(N, size=min(n_drive, N), replace=False)
        info = dict(names=["er"], sizes=[N], offsets=[0, N], N=N, drive=drive,
                    bridge={}, fly_offset=0, kinds={"er": "er"}, bridge_sigma=0.0)
        return Wer.tocsr(), info, False, False

    if arm in ("GATED_REAL", "GATED_REAL_noLeak"):
        Wf, sf, lf, _ = F.make_block("fly_cx", "real", seed)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "real", "human": "real"}
        return W.tocsr(), info, shuffle_myelin, True

    if arm == "GATED_ER":
        Wf0, sf, lf, _ = F.load_mod("fly_cx")
        Wh0, sh, lh, _ = F.load_mod("human")
        g1 = np.random.default_rng(seed + 13 + hash("fly_cx") % 1000)
        g2 = np.random.default_rng(seed + 13 + hash("human") % 1000)
        Wf = F.scale(F.er_match(Wf0, sf, g1), 1.0)
        Wh = F.scale(F.er_match(Wh0, sh, g2), 1.0)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "er", "human": "er"}
        info["names"] = ["fly_cx", "human"]
        return W.tocsr(), info, True, True

    if arm == "GATED_EI":
        Wf0, sf, lf, _ = F.load_mod("fly_cx")
        g1 = np.random.default_rng(seed + 13 + hash("fly_cx") % 1000)
        Wf = F.scale(ei_class_shuffle(Wf0, sf, g1), 1.0)
        Wh, sh, lh, _ = F.make_block("human", "real", seed)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed, g)
        info["kinds"] = {"fly_cx": "ei_shuf", "human": "real"}
        return W.tocsr(), info, shuffle_myelin, True

    raise KeyError(arm)


def _slices(info):
    fr = fly_range(info)
    hr = human_range(info)
    if fr is None or hr is None:
        return None, None
    return slice(fr[0], fr[1]), slice(hr[0], hr[1])


def fit_gate_pred(X, y, fly_sl, hum_sl, alpha, wash, tr, te_start):
    """Train experts + convex λ on [wash:wash+tr]; predict on te_start:."""
    Xtr = X[wash:wash + tr]
    ytr = y[wash:wash + tr]
    Xte = X[te_start:]
    Xf_tr, Xh_tr = Xtr[:, fly_sl], Xtr[:, hum_sl]
    Bf = K.ridge(Xf_tr, ytr, alpha)
    Bh = K.ridge(Xh_tr, ytr, alpha)
    pf_tr = K.pred(Xf_tr, Bf)
    ph_tr = K.pred(Xh_tr, Bh)
    d = pf_tr - ph_tr
    resid = ytr - ph_tr
    if d.ndim == 1:
        den = float(np.dot(d, d)) + 1e-12
        num = float(np.dot(resid, d))
    else:
        den = float(np.sum(d * d)) + 1e-12
        num = float(np.sum(resid * d))
    lam = float(np.clip(num / den, 0.0, 1.0))
    pf_te = K.pred(Xte[:, fly_sl], Bf)
    ph_te = K.pred(Xte[:, hum_sl], Bh)
    pred = lam * pf_te + (1.0 - lam) * ph_te
    return pred, lam


def score_from_pred(task, pred, yt):
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(yt)):
        return float("nan"), None
    if task == "MC":
        D = yt.shape[1]
        return float(sum(np.nan_to_num(np.corrcoef(pred[:, k], yt[:, k])[0, 1]) ** 2 for k in range(D))), None
    if task == "PC_HARD":
        return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12)), None
    if np.std(yt) < 1e-12:
        return float("nan"), None
    nrmse = float(np.sqrt(np.mean((pred - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse):
        return float("nan"), None
    if task == "XOR_DELAY":
        return float(1.0 - nrmse), None
    return nrmse, None  # NARMA


def _make_U_y(task, rng, n, in_idx):
    if task == "XOR_DELAY":
        T = WASH + TR + TE
        a = rng.integers(0, 2, T).astype(np.float64)
        b = rng.integers(0, 2, T).astype(np.float64)
        y = np.zeros(T)
        for t in range(XOR_D, T):
            y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
        U = np.stack([2 * a - 1, 2 * b - 1], 1)
        Win, bias = masked_win(n, 2, in_idx, rng)
        return U, y, Win, bias, WASH, TR, WASH + TR
    if task == "NARMA":
        T = WASH + TR + TE
        u = rng.uniform(0, 0.5, T)
        y = np.zeros(T)
        for i in range(10, T - 1):
            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
        Win, bias = masked_win(n, 1, in_idx, rng)
        return u, y, Win, bias, WASH, TR, WASH + TR
    if task == "MC":
        T = WASH + TR + TE
        u = rng.uniform(-1, 1, T)
        Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
        Win, bias = masked_win(n, 1, in_idx, rng)
        return u, Y, Win, bias, WASH, TR, WASH + TR
    if task == "PC_HARD":
        T = 1200
        theta = np.linspace(0, 8 * np.pi, T)
        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
        idx = np.asarray(in_idx)
        rng2 = np.random.default_rng(int(rng.integers(1e9)))
        k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
        Win, bias = masked_win(n, 2, drive_idx, rng)
        u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
        wash, tr = 200, 800
        return u_drive, u_clean, Win, bias, wash, tr - wash, tr
    raise KeyError(task)


TASK_SEED_OFF = {"MC": 0, "NARMA": 2, "PC_HARD": 4, "XOR_DELAY": 6}


def run_one(arm, seed, task, alpha, hyper):
    """hyper: dict with gamma_fly, gamma_human, leak_fly, leak_human, leak_single."""
    t0 = time.time()
    W, info, shuf_m, gated = build_wiring(arm, seed)
    mf, med = mf_filled()
    delay, dmax = delays_for(W, info, seed, mf, shuffle_myelin=shuf_m)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    U, y, Win, bias, wash, tr, te_start = _make_U_y(task, rng, W.shape[0], info["drive"])

    if gated:
        m_fly, m_hum = TASK_CUE[task]
        gf = 1.0 + float(hyper["gamma_fly"]) * m_fly
        gh = 1.0 + float(hyper["gamma_human"]) * m_hum
        W, Win = scale_module_gains(W, Win, info, gf, gh)
        if arm == "GATED_REAL_noLeak":
            leak = leak_vector(info, 0.2, 0.2, 0.2)
            lf, lh = 0.2, 0.2
        else:
            lf = float(hyper["leak_fly"]); lh = float(hyper["leak_human"])
            leak = leak_vector(info, lf, lh, 0.2)
    else:
        lf = lh = float(hyper.get("leak_single", 0.2))
        leak = lf
        gf = gh = 1.0

    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    if not np.all(np.isfinite(X)):
        return dict(arm=arm, seed=int(seed), task=task, value=float("nan"),
                    n=int(W.shape[0]), alpha=float(alpha), sec=time.time() - t0,
                    error="nonfinite_X", source="gated_team")

    fly_sl, hum_sl = _slices(info)
    lam = None
    if gated and fly_sl is not None:
        if task == "PC_HARD":
            pred, lam = fit_gate_pred(X, y, fly_sl, hum_sl, alpha, wash, tr, te_start)
            yt = y[te_start:]
        else:
            pred, lam = fit_gate_pred(X, y, fly_sl, hum_sl, alpha, wash, tr, te_start)
            yt = y[te_start:]
        val, _ = score_from_pred(task, pred, yt)
    else:
        # single ridge
        if task == "PC_HARD":
            B = K.ridge(X[wash:te_start], y[wash:te_start], alpha)
            pred = K.pred(X[te_start:], B)
            yt = y[te_start:]
        else:
            B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], alpha)
            pred = K.pred(X[wash + tr:], B)
            yt = y[wash + tr:]
        val, _ = score_from_pred(task, pred, yt)

    return dict(
        arm=arm, seed=int(seed), task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        gamma_fly=float(hyper.get("gamma_fly", 0)), gamma_human=float(hyper.get("gamma_human", 0)),
        leak_fly=float(lf), leak_human=float(lh),
        g_eff_fly=float(gf), g_eff_human=float(gh),
        lam=None if lam is None else float(lam),
        delay_mean=float(np.mean(delay)), myelin_median=med,
        shuffle_myelin=bool(shuf_m), gated=bool(gated),
        sec=time.time() - t0, has_fastcore2=_HAS_FC2, source="gated_team",
    )


def _job(args):
    try:
        arm, seed, task, alpha, hyper = args
        return run_one(arm, seed, task, alpha, hyper)
    except Exception as e:
        import traceback
        arm, seed, task, alpha, hyper = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="gated_team")


def _score_states(arm, seed, task, alpha, hyper):
    """Used in tune: same as run_one value."""
    r = run_one(arm, seed, task, alpha, hyper)
    return r.get("value", float("nan")), r


def _default_hyper(arm, choice=None):
    h = dict(gamma_fly=1.0, gamma_human=1.0, leak_fly=0.35, leak_human=0.12, leak_single=0.2)
    if choice and arm in choice.get("hypers", {}):
        h.update(choice["hypers"][arm])
    if arm == "GATED_REAL_noLeak":
        h["leak_fly"] = 0.2
        h["leak_human"] = 0.2
    return h


def tune_all():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"TUNE arms={ALL_ARMS} tasks={ALL_TASKS} seeds={TUNE_SEEDS} fc2={_HAS_FC2}", flush=True)
    choice = {
        "alphas": {}, "hypers": {}, "grid_alpha": ALPHA_GRID,
        "grid_gamma": GAMMA_GRID, "grid_leak_fly": LEAK_FLY_GRID, "grid_leak_hum": LEAK_HUM_GRID,
        "tune_seeds": TUNE_SEEDS, "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_gated_team.md",
        "composite_z": {}, "tune_scores": {},
    }

    def mean_score(arm, task, alpha, hyper):
        jobs = [(arm, seed, task, alpha, hyper) for seed in TUNE_SEEDS]
        workers = int(os.environ.get("WORKERS", "2"))
        if workers > 1 and len(jobs) > 1:
            with Pool(workers) as pool:
                rows = list(pool.map(_job, jobs))
        else:
            rows = [_job(j) for j in jobs]
        vals = [r.get("value", float("nan")) for r in rows]
        return float(np.nanmean(vals)), vals

    # --- 1) Alone / TEAM / ER4962: tune leak_single then alpha per task ---
    for arm in ALONE_ARMS:
        print(f"== tune leaks/alpha {arm}", flush=True)
        # pick leak on XOR+NARMA composite proxy with alpha=1e-6
        best_leak, best_key = 0.2, -1e99
        for leak in LEAK_SINGLE_GRID:
            h = dict(gamma_fly=0, gamma_human=0, leak_fly=leak, leak_human=leak, leak_single=leak)
            mx, _ = mean_score(arm, "XOR_DELAY", 1e-6, h)
            mn, _ = mean_score(arm, "NARMA", 1e-6, h)
            # higher XOR better, lower NARMA better → proxy
            key = (mx if np.isfinite(mx) else -1) - (mn if np.isfinite(mn) else 1)
            print(f"  leak={leak} xor={mx:.4f} narma={mn:.4f} key={key:.4f}", flush=True)
            if key > best_key:
                best_key, best_leak = key, leak
        choice["hypers"][arm] = dict(gamma_fly=0.0, gamma_human=0.0,
                                     leak_fly=best_leak, leak_human=best_leak, leak_single=best_leak)
        choice["alphas"][arm] = {}
        choice["tune_scores"][arm] = {}
        h = choice["hypers"][arm]
        for task in ALL_TASKS:
            scores = {}
            for a in ALPHA_GRID:
                m, vals = mean_score(arm, task, a, h)
                scores[a] = (m, vals)
                print(f"  alpha {arm} {task} a={a} mean={m:.4f}", flush=True)
            if task == "NARMA":
                best = min(ALPHA_GRID, key=lambda a: (scores[a][0] if np.isfinite(scores[a][0]) else 1e99, a))
            else:
                best = max(ALPHA_GRID, key=lambda a: (scores[a][0] if np.isfinite(scores[a][0]) else -1e99, -a))
            choice["alphas"][arm][task] = {"alpha": best, "means": {str(a): scores[a][0] for a in ALPHA_GRID}}
            choice["tune_scores"][arm][task] = scores[best][1]
            print(f"LOCK alpha {arm} {task} -> {best}", flush=True)
            cache.write_text(json.dumps(choice, indent=2))

    # --- 2) Gated arms: leaks → gamma → alpha ---
    for arm in GATED_ARMS:
        print(f"== tune gated {arm}", flush=True)
        if arm == "GATED_REAL_noLeak":
            best_lf, best_lh = 0.2, 0.2
        else:
            best_lf, best_lh, best_key = 0.35, 0.12, -1e99
            for lf in LEAK_FLY_GRID:
                for lh in LEAK_HUM_GRID:
                    h = dict(gamma_fly=1.0, gamma_human=1.0, leak_fly=lf, leak_human=lh, leak_single=0.2)
                    mx, _ = mean_score(arm, "XOR_DELAY", 1e-6, h)
                    mn, _ = mean_score(arm, "NARMA", 1e-6, h)
                    key = (mx if np.isfinite(mx) else -1) - (mn if np.isfinite(mn) else 1)
                    print(f"  leaks lf={lf} lh={lh} xor={mx:.4f} narma={mn:.4f} key={key:.4f}", flush=True)
                    if key > best_key:
                        best_key, best_lf, best_lh = key, lf, lh
        # gamma search
        best_gf, best_gh, best_key = 1.0, 1.0, -1e99
        for gf in GAMMA_GRID:
            for gh in GAMMA_GRID:
                h = dict(gamma_fly=gf, gamma_human=gh, leak_fly=best_lf, leak_human=best_lh, leak_single=0.2)
                mx, _ = mean_score(arm, "XOR_DELAY", 1e-6, h)
                mn, _ = mean_score(arm, "NARMA", 1e-6, h)
                key = (mx if np.isfinite(mx) else -1) - (mn if np.isfinite(mn) else 1)
                print(f"  gamma gf={gf} gh={gh} xor={mx:.4f} narma={mn:.4f} key={key:.4f}", flush=True)
                if key > best_key:
                    best_key, best_gf, best_gh = key, gf, gh
        choice["hypers"][arm] = dict(gamma_fly=best_gf, gamma_human=best_gh,
                                     leak_fly=best_lf, leak_human=best_lh, leak_single=0.2)
        choice["alphas"][arm] = {}
        choice["tune_scores"][arm] = {}
        h = choice["hypers"][arm]
        for task in ALL_TASKS:
            scores = {}
            for a in ALPHA_GRID:
                m, vals = mean_score(arm, task, a, h)
                scores[a] = (m, vals)
                print(f"  alpha {arm} {task} a={a} mean={m:.4f}", flush=True)
            if task == "NARMA":
                best = min(ALPHA_GRID, key=lambda a: (scores[a][0] if np.isfinite(scores[a][0]) else 1e99, a))
            else:
                best = max(ALPHA_GRID, key=lambda a: (scores[a][0] if np.isfinite(scores[a][0]) else -1e99, -a))
            choice["alphas"][arm][task] = {"alpha": best, "means": {str(a): scores[a][0] for a in ALPHA_GRID}}
            choice["tune_scores"][arm][task] = scores[best][1]
            print(f"LOCK alpha {arm} {task} -> {best}", flush=True)
            cache.write_text(json.dumps(choice, indent=2))

    # composite z from all tune scores at locked alpha
    xor_vals, narma_vals = [], []
    for arm in ALL_ARMS:
        xor_vals.extend([v for v in choice["tune_scores"][arm].get("XOR_DELAY", []) if np.isfinite(v)])
        narma_vals.extend([v for v in choice["tune_scores"][arm].get("NARMA", []) if np.isfinite(v)])
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
    print("hypers", json.dumps(choice["hypers"], indent=2), flush=True)
    return choice


def _load_done():
    done = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                done.add((r["arm"], r["seed"], r["task"]))
            except Exception:
                pass
    return done


def smoke():
    out = []
    h = dict(gamma_fly=1.0, gamma_human=1.0, leak_fly=0.35, leak_human=0.12, leak_single=0.2)
    for arm in ("GATED_REAL", "FLY_CX", "HUMAN", "TEAM", "ER4962"):
        for task in ("XOR_DELAY", "NARMA"):
            r = run_one(arm, 700, task, 1e-6, h)
            out.append(r)
            print("SMOKE", {k: r.get(k) for k in
                            ("arm", "task", "value", "n", "lam", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")


def build_jobs(choice, phase="all"):
    jobs = []
    tasks = ALL_TASKS if phase == "all" else (PRIMARY_TASKS if phase == "primary" else SECONDARY_TASKS)
    for arm in ALL_ARMS:
        h = _default_hyper(arm, choice)
        for task in tasks:
            a = choice["alphas"][arm][task]["alpha"]
            for seed in SEEDS:
                jobs.append((arm, seed, task, a, h))
    return jobs


def run_jobs(jobs, label="run"):
    done = _load_done()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"{label} jobs={len(jobs)} skip={len(done)} workers={workers} fc2={_HAS_FC2}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = _job(j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), "lam=", r.get("lam"), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"),
                          round(r.get("sec", 0) or 0, 1), "lam=", r.get("lam"), r.get("error", ""), flush=True)


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "tune":
        tune_all(); return
    choice = tune_all()
    phase = os.environ.get("PHASE", "all")
    if phase == "primary":
        run_jobs(build_jobs(choice, "primary"), "primary")
        (OUT / "primary.done").write_text("ok\n"); return
    if phase == "secondary":
        run_jobs(build_jobs(choice, "secondary"), "secondary")
        (OUT / "secondary.done").write_text("ok\n"); return
    run_jobs(build_jobs(choice, "primary"), "primary")
    (OUT / "primary.done").write_text("ok\n")
    run_jobs(build_jobs(choice, "secondary"), "secondary")
    (OUT / "secondary.done").write_text("ok\n")
    (OUT / "gated_team.done").write_text("ok\n")
    print("gated_team done", flush=True)


if __name__ == "__main__":
    main()

"""MACAQUE FLN reservoir + FRANKEN graft. Prereg: PREREG_macaque.md."""
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
sys.path.insert(0, str(_HERE))
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import frankenstein as F
import fast_kernels as K
from build_macaque_module import expand, _row_sum_shuffle, TEMPLATE_SEED, DMAX as MAC_DMAX, FLN_PATH

try:
    import fastcore2  # noqa: F401
    _HAS_FC2 = True
except Exception:
    _HAS_FC2 = False

H = _HERE
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
MOD = DATA / "macaque_module.npz"
META_PATH = DATA / "macaque_module_meta.json"

SEEDS = list(range(600, 624))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 8
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
XOR_D = 5

ALL_ARMS = [
    "MACAQUE", "MACAQUE_ER", "MACAQUE_SHUF", "DENS",
    "FLY_CX", "FRANKEN", "FRANKEN+MACAQUE", "FRANKEN+ER_addon",
]
PRIMARY_TASKS = ["XOR_DELAY", "NARMA"]
SECONDARY_TASKS = ["MC", "PC_HARD"]
ALL_TASKS = PRIMARY_TASKS + SECONDARY_TASKS
TASK_SEED_OFF = {"MC": 0, "NARMA": 2, "PC_HARD": 4, "XOR_DELAY": 6}

_CACHE = {}


def load_macaque_npz(path=MOD):
    z = np.load(path, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    delay_map = {(int(r), int(c)): int(d) for r, c, d in zip(z["edge_row"], z["edge_col"], z["edge_delay"])}
    return W, z["sign"].astype(np.int8), z["layer"].astype(str), z["cell_type"].astype(str), delay_map, int(z["dmax"][0])


def delays_from_map(W, delay_map, seed, fixed=False, dmax=DMAX):
    Wc = W.tocoo()
    g = np.random.default_rng(seed + 99)
    if fixed:
        return g.integers(1, dmax + 1, Wc.nnz).astype(np.int32), dmax
    out = np.array([delay_map.get((int(r), int(c)), int(g.integers(1, dmax + 1)))
                    for r, c in zip(Wc.row, Wc.col)], dtype=np.int32)
    return out, dmax


def delays_fixed(W, seed, dmax=DMAX):
    g = np.random.default_rng(seed + 99)
    return g.integers(1, dmax + 1, W.nnz).astype(np.int32), dmax


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


def get_macaque_real():
    if "mac_real" not in _CACHE:
        W, sign, layer, ct, dmap, dmax = load_macaque_npz()
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        _CACHE["mac_real"] = dict(W=W, sign=sign, layer=layer, ct=ct, dmap=dmap, dmax=dmax, meta=meta)
    return _CACHE["mac_real"]


def build_shuf_macaque(seed):
    """Area-FLN row-sum shuffle, re-expand (prereg)."""
    key = f"shuf_{seed}"
    if key in _CACHE:
        return _CACHE[key]
    z = np.load(FLN_PATH, allow_pickle=True)
    FLN = z["FLN"].astype(np.float64).copy()
    DIST = z["DIST_mm"].astype(np.float64)
    areas = z["areas"]
    np.fill_diagonal(FLN, 0.0)
    g = np.random.default_rng(seed + 10007)
    FLNs = _row_sum_shuffle(FLN, g)
    W, sign, layer, ct, area_of, ed, row, col, meta = expand(
        FLNs, DIST, areas, seed=TEMPLATE_SEED + 1000 + (seed % 97), tag="shuf")
    dmap = {(int(r), int(c)): int(d) for r, c, d in zip(row, col, ed)}
    out = dict(W=W, sign=sign, layer=layer, ct=ct, dmap=dmap, dmax=MAC_DMAX, meta=meta)
    _CACHE[key] = out
    return out


def scale(W, rho=RHO):
    r = float(K.radius_cached(W))
    return W * (rho / (r + 1e-12))


def _drive_from_layer(layer, N):
    L4 = np.where(layer == "L4")[0]
    if len(L4) == 0:
        L4 = np.arange(min(200, N))
    return L4


def graft_with_macaque(mac_W, mac_layer, seed, mac_name="macaque"):
    """FRANKEN blocks + macaque addon via frankenstein.graft_blocks."""
    g = np.random.default_rng(seed + 7)
    blocks = []
    for name in ("mouse", "human", "fly_cx"):
        Wm, sign, layer, ct = F.make_block(name, "real", seed)
        blocks.append((name, Wm, layer))
    # macaque already at ρ≈raw; scale to ρ=1 like make_block
    Wm = scale(mac_W, 1.0)
    blocks.append((mac_name, Wm, mac_layer))
    return F.graft_blocks(blocks, seed, g)


class DelayReservoirFast:
    def __init__(self, W, delay, dmax):
        W = W.tocoo()
        self.n = W.shape[0]; self.dmax = int(dmax)
        self.row = W.row.copy(); self.col = W.col.copy()
        self.data = W.data.astype(np.float64).copy()
        self.delay = delay.astype(np.int32).copy()
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = self.delay == d
            if m.any():
                self.buckets.append(sp.csr_matrix((self.data[m], (self.row[m], self.col[m])), shape=(self.n, self.n)))
            else:
                self.buckets.append(None)
        self.buf = np.zeros((self.dmax + 1, self.n), dtype=np.float64)
        self.t = 0

    def step(self, x_in):
        self.t += 1
        slot = self.t % (self.dmax + 1)
        h = np.zeros(self.n, dtype=np.float64)
        for di, Wd in enumerate(self.buckets):
            if Wd is None: continue
            h += Wd @ self.buf[(self.t - (di + 1)) % (self.dmax + 1)]
        y = np.tanh(h + x_in)
        self.buf[slot] = y
        return y


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak):
    dr = DelayReservoirFast(W, delay, dmax)
    U = np.asarray(U, float); Win = np.asarray(Win, float); b = np.asarray(b, float)
    n = W.shape[0]; X = np.empty((len(U), n), float); x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        y = dr.step(drive)
        x = (1.0 - leak) * x + leak * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x
    return X


def masked_win(n, n_in, drive_idx, rng):
    Win = np.zeros((n, n_in))
    idx = np.asarray(drive_idx)
    if len(idx) == 0:
        idx = np.arange(min(200, n))
    Win[idx] = rng.uniform(-0.5, 0.5, (len(idx), n_in))
    b = rng.uniform(-0.1, 0.1, n)
    return Win, b


def build_wiring(arm, seed):
    """Returns W (ρ=0.9), info, delay_mode."""
    if arm == "MACAQUE":
        m = get_macaque_real()
        W = scale(m["W"], RHO)
        drive = _drive_from_layer(m["layer"], W.shape[0])
        return W, dict(drive=drive, dmap=m["dmap"], dmax=m["dmax"], meta=m["meta"], N=W.shape[0]), "dist"
    if arm == "MACAQUE_ER":
        m = get_macaque_real()
        g = np.random.default_rng(seed + 13)
        W = scale(er_match(m["W"], m["sign"], g), RHO)
        drive = _drive_from_layer(m["layer"], W.shape[0])
        return W, dict(drive=drive, dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    if arm == "MACAQUE_SHUF":
        m = build_shuf_macaque(seed)
        W = scale(m["W"], RHO)
        drive = _drive_from_layer(m["layer"], W.shape[0])
        return W, dict(drive=drive, dmap=m["dmap"], dmax=m["dmax"], N=W.shape[0]), "dist"
    if arm == "DENS":
        m = get_macaque_real()
        g = np.random.default_rng(seed + 17)
        W = scale(dens_match(m["W"], m["sign"], g), RHO)
        drive = _drive_from_layer(m["layer"], W.shape[0])
        return W, dict(drive=drive, dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    if arm == "FLY_CX":
        Wm, sign, layer, ct = F.make_block("fly_cx", "real", seed)
        W = scale(Wm, RHO)
        L4, _, _ = F.layer_sets(layer, Wm.shape[0])
        return W, dict(drive=L4, dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    if arm == "FRANKEN":
        W, info = F.build_arm("FRANKEN", seed)
        return W, dict(drive=info["drive"], dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    if arm == "FRANKEN+MACAQUE":
        m = get_macaque_real()
        W, info = graft_with_macaque(m["W"], m["layer"], seed, "macaque")
        return W, dict(drive=info["drive"], dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    if arm == "FRANKEN+ER_addon":
        m = get_macaque_real()
        g = np.random.default_rng(seed + 19)
        Wer = er_match(m["W"], m["sign"], g)
        W, info = graft_with_macaque(Wer, m["layer"], seed, "macaque_er")
        return W, dict(drive=info["drive"], dmap={}, dmax=DMAX, N=W.shape[0]), "fixed"
    raise KeyError(arm)


def make_delays(W, info, seed, mode):
    if mode == "dist":
        return delays_from_map(W, info.get("dmap", {}), seed, fixed=False, dmax=info.get("dmax", DMAX))
    return delays_fixed(W, seed, dmax=info.get("dmax", DMAX))


def score_xor(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    a = rng.integers(0, 2, T).astype(np.float64); b = rng.integers(0, 2, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(XOR_D, T):
        y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
    U = np.stack([2 * a - 1, 2 * b - 1], 1)
    Win, bias = masked_win(W.shape[0], 2, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B); yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12: return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan")


def score_narma(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B); yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12: return float("nan")
    return float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))


def score_mc(W, delay, dmax, leak, rng, in_idx, alpha, D=100):
    T = WASH + TR + TE
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], alpha)
    P = K.pred(X[WASH + TR:], B); Yt = Y[WASH + TR:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def score_pc_hard(W, delay, dmax, leak, rng, in_idx, alpha):
    n = W.shape[0]; T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = np.asarray(in_idx)
    rng2 = np.random.default_rng(int(rng.integers(1e9)))
    k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
    drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
    Win, b = masked_win(n, 2, drive_idx, rng)
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak)
    wash, tr = 200, 800
    B = K.ridge(X[wash:tr], u_clean[wash:tr], alpha)
    pred = K.pred(X[tr:], B); yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


TASK_FN = {"XOR_DELAY": score_xor, "NARMA": score_narma, "MC": score_mc, "PC_HARD": score_pc_hard}


def run_one(arm, seed, task, alpha):
    t0 = time.time()
    W, info, mode = build_wiring(arm, seed)
    delay, dmax = make_delays(W, info, seed, mode)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    val = TASK_FN[task](W, delay, dmax, LEAK, rng, info["drive"], alpha)
    return dict(
        arm=arm, seed=int(seed), task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha), delay_mode=mode,
        delay_mean=float(np.mean(delay)), sec=time.time() - t0,
        has_fastcore2=_HAS_FC2, source="macaque",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="macaque")


def _score_from_XY(task, X, y, alpha):
    if X is None or not np.all(np.isfinite(X)): return float("nan")
    if task == "MC":
        B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
        P = K.pred(X[WASH + TR:], B); Yt = y[WASH + TR:]
        return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(Yt.shape[1])))
    if task == "PC_HARD":
        wash, tr = 200, 800
        B = K.ridge(X[wash:tr], y[wash:tr], alpha)
        pred = K.pred(X[tr:], B); yt = y[tr:]
        return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B); yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12: return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse): return float("nan")
    return float(1.0 - nrmse) if task == "XOR_DELAY" else nrmse


def _make_states(arm, seed, task):
    W, info, mode = build_wiring(arm, seed)
    delay, dmax = make_delays(W, info, seed, mode)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    if task == "XOR_DELAY":
        T = WASH + TR + TE
        a = rng.integers(0, 2, T).astype(np.float64); b = rng.integers(0, 2, T).astype(np.float64)
        y = np.zeros(T)
        for t in range(XOR_D, T):
            y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
        U = np.stack([2 * a - 1, 2 * b - 1], 1)
        Win, bias = masked_win(W.shape[0], 2, info["drive"], rng)
        return run_delay_reservoir(W, delay, dmax, Win, bias, U, LEAK), y, W.shape[0]
    if task == "NARMA":
        T = WASH + TR + TE
        u = rng.uniform(0, 0.5, T); y = np.zeros(T)
        for i in range(10, T - 1):
            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
        Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
        return run_delay_reservoir(W, delay, dmax, Win, b, u, LEAK), y, W.shape[0]
    if task == "MC":
        T = WASH + TR + TE
        u = rng.uniform(-1, 1, T)
        Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
        X = run_delay_reservoir(W, delay, dmax, Win, b, u, LEAK)
        Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
        return X, Y, W.shape[0]
    if task == "PC_HARD":
        n = W.shape[0]; T = 1200
        theta = np.linspace(0, 8 * np.pi, T)
        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
        idx = np.asarray(info["drive"])
        rng2 = np.random.default_rng(int(rng.integers(1e9)))
        k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
        Win, b = masked_win(n, 2, drive_idx, rng)
        u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
        return run_delay_reservoir(W, delay, dmax, Win, b, u_drive, LEAK), u_clean, n
    raise KeyError(task)


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    m = get_macaque_real()
    (OUT / "design_meta.json").write_text(json.dumps(m["meta"], indent=2))
    print(f"TUNE arms={ALL_ARMS} mac_N={m['meta'].get('N')} nnz={m['meta'].get('nnz')} fc2={_HAS_FC2}", flush=True)
    choice = {
        "alphas": {}, "grid": ALPHA_GRID, "tune_seeds": TUNE_SEEDS,
        "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_macaque.md",
        "design_meta": m["meta"], "composite_z": {}, "tune_scores": {},
    }
    xor_vals, narma_vals = [], []
    for arm in ALL_ARMS:
        choice["alphas"][arm] = {}; choice["tune_scores"][arm] = {}
        for task in ALL_TASKS:
            scores = {a: [] for a in ALPHA_GRID}
            for seed in TUNE_SEEDS:
                t0 = time.time()
                X, y, n = _make_states(arm, seed, task)
                for alpha in ALPHA_GRID:
                    scores[alpha].append(_score_from_XY(task, X, y, alpha))
                print(f"tune {arm} {task} seed={seed} n={n} {time.time()-t0:.1f}s", flush=True)
            means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
            if task == "NARMA":
                best = min(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else 1e99, a))
            else:
                best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
            choice["alphas"][arm][task] = {"alpha": best, "means": means}
            locked = [scores[best][i] for i in range(len(TUNE_SEEDS))]
            choice["tune_scores"][arm][task] = locked
            if task == "XOR_DELAY":
                xor_vals.extend([v for v in locked if np.isfinite(v)])
            if task == "NARMA":
                narma_vals.extend([v for v in locked if np.isfinite(v)])
            print(f"LOCK alpha {arm} {task} -> {best}", flush=True)
            cache.write_text(json.dumps(choice, indent=2))
    xor_vals = np.asarray(xor_vals, float); narma_vals = np.asarray(narma_vals, float)
    choice["composite_z"] = dict(
        mu_xor=float(np.mean(xor_vals)) if len(xor_vals) else 0.0,
        sd_xor=float(np.std(xor_vals, ddof=1)) if len(xor_vals) > 1 else 1.0,
        mu_narma=float(np.mean(narma_vals)) if len(narma_vals) else 0.0,
        sd_narma=float(np.std(narma_vals, ddof=1)) if len(narma_vals) > 1 else 1.0,
        n_xor=int(len(xor_vals)), n_narma=int(len(narma_vals)),
        source="tune seeds × all arms, locked before eval",
    )
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
    m = get_macaque_real()
    (OUT / "design_meta.json").write_text(json.dumps(m["meta"], indent=2))
    print("MACAQUE meta", m["meta"], flush=True)
    out = []
    for arm in ("MACAQUE", "MACAQUE_ER", "FLY_CX", "FRANKEN+MACAQUE", "FRANKEN+ER_addon"):
        for task in ("XOR_DELAY", "NARMA"):
            r = run_one(arm, 600, task, 1e-6)
            out.append(r)
            print("SMOKE", {k: r.get(k) for k in ("arm", "task", "value", "n", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")


def build_jobs(alphas, phase="all"):
    tasks = ALL_TASKS if phase == "all" else (PRIMARY_TASKS if phase == "primary" else SECONDARY_TASKS)
    jobs = []
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
    print(f"{label} jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
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
    (OUT / "macaque.done").write_text("ok\n")
    print("macaque done", flush=True)


if __name__ == "__main__":
    main()

"""CHIMERA_DESIGN synthetic reservoir. Prereg: PREREG_designed_brain.md."""
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
import scipy.sparse.linalg as sla
from multiprocessing import Pool
import frankenstein as F
import fast_kernels as K

try:
    import fastcore2  # noqa: F401
    _HAS_FC2 = True
except Exception:
    _HAS_FC2 = False

H = _HERE
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
MYELIN = DATA / "human_myelin.npz"
STATS_PATH = OUT / "extracted_stats.json"

SEEDS = list(range(500, 524))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
XOR_D = 5
DESIGN_N = 4000
DESIGN_GRAPH_SEED = 7  # frozen template

ALL_ARMS = [
    "DESIGN", "DESIGN_EI_ONLY", "DESIGN_MYELIN_ONLY",
    "FLY_CX", "HUMAN", "TEAM_ER", "ER", "DENS", "SHUF",
]
PRIMARY_TASKS = ["XOR_DELAY", "NARMA"]
SECONDARY_TASKS = ["MC", "PC_HARD"]
ALL_TASKS = PRIMARY_TASKS + SECONDARY_TASKS
TASK_SEED_OFF = {"MC": 0, "NARMA": 2, "PC_HARD": 4, "XOR_DELAY": 6}

# ---------------------------------------------------------------------------
# Extract / load stats
# ---------------------------------------------------------------------------

def ensure_stats():
    if STATS_PATH.exists():
        return json.loads(STATS_PATH.read_text())
    raise FileNotFoundError("extracted_stats.json missing — run extract before eval")


def load_human_mf_by_layer():
    hz = np.load(DATA / "human_module.npz", allow_pickle=True)
    mfz = np.load(MYELIN, allow_pickle=True)
    mf = mfz["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf = mf.copy(); mf[~np.isfinite(mf)] = med
    layer = hz["layer"].astype(str)
    pools = {}
    for L in sorted(set(layer)):
        pools[L] = mf[layer == L].astype(np.float64)
    pools["__all__"] = mf
    pools["__median__"] = med
    return pools


# ---------------------------------------------------------------------------
# CHIMERA generator (frozen seed 7)
# ---------------------------------------------------------------------------
_DESIGN_CACHE = {}


def _spectral_frac(W, k=64):
    try:
        ev = np.abs(sla.eigs(W.astype(float), k=min(k, W.shape[0] - 2),
                             which="LM", return_eigenvectors=False, maxiter=4000))
        return float((ev > 0.5).mean())
    except Exception:
        return float("nan")


def _temper_spectral(W, target_max=0.40, max_iter=8):
    W = W.tocsr()
    W = F.scale(W, 1.0)
    for _ in range(max_iter):
        frac = _spectral_frac(W)
        if not np.isfinite(frac) or frac <= target_max:
            break
        Wc = W.tocoo()
        mag = np.abs(Wc.data)
        thr = np.median(mag)
        small = mag <= thr
        new = mag.copy(); new[small] *= 0.7
        W = sp.csr_matrix((new * np.sign(Wc.data), (Wc.row, Wc.col)), shape=W.shape)
        W = F.scale(W, 1.0)
    return W, _spectral_frac(W)


def generate_chimera(n=DESIGN_N, seed=DESIGN_GRAPH_SEED):
    """Return dict with W (ρ=1), sign, layer, mf, meta. No copied edges."""
    stats = ensure_stats()
    g = np.random.default_rng(seed)
    cx = stats["cx"]
    frac_E = float(cx["frac_E"])
    n_E = int(round(frac_E * n))
    n_I = n - n_E
    sign = np.ones(n, dtype=np.int8)
    sign[n_E:] = -1
    g.shuffle(sign)  # random positions, preserve counts

    # Layer mix: CX-like + thin L6 for myelin gradient (human depth)
    # CX props ~ L2/3 0.757, L4 0.139, L5 0.104 → carve 8% of L2/3 into L6
    props = {"L2/3": 0.70, "L4": 0.14, "L5": 0.10, "L6": 0.06}
    layers = np.array(["L2/3"] * n, dtype=object)
    order = g.permutation(n)
    cut = 0
    for L, p in props.items():
        k = int(round(p * n))
        layers[order[cut:cut + k]] = L
        cut += k

    # Build SBM from CX block densities + magnitude bootstrap from CX classes
    z = np.load(DATA / "fly_cx_module.npz", allow_pickle=True)
    Wcx = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    scx = z["sign"].astype(np.int8)
    Wcx_c = Wcx.tocoo()
    mag_pools = {}
    for a in (-1, 1):
        for b in (-1, 1):
            m = (scx[Wcx_c.row] == a) & (scx[Wcx_c.col] == b)
            mag_pools[(a, b)] = np.abs(Wcx_c.data[m]) if m.any() else np.array([0.003])

    dens = {}
    for key, blk in cx["ei_blocks"].items():
        a, b = key.split("->")
        dens[(int(a), int(b))] = float(blk["dens"])

    rows, cols, data = [], [], []
    idx_E = np.where(sign == 1)[0]
    idx_I = np.where(sign == -1)[0]
    class_idx = {1: idx_E, -1: idx_I}
    for a in (1, -1):
        for b in (1, -1):
            pre = class_idx[a]; post = class_idx[b]
            d = dens[(a, b)]
            # expected edges
            n_pos = len(pre) * len(post) - (len(pre) if a == b else 0)
            n_e = int(round(d * n_pos))
            if n_e <= 0:
                continue
            # sample pairs without replacement-ish
            if a == b:
                # sample then reject diagonals
                got = 0
                chunk = max(n_e * 2, n_e + 1000)
                pr, po = [], []
                while got < n_e:
                    rr = g.choice(pre, size=chunk, replace=True)
                    cc = g.choice(post, size=chunk, replace=True)
                    ok = rr != cc
                    rr, cc = rr[ok], cc[ok]
                    take = min(n_e - got, len(rr))
                    pr.append(rr[:take]); po.append(cc[:take])
                    got += take
                rr = np.concatenate(pr)[:n_e]; cc = np.concatenate(po)[:n_e]
            else:
                rr = g.choice(pre, size=n_e, replace=True)
                cc = g.choice(post, size=n_e, replace=True)
            mags = g.choice(mag_pools[(a, b)], size=n_e, replace=True)
            rows.append(rr); cols.append(cc); data.append(mags * float(a))  # out-sign = pre
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # drop exact duplicate accumulations ok; remove diagonal if any
    W.setdiag(0); W.eliminate_zeros()
    W = F.scale(W, 1.0)
    W, frac = _temper_spectral(W, target_max=0.40, max_iter=8)

    # Myelin frac by layer from human pools
    pools = load_human_mf_by_layer()
    mf = np.zeros(n, dtype=np.float64)
    for L in props:
        idx = np.where(layers == L)[0]
        pool = pools.get(L, pools["__all__"])
        mf[idx] = g.choice(pool, size=len(idx), replace=True)
    med = float(pools["__median__"])
    mf[~np.isfinite(mf)] = med
    mf = np.clip(mf, 0.0, 1.0)

    meta = dict(
        N=n, nnz=int(W.nnz), frac_E=float((sign > 0).mean()),
        spectral_frac=frac, dens= {f"{a}->{b}": dens[(a, b)] for a in (1, -1) for b in (1, -1)},
        layer_counts={L: int((layers == L).sum()) for L in props},
        mf_mean=float(mf.mean()), graph_seed=seed,
        plain_words=(
            f"A synthetic {n}-neuron directed network whose excitatory/inhibitory labels "
            f"and four E/I connection-class densities+weight magnitudes are matched to FlyWire CX "
            f"statistics (stochastic block model; no copied edges). Neurons get cortical-like "
            f"layers; each neuron's myelin fraction is drawn from the real H01 distribution for "
            f"that layer, then edge delays follow the same myelin→delay map used in the human "
            f"patch (more myelin → shorter delay). Weak spectral tempering keeps the bulk of "
            f"large eigenvalues from filling like an ER graph."
        ),
    )
    return dict(W=W.tocsr(), sign=sign, layer=np.asarray(layers, dtype=str), mf=mf, meta=meta)


def get_design(kind="full"):
    """kind in full|ei_only_graph — graph cached; myelin is always stored."""
    key = "full"
    if key not in _DESIGN_CACHE:
        _DESIGN_CACHE[key] = generate_chimera()
    return _DESIGN_CACHE[key]


# ---------------------------------------------------------------------------
# Delays / reservoir / tasks (aligned with team_franken)
# ---------------------------------------------------------------------------

def delays_myelin_rule(W, mf, seed):
    Wcoo = W.tocoo()
    pre = Wcoo.col  # source = pre (same convention as xor_rep REALMYELIN)
    d_neuron = (1 + np.rint((1.0 - np.clip(mf, 0, 1)) * (DMAX - 1))).astype(np.int32)
    delay = d_neuron[pre]
    return delay, DMAX


def delays_fixed(W, seed):
    g = np.random.default_rng(seed + 909)
    return g.integers(1, DMAX + 1, W.tocoo().nnz).astype(np.int32), DMAX


def delays_realmyelin_human(W, info, seed, mf):
    Wcoo = W.tocoo(); n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    g = np.random.default_rng(seed + 909)
    pre = Wcoo.col
    if "human" not in info["names"]:
        delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
        return delay, DMAX
    i = info["names"].index("human")
    h0, h1 = int(info["offsets"][i]), int(info["offsets"][i + 1])
    assert (h1 - h0) == len(mf)
    mf = np.clip(mf, 0.0, 1.0)
    d_neuron = (1 + np.rint((1.0 - mf) * (DMAX - 1))).astype(np.int32)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    other = ~mask
    delay[other] = g.integers(1, DMAX + 1, int(other.sum())).astype(np.int32)
    return delay, DMAX


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


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------
_MF_HUMAN = None


def human_mf():
    global _MF_HUMAN
    if _MF_HUMAN is None:
        z = np.load(MYELIN, allow_pickle=True)
        mf = z["myelin_frac"].astype(np.float64)
        med = float(np.nanmedian(mf)) if np.isfinite(np.nanmedian(mf)) else 0.0
        mf = mf.copy(); mf[~np.isfinite(mf)] = med
        _MF_HUMAN = (mf, med)
    return _MF_HUMAN


def graft_zero_bridge(blocks, seed):
    names = [b[0] for b in blocks]
    Ws = [b[1] for b in blocks]
    layers = [b[2] for b in blocks]
    sizes = [W.shape[0] for W in Ws]
    N = sum(sizes); off = np.cumsum([0] + sizes)
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
    return W, dict(names=names, sizes=sizes, offsets=[int(x) for x in off], N=N,
                   drive=drive, bridge={}, fly_offset=0)


def build_wiring(arm, seed):
    """Returns W (ρ=0.9), info, delay_mode in {'design_myelin','fixed','human_realmyelin','team_er_shuf'}."""
    g = np.random.default_rng(seed + 7)

    if arm in ("DESIGN", "DESIGN_EI_ONLY"):
        D = get_design()
        W = F.scale(D["W"], RHO)
        L4, _, _ = F.layer_sets(D["layer"], D["W"].shape[0])
        info = dict(names=["design"], sizes=[D["W"].shape[0]], offsets=[0, D["W"].shape[0]],
                    N=D["W"].shape[0], drive=L4, bridge={}, mf=D["mf"], sign=D["sign"],
                    layer=D["layer"], kinds={"design": arm})
        mode = "design_myelin" if arm == "DESIGN" else "fixed"
        return W.tocsr(), info, mode

    if arm == "DESIGN_MYELIN_ONLY":
        D = get_design()
        key = "er_twin"
        if key not in _DESIGN_CACHE:
            g2 = np.random.default_rng(DESIGN_GRAPH_SEED + 99)  # frozen ER twin
            Wer = F.scale(F.er_match(D["W"], D["sign"], g2), 1.0)
            _DESIGN_CACHE[key] = Wer
        Wer = _DESIGN_CACHE[key]
        W = F.scale(Wer, RHO)
        L4, _, _ = F.layer_sets(D["layer"], W.shape[0])
        info = dict(names=["design"], sizes=[W.shape[0]], offsets=[0, W.shape[0]],
                    N=W.shape[0], drive=L4, bridge={}, mf=D["mf"], sign=D["sign"],
                    layer=D["layer"], kinds={"design": "er_myelin"})
        return W.tocsr(), info, "design_myelin"

    if arm in ("ER", "DENS", "SHUF"):
        D = get_design()
        W0, sign = D["W"], D["sign"]
        g2 = np.random.default_rng(seed + 13)
        if arm == "ER":
            Wm = F.er_match(W0, sign, g2)
        elif arm == "DENS":
            Wm = F.dens_match(W0, sign, g2)
        else:
            Wm = F.degree_shuffle(W0, sign, g2)
        W = F.scale(F.scale(Wm, 1.0), RHO)
        L4, _, _ = F.layer_sets(D["layer"], W.shape[0])
        info = dict(names=["design"], sizes=[W.shape[0]], offsets=[0, W.shape[0]],
                    N=W.shape[0], drive=L4, bridge={}, kinds={"design": arm.lower()})
        return W.tocsr(), info, "fixed"

    if arm == "FLY_CX":
        W, info = F.build_arm("FLY_CX", seed, g)
        return W.tocsr(), info, "fixed"

    if arm == "HUMAN":
        Wm, sign, layer, ct = F.make_block("human", "real", seed)
        W = F.scale(Wm, RHO)
        L4, _, _ = F.layer_sets(layer, Wm.shape[0])
        info = dict(names=["human"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                    N=Wm.shape[0], drive=L4, bridge={}, kinds={"human": "real"})
        return W.tocsr(), info, "human_realmyelin"

    if arm == "TEAM_ER":
        Wf0, sf, lf, _ = F.load_mod("fly_cx")
        Wh0, sh, lh, _ = F.load_mod("human")
        g1 = np.random.default_rng(seed + 13 + hash("fly_cx") % 1000)
        g2 = np.random.default_rng(seed + 13 + hash("human") % 1000)
        Wf = F.scale(F.er_match(Wf0, sf, g1), 1.0)
        Wh = F.scale(F.er_match(Wh0, sh, g2), 1.0)
        W, info = graft_zero_bridge([("fly_cx", Wf, lf), ("human", Wh, lh)], seed)
        info["kinds"] = {"fly_cx": "er", "human": "er"}
        return W.tocsr(), info, "team_er_shuf"

    raise KeyError(arm)


def make_delays(W, info, seed, mode):
    if mode == "design_myelin":
        return delays_myelin_rule(W, info["mf"], seed)
    if mode == "fixed":
        return delays_fixed(W, seed)
    if mode == "human_realmyelin":
        mf, _ = human_mf()
        return delays_realmyelin_human(W, info, seed, mf)
    if mode == "team_er_shuf":
        mf, _ = human_mf()
        # shuffle mf then apply like team_franken TEAM_ER
        mf_use = mf.copy()
        np.random.default_rng(seed + 4242).shuffle(mf_use)
        return delays_realmyelin_human(W, info, seed, mf_use)
    raise KeyError(mode)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_xor(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    a = rng.integers(0, 2, T).astype(np.float64)
    b = rng.integers(0, 2, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(XOR_D, T):
        y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
    U = np.stack([2 * a - 1, 2 * b - 1], 1)
    Win, bias = masked_win(W.shape[0], 2, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    if not np.all(np.isfinite(X)): return float("nan")
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
    if not np.all(np.isfinite(y)): return float("nan")
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    if not np.all(np.isfinite(X)): return float("nan")
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B); yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12: return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


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
    wash, tr = 200, 800
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak)
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
        has_fastcore2=_HAS_FC2, source="designed_brain",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="designed_brain")


# ---------------------------------------------------------------------------
# Tune / run
# ---------------------------------------------------------------------------

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
    # freeze design meta
    D = get_design()
    (OUT / "design_meta.json").write_text(json.dumps(D["meta"], indent=2))
    print(f"TUNE arms={ALL_ARMS} design_N={D['meta']['N']} nnz={D['meta']['nnz']} "
          f"fracE={D['meta']['frac_E']:.3f} spec={D['meta']['spectral_frac']} fc2={_HAS_FC2}", flush=True)
    choice = {
        "alphas": {}, "grid": ALPHA_GRID, "tune_seeds": TUNE_SEEDS,
        "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_designed_brain.md",
        "design_meta": D["meta"], "composite_z": {}, "tune_scores": {},
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
    D = get_design()
    (OUT / "design_meta.json").write_text(json.dumps(D["meta"], indent=2))
    print("DESIGN meta", D["meta"], flush=True)
    out = []
    for arm in ("DESIGN", "DESIGN_EI_ONLY", "DESIGN_MYELIN_ONLY", "ER", "FLY_CX"):
        for task in ("XOR_DELAY", "NARMA"):
            r = run_one(arm, 500, task, 1e-6)
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
    (OUT / "designed_brain.done").write_text("ok\n")
    print("designed_brain done", flush=True)


if __name__ == "__main__":
    main()

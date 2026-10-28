"""Trained bridges on FRANKEN. Prereg: PREREGISTRATION_trained_bridges.md
Do NOT touch mega_franken.
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # regions/<exp> -> repo root
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

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"
JF = OUT / "runs.jsonl"
MYELIN = DATA / "human_myelin.npz"

SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
WASH, TR, TE = 200, 1500, 500
# short fitness rollouts (train only)
FW, FTR, FTE = 100, 800, 300
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
WIRINGS = ["FRANKEN", "ER", "SHUF"]
CONDS = ["FIXED", "ES"]
TASKS = ["MC", "NARMA", "PC_HARD", "XOR_DELAY"]
XOR_DELAY_D = 5
ES_G = 10          # generations
ES_SIGMA_FRAC = 0.05
ES_LR = 0.5
WIRING_TO_ARM = {"FRANKEN": "FRANKEN", "ER": "ER", "SHUF": "SHUF"}


def arm_key(wiring, cond):
    return f"{wiring}_{cond}"


ALL_ARMS = [arm_key(w, c) for w in WIRINGS for c in CONDS]


def parse_arm(arm):
    for w in WIRINGS:
        for c in CONDS:
            if arm == arm_key(w, c):
                return w, c
    raise KeyError(arm)


# ---------------------------------------------------------------------------
# Myelin / delays (franken_realmyelin)
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


def delays_realmyelin(W, info, seed, mf):
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
    mf = np.clip(mf, 0.0, 1.0)
    d_neuron = (1 + np.rint((1.0 - mf) * (DMAX - 1))).astype(np.int32)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    other = ~mask
    delay[other] = g.integers(1, DMAX + 1, int(other.sum())).astype(np.int32)
    return delay, DMAX


# ---------------------------------------------------------------------------
# Delay reservoir
# ---------------------------------------------------------------------------
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


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


def scale(W, rho=RHO):
    r = float(K.radius_cached(W))
    return W * (rho / (r + 1e-12)), float(r)


# ---------------------------------------------------------------------------
# Build wiring + bridge mask
# ---------------------------------------------------------------------------
def build_wiring(wiring, seed, g=None):
    if g is None:
        g = np.random.default_rng(seed + 7)
    arm = WIRING_TO_ARM[wiring]
    W, info = F.build_arm(arm, seed, g)
    return W.tocsr(), info


def bridge_edge_mask(W, info):
    """True for edges whose pre and post are in different blocks."""
    Wcoo = W.tocoo()
    row, col = Wcoo.row, Wcoo.col
    mask = np.ones(Wcoo.nnz, dtype=bool)
    offs = info["offsets"]
    for i in range(len(info["names"])):
        a, b = int(offs[i]), int(offs[i + 1])
        inside = (row >= a) & (row < b) & (col >= a) & (col < b)
        mask &= ~inside
    return mask, Wcoo


def freeze_blocks(W, info):
    """Return list of CSR block diagonals (frozen copies)."""
    blocks = []
    offs = info["offsets"]
    W = W.tocsr()
    for i in range(len(info["names"])):
        a, b = int(offs[i]), int(offs[i + 1])
        blocks.append(W[a:b, a:b].tocsr().copy())
    return blocks


def apply_theta(template_coo, bridge_idx, theta, frozen_blocks, info, shape):
    """Write θ into bridges, snap blocks, soft-scale bridges if ρ>0.9."""
    data = template_coo.data.astype(np.float64).copy()
    data[bridge_idx] = np.asarray(theta, dtype=np.float64)
    W = sp.csr_matrix(
        (data, (template_coo.row, template_coo.col)), shape=shape
    ).tolil()
    offs = info["offsets"]
    for i, B in enumerate(frozen_blocks):
        a, b = int(offs[i]), int(offs[i + 1])
        W[a:b, a:b] = B
    W = W.tocsr()
    r = float(K.radius_cached(W))
    if r > RHO and r > 1e-12:
        # scale bridge entries only
        coo = W.tocoo()
        br = bridge_edge_mask(W, info)[0]
        new_data = coo.data.copy()
        new_data[br] *= (RHO / r)
        W = sp.csr_matrix((new_data, (coo.row, coo.col)), shape=shape)
        # snap blocks again (bridge scale shouldn't touch them, but be safe)
        W = W.tolil()
        for i, B in enumerate(frozen_blocks):
            a, b = int(offs[i]), int(offs[i + 1])
            W[a:b, a:b] = B
        W = W.tocsr()
        r = float(K.radius_cached(W))
    return W, r


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def mc_score(W, delay, dmax, leak, rng, in_idx, alpha, wash, tr, te, D=100):
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
    P = K.pred(X[wash + tr:], B)
    Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_score(W, delay, dmax, leak, rng, in_idx, alpha, wash, tr, te):
    T = wash + tr + te
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
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], alpha)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def pc_hard_score(W, delay, dmax, leak, rng, in_idx, alpha):
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


def xor_delay_score(W, delay, dmax, leak, rng, in_idx, alpha, wash=WASH, tr=TR, te=TE, d=XOR_DELAY_D):
    """2-input delayed XOR. Target y_t = a_{t-d} XOR b_{t-d} in {+1,-1}. Metric: 1-NRMSE."""
    T = wash + tr + te
    a = rng.integers(0, 2, T).astype(np.float64)
    b = rng.integers(0, 2, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(d, T):
        y[t] = 1.0 if (int(a[t - d]) ^ int(b[t - d])) else -1.0
    U = np.stack([2 * a - 1, 2 * b - 1], 1)  # ±1 inputs
    Win, bias = masked_win(W.shape[0], 2, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], alpha)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse):
        return float("nan")
    return float(1.0 - nrmse)


def fitness_of(W, delay, dmax, leak, in_idx, alpha, gen, seed):
    """Short MC/20 - NARMA. Deterministic given (gen, seed)."""
    rng_mc = np.random.default_rng(seed + 1000 + gen * 17)
    rng_na = np.random.default_rng(seed + 2000 + gen * 19)
    mc = mc_score(W, delay, dmax, leak, rng_mc, in_idx, alpha, FW, FTR, FTE, D=50)
    na = narma_score(W, delay, dmax, leak, rng_na, in_idx, alpha, FW, FTR, FTE)
    if not np.isfinite(mc) or not np.isfinite(na):
        return -1e6
    return float(mc / 20.0 - na)


# ---------------------------------------------------------------------------
# ES train bridges
# ---------------------------------------------------------------------------
def es_train(W0, info, delay, dmax, alpha, seed, n_gen=ES_G):
    """Antithetic ES on bridge weights. Returns (W_final, log)."""
    mask, coo0 = bridge_edge_mask(W0, info)
    if int(mask.sum()) == 0:
        mf0, _ = mf_filled()
        d0, dm0 = delays_realmyelin(W0, info, seed, mf0)
        return W0.tocsr(), dict(n_bridge=0, gens=[], note="no bridges",
                                final_rho=float(K.radius_cached(W0)),
                                n_gen=0, n_fitness_evals=0, method="antithetic_ES"), d0
    frozen = freeze_blocks(W0, info)
    # template coo shares structure; delays align with this order
    template = W0.tocoo()
    # Ensure delay aligns with template order: rebuild delay from same W0
    mf, _ = mf_filled()
    delay, dmax = delays_realmyelin(W0, info, seed, mf)
    # Re-get mask on W0 after tocsr consistency
    W0 = W0.tocsr()
    mask, coo0 = bridge_edge_mask(W0, info)
    template = coo0
    theta = template.data[mask].astype(np.float64).copy()
    rms = float(np.sqrt(np.mean(theta ** 2))) if len(theta) else 1e-3
    sigma = max(ES_SIGMA_FRAC * rms, 1e-3)
    in_idx = info["drive"]
    log = []
    g = np.random.default_rng(seed + 4242)
    shape = W0.shape
    W_cur = W0
    rho_cur = float(K.radius_cached(W_cur))
    for gen in range(n_gen):
        eps = g.standard_normal(len(theta))
        Wp, rp = apply_theta(template, mask, theta + sigma * eps, frozen, info, shape)
        Wm, rm = apply_theta(template, mask, theta - sigma * eps, frozen, info, shape)
        # delays must match edge order of Wp/Wm — rebuild each time (structure fixed)
        delay_p, _ = delays_realmyelin(Wp, info, seed, mf)
        delay_m, _ = delays_realmyelin(Wm, info, seed, mf)
        fp = fitness_of(Wp, delay_p, dmax, LEAK, in_idx, alpha, gen, seed)
        fm = fitness_of(Wm, delay_m, dmax, LEAK, in_idx, alpha, gen, seed)
        theta = theta + ES_LR * 0.5 * (fp - fm) * eps
        W_cur, rho_cur = apply_theta(template, mask, theta, frozen, info, shape)
        # refresh template data from W_cur so next write is consistent? 
        # Keep original template structure; theta is absolute bridge weights.
        log.append(dict(gen=gen, f_plus=fp, f_minus=fm, f_mean=0.5 * (fp + fm),
                        rho=float(rho_cur), sigma=float(sigma)))
    delay_f, _ = delays_realmyelin(W_cur, info, seed, mf)
    return W_cur, dict(
        n_bridge=int(mask.sum()), gens=log, sigma=float(sigma),
        final_rho=float(rho_cur), n_gen=n_gen, n_fitness_evals=n_gen * 2,
        method="antithetic_ES",
    ), delay_f


# ---------------------------------------------------------------------------
# Tune / run
# ---------------------------------------------------------------------------
def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"TUNE alphas arms={ALL_ARMS} seeds={TUNE_SEEDS} fc2={_HAS_FC2}", flush=True)
    mf, med = mf_filled()
    choice = {
        "alphas": {},
        "grid": ALPHA_GRID,
        "criterion": "mean MC @ full train",
        "tune_seeds": TUNE_SEEDS,
        "myelin_median": med,
        "has_fastcore2": _HAS_FC2,
        "note": "FIXED and ES share alpha per wiring (tuned on FIXED init; ES starts from same)",
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Tune per wiring on FIXED init only; ES inherits (fair: alpha not fit to trained bridges)
    for wiring in WIRINGS:
        arm_fixed = arm_key(wiring, "FIXED")
        arm_es = arm_key(wiring, "ES")
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            t0 = time.time()
            W, info = build_wiring(wiring, seed)
            delay, dmax = delays_realmyelin(W, info, seed, mf)
            rng = np.random.default_rng(seed + 0)
            T = WASH + TR + TE
            u = rng.uniform(-1, 1, T)
            Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
            X = run_delay_reservoir(W, delay, dmax, Win, b, u, LEAK)
            D = 100
            Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
            for a in ALPHA_GRID:
                B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], a)
                P = K.pred(X[WASH + TR:], B)
                Yt = Y[WASH + TR:]
                val = float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
                scores[a].append(val)
            print(f"tune {wiring} seed={seed} n={W.shape[0]} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
        for arm in (arm_fixed, arm_es):
            choice["alphas"][arm] = {"alpha": best, "means": means, "wiring": wiring,
                                     "tuned_on": "FIXED_init"}
        print(f"LOCK alpha {wiring} -> {best} (both FIXED and ES)", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, task, alpha, n_gen=ES_G):
    t0 = time.time()
    wiring, cond = parse_arm(arm)
    mf, med = mf_filled()
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4, "XOR_DELAY": 6}[task] + 17)
    W0, info = build_wiring(wiring, seed)
    delay0, dmax = delays_realmyelin(W0, info, seed, mf)
    train_log = None
    if cond == "ES":
        # Train once per (wiring, seed); cache
        cache_path = OUT / f"es_cache_{wiring}_seed{seed}_g{n_gen}.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            W = sp.csr_matrix(
                (z["W_data"], z["W_indices"], z["W_indptr"]),
                shape=tuple(int(x) for x in z["W_shape"]),
            )
            delay = z["delay"]
            dmax = int(z["dmax"])
            train_log = json.loads(str(z["log"]))
        else:
            W, train_log, delay = es_train(W0, info, delay0, dmax, alpha, seed, n_gen=n_gen)
            dmax = DMAX
            np.savez(
                cache_path,
                W_data=W.data, W_indices=W.indices, W_indptr=W.indptr,
                W_shape=np.array(W.shape), delay=delay, dmax=dmax,
                log=np.array(json.dumps(train_log)),
            )
    else:
        W, delay = W0, delay0

    in_idx = info["drive"]
    if task == "MC":
        val = mc_score(W, delay, dmax, LEAK, g, in_idx, alpha, WASH, TR, TE)
    elif task == "NARMA":
        val = narma_score(W, delay, dmax, LEAK, g, in_idx, alpha, WASH, TR, TE)
    elif task == "PC_HARD":
        val = pc_hard_score(W, delay, dmax, LEAK, g, in_idx, alpha)
    else:
        val = xor_delay_score(W, delay, dmax, LEAK, g, in_idx, alpha)

    row = dict(
        arm=arm, wiring=wiring, cond=cond, seed=int(seed), task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        delay_mean=float(np.mean(delay)), myelin_median=med,
        sec=time.time() - t0, has_fastcore2=_HAS_FC2,
        source="trained_bridges", n_gen=int(n_gen) if cond == "ES" else 0,
    )
    if train_log is not None:
        row["n_bridge"] = train_log.get("n_bridge")
        row["final_rho"] = train_log.get("final_rho")
        row["n_fitness_evals"] = train_log.get("n_fitness_evals")
        # last gen fitness mean
        gens = train_log.get("gens") or []
        if gens:
            row["train_f_first"] = gens[0].get("f_mean")
            row["train_f_last"] = gens[-1].get("f_mean")
    return row


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args[:4]
        n_gen = args[4] if len(args) > 4 else ES_G
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    n_gen=n_gen, error=str(e)[:400],
                    traceback=traceback.format_exc()[-800:],
                    source="trained_bridges")


def smoke():
    """1 seed, FRANKEN_FIXED + FRANKEN_ES (G=2), MC only."""
    mf, _ = mf_filled()
    out = []
    for arm, n_gen in [("FRANKEN_FIXED", 0), ("FRANKEN_ES", 2)]:
        t0 = time.time()
        r = run_one(arm, 0, "MC", 1e-6, n_gen=n_gen)
        r["smoke_sec"] = time.time() - t0
        out.append(r)
        print("SMOKE", {k: r.get(k) for k in
              ("arm", "value", "n", "sec", "n_bridge", "final_rho", "train_f_first", "train_f_last")},
              flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")
    return out


def main():
    mode = os.environ.get("MODE", "full")  # smoke | tune | full
    n_gen = int(os.environ.get("ES_G", str(ES_G)))
    if mode == "smoke":
        smoke()
        return
    if mode == "tune":
        tune_alphas()
        return
    alphas = tune_alphas()
    only = os.environ.get("ONLY_ARMS", "")
    arms = [a.strip() for a in only.split(",") if a.strip()] if only else ALL_ARMS
    # Order: FIXED first (fast), then ES (so cache builds), tasks grouped
    arms_sorted = sorted(arms, key=lambda a: (0 if a.endswith("_FIXED") else 1, a))
    jobs = []
    for arm in arms_sorted:
        a = alphas["alphas"][arm]["alpha"]
        for seed in SEEDS:
            for task in TASKS:
                jobs.append((arm, seed, task, a, n_gen))
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
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "1"))
    print(f"trained_bridges jobs={len(jobs)} skip={len(done)} workers={workers} "
          f"arms={len(arms)} fc2={_HAS_FC2} ES_G={n_gen}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = _job(j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r.get("arm"), r.get("seed"), r.get("task"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("arm"), r.get("seed"), r.get("task"), r.get("value"),
                          round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
    (OUT / "trained_bridges.done").write_text("ok\n")
    print("trained_bridges done", flush=True)


if __name__ == "__main__":
    main()

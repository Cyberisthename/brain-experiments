"""Developmental maturation suite. Prereg: PREREGISTRATION_mature.md.
Reuses frankenstein builders + fast_kernels. Nice'd; don't starve hybrid/robust.
"""
from __future__ import annotations
import os, sys, json, time, math
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/regions/frankenstein")
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from multiprocessing import Pool
import frankenstein as F
import fast_kernels as K

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
ARMS = ["FLY_CX", "FRANKEN", "ER", "SHUF", "DENS_MATCHED"]
CONDITIONS = ["none", "hebb", "hebb_prune", "full"]
AGES = [0, 1000, 5000, 20000, 50000]
LEAK = 0.2
RHO = 0.9
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
MIN_KEEP_FRAC = 0.50
EPS_SIGN = 1e-12

# defaults; overwritten by out/tune.json after tuning
DEFAULT_HP = dict(
    eta=1e-4,
    prune_frac=0.02,
    prune_every=1000,
    delay_ema=0.1,
    dmax=DMAX,
)


def load_hp():
    p = OUT / "tune.json"
    if p.exists():
        d = json.loads(p.read_text())
        hp = dict(DEFAULT_HP)
        hp.update({k: d[k] for k in DEFAULT_HP if k in d})
        return hp
    return dict(DEFAULT_HP)


def build(arm, seed, g=None):
    if g is None:
        g = np.random.default_rng(seed + 7)
    _orig = F.scale
    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale
    try:
        W, info = F.build_arm(arm, seed, g)
    finally:
        F.scale = _orig
    return W.tocsr(), info


def rescale(W, rho=RHO):
    r = float(K.radius_cached(W))
    return (W * (rho / (r + 1e-12))).tocsr()


def preserve_signs(W, sign0):
    """Clip any sign flips back to ε·sign0."""
    W = W.tocsr()
    s = np.sign(W.data)
    s0 = np.sign(sign0)
    flip = (s * s0) < 0
    if flip.any():
        W.data[flip] = EPS_SIGN * s0[flip]
    # also kill exact zeros that lost sign info
    zero = W.data == 0
    if zero.any():
        W.data[zero] = EPS_SIGN * s0[zero]
    return W


def make_dev_stream(T, n_ch, rng):
    """Structured developmental inputs — NOT the test tasks.
    Returns U shape (T, n_ch).
    """
    t = np.arange(T, dtype=np.float64)
    U = np.zeros((T, n_ch), dtype=np.float64)
    # (i) smooth low-freq sinusoids on first third of channels
    n1 = max(1, n_ch // 3)
    for c in range(n1):
        f = 0.002 + 0.003 * (c / max(1, n1 - 1))
        ph = rng.uniform(0, 2 * np.pi)
        U[:, c] = np.sin(2 * np.pi * f * t + ph)
    # (ii) periodic multi-frequency phase patterns
    n2 = max(1, n_ch // 3)
    for c in range(n2):
        f1 = 0.01 + 0.02 * (c / max(1, n2 - 1))
        f2 = 3 * f1
        U[:, n1 + c] = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.cos(2 * np.pi * f2 * t + c)
    # (iii) sparse event pulses on remaining channels
    n3 = n_ch - n1 - n2
    for c in range(max(0, n3)):
        rate = 0.02 + 0.03 * rng.random()
        spikes = rng.random(T) < rate
        pulse = np.zeros(T)
        pulse[spikes] = rng.uniform(0.5, 1.5, spikes.sum())
        # short exponential decay envelope
        for i in np.where(spikes)[0]:
            end = min(T, i + 8)
            pulse[i:end] = np.maximum(pulse[i:end], pulse[i] * np.exp(-0.4 * np.arange(end - i)))
        U[:, n1 + n2 + c] = pulse
    # mild global amplitude jitter
    U *= rng.uniform(0.8, 1.2)
    return U


def masked_win_multi(n, n_ch, in_idx, rng, scale=0.5):
    Win = np.zeros((n, n_ch))
    idx = np.asarray(in_idx)
    if len(idx) == 0:
        Win[:] = rng.uniform(-scale, scale, (n, n_ch))
    else:
        Win[idx] = rng.uniform(-scale, scale, (len(idx), n_ch))
    b = rng.uniform(-0.1, 0.1, n)
    return Win, b


def init_delays(W, rng, dmax=DMAX):
    nnz = W.nnz
    return rng.integers(1, dmax + 1, nnz).astype(np.int32)


def split_delay_bins(W, delays, dmax):
    """Return list of CSR mats W_d for delay d=1..dmax (0 unused)."""
    W = W.tocsr()
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    cols = W.indices
    data = W.data
    bins = [None]
    for d in range(1, dmax + 1):
        m = delays == d
        if not m.any():
            bins.append(sp.csr_matrix(W.shape, dtype=np.float64))
        else:
            bins.append(sp.csr_matrix((data[m], (rows[m], cols[m])), shape=W.shape))
    return bins


def run_reservoir_delayed(bins, Win, b, U, leak, dmax):
    n = bins[1].shape[0]
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    T = len(U)
    X = np.empty((T, n), dtype=np.float64)
    buf = np.zeros((dmax + 1, n), dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    multi = U.ndim == 2
    for i in range(T):
        h = np.zeros(n, dtype=np.float64)
        for d in range(1, dmax + 1):
            src = buf[(i - d) % (dmax + 1)]
            h += bins[d] @ src
        drive = (Win @ U[i]) if multi else (Win[:, 0] * U[i])
        x = (1.0 - leak) * x + leak * np.tanh(h + drive + b)
        buf[i % (dmax + 1)] = x
        X[i] = x
    return X


def run_res(W, Win, b, U, leak, delays=None, dmax=DMAX):
    if delays is None:
        return K.run_reservoir_csr(W, Win, b, U, leak)
    bins = split_delay_bins(W, delays, dmax)
    return run_reservoir_delayed(bins, Win, b, U, leak, dmax)


def prune_weakest(W, usage, frac, sign0, min_keep_frac=MIN_KEEP_FRAC):
    """Drop weakest frac of edges by |w|·usage; keep ≥ min_keep_frac of original nnz.
    usage aligned with W.data order. Always returns (W, sign0, usage).
    """
    W = W.tocsr()
    nnz0 = len(sign0)
    keep_floor = int(math.ceil(min_keep_frac * nnz0))
    n_drop = int(math.floor(frac * W.nnz))
    n_keep = max(keep_floor, W.nnz - n_drop)
    if n_keep >= W.nnz:
        return W, sign0, usage
    score = np.abs(W.data) * (usage + 1e-6)
    order = np.argsort(score)  # weakest first
    keep_idx = np.sort(order[-n_keep:])
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    cols = W.indices
    data = W.data
    # rebuild sign0 alignment: we track sign0 on current edges only via parallel array
    W2 = sp.csr_matrix((data[keep_idx], (rows[keep_idx], cols[keep_idx])), shape=W.shape)
    new_sign0 = sign0[keep_idx] if len(sign0) == len(data) else np.sign(data[keep_idx])
    return W2.tocsr(), new_sign0, usage[keep_idx]


def adapt_delays(delays, co, rng_shuf=None, dmax=DMAX):
    """Myelin ADAPT rule on coactivity per-edge array `co`."""
    med = np.median(co)
    p20 = np.percentile(co, 20)
    shorten = co > med
    lengthen = co < p20
    delays = delays.copy()
    delays[shorten] = np.maximum(1, delays[shorten] - 1)
    delays[lengthen] = np.minimum(dmax, delays[lengthen] + 1)
    return delays



def mature_segment(W, sign0, Win, b, U, leak, hp, condition, delays, usage, pre_trace, post_trace):
    """Run |U| developmental steps with plasticity per condition.
    Optimized: cache CSR row indices; rebuild delay bins sparingly.
    """
    do_hebb = condition in ("hebb", "hebb_prune", "full")
    do_prune = condition in ("hebb_prune", "full")
    do_delay = condition == "full"
    eta = float(hp["eta"])
    prune_frac = float(hp["prune_frac"])
    prune_every = int(hp["prune_every"])
    delay_ema = float(hp["delay_ema"])
    dmax = int(hp["dmax"])
    W = W.tocsr()
    n = W.shape[0]
    x = np.zeros(n, dtype=np.float64)
    if usage is None or len(usage) != W.nnz:
        usage = np.zeros(W.nnz, dtype=np.float64)
    if pre_trace is None:
        pre_trace = np.zeros(n, dtype=np.float64)
    if post_trace is None:
        post_trace = np.zeros(n, dtype=np.float64)
    if do_delay and delays is None:
        raise ValueError("delays required for full")
    if do_delay and len(delays) != W.nnz:
        delays = init_delays(W, np.random.default_rng(0), dmax)

    def _rows_of(Wm):
        return np.repeat(np.arange(Wm.shape[0]), np.diff(Wm.indptr))

    rows = _rows_of(W)
    cols = W.indices  # view
    buf = np.zeros((dmax + 1, n), dtype=np.float64) if do_delay else None
    bins = split_delay_bins(W, delays, dmax) if do_delay else None
    U = np.asarray(U, dtype=np.float64)
    multi = U.ndim == 2
    steps_since_prune = 0
    BIN_EVERY = 100  # rebuild delay bins / adapt delays
    USAGE_EVERY = 1

    for i in range(len(U)):
        h_pre = x.copy()
        drive = (Win @ U[i]) if multi else (Win[:, 0] * U[i])
        if do_delay:
            h = bins[1] @ buf[(i - 1) % (dmax + 1)]
            for d in range(2, dmax + 1):
                h = h + bins[d] @ buf[(i - d) % (dmax + 1)]
            x = (1.0 - leak) * x + leak * np.tanh(h + drive + b)
            buf[i % (dmax + 1)] = x
        else:
            x = (1.0 - leak) * x + leak * np.tanh(W @ h_pre + drive + b)

        if do_hebb and eta > 0:
            K.oja_update_csr_fast(W, h_pre, x, eta)
            if (i + 1) % 10 == 0:
                s = np.sign(W.data)
                flip = s * sign0 < 0
                if flip.any():
                    W.data[flip] = EPS_SIGN * sign0[flip]

        if do_prune and (i % USAGE_EVERY == 0):
            if len(usage) != W.nnz:
                usage = np.zeros(W.nnz, dtype=np.float64)
                rows = _rows_of(W)
                cols = W.indices
            co = np.abs(h_pre[cols] * x[rows])
            usage *= (1 - delay_ema)
            usage += delay_ema * co

        if do_delay and (i + 1) % BIN_EVERY == 0:
            pre_trace = (1 - delay_ema) * pre_trace + delay_ema * (h_pre ** 2)
            post_trace = (1 - delay_ema) * post_trace + delay_ema * (x ** 2)
            if len(delays) != W.nnz:
                delays = init_delays(W, np.random.default_rng(i + 1), dmax)
            rows = _rows_of(W); cols = W.indices
            co_d = pre_trace[cols] * post_trace[rows]
            delays = adapt_delays(delays, co_d, dmax=dmax)
            bins = split_delay_bins(W, delays, dmax)

        steps_since_prune += 1
        if do_prune and steps_since_prune >= prune_every:
            W, sign0, usage = prune_weakest(W, usage, prune_frac, sign0)
            W = rescale(W, RHO)
            sign0 = np.sign(W.data)
            rows = _rows_of(W); cols = W.indices
            if do_delay:
                rng_d = np.random.default_rng((i + 13) % (2**31))
                delays = init_delays(W, rng_d, dmax)
                bins = split_delay_bins(W, delays, dmax)
            steps_since_prune = 0
    return W, sign0, delays, usage, pre_trace, post_trace


# -------------------- evals --------------------
def mc_from_X(X, u, wash=200, tr=1500, te=500, D=100):
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = K.pred(X[wash + tr:], B)
    Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_from_X(X, y, wash=200, tr=1500, te=500):
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return float("nan")
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def pc_hard(W, leak, rng, in_idx, delays=None):
    n = W.shape[0]
    T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = np.asarray(in_idx)
    rng2 = np.random.default_rng(int(rng.integers(1e9)))
    k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
    drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
    Win, b = F.masked_win(n, 2, drive_idx, rng)
    wash, tr = 200, 800
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = run_res(W, Win, b, u_drive, leak, delays=delays)
    Wout = np.linalg.lstsq(X[wash:tr], u_clean[wash:tr], rcond=None)[0]
    pred = X[tr:] @ Wout
    yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


def eval_all(W, info, seed, leak, delays=None):
    """MC, NARMA, PC_HARD, lesion20 retention (MC)."""
    drive = info["drive"]
    n = W.shape[0]
    out = {}
    # MC
    rng = np.random.default_rng(seed + 0)
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = F.masked_win(n, 1, drive, rng)
    X = run_res(W, Win, b, u, leak, delays=delays)
    mc = mc_from_X(X, u, wash, tr, te, D)
    out["MC"] = mc
    # lesion 20%
    rng_l = np.random.default_rng(seed + 1000)
    k = max(1, int(round(0.20 * n)))
    les = rng_l.choice(n, size=min(k, n - 1), replace=False)
    keep = np.ones(n, dtype=bool); keep[les] = False
    Win_l = Win.copy(); b_l = b.copy()
    Win_l[les] = 0; b_l[les] = 0
    # delayed lesion: zero state of lesioned each step
    if delays is None:
        Xl = K.run_reservoir_csr(W, Win_l, b_l, u, leak)
        # silence — re-sim with mask
        Xl = np.empty_like(X)
        x = np.zeros(n)
        win0 = Win_l[:, 0]
        for i in range(T):
            x = (1 - leak) * x + leak * np.tanh(W @ x + win0 * u[i] + b_l)
            x[~keep] = 0
            Xl[i] = x
    else:
        bins = split_delay_bins(W, delays, DMAX)
        Xl = np.empty((T, n))
        buf = np.zeros((DMAX + 1, n))
        x = np.zeros(n)
        for i in range(T):
            h = np.zeros(n)
            for d in range(1, DMAX + 1):
                h += bins[d] @ buf[(i - d) % (DMAX + 1)]
            x = (1 - leak) * x + leak * np.tanh(h + Win_l[:, 0] * u[i] + b_l)
            x[~keep] = 0
            buf[i % (DMAX + 1)] = x
            Xl[i] = x
    mc_l = mc_from_X(Xl, u, wash, tr, te, D)
    out["LES20_MC"] = mc_l
    out["LES20_RET"] = float(mc_l / mc) if (np.isfinite(mc) and abs(mc) > 1e-12) else float("nan")
    # NARMA
    rng = np.random.default_rng(seed + 2)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = F.masked_win(n, 1, drive, rng)
    X = run_res(W, Win, b, u, leak, delays=delays)
    out["NARMA"] = narma_from_X(X, y, wash, tr, te)
    # PC_HARD
    rng = np.random.default_rng(seed + 4)
    out["PC_HARD"] = pc_hard(W, leak, rng, drive, delays=delays)
    return out


def run_one(arm, seed, condition, hp=None, max_age=None):
    """One arm×seed×condition through all ages (or up to max_age)."""
    t0 = time.time()
    hp = hp or load_hp()
    ages = [a for a in AGES if max_age is None or a <= max_age]
    g = np.random.default_rng(seed + 7)
    W, info = build(arm, seed, g)
    W = W.tocsr()
    sign0 = W.data.copy()  # initial signed weights for sign lock (values, not just signs)
    sign0 = np.sign(sign0)
    delays = init_delays(W, np.random.default_rng(seed + 99), hp["dmax"]) if condition == "full" else None
    usage = np.zeros(W.nnz, dtype=np.float64)
    pre_trace = post_trace = None
    # developmental Win (multi-channel)
    n_ch = 6
    rng_dev = np.random.default_rng(seed + 12345)
    Win_dev, b_dev = masked_win_multi(W.shape[0], n_ch, info["drive"], rng_dev)
    rows = []
    # age 0
    ev = eval_all(W, info, seed, LEAK, delays=(delays if condition == "full" else None))
    rows.append(dict(arm=arm, seed=seed, condition=condition, age=0, **ev,
                     nnz=int(W.nnz), n=int(W.shape[0]), sec=time.time() - t0, hp=hp))
    if condition == "none":
        # W unchanged — replicate age-0 scores for schema
        for age in ages:
            if age == 0:
                continue
            rows.append(dict(arm=arm, seed=seed, condition=condition, age=age, **ev,
                             nnz=int(W.nnz), n=int(W.shape[0]), sec=time.time() - t0, hp=hp,
                             note="none_shortcircuit"))
        return rows
    # progressive segments between ages
    for ai in range(1, len(ages)):
        steps = ages[ai] - ages[ai - 1]
        U = make_dev_stream(steps, n_ch, np.random.default_rng(seed + 777 + ages[ai]))
        W, sign0, delays, usage, pre_trace, post_trace = mature_segment(
            W, sign0, Win_dev, b_dev, U, LEAK, hp, condition, delays, usage, pre_trace, post_trace
        )
        W = rescale(W, RHO)
        sign0 = np.sign(W.data)
        if delays is not None and len(delays) != W.nnz:
            delays = init_delays(W, np.random.default_rng(seed + ages[ai]), hp["dmax"])
        ev = eval_all(W, info, seed, LEAK, delays=(delays if condition == "full" else None))
        rows.append(dict(arm=arm, seed=seed, condition=condition, age=ages[ai], **ev,
                         nnz=int(W.nnz), n=int(W.shape[0]), sec=time.time() - t0, hp=hp))
    return rows


def _job(args):
    arm, seed, condition = args[:3]
    hp = args[3] if len(args) > 3 else None
    max_age = args[4] if len(args) > 4 else None
    try:
        return run_one(arm, seed, condition, hp=hp, max_age=max_age)
    except Exception as e:
        import traceback
        return [dict(arm=arm, seed=seed, condition=condition, age=-1,
                     error=str(e)[:400], traceback=traceback.format_exc()[-800:])]


def tune():
    """Pick η / prune_frac on tune seeds 90-92; write out/tune.json BEFORE test seeds."""
    cache = OUT / "tune.json"
    if cache.exists() and not os.environ.get("RETUNE"):
        print("tune cache hit", cache, flush=True)
        return json.loads(cache.read_text())
    # eta grid on hebb@5k; prune_frac locked at 0.02 (only affects hebb_prune/full)
    etas = [3e-5, 1e-4, 3e-4]
    pf = 0.02
    best = None
    best_score = -1e9
    results = []
    for eta in etas:
        hp = dict(DEFAULT_HP, eta=eta, prune_frac=pf)
        mcs = []
        ok = True
        for arm in ("FLY_CX", "FRANKEN", "ER"):
            for ts in TUNE_SEEDS:
                rows = run_one(arm, ts, "hebb", hp=hp, max_age=5000)
                r5 = [r for r in rows if r.get("age") == 5000]
                r0 = [r for r in rows if r.get("age") == 0]
                if not r5 or not np.isfinite(r5[0].get("MC", float("nan"))):
                    ok = False
                    mcs.append(float("nan"))
                    break
                if not np.isfinite(r5[0].get("NARMA", float("nan"))):
                    ok = False
                mcs.append(r5[0]["MC"])
                results.append(dict(eta=eta, prune_frac=pf, arm=arm, seed=ts,
                                    MC0=r0[0]["MC"], MC5=r5[0]["MC"],
                                    NARMA5=r5[0].get("NARMA"), ok=ok))
            if not ok:
                break
        if not ok:
            score = -1e9
        else:
            score = float(np.nanmean(mcs)) - 0.1 * abs(eta - 1e-4) * 1e4
        print(f"tune eta={eta} pf={pf} score={score:.3f} ok={ok} mcs={[round(x,3) if x==x else None for x in mcs]}", flush=True)
        if score > best_score:
            best_score = score
            best = hp
    if best is None:
        best = dict(DEFAULT_HP)
        best["tune_fallback"] = True
    best["tune_seeds"] = TUNE_SEEDS
    best["tune_score"] = best_score
    best["tuned_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cache.write_text(json.dumps(best, indent=2))
    (OUT / "tune_grid.jsonl").write_text("\n".join(json.dumps(r, default=float) for r in results) + "\n")
    print("locked hp", best, flush=True)
    return best


def main():
    mode = os.environ.get("MODE", "full")  # tune | full | smoke
    if mode == "tune":
        tune()
        return
    hp = load_hp()
    if not (OUT / "tune.json").exists():
        print("no tune.json — running tune first", flush=True)
        hp = tune()
    max_age = int(os.environ["MAX_AGE"]) if os.environ.get("MAX_AGE") else None
    conds = [c.strip() for c in os.environ.get("ONLY_COND", "").split(",") if c.strip()] or CONDITIONS
    arms = [a.strip() for a in os.environ.get("ONLY_ARM", "").split(",") if a.strip()] or ARMS
    seeds = SEEDS
    if os.environ.get("ONLY_SEEDS"):
        seeds = [int(s) for s in os.environ["ONLY_SEEDS"].split(",")]
    jobs = [(a, s, c, hp, max_age) for a in arms for s in seeds for c in conds]
    JF = OUT / "runs.jsonl"
    done = set()
    if JF.exists():
        for line in JF.open():
            try:
                r = json.loads(line)
                if r.get("error") or r.get("age", 0) < 0:
                    continue
                # done if we have the final age for that triple
                key = (r["arm"], r["seed"], r["condition"])
                final = max_age if max_age is not None else AGES[-1]
                if r.get("age") == final:
                    done.add(key)
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "1"))
    print(f"mature jobs={len(jobs)} skip={len(done)} workers={workers} hp={hp} max_age={max_age}", flush=True)
    with JF.open("a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                rows = _job(j)
                for r in rows:
                    f.write(json.dumps(r, default=float) + "\n")
                f.flush()
                print(j[0], j[1], j[2], "nrows", len(rows),
                      rows[-1].get("MC"), rows[-1].get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for rows in pool.imap_unordered(_job, jobs):
                    for r in rows:
                        f.write(json.dumps(r, default=float) + "\n")
                    f.flush()
                    print(rows[0].get("arm"), rows[0].get("seed"), rows[0].get("condition"),
                          "nrows", len(rows), rows[-1].get("MC"), rows[-1].get("error", ""), flush=True)
    (OUT / "mature.done").write_text("ok\n")
    print("mature batch done", flush=True)


if __name__ == "__main__":
    main()

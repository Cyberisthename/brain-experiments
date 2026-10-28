"""Mega-Frankenstein: stitch more sections + bridge σ / realistic ports.
Prereg: PREREGISTRATION_mega_franken.md
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

# Opt-in speedup (numerics unchanged when batch=1)
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
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
SIGMAS = [0.05, 0.2, 0.5]
BRIDGE_MODES = ["UNIF", "REALISTIC"]
TASKS = ["MC", "NARMA", "PC_HARD"]

# stack name -> list of block names (fly_cx always hub)
STACKS = {
    "FRANKEN": ["mouse", "human", "fly_cx"],
    "FRANKEN+FINCH": ["mouse", "human", "finch", "fly_cx"],
    "FRANKEN+WORM": ["mouse", "human", "worm", "fly_cx"],
    "MEGA": ["mouse", "human", "finch", "worm", "octo", "fly_cx"],
}
# which blocks are "added" (eligible for ER/SHUF controls)
ADDED = {
    "FRANKEN": [],
    "FRANKEN+FINCH": ["finch"],
    "FRANKEN+WORM": ["worm"],
    "MEGA": ["finch", "worm", "octo"],
}
CTRLS = {
    "FRANKEN": ["REAL"],  # baseline only
    "FRANKEN+FINCH": ["REAL", "ER", "SHUF"],
    "FRANKEN+WORM": ["REAL", "ER", "SHUF"],
    "MEGA": ["REAL", "ER", "SHUF"],
}

MOD_FILES = {
    "mouse": "mouse_module.npz",
    "human": "human_module.npz",
    "fly_cx": "fly_cx_module.npz",
    "finch": "finch_module.npz",
    "worm": "worm_module.npz",
    "octo": "octo_module.npz",  # SYNTHETIC
}


def arm_key(stack, ctrl, sigma, bridge):
    if stack == "FRANKEN":
        return f"FRANKEN__s{sigma:g}__{bridge}"
    return f"{stack}_{ctrl}__s{sigma:g}__{bridge}"


def parse_arm(arm):
    """Return stack, ctrl, sigma, bridge."""
    # e.g. FRANKEN+FINCH_ER__s0.2__UNIF  or FRANKEN__s0.05__REALISTIC
    if "__s" not in arm:
        raise KeyError(arm)
    left, rest = arm.split("__s", 1)
    sig_s, bridge = rest.split("__", 1)
    sigma = float(sig_s)
    if left == "FRANKEN":
        return "FRANKEN", "REAL", sigma, bridge
    # left like FRANKEN+FINCH_ER or MEGA_SHUF or FRANKEN+FINCH_REAL
    for stack in ("FRANKEN+FINCH", "FRANKEN+WORM", "MEGA"):
        if left == stack or left.startswith(stack + "_"):
            if left == stack:
                ctrl = "REAL"
            else:
                ctrl = left[len(stack) + 1:]  # REAL / ER / SHUF
            return stack, ctrl, sigma, bridge
    raise KeyError(arm)


def all_arms():
    arms = []
    for stack, ctrls in CTRLS.items():
        for ctrl in ctrls:
            for sigma in SIGMAS:
                for bridge in BRIDGE_MODES:
                    arms.append(arm_key(stack, ctrl, sigma, bridge))
    return arms


ALL_ARMS = all_arms()


def load_mod(name):
    z = np.load(DATA / MOD_FILES[name], allow_pickle=True)
    W = sp.csr_matrix(
        (z["W_data"].astype(np.float64), z["W_indices"], z["W_indptr"]),
        shape=tuple(int(x) for x in z["W_shape"]),
    )
    sign = z["sign"].astype(np.int8)
    layer = z["layer"].astype(str) if "layer" in z.files else np.array(["L2/3"] * W.shape[0])
    ct = z["cell_type"].astype(str) if "cell_type" in z.files else np.array(["unk"] * W.shape[0])
    return W, sign, layer, ct


def scale(W, rho=RHO):
    r = float(K.radius_cached(W))
    return W * (rho / (r + 1e-12))


def make_block(name, kind, seed):
    W, sign, layer, ct = load_mod(name)
    g = np.random.default_rng(seed + 13 + hash(name) % 1000 + hash(kind) % 97)
    if kind == "er":
        W = F.er_match(W, sign, g)
    elif kind == "shuf":
        W = F.degree_shuffle(W, sign, g)
    elif kind == "real":
        pass
    else:
        raise KeyError(kind)
    return scale(W, 1.0), sign, layer, ct


def ports_for(name, layer, ct, N, bridge_mode="UNIF"):
    """Return (in_ports, out_ports) local indices for bridging.

    UNIF: cortical L4/L5 if present; else index-proxy (frankenstein fallback).
    REALISTIC: type-aware ports for finch (HVC/LMAN out, MSN in) and worm
    (command out, sensory in). Cortical blocks stay L4/L5.
    """
    layer = np.asarray(layer).astype(str)
    ct = np.asarray(ct).astype(str)
    if name in ("mouse", "human", "octo", "fly_cx"):
        L4, L5, _ = F.layer_sets(layer, N)
        return L4.astype(int), L5.astype(int)
    if bridge_mode == "REALISTIC" and name == "finch":
        out_p = np.where(np.isin(ct, ["HVC", "LMAN"]))[0]
        in_p = np.where(ct == "MSN")[0]
        if len(out_p) == 0:
            out_p = np.arange(N // 2, N)
        if len(in_p) == 0:
            in_p = np.arange(min(200, N))
        return in_p.astype(int), out_p.astype(int)
    if bridge_mode == "REALISTIC" and name == "worm":
        in_p = np.where(layer == "L4")[0]
        out_p = np.where(layer == "L5")[0]
        if len(in_p) == 0:
            in_p = np.arange(min(40, N))
        if len(out_p) == 0:
            out_p = np.arange(max(0, N - 40), N)
        return in_p.astype(int), out_p.astype(int)
    # UNIF (or unknown): index proxies — first ~max(200,N//10) in, last half out
    n_in = max(200, N // 10)
    in_p = np.arange(min(n_in, N))
    out_p = np.arange(N // 2, N)
    return in_p.astype(int), out_p.astype(int)


def graft(blocks, seed, sigma, bridge_mode, g=None):
    """blocks: list of (name, Wm, layer, ct). Hub=fly_cx."""
    if g is None:
        g = np.random.default_rng(seed + 7)
    names = [b[0] for b in blocks]
    Ws = [b[1] for b in blocks]
    layers = [b[2] for b in blocks]
    cts = [b[3] for b in blocks]
    sizes = [W.shape[0] for W in Ws]
    N = sum(sizes)
    off = np.cumsum([0] + sizes)
    W = sp.lil_matrix((N, N))
    for i, Wi in enumerate(Ws):
        a, b = off[i], off[i + 1]
        W[a:b, a:b] = Wi
    ifx = names.index("fly_cx")
    fly_off = off[ifx]
    fly_in, fly_out = ports_for("fly_cx", layers[ifx], cts[ifx], sizes[ifx], bridge_mode)
    # hub targets: union of fly in+out (UNIF) or fly_out-ish; keep frankenstein style
    tgt_local = np.unique(np.concatenate([fly_in, fly_out]))
    tgt = tgt_local + fly_off
    bridge_info = {}
    for i, name in enumerate(names):
        if name == "fly_cx":
            continue
        in_p, out_p = ports_for(name, layers[i], cts[i], sizes[i], bridge_mode)
        L4g = in_p + off[i]
        L5g = out_p + off[i]
        n_out = min(3000, max(100, len(out_p) * 4))
        n_in = min(3000, max(100, len(in_p) * 4))
        br_from = g.choice(L5g, n_out, replace=True)
        br_to = g.choice(tgt, n_out, replace=True)
        for r, c, d in zip(br_to, br_from, g.standard_normal(n_out) * sigma):
            W[r, c] += d
        a_from = g.choice(tgt, n_in, replace=True)
        a_to = g.choice(L4g, n_in, replace=True)
        for r, c, d in zip(a_to, a_from, g.standard_normal(n_in) * sigma):
            W[r, c] += d
        bridge_info[name] = dict(
            n_in=int(n_in), n_out=int(n_out),
            n_in_ports=int(len(in_p)), n_out_ports=int(len(out_p)),
            offset=int(off[i]), bridge_mode=bridge_mode, sigma=float(sigma),
        )
    W = scale(W.tocsr(), RHO)
    # drive = union of all in-ports
    drive = []
    for i, name in enumerate(names):
        in_p, _ = ports_for(name, layers[i], cts[i], sizes[i], bridge_mode)
        drive.append(in_p + off[i])
    drive = np.unique(np.concatenate(drive)) if drive else np.arange(min(200, N))
    info = dict(
        names=names, sizes=sizes, offsets=[int(x) for x in off], N=int(N),
        drive=drive, bridge=bridge_info, fly_offset=int(fly_off),
        sigma=float(sigma), bridge_mode=bridge_mode,
        synthetic_octo=("octo" in names),
    )
    return W, info


def kinds_for(stack, ctrl):
    """Per-block kind: base always real; added follow ctrl."""
    kinds = {}
    added = set(ADDED[stack])
    for name in STACKS[stack]:
        if name in added:
            kinds[name] = {"REAL": "real", "ER": "er", "SHUF": "shuf"}[ctrl]
        else:
            kinds[name] = "real"
    return kinds


def build_arm(stack, ctrl, sigma, bridge, seed, g=None):
    if g is None:
        g = np.random.default_rng(seed + 7)
    kinds = kinds_for(stack, ctrl)
    blocks = []
    for name in STACKS[stack]:
        Wm, sign, layer, ct = make_block(name, kinds[name], seed)
        blocks.append((name, Wm, layer, ct))
    W, info = graft(blocks, seed, sigma, bridge, g)
    info["kinds"] = kinds
    info["stack"] = stack
    info["ctrl"] = ctrl
    return W, info


def load_myelin():
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf_filled = mf.copy()
    mf_filled[~np.isfinite(mf_filled)] = med
    return mf_filled, med


_MF = None


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
    """FIXED on non-human-pre; real H01 myelin on human-pre."""
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


def mc_task(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak)
    D = 100
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[WASH:WASH + TR], Y[WASH:WASH + TR], alpha)
    P = K.pred(X[WASH + TR:], B)
    Yt = Y[WASH + TR:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_task(W, delay, dmax, leak, rng, in_idx, alpha):
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


def pc_hard_task(W, delay, dmax, leak, rng, in_idx, alpha):
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


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"TUNE alphas n_arms={len(ALL_ARMS)} seeds={TUNE_SEEDS} fc2={_HAS_FC2}", flush=True)
    mf, med = mf_filled()
    choice = {
        "alphas": {},
        "grid": ALPHA_GRID,
        "criterion": "mean MC @ full train",
        "tune_seeds": TUNE_SEEDS,
        "myelin_median": med,
        "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for ai, arm in enumerate(ALL_ARMS):
        stack, ctrl, sigma, bridge = parse_arm(arm)
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            t0 = time.time()
            W, info = build_arm(stack, ctrl, sigma, bridge, seed)
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
            print(f"tune [{ai+1}/{len(ALL_ARMS)}] {arm} seed={seed} n={W.shape[0]} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
        choice["alphas"][arm] = {"alpha": best, "means": means,
                                 "stack": stack, "ctrl": ctrl, "sigma": sigma, "bridge": bridge}
        print(f"LOCK alpha {arm} -> {best}", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, task, alpha):
    t0 = time.time()
    stack, ctrl, sigma, bridge = parse_arm(arm)
    mf, med = mf_filled()
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
    W, info = build_arm(stack, ctrl, sigma, bridge, seed, g)
    delay, dmax = delays_realmyelin(W, info, seed, mf)
    in_idx = info["drive"]
    if task == "MC":
        val = mc_task(W, delay, dmax, LEAK, g, in_idx, alpha)
    elif task == "NARMA":
        val = narma_task(W, delay, dmax, LEAK, g, in_idx, alpha)
    else:
        val = pc_hard_task(W, delay, dmax, LEAK, g, in_idx, alpha)
    return dict(
        arm=arm, stack=stack, ctrl=ctrl, sigma=float(sigma), bridge=bridge,
        seed=seed, task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        delay_mean=float(delay.mean()), myelin_median=med,
        synthetic_octo=bool(info.get("synthetic_octo")),
        kinds=info.get("kinds"),
        sec=time.time() - t0,
        has_fastcore2=_HAS_FC2,
        source="mega_franken",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="mega_franken")


def smoke():
    """1 seed, σ=0.05 UNIF, FRANKEN + FRANKEN+FINCH_REAL + MEGA_REAL, MC only."""
    mf, _ = mf_filled()
    arms = [
        arm_key("FRANKEN", "REAL", 0.05, "UNIF"),
        arm_key("FRANKEN+FINCH", "REAL", 0.05, "UNIF"),
        arm_key("MEGA", "REAL", 0.05, "UNIF"),
    ]
    out = []
    for arm in arms:
        stack, ctrl, sigma, bridge = parse_arm(arm)
        t0 = time.time()
        W, info = build_arm(stack, ctrl, sigma, bridge, 0)
        delay, dmax = delays_realmyelin(W, info, 0, mf)
        rng = np.random.default_rng(17)
        val = mc_task(W, delay, dmax, LEAK, rng, info["drive"], 1e-6)
        row = dict(arm=arm, n=int(W.shape[0]), mc=val, sec=time.time() - t0,
                   synthetic_octo=info.get("synthetic_octo"), sizes=info["sizes"], names=info["names"])
        out.append(row)
        print("SMOKE", row, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2))
    (OUT / "smoke.done").write_text("ok\n")
    return out


def main():
    mode = os.environ.get("MODE", "full")  # smoke | tune | full
    if mode == "smoke":
        smoke()
        return
    if mode == "tune":
        tune_alphas()
        return
    alphas = tune_alphas()
    only = os.environ.get("ONLY_ARMS", "")
    arms = [a.strip() for a in only.split(",") if a.strip()] if only else ALL_ARMS
    jobs = []
    for arm in arms:
        a = alphas["alphas"][arm]["alpha"]
        for seed in SEEDS:
            for task in TASKS:
                jobs.append((arm, seed, task, a))
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
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"mega_franken jobs={len(jobs)} skip={len(done)} workers={workers} arms={len(arms)} fc2={_HAS_FC2}", flush=True)
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
    (OUT / "mega_franken.done").write_text("ok\n")
    print("mega_franken done", flush=True)


if __name__ == "__main__":
    main()

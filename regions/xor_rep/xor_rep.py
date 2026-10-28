"""XOR_DELAY confirmatory replication. Prereg: PREREG_xor_rep.md (locked before eval)."""
from __future__ import annotations
import os, sys, json, time, math
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

# Point frankenstein loader at our data symlinks only if needed — F.DATA is frankenstein/data
# which already has modules. Myelin is local.

SEEDS = list(range(200, 224))          # 24 fresh
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
PRIMARY_ARMS = ["FRANKEN", "ER", "SHUF"]
ALL_ARMS = ["FRANKEN", "ER", "SHUF", "DENS", "FLY_CX", "HUMAN", "SPEC"]
XOR_D_PRIMARY = 5
XOR_D_SECONDARY = [2, 3, 8, 10]
DELAY_MODES_PRIMARY = ["REALMYELIN"]
DELAY_MODES_SECONDARY = ["NODELAY", "FIXED"]  # vs REALMYELIN at d=5, primary arms

# ---------------------------------------------------------------------------
# Delay reservoir — copied from trained_bridges (do not change numerics)
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
    """Identical to trained_bridges.delays_realmyelin."""
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


def delays_fixed(W, seed):
    Wcoo = W.tocoo()
    g = np.random.default_rng(seed + 909)
    delay = g.integers(1, DMAX + 1, Wcoo.nnz).astype(np.int32)
    return delay, DMAX


def delays_nodelay(W):
    return np.ones(W.tocoo().nnz, dtype=np.int32), 1


def make_delays(W, info, seed, mode, mf):
    if mode == "REALMYELIN":
        return delays_realmyelin(W, info, seed, mf)
    if mode == "FIXED":
        return delays_fixed(W, seed)
    if mode == "NODELAY":
        return delays_nodelay(W)
    raise KeyError(mode)


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


def xor_delay_score(W, delay, dmax, leak, rng, in_idx, alpha, wash=WASH, tr=TR, te=TE, d=XOR_D_PRIMARY):
    """BIT-EXACT copy of trained_bridges.xor_delay_score. Metric: 1-NRMSE."""
    T = wash + tr + te
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
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], alpha)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse):
        return float("nan")
    return float(1.0 - nrmse)


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------
def _permute_block_break_layers(Wm, seed, name):
    """Spectrum-preserving: permute W rows/cols; caller keeps original layer labels."""
    n = Wm.shape[0]
    g = np.random.default_rng(seed + 7777 + (hash(name) % 100000))
    perm = g.permutation(n)
    # P W P.T via index remap on coo
    Wc = Wm.tocoo()
    Wp = sp.csr_matrix((Wc.data, (perm[Wc.row], perm[Wc.col])), shape=(n, n))
    # rescale ρ=1 (should already be ~1; numerical)
    r = float(K.radius_cached(Wp))
    if r > 1e-12:
        Wp = Wp * (1.0 / r)
    return Wp


def build_wiring(arm, seed, lesion=None):
    """lesion in {None, 'FLY', 'HUMAN', 'BRIDGES'}."""
    g = np.random.default_rng(seed + 7)
    if arm == "HUMAN":
        Wm, sign, layer, ct = F.make_block("human", "real", seed)
        W = F.scale(Wm, RHO)
        L4, L5, _ = F.layer_sets(layer, Wm.shape[0])
        info = dict(names=["human"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                    N=Wm.shape[0], drive=L4, bridge={}, fly_offset=0,
                    kinds={"human": "real"})
        return W.tocsr(), info
    if arm == "FLY_CX":
        return F.build_arm("FLY_CX", seed, g)
    if arm == "DENS":
        return F.build_arm("DENS_MATCHED", seed, g)
    if arm == "SPEC":
        # real blocks, each W node-permuted while layer labels stay — breaks
        # connectivity↔layer alignment, preserves block spectra.
        blocks = []
        for name in F.BLOCKS:
            Wm, sign, layer, ct = F.make_block(name, "real", seed)
            Wm = _permute_block_break_layers(Wm, seed, name)
            blocks.append((name, Wm, layer))
        W, info = F.graft_blocks(blocks, seed, g)
        info["kinds"] = {n: "spec_permute" for n in F.BLOCKS}
        return W.tocsr(), info
    if arm in ("FRANKEN", "ER", "SHUF"):
        W, info = F.build_arm(arm, seed, g)
    else:
        raise KeyError(arm)

    if lesion is None:
        return W.tocsr(), info
    W = W.tolil()
    offs = info["offsets"]
    names = info["names"]
    if lesion == "FLY" and "fly_cx" in names:
        i = names.index("fly_cx")
        a, b = int(offs[i]), int(offs[i + 1])
        W[a:b, a:b] = 0
    elif lesion == "HUMAN" and "human" in names:
        i = names.index("human")
        a, b = int(offs[i]), int(offs[i + 1])
        W[a:b, a:b] = 0
    elif lesion == "BRIDGES":
        # zero inter-block edges
        for i, name in enumerate(names):
            a, b = int(offs[i]), int(offs[i + 1])
            # keep diagonal block; zero everything else involving these rows carefully
            pass
        # clearer: start from block-diag only
        W2 = sp.lil_matrix(W.shape)
        for i in range(len(names)):
            a, b = int(offs[i]), int(offs[i + 1])
            W2[a:b, a:b] = W[a:b, a:b]
        W = W2
    else:
        raise KeyError(lesion)
    W = sp.csr_matrix(W)
    r = float(K.radius_cached(W))
    if r > 1e-12:
        W = W * (RHO / r)
    return W.tocsr(), info


# ---------------------------------------------------------------------------
# Tune / run
# ---------------------------------------------------------------------------
def tune_alphas(arms=None):
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    arms = arms or ALL_ARMS
    print(f"TUNE alphas arms={arms} seeds={TUNE_SEEDS} fc2={_HAS_FC2}", flush=True)
    mf, med = mf_filled()
    choice = {
        "alphas": {},
        "grid": ALPHA_GRID,
        "criterion": "mean XOR_DELAY d=5 REALMYELIN @ full train",
        "tune_seeds": TUNE_SEEDS,
        "myelin_median": med,
        "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_xor_rep.md",
    }
    for arm in arms:
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            t0 = time.time()
            W, info = build_wiring(arm, seed)
            delay, dmax = make_delays(W, info, seed, "REALMYELIN", mf)
            # one trajectory; sweep alpha on same X
            rng = np.random.default_rng(seed + 6 + 17)
            T = WASH + TR + TE
            a_bits = rng.integers(0, 2, T).astype(np.float64)
            b_bits = rng.integers(0, 2, T).astype(np.float64)
            y = np.zeros(T)
            d = XOR_D_PRIMARY
            for t in range(d, T):
                y[t] = 1.0 if (int(a_bits[t - d]) ^ int(b_bits[t - d])) else -1.0
            U = np.stack([2 * a_bits - 1, 2 * b_bits - 1], 1)
            Win, bias = masked_win(W.shape[0], 2, info["drive"], rng)
            X = run_delay_reservoir(W, delay, dmax, Win, bias, U, LEAK)
            for alpha in ALPHA_GRID:
                if not np.all(np.isfinite(X)):
                    scores[alpha].append(float("nan"))
                    continue
                B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
                p = K.pred(X[WASH + TR:], B)
                yt = y[WASH + TR:]
                if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
                    scores[alpha].append(float("nan"))
                    continue
                nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
                scores[alpha].append(float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan"))
            print(f"tune {arm} seed={seed} n={W.shape[0]} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
        choice["alphas"][arm] = {"alpha": best, "means": means}
        print(f"LOCK alpha {arm} -> {best}", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, xor_d, delay_mode, alpha, lesion=None):
    t0 = time.time()
    mf, med = mf_filled()
    W, info = build_wiring(arm, seed, lesion=lesion)
    delay, dmax = make_delays(W, info, seed, delay_mode, mf)
    # RNG scheme matches trained_bridges XOR: seed + 6 + 17
    rng = np.random.default_rng(seed + 6 + 17)
    val = xor_delay_score(W, delay, dmax, LEAK, rng, info["drive"], alpha, d=xor_d)
    return dict(
        arm=arm, seed=int(seed), task="XOR_DELAY", xor_d=int(xor_d),
        delay_mode=delay_mode, lesion=lesion or "NONE",
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        delay_mean=float(np.mean(delay)), myelin_median=med,
        sec=time.time() - t0, has_fastcore2=_HAS_FC2,
        source="xor_rep",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, xor_d, delay_mode, alpha = args[:5]
        lesion = args[5] if len(args) > 5 else None
        return dict(arm=arm, seed=seed, task="XOR_DELAY", xor_d=xor_d,
                    delay_mode=delay_mode, lesion=lesion or "NONE",
                    value=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-800:], source="xor_rep")


def _done_key(r):
    return (r["arm"], r["seed"], r.get("xor_d"), r.get("delay_mode"), r.get("lesion", "NONE"))


def _load_done():
    done = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                done.add(_done_key(r))
            except Exception:
                pass
    return done


def smoke():
    mf, _ = mf_filled()
    out = []
    for arm in ("FRANKEN", "ER", "HUMAN"):
        t0 = time.time()
        r = run_one(arm, 200, 5, "REALMYELIN", 1e-6)
        r["smoke_sec"] = time.time() - t0
        out.append(r)
        print("SMOKE", {k: r.get(k) for k in ("arm", "value", "n", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")
    return out


def build_jobs(alphas, phase="primary"):
    jobs = []
    if phase in ("primary", "all"):
        for arm in ALL_ARMS:
            a = alphas["alphas"][arm]["alpha"]
            for seed in SEEDS:
                jobs.append((arm, seed, XOR_D_PRIMARY, "REALMYELIN", a, None))
    if phase in ("secondary", "all"):
        for arm in PRIMARY_ARMS:
            a = alphas["alphas"][arm]["alpha"]
            for seed in SEEDS:
                for d in XOR_D_SECONDARY:
                    jobs.append((arm, seed, d, "REALMYELIN", a, None))
                for mode in DELAY_MODES_SECONDARY:
                    jobs.append((arm, seed, XOR_D_PRIMARY, mode, a, None))
    if phase == "mechanism":
        # FRANKEN lesions, seeds 200-207
        a = alphas["alphas"]["FRANKEN"]["alpha"]
        for seed in range(200, 208):
            jobs.append(("FRANKEN", seed, XOR_D_PRIMARY, "REALMYELIN", a, None))
            for les in ("FLY", "HUMAN", "BRIDGES"):
                jobs.append(("FRANKEN", seed, XOR_D_PRIMARY, "REALMYELIN", a, les))
    return jobs


def run_jobs(jobs, label="run"):
    done = _load_done()
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], j[5] or "NONE") not in done]
    workers = int(os.environ.get("WORKERS", "1"))
    print(f"{label} jobs={len(jobs)} skip={len(done)} workers={workers} fc2={_HAS_FC2}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                r = _job(j)
                f.write(json.dumps(r, default=float) + "\n"); f.flush()
                print(r.get("arm"), r.get("seed"), r.get("xor_d"), r.get("delay_mode"),
                      r.get("lesion"), r.get("value"), round(r.get("sec", 0) or 0, 1),
                      r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("arm"), r.get("seed"), r.get("xor_d"), r.get("delay_mode"),
                          r.get("lesion"), r.get("value"), round(r.get("sec", 0) or 0, 1),
                          r.get("error", ""), flush=True)


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "tune":
        tune_alphas(); return
    alphas = tune_alphas()
    phase = os.environ.get("PHASE", "all")  # primary | secondary | mechanism | all
    if phase == "mechanism":
        run_jobs(build_jobs(alphas, "mechanism"), "mechanism")
        (OUT / "mechanism.done").write_text("ok\n")
        return
    if phase == "primary":
        run_jobs(build_jobs(alphas, "primary"), "primary")
        (OUT / "primary.done").write_text("ok\n")
        return
    if phase == "secondary":
        run_jobs(build_jobs(alphas, "secondary"), "secondary")
        (OUT / "secondary.done").write_text("ok\n")
        return
    # full = primary then secondary
    run_jobs(build_jobs(alphas, "primary"), "primary")
    (OUT / "primary.done").write_text("ok\n")
    run_jobs(build_jobs(alphas, "secondary"), "secondary")
    (OUT / "secondary.done").write_text("ok\n")
    (OUT / "xor_rep.done").write_text("ok\n")
    print("xor_rep done", flush=True)


if __name__ == "__main__":
    main()

"""FLY_CX alone XOR_DELAY vs own matched controls. Prereg: PREREG_xor_cx.md."""
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
    import fastcore2 as FC2  # noqa: F401
    _HAS_FC2 = True
except Exception:
    _HAS_FC2 = False

H = _HERE
OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"

SEEDS = list(range(300, 324))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
PRIMARY_ARMS = ["CX", "ER", "SHUF", "DENS"]
ALL_ARMS = ["CX", "ER", "SHUF", "DENS", "EI_SHUF", "SPEC"]
XOR_D_PRIMARY = 5
XOR_D_SECONDARY = [2, 3, 8, 10]


# ---------------------------------------------------------------------------
# Delay reservoir + XOR — bit-exact with xor_rep / trained_bridges
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


def delays_fixed(W, seed):
    Wcoo = W.tocoo()
    g = np.random.default_rng(seed + 909)
    return g.integers(1, DMAX + 1, Wcoo.nnz).astype(np.int32), DMAX


def xor_delay_score(W, delay, dmax, leak, rng, in_idx, alpha, wash=WASH, tr=TR, te=TE, d=XOR_D_PRIMARY):
    """BIT-EXACT copy of trained_bridges / xor_rep xor_delay_score."""
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
def ei_class_shuffle(W, sign, g):
    """Rewire within (sign_pre, sign_post) classes; keep magnitudes; apply out-sign."""
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
            # candidate posts with sign sq
            cands = np.where(sign == s_post_v)[0]
            if len(cands) == 0:
                continue
            # reshuffle existing posts among class, then fix loops
            posts = col[idx].copy()
            g.shuffle(posts)
            # replace loops / invalid
            for _ in range(8):
                bad = row[idx] == posts
                if not bad.any():
                    break
                posts[bad] = g.choice(cands, size=int(bad.sum()), replace=True)
            new_col[idx] = posts
    out_data = data * sign[row].astype(float)
    return sp.csr_matrix((out_data, (row, new_col)), shape=(N, N))


def spectral_frac(W, k=64):
    try:
        ev = np.abs(sla.eigs(W.astype(float), k=min(k, W.shape[0] - 2),
                             which="LM", return_eigenvectors=False, maxiter=5000))
        return float((ev > 0.5).mean()), float(ev.mean())
    except Exception:
        return float("nan"), float("nan")


def make_spec_match(W_real_scaled, sign, g, target_frac, max_iter=25):
    """dens_match + shrink smallest half of |w| until top-64 frac>0.5 ≈ target."""
    W = F.dens_match(W_real_scaled, sign, g).tocsr()
    W = F.scale(W, 1.0)
    for _ in range(max_iter):
        frac, _ = spectral_frac(W)
        if not np.isfinite(frac):
            break
        if abs(frac - target_frac) <= 0.05:
            break
        if frac <= target_frac:
            break  # already as sparse-bulk as we can get this way
        Wc = W.tocoo()
        mag = np.abs(Wc.data)
        thr = np.median(mag)
        small = mag <= thr
        new = mag.copy()
        new[small] *= 0.7
        data = new * np.sign(Wc.data)
        # keep signs from data; rebuild
        W = sp.csr_matrix((data, (Wc.row, Wc.col)), shape=W.shape)
        W = F.scale(W, 1.0)
    return W


def build_wiring(arm, seed):
    W0, sign, layer, ct = F.load_mod("fly_cx")
    g = np.random.default_rng(seed + 13)
    if arm == "CX":
        Wm = F.scale(W0, 1.0)
    elif arm == "ER":
        Wm = F.scale(F.er_match(W0, sign, g), 1.0)
    elif arm == "SHUF":
        Wm = F.scale(F.degree_shuffle(W0, sign, g), 1.0)
    elif arm == "DENS":
        Wm = F.scale(F.dens_match(W0, sign, g), 1.0)
    elif arm == "EI_SHUF":
        Wm = F.scale(ei_class_shuffle(W0, sign, g), 1.0)
    elif arm == "SPEC":
        # target from real
        Wreal = F.scale(W0, 1.0)
        tgt, _ = spectral_frac(Wreal)
        if not np.isfinite(tgt):
            tgt = 0.28
        Wm = make_spec_match(W0, sign, g, tgt)
    else:
        raise KeyError(arm)
    W = F.scale(Wm, RHO)
    L4, L5, _ = F.layer_sets(layer, Wm.shape[0])
    info = dict(names=["fly_cx"], sizes=[Wm.shape[0]], offsets=[0, Wm.shape[0]],
                N=Wm.shape[0], drive=L4, bridge={}, fly_offset=0, kinds={"fly_cx": arm})
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
    choice = {
        "alphas": {}, "grid": ALPHA_GRID,
        "criterion": "mean XOR_DELAY d=5 FIXED",
        "tune_seeds": TUNE_SEEDS, "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_xor_cx.md",
        "spectral_audit": {},
    }
    for arm in arms:
        scores = {a: [] for a in ALPHA_GRID}
        fracs = []
        for seed in TUNE_SEEDS:
            t0 = time.time()
            W, info = build_wiring(arm, seed)
            frac, mean_ev = spectral_frac(W)
            fracs.append(dict(seed=seed, frac_gt_0_5=frac, mean_abs_eig=mean_ev))
            delay, dmax = delays_fixed(W, seed)
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
                    scores[alpha].append(float("nan")); continue
                B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
                p = K.pred(X[WASH + TR:], B)
                yt = y[WASH + TR:]
                if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
                    scores[alpha].append(float("nan")); continue
                nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
                scores[alpha].append(float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan"))
            print(f"tune {arm} seed={seed} n={W.shape[0]} frac>0.5={frac} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
        choice["alphas"][arm] = {"alpha": best, "means": means}
        choice["spectral_audit"][arm] = fracs
        print(f"LOCK alpha {arm} -> {best}", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, xor_d, alpha):
    t0 = time.time()
    W, info = build_wiring(arm, seed)
    delay, dmax = delays_fixed(W, seed)
    rng = np.random.default_rng(seed + 6 + 17)
    val = xor_delay_score(W, delay, dmax, LEAK, rng, info["drive"], alpha, d=xor_d)
    return dict(
        arm=arm, seed=int(seed), task="XOR_DELAY", xor_d=int(xor_d),
        delay_mode="FIXED", value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha),
        delay_mean=float(np.mean(delay)),
        sec=time.time() - t0, has_fastcore2=_HAS_FC2, source="xor_cx",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, xor_d, alpha = args
        return dict(arm=arm, seed=seed, task="XOR_DELAY", xor_d=xor_d,
                    delay_mode="FIXED", value=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-800:], source="xor_cx")


def _load_done():
    done = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if r.get("error"): continue
                done.add((r["arm"], r["seed"], r.get("xor_d")))
            except Exception:
                pass
    return done


def smoke():
    out = []
    for arm in ("CX", "ER", "EI_SHUF", "SPEC"):
        r = run_one(arm, 300, 5, 1e-6)
        out.append(r)
        print("SMOKE", {k: r.get(k) for k in ("arm", "value", "n", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")


def build_jobs(alphas, phase="all"):
    jobs = []
    if phase in ("primary", "all"):
        for arm in ALL_ARMS:
            a = alphas["alphas"][arm]["alpha"]
            for seed in SEEDS:
                jobs.append((arm, seed, XOR_D_PRIMARY, a))
    if phase in ("secondary", "all"):
        for arm in PRIMARY_ARMS:
            a = alphas["alphas"][arm]["alpha"]
            for seed in SEEDS:
                for d in XOR_D_SECONDARY:
                    jobs.append((arm, seed, d, a))
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
                print(r.get("arm"), r.get("seed"), r.get("xor_d"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r, default=float) + "\n"); f.flush()
                    print(r.get("arm"), r.get("seed"), r.get("xor_d"), r.get("value"),
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
    (OUT / "xor_cx.done").write_text("ok\n")
    print("xor_cx done", flush=True)


if __name__ == "__main__":
    main()

"""Few-shot + noise with matched delays. Prereg: PREREGISTRATION_fewshot_delay.md."""
from __future__ import annotations
import os, sys, json, time, math
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/regions/frankenstein")
sys.path.insert(0, "/workspace/flybrain")
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from pathlib import Path
import frankenstein as F
import fast_kernels as K

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
MYELIN = H / "data" / "human_myelin.npz"
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
WASH, MAX_TR, TE = 200, 3000, 500
N_TRAINS = [25, 50, 100, 200, 500, 1000, 3000]
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]

WIRING_ARMS = ["FLY_CX", "FRANKEN", "ER", "SHUF", "DENS"]
# full arm names = f"{wiring}_{NOD|DEL}"
def arm_name(wiring, delay_mode):
    return f"{wiring}_{delay_mode}"

ALL_ARMS = [arm_name(w, d) for w in WIRING_ARMS for d in ("NOD", "DEL")]
TASKS = ["MC", "NARMA", "PC_HARD"]


def patch_scale():
    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale


patch_scale()


def load_myelin():
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf_filled = mf.copy()
    mf_filled[~np.isfinite(mf_filled)] = med
    return mf_filled, med


def myelin_to_delay(mf, dmax=DMAX):
    mf = np.clip(mf, 0.0, 1.0)
    return (1 + np.rint((1.0 - mf) * (dmax - 1))).astype(np.int32)


_MF = None
def mf_filled():
    global _MF
    if _MF is None:
        _MF = load_myelin()
    return _MF


def human_range(info):
    names = info["names"]
    offs = info["offsets"]
    if "human" not in names:
        return None
    i = names.index("human")
    return int(offs[i]), int(offs[i + 1])


def build_wiring(wiring, seed, g):
    """wiring in FLY_CX|FRANKEN|ER|SHUF|DENS. ER/SHUF/DENS are frankenstein-sized."""
    if wiring == "FLY_CX":
        return F.build_arm("FLY_CX", seed, g)
    arm_map = {"FRANKEN": "FRANKEN", "ER": "ER", "SHUF": "SHUF", "DENS": "DENS_MATCHED"}
    return F.build_arm(arm_map[wiring], seed, g)


def delays_for(wiring, delay_mode, W, info, seed, mf):
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    if delay_mode == "NOD":
        return delay, DMAX
    g = np.random.default_rng(seed + 909)
    pre = Wcoo.col
    hr = human_range(info)
    if wiring == "FLY_CX" or hr is None:
        delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
        return delay, DMAX
    # frankenstein-sized: real myelin on human-pre + FIXED elsewhere (identical rule for ER/SHUF/DENS)
    h0, h1 = hr
    assert (h1 - h0) == len(mf), (h1 - h0, len(mf))
    d_neuron = myelin_to_delay(mf)
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


def run_plain(W, Win, b, U, leak):
    return K.run_reservoir_csr(W, Win, b, U, leak)


def rollout(W, delay, dmax, delay_mode, Win, b, U, leak):
    if delay_mode == "NOD":
        return run_plain(W, Win, b, U, leak)
    return run_delay_reservoir(W, delay, dmax, Win, b, U, leak)


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


def make_mc_u(rng, wash=WASH, tr=MAX_TR, te=TE):
    T = wash + tr + te
    return rng.uniform(-1, 1, T)


def make_narma(rng, wash=WASH, tr=MAX_TR, te=TE):
    T = wash + tr + te
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    return u, y


def score_mc(X, u, n_train, alpha, wash=WASH, tr=MAX_TR, te=TE, D=100):
    n_train = min(int(n_train), tr)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[wash:wash + n_train], Y[wash:wash + n_train], alpha)
    P = K.pred(X[wash + tr:], B)
    Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def score_narma(X, y, n_train, alpha, wash=WASH, tr=MAX_TR, te=TE):
    n_train = min(int(n_train), tr)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return float("nan")
    B = K.ridge(X[wash:wash + n_train], y[wash:wash + n_train], alpha)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def score_pc_hard(X, u_clean, n_train, alpha, wash=200, tr_end=1000):
    """X from noisy drive; targets are clean. Train window [wash, wash+n_train) clipped to [wash, tr_end)."""
    n_train = min(int(n_train), tr_end - wash)
    sl = slice(wash, wash + n_train)
    B = K.ridge(X[sl], u_clean[sl], alpha)
    pred = K.pred(X[tr_end:], B)
    yt = u_clean[tr_end:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


def participation_ratio(X):
    """PR of sample covariance; X is (T, N). Uses Gram if T < N."""
    Xc = X - X.mean(0, keepdims=True)
    T, N = Xc.shape
    if T < 2:
        return float("nan"), float("nan"), float("nan")
    if T <= N:
        G = (Xc @ Xc.T) / (T - 1)
        # eig of Gram; nonzero eigs match covariance
        ev = np.linalg.eigvalsh(G)
    else:
        C = (Xc.T @ Xc) / (T - 1)
        ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    if s < 1e-18:
        return 0.0, 0.0, 0.0
    pr = float((s * s) / (np.square(ev).sum() + 1e-30))
    order = np.sort(ev)[::-1]
    top10 = float(order[: min(10, len(order))].sum() / s)
    # spectral flatness: exp(mean log) / mean
    pos = order[order > 1e-18]
    if len(pos) == 0:
        flat = 0.0
    else:
        flat = float(np.exp(np.mean(np.log(pos))) / (pos.mean() + 1e-30))
    return pr, top10, flat


def parse_arm(arm):
    # FLY_CX_NOD / FRANKEN_DEL / DENS_DEL ...
    if arm.endswith("_NOD"):
        return arm[:-4], "NOD"
    if arm.endswith("_DEL"):
        return arm[:-4], "DEL"
    raise KeyError(arm)


def build_arm_full(arm, seed, task_tag=0):
    wiring, delay_mode = parse_arm(arm)
    g = np.random.default_rng(seed + 7 + task_tag)
    W, info = build_wiring(wiring, seed, g)
    mf, med = mf_filled()
    delay, dmax = delays_for(wiring, delay_mode, W, info, seed, mf)
    return W, info, delay, dmax, delay_mode, med, float(delay.mean())


def retention(score, intact, higher_better):
    if intact is None or score is None:
        return float("nan")
    if not (np.isfinite(score) and np.isfinite(intact)):
        return float("nan")
    if higher_better:
        if abs(intact) < 1e-12:
            return float("nan")
        return float(score / intact)
    if abs(score) < 1e-12:
        return 2.0
    return float(np.clip(intact / score, 0.0, 2.0))


# -------------------- ALPHA TUNE --------------------
def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("TUNE alphas arms", ALL_ARMS, "seeds", TUNE_SEEDS, flush=True)
    choice = {"alphas": {}, "grid": ALPHA_GRID, "criterion": "mean MC @ n_train=100",
              "tune_seeds": TUNE_SEEDS,
              "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for arm in ALL_ARMS:
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            W, info, delay, dmax, delay_mode, med, dmean = build_arm_full(arm, seed, task_tag=0)
            rng = np.random.default_rng(seed + 0)
            u = make_mc_u(rng)
            Win, b = masked_win(W.shape[0], 1, info["drive"], rng)
            X = rollout(W, delay, dmax, delay_mode, Win, b, u, LEAK)
            for a in ALPHA_GRID:
                val = score_mc(X, u, 100, a)
                scores[a].append(val)
                with open(OUT / "tune_alpha.jsonl", "a") as f:
                    f.write(json.dumps(dict(arm=arm, seed=seed, alpha=a, value=val,
                                            n_train=100, task="MC")) + "\n")
            print(f"tune {arm} seed={seed} done", flush=True)
        means = {a: float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[a] if np.isfinite(means[a]) else -1e99, -a))
        choice["alphas"][arm] = {"alpha": best, "means": means}
        print(f"LOCK alpha {arm} -> {best} means={means}", flush=True)
    cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


# -------------------- FEWSHOT + MECHANISM --------------------
def run_fewshot_one(arm, seed, alpha):
    t0 = time.time()
    rows = []
    for task in TASKS:
        tag = {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task]
        W, info, delay, dmax, delay_mode, med, dmean = build_arm_full(arm, seed, task_tag=tag)
        rng = np.random.default_rng(seed + tag)
        n = W.shape[0]
        drive = info["drive"]
        if task == "MC":
            u = make_mc_u(rng)
            Win, b = masked_win(n, 1, drive, rng)
            X = rollout(W, delay, dmax, delay_mode, Win, b, u, LEAK)
            for nt in N_TRAINS:
                val = score_mc(X, u, nt, alpha)
                rows.append(dict(
                    test="fewshot", arm=arm, seed=seed, task=task,
                    n_train=nt, value=val, alpha=alpha, n=n,
                    delay_mode=delay_mode, delay_mean=dmean,
                    myelin_median=med, sec=time.time() - t0,
                ))
            # mechanism from MC states
            pr, top10, flat = participation_ratio(X[WASH:WASH + MAX_TR])
            rows.append(dict(
                test="mechanism", arm=arm, seed=seed, task="MC",
                PR=pr, top10_frac=top10, spectral_flatness=flat,
                n=n, delay_mode=delay_mode, alpha=alpha, sec=time.time() - t0,
            ))
        elif task == "NARMA":
            u, y = make_narma(rng)
            if not np.all(np.isfinite(y)):
                for nt in N_TRAINS:
                    rows.append(dict(test="fewshot", arm=arm, seed=seed, task=task,
                                     n_train=nt, value=float("nan"), alpha=alpha, n=n,
                                     delay_mode=delay_mode, narma_nan=True, sec=time.time() - t0))
            else:
                Win, b = masked_win(n, 1, drive, rng)
                X = rollout(W, delay, dmax, delay_mode, Win, b, u, LEAK)
                for nt in N_TRAINS:
                    val = score_narma(X, y, nt, alpha)
                    rows.append(dict(
                        test="fewshot", arm=arm, seed=seed, task=task,
                        n_train=nt, value=val, alpha=alpha, n=n,
                        delay_mode=delay_mode, delay_mean=dmean, sec=time.time() - t0,
                    ))
        else:  # PC_HARD
            T = 1200
            theta = np.linspace(0, 8 * np.pi, T)
            u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
            idx = np.asarray(drive)
            rng2 = np.random.default_rng(int(rng.integers(1e9)))
            k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
            drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
            Win, b = masked_win(n, 2, drive_idx, rng)
            wash, tr_end = 200, 800
            # few-shot n_trains clipped to train window length 600
            pc_ns = [nt for nt in N_TRAINS if nt <= (tr_end - wash)] + ([tr_end - wash] if (tr_end - wash) not in N_TRAINS else [])
            pc_ns = sorted(set(pc_ns))
            u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
            X = rollout(W, delay, dmax, delay_mode, Win, b, u_drive, LEAK)
            for nt in pc_ns:
                val = score_pc_hard(X, u_clean, nt, alpha, wash=wash, tr_end=tr_end)
                rows.append(dict(
                    test="fewshot", arm=arm, seed=seed, task=task,
                    n_train=nt, value=val, alpha=alpha, n=n,
                    delay_mode=delay_mode, delay_mean=dmean, sec=time.time() - t0,
                ))
    return rows


# -------------------- NOISE --------------------
def run_noise_one(arm, seed, alpha):
    t0 = time.time()
    rows = []
    for task in ("MC", "NARMA"):
        tag = 0 if task == "MC" else 2
        W, info, delay, dmax, delay_mode, med, dmean = build_arm_full(arm, seed, task_tag=tag + 50)
        rng = np.random.default_rng(seed + tag)
        n = W.shape[0]
        drive = info["drive"]
        higher = task == "MC"
        if task == "MC":
            u = make_mc_u(rng)
            target = u
        else:
            u, y = make_narma(rng)
            target = y
        Win, b = masked_win(n, 1, drive, rng)
        # clean
        X0 = rollout(W, delay, dmax, delay_mode, Win, b, u, LEAK)
        if task == "MC":
            intact = score_mc(X0, target, MAX_TR, alpha)
        else:
            intact = score_narma(X0, target, MAX_TR, alpha)
        rets = []
        for sig in NOISE_SIGMAS:
            rng_n = np.random.default_rng(seed + 4000 + int(sig * 100) + tag)
            if sig == 0:
                val = intact
            else:
                u_noisy = u + sig * rng_n.standard_normal(u.shape)
                Xi = rollout(W, delay, dmax, delay_mode, Win, b, u_noisy, LEAK)
                if task == "MC":
                    val = score_mc(Xi, target, MAX_TR, alpha)  # clean target
                else:
                    val = score_narma(Xi, target, MAX_TR, alpha)
            ret = retention(val, intact, higher)
            if sig > 0:
                rets.append((sig, ret))
            rows.append(dict(
                test="noise", arm=arm, seed=seed, task=task,
                channel="input", sigma=sig, value=val, retention=ret,
                intact=intact, alpha=alpha, n=n, delay_mode=delay_mode,
                sec=time.time() - t0,
            ))
        xs = np.array([p[0] for p in rets], float) if rets else np.array([])
        ys = np.array([p[1] for p in rets], float) if rets else np.array([])
        auc = float(np.trapezoid(ys, xs)) if len(xs) and np.all(np.isfinite(ys)) else float("nan")
        mean_ret = float(np.nanmean(ys)) if len(ys) else float("nan")
        rows.append(dict(
            test="noise_auc", arm=arm, seed=seed, task=task,
            channel="input", auc=auc, mean_retention=mean_ret,
            intact=intact, alpha=alpha, n=n, delay_mode=delay_mode,
            sec=time.time() - t0,
        ))
    return rows


def _job_fewshot(args):
    arm, seed, alpha = args
    try:
        return run_fewshot_one(arm, seed, alpha)
    except Exception as e:
        import traceback
        return [dict(test="fewshot", arm=arm, seed=seed, error=str(e)[:400],
                     traceback=traceback.format_exc()[-600:])]


def _job_noise(args):
    arm, seed, alpha = args
    try:
        return run_noise_one(arm, seed, alpha)
    except Exception as e:
        import traceback
        return [dict(test="noise", arm=arm, seed=seed, error=str(e)[:400],
                     traceback=traceback.format_exc()[-600:])]


def _run_pool(jobs, worker_fn, jf, label):
    done = set()
    if jf.exists():
        for line in jf.open():
            try:
                r = json.loads(line)
                if r.get("error"):
                    continue
                if label == "fewshot" and r.get("test") == "fewshot" and r.get("task") == "MC" and r.get("n_train") == N_TRAINS[0]:
                    done.add((r["arm"], r["seed"]))
                elif label == "noise" and r.get("test") == "noise_auc" and r.get("task") == "MC":
                    done.add((r["arm"], r["seed"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"{label} jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
    with jf.open("a") as f:
        if workers <= 1 or len(jobs) <= 1:
            for j in jobs:
                rows = worker_fn(j)
                for r in rows:
                    f.write(json.dumps(r, default=float) + "\n")
                f.flush()
                print("done", label, j[0], j[1], flush=True)
        else:
            with Pool(workers) as pool:
                for rows in pool.imap_unordered(worker_fn, jobs):
                    for r in rows:
                        f.write(json.dumps(r, default=float) + "\n")
                    f.flush()
                    print("done", label, rows[0].get("arm"), rows[0].get("seed"),
                          rows[0].get("error", ""), flush=True)


def main():
    mode = os.environ.get("MODE", "full")  # tune | fewshot | noise | full
    choice = tune_alphas()
    alphas = {a: choice["alphas"][a]["alpha"] for a in ALL_ARMS}
    if mode == "tune":
        print("tune-only done", flush=True)
        return
    if mode in ("fewshot", "full"):
        jobs = [(a, s, alphas[a]) for a in ALL_ARMS for s in SEEDS]
        _run_pool(jobs, _job_fewshot, OUT / "fewshot.jsonl", "fewshot")
        (OUT / "fewshot.done").write_text("ok\n")
    if mode in ("noise", "full"):
        jobs = [(a, s, alphas[a]) for a in ALL_ARMS for s in SEEDS]
        _run_pool(jobs, _job_noise, OUT / "noise.jsonl", "noise")
        (OUT / "noise.done").write_text("ok\n")
    (OUT / "fewshot_delay.done").write_text("ok\n")
    print("fewshot_delay done", flush=True)


if __name__ == "__main__":
    main()

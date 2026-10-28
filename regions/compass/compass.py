"""Fly CX angular path integration (compass). Prereg: PREREGISTRATION_compass.md."""
from __future__ import annotations
import os, sys, json, time, math
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/compress")
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
DATA = H / "data"
JF = OUT / "runs.jsonl"
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1200, 400
SIGMA_OMEGA = 0.15
OU_TAU = 20.0
LANDMARK_ON, LANDMARK_OFF = 50, 150
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
WIRINGS = ["FLY_CX", "ER", "SHUF", "DENS"]
DELAYS = ["NOD", "DEL"]
TASKS = ["HEADING", "DRIFT", "GEN"]

def arm_name(w, d):
    return f"{w}_{d}"

ALL_ARMS = [arm_name(w, d) for w in WIRINGS for d in DELAYS]


def patch_scale():
    def _scale(W, rho=RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale


patch_scale()


def load_types():
    z = np.load(DATA / "fly_cx_types.npz", allow_pickle=True)
    pen = np.where(z["pen_mask"].astype(bool))[0]
    epg = np.where(z["epg_mask"].astype(bool))[0]
    return pen.astype(int), epg.astype(int)


PEN_IDX, EPG_IDX = load_types()


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


def rollout(W, delay, dmax, delay_mode, Win, b, U, leak):
    if delay_mode == "NOD":
        return K.run_reservoir_csr(W, Win, b, U, leak)
    return run_delay_reservoir(W, delay, dmax, Win, b, U, leak)


def build_wiring(wiring, seed, g):
    Wm, sign, layer, ct = F.make_block("fly_cx", "real", seed)
    if wiring == "FLY_CX":
        W = F.scale(Wm, RHO)
        kind = "real"
    elif wiring == "ER":
        W = F.scale(F.er_match(Wm, sign, g), RHO)
        kind = "er"
    elif wiring == "SHUF":
        W = F.scale(F.degree_shuffle(Wm, sign, g), RHO)
        kind = "shuf"
    elif wiring == "DENS":
        W = F.scale(F.dens_match(Wm, sign, g), RHO)
        kind = "dens"
    else:
        raise KeyError(wiring)
    return W, dict(N=W.shape[0], kind=kind, wiring=wiring)


def delays_for(delay_mode, W, seed):
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    if delay_mode == "NOD":
        return delay, DMAX
    g = np.random.default_rng(seed + 909)
    delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
    return delay, DMAX


def injection_sets(wiring, seed, N):
    """Real PEN/EPG for FLY_CX; random same-size sets for controls."""
    g = np.random.default_rng(seed + 4242)
    n_pen, n_epg = len(PEN_IDX), len(EPG_IDX)
    if wiring == "FLY_CX":
        return PEN_IDX.copy(), EPG_IDX.copy()
    pen = g.choice(N, size=n_pen, replace=False)
    # landmark set disjoint when possible
    remaining = np.setdiff1d(np.arange(N), pen, assume_unique=False)
    if len(remaining) >= n_epg:
        epg = g.choice(remaining, size=n_epg, replace=False)
    else:
        epg = g.choice(N, size=n_epg, replace=False)
    return pen.astype(int), epg.astype(int)


def landmark_mask(T, on=LANDMARK_ON, off=LANDMARK_OFF):
    m = np.zeros(T, dtype=np.float64)
    t = 0
    while t < T:
        m[t: min(T, t + on)] = 1.0
        t += on + off
    return m


def make_trajectory(rng, T, sigma=SIGMA_OMEGA, tau=OU_TAU):
    """OU angular velocity + integrated heading."""
    omega = np.zeros(T)
    x = 0.0
    a = math.exp(-1.0 / tau)
    s = sigma * math.sqrt(max(1e-12, 1.0 - a * a))
    for t in range(T):
        x = a * x + s * rng.standard_normal()
        omega[t] = x
    theta = np.cumsum(omega)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return omega, theta, cos_t, sin_t


def build_drive(omega, cos_t, sin_t, cue_mask, pen_idx, epg_idx, n, rng):
    """U is (T, 3): [omega, cue*cos, cue*sin]. Win maps to neurons."""
    T = len(omega)
    U = np.column_stack([omega, cue_mask * cos_t, cue_mask * sin_t])
    Win = np.zeros((n, 3), dtype=np.float64)
    # ω channel → PEN set
    w_pen = rng.uniform(-0.5, 0.5, len(pen_idx))
    Win[pen_idx, 0] = w_pen
    # landmark channels → EPG set
    w_epg = rng.uniform(-0.5, 0.5, (len(epg_idx), 2))
    Win[epg_idx, 1:] = w_epg
    b = rng.uniform(-0.1, 0.1, n)
    return U, Win, b


def circular_rmse(pred, cos_t, sin_t):
    """pred (T,2) → mean circular error vs (cos,sin)."""
    p = np.asarray(pred, dtype=np.float64)
    nrm = np.linalg.norm(p, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    p = p / nrm
    dot = np.clip(p[:, 0] * cos_t + p[:, 1] * sin_t, -1.0, 1.0)
    err = np.arccos(dot)
    # zero / nonfinite predictions → π
    bad = ~np.isfinite(err) | ~np.isfinite(nrm[:, 0])
    err = err.copy()
    err[bad] = math.pi
    return float(np.mean(err))


def score_heading(X, cos_t, sin_t, alpha, wash=WASH, tr=TR, te=TE):
    Y = np.column_stack([cos_t, sin_t])
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
    pred = K.pred(X[wash + tr: wash + tr + te], B)
    return circular_rmse(pred, cos_t[wash + tr: wash + tr + te], sin_t[wash + tr: wash + tr + te])


def parse_arm(arm):
    for d in DELAYS:
        suf = "_" + d
        if arm.endswith(suf):
            return arm[: -len(suf)], d
    raise KeyError(arm)


def build_arm_full(arm, seed):
    wiring, delay_mode = parse_arm(arm)
    g = np.random.default_rng(seed + 7)
    # F.make_block already loads from frankenstein DATA — symlink or set path
    # ensure frankenstein finds fly_cx_module: we copied to compass/data but
    # frankenstein loads from its own data/. Use frankenstein's copy (identical).
    W, info = build_wiring(wiring, seed, g)
    delay, dmax = delays_for(delay_mode, W, seed)
    pen, epg = injection_sets(wiring, seed, W.shape[0])
    return W, info, delay, dmax, delay_mode, pen, epg


def run_task(arm, seed, task, alpha):
    t0 = time.time()
    W, info, delay, dmax, delay_mode, pen, epg = build_arm_full(arm, seed)
    n = W.shape[0]
    T = WASH + TR + TE
    rng = np.random.default_rng(seed + {"HEADING": 11, "DRIFT": 22, "GEN": 33}[task])

    if task == "HEADING":
        omega, theta, cos_t, sin_t = make_trajectory(rng, T, sigma=SIGMA_OMEGA)
        cue = landmark_mask(T)
        U, Win, b = build_drive(omega, cos_t, sin_t, cue, pen, epg, n, rng)
        X = rollout(W, delay, dmax, delay_mode, Win, b, U, LEAK)
        val = score_heading(X, cos_t, sin_t, alpha)
    elif task == "DRIFT":
        # train with landmarks, test without: build one long traj; cue off on test
        omega, theta, cos_t, sin_t = make_trajectory(rng, T, sigma=SIGMA_OMEGA)
        cue = landmark_mask(T)
        cue[WASH + TR:] = 0.0  # test: landmarks OFF
        U, Win, b = build_drive(omega, cos_t, sin_t, cue, pen, epg, n, rng)
        X = rollout(W, delay, dmax, delay_mode, Win, b, U, LEAK)
        val = score_heading(X, cos_t, sin_t, alpha)
    elif task == "GEN":
        # train at σ, test at 2σ — two phases: generate train traj then test traj
        # Simpler fair protocol: single traj with train portion at σ and test at 2σ
        # by rescaling omega after train boundary and rebuilding theta for test only.
        omega_tr, _, cos_tr, sin_tr = make_trajectory(rng, WASH + TR, sigma=SIGMA_OMEGA)
        omega_te, _, cos_te, sin_te = make_trajectory(
            np.random.default_rng(seed + 99), TE, sigma=2.0 * SIGMA_OMEGA
        )
        omega = np.concatenate([omega_tr, omega_te])
        cos_t = np.concatenate([cos_tr, cos_te])
        sin_t = np.concatenate([sin_tr, sin_te])
        # continuous heading across boundary: rotate test heading to match train end
        # (avoid artificial jump). Align first test angle to last train angle.
        if len(cos_tr):
            ang0 = math.atan2(sin_tr[-1], cos_tr[-1])
            ang_te0 = math.atan2(sin_te[0], cos_te[0])
            dang = ang0 - ang_te0
            c, s = math.cos(dang), math.sin(dang)
            cos_te2 = c * cos_te - s * sin_te
            sin_te2 = s * cos_te + c * sin_te
            cos_t = np.concatenate([cos_tr, cos_te2])
            sin_t = np.concatenate([sin_tr, sin_te2])
        cue = landmark_mask(T)
        U, Win, b = build_drive(omega, cos_t, sin_t, cue, pen, epg, n, rng)
        X = rollout(W, delay, dmax, delay_mode, Win, b, U, LEAK)
        val = score_heading(X, cos_t, sin_t, alpha)
    else:
        raise KeyError(task)

    return dict(
        arm=arm, seed=seed, task=task, value=float(val) if np.isfinite(val) else float("nan"),
        alpha=float(alpha), n=int(n), sec=time.time() - t0,
        delay_mode=delay_mode, wiring=info["wiring"], kind=info["kind"],
        n_pen=int(len(pen)), n_epg=int(len(epg)),
        delay_mean=float(delay.mean()), source="compass",
    )


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("TUNE alphas arms", ALL_ARMS, "seeds", TUNE_SEEDS, flush=True)
    choice = {
        "alphas": {}, "grid": ALPHA_GRID,
        "criterion": "argmin mean HEADING circular RMSE",
        "tune_seeds": TUNE_SEEDS,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for arm in ALL_ARMS:
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            # one rollout, score all alphas
            W, info, delay, dmax, delay_mode, pen, epg = build_arm_full(arm, seed)
            n = W.shape[0]
            T = WASH + TR + TE
            rng = np.random.default_rng(seed + 11)
            omega, theta, cos_t, sin_t = make_trajectory(rng, T, sigma=SIGMA_OMEGA)
            cue = landmark_mask(T)
            U, Win, b = build_drive(omega, cos_t, sin_t, cue, pen, epg, n, rng)
            t0 = time.time()
            X = rollout(W, delay, dmax, delay_mode, Win, b, U, LEAK)
            for a in ALPHA_GRID:
                val = score_heading(X, cos_t, sin_t, a)
                scores[a].append(val)
                with open(OUT / "tune_alpha.jsonl", "a") as f:
                    f.write(json.dumps(dict(arm=arm, seed=seed, alpha=a, value=val,
                                            task="HEADING")) + "\n")
            print(f"tune {arm} seed={seed} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = min(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else 1e99, a))
        choice["alphas"][arm] = {"alpha": best, "means": means}
        print(f"LOCKED alpha {arm}={best} means={means}", flush=True)
    cache.write_text(json.dumps(choice, indent=2))
    return choice


def _job(args):
    arm, seed, task, alpha = args
    try:
        return run_task(arm, seed, task, alpha)
    except Exception as e:
        import traceback
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-600:],
                    source="compass")


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "tune":
        tune_alphas()
        (OUT / "tune.done").write_text("ok\n")
        return
    alphas = tune_alphas()
    only = os.environ.get("ONLY_ARMS", "")
    arms = [a.strip() for a in only.split(",") if a.strip()] or ALL_ARMS
    only_t = os.environ.get("ONLY_TASKS", "")
    tasks = [t.strip() for t in only_t.split(",") if t.strip()] or TASKS
    seeds = SEEDS
    if os.environ.get("SMOKE_SEED") is not None:
        seeds = [int(os.environ["SMOKE_SEED"])]
    jobs = []
    for arm in arms:
        a = alphas["alphas"][arm]["alpha"]
        for s in seeds:
            for t in tasks:
                jobs.append((arm, s, t, a))
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
    print(f"compass jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
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
    (OUT / "compass.done").write_text("ok\n")
    print("compass done", flush=True)


if __name__ == "__main__":
    main()

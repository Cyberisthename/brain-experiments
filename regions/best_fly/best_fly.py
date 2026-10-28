"""Smartest-fly stack. Prereg: PREREGISTRATION_best_fly.md. Tune 90-92 then eval 0-7."""
from __future__ import annotations
import os, sys, json, time
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
JF = OUT / "runs.jsonl"
MYELIN = H / "data" / "human_myelin.npz"
SEEDS = list(range(8))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5

# Fixed eval arms (always run)
FIXED_ARMS = ["FLY_CX_PLAIN", "FLY_CX_FIXED", "FRANKEN_REALMYELIN"]
# BEST + controls filled after tune
BEST_ARMS = ["BEST", "ER_BEST", "SHUF_BEST", "DENS_BEST"]
TASKS = ["MC", "NARMA", "PC_HARD"]

CANDIDATES = [
    "fly_cx_fixed",
    "franken_fixed",
    "franken_realmyelin",
    "franken_realmyelin_only",
]


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


def patch_scale():
    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale


patch_scale()


def human_range(info):
    names = info["names"]
    offs = info["offsets"]
    if "human" not in names:
        return None
    i = names.index("human")
    return int(offs[i]), int(offs[i + 1])


def build_wiring(kind, seed, g):
    """kind: fly_cx | franken | er | shuf | dens  (size follows franken blocks unless fly_cx)."""
    if kind == "fly_cx":
        return F.build_arm("FLY_CX", seed, g)
    arm_map = {"franken": "FRANKEN", "er": "ER", "shuf": "SHUF", "dens": "DENS_MATCHED"}
    return F.build_arm(arm_map[kind], seed, g)


def delays_for(recipe, W, info, seed, mf_filled):
    """recipe in CANDIDATES or plain/fixed/realmyelin labels."""
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    g = np.random.default_rng(seed + 909)
    pre = Wcoo.col
    hr = human_range(info)

    if recipe in ("plain", "FLY_CX_PLAIN"):
        return delay, DMAX

    if recipe in ("fixed", "fly_cx_fixed", "franken_fixed", "FLY_CX_FIXED"):
        delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
        return delay, DMAX

    # real myelin variants
    if hr is None:
        # no human block — fall back to FIXED
        delay[:] = g.integers(1, DMAX + 1, n_e).astype(np.int32)
        return delay, DMAX
    h0, h1 = hr
    assert (h1 - h0) == len(mf_filled), (h1 - h0, len(mf_filled))
    d_neuron = myelin_to_delay(mf_filled)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]

    if recipe in ("franken_realmyelin", "FRANKEN_REALMYELIN", "realmyelin"):
        # FIXED on non-human-pre
        other = ~mask
        delay[other] = g.integers(1, DMAX + 1, int(other.sum())).astype(np.int32)
    # franken_realmyelin_only: leave non-human at 1
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


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak, seed=0):
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


def mc_task(W, delay, dmax, leak, rng, in_idx, seed):
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak, seed=seed)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = K.pred(X[wash + tr:], B)
    Yt = Y[wash + tr:]
    return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))


def narma_task(W, delay, dmax, leak, rng, in_idx, seed):
    wash, tr, te = 200, 1500, 500
    T = wash + tr + te
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    if not np.all(np.isfinite(y)):
        return float("nan")
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, b, u, leak, seed=seed)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6)
    p = K.pred(X[wash + tr:], B)
    yt = y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def pc_hard_task(W, delay, dmax, leak, rng, in_idx, seed):
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
    X = run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak, seed=seed)
    Wout = np.linalg.lstsq(X[wash:tr], u_clean[wash:tr], rcond=None)[0]
    pred = X[tr:] @ Wout
    yt = u_clean[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


def arm_to_spec(arm, best_choice):
    """Return (wiring_kind, delay_recipe)."""
    if arm == "FLY_CX_PLAIN":
        return "fly_cx", "plain"
    if arm == "FLY_CX_FIXED":
        return "fly_cx", "fixed"
    if arm == "FRANKEN_REALMYELIN":
        return "franken", "franken_realmyelin"
    if arm == "BEST":
        c = best_choice["candidate"]
        if c.startswith("fly_cx"):
            return "fly_cx", c
        return "franken", c
    if arm in ("ER_BEST", "SHUF_BEST", "DENS_BEST"):
        c = best_choice["candidate"]
        wiring = {"ER_BEST": "er", "SHUF_BEST": "shuf", "DENS_BEST": "dens"}[arm]
        if c.startswith("fly_cx"):
            # fly-sized controls: build fly_cx then replace with random of same size
            return "fly_" + wiring, c  # special
        return wiring, c
    raise KeyError(arm)


def build_fly_control(ctrl, seed, g):
    """fly-sized ER/SHUF/DENS from fly_cx module."""
    Wm, sign, layer, ct = F.make_block("fly_cx", "real", seed)
    if ctrl == "er":
        W = F.er_match(Wm, sign, g)
    elif ctrl == "shuf":
        W = F.degree_shuffle(Wm, sign, g)
    elif ctrl == "dens":
        W = F.dens_match(Wm, sign, g)
    else:
        raise KeyError(ctrl)
    W = F.scale(W, F.RHO)
    L4, L5, _ = F.layer_sets(layer, W.shape[0])
    return W, dict(names=["fly_cx"], sizes=[W.shape[0]], offsets=[0, W.shape[0]],
                   N=W.shape[0], drive=L4, bridge={}, fly_offset=0,
                   kinds={"fly_cx": ctrl})


_MF = None

def mf_filled():
    global _MF
    if _MF is None:
        _MF = load_myelin()
    return _MF


def run_one(arm, seed, task, best_choice):
    t0 = time.time()
    mf, med = mf_filled()
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
    wiring, recipe = arm_to_spec(arm, best_choice)
    if wiring.startswith("fly_") and wiring != "fly_cx":
        ctrl = wiring.split("_", 1)[1]  # er/shuf/dens
        W, info = build_fly_control(ctrl, seed, g)
    else:
        W, info = build_wiring(wiring, seed, g)
    delay, dmax = delays_for(recipe, W, info, seed, mf)
    in_idx = info["drive"]
    if task == "MC":
        val = mc_task(W, delay, dmax, LEAK, g, in_idx, seed)
    elif task == "NARMA":
        val = narma_task(W, delay, dmax, LEAK, g, in_idx, seed)
    else:
        val = pc_hard_task(W, delay, dmax, LEAK, g, in_idx, seed)
    out = dict(
        arm=arm, seed=seed, task=task,
        value=(None if (isinstance(val, float) and val != val) else float(val)),
        n=int(W.shape[0]), sec=time.time() - t0,
        wiring=wiring, recipe=recipe,
        best_candidate=best_choice.get("candidate"),
        myelin_median=med,
        delay_mean=float(delay.mean()),
        source="best_fly",
    )
    if isinstance(val, float) and val != val:
        out["value"] = float("nan")
        out["narma_nan"] = True
    return out


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, best_choice = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-600:],
                    source="best_fly")


def score_candidate(cand_scores):
    """cand_scores: dict task -> list of values."""
    mc = float(np.nanmean(cand_scores.get("MC", [np.nan])))
    na = float(np.nanmean(cand_scores.get("NARMA", [np.nan])))
    pc = float(np.nanmean(cand_scores.get("PC_HARD", [np.nan])))
    if not np.isfinite(mc) or not np.isfinite(na) or not np.isfinite(pc):
        return float("-inf"), dict(MC=mc, NARMA=na, PC_HARD=pc, S=float("-inf"))
    S = mc / 10.0 - na + pc
    return S, dict(MC=mc, NARMA=na, PC_HARD=pc, S=S)


def tune_best():
    cache = OUT / "best_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print("TUNE start candidates", CANDIDATES, "seeds", TUNE_SEEDS, flush=True)
    results = {c: {"MC": [], "NARMA": [], "PC_HARD": []} for c in CANDIDATES}
    # temporary best_choice stub so arm_to_spec works during tune via direct recipe
    for cand in CANDIDATES:
        for seed in TUNE_SEEDS:
            for task in TASKS:
                # map candidate → arm-like run without BEST indirection
                if cand == "fly_cx_fixed":
                    wiring, recipe = "fly_cx", "fly_cx_fixed"
                elif cand == "franken_fixed":
                    wiring, recipe = "franken", "franken_fixed"
                elif cand == "franken_realmyelin":
                    wiring, recipe = "franken", "franken_realmyelin"
                else:
                    wiring, recipe = "franken", "franken_realmyelin_only"
                t0 = time.time()
                mf, med = mf_filled()
                g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
                W, info = build_wiring(wiring, seed, g)
                delay, dmax = delays_for(recipe, W, info, seed, mf)
                in_idx = info["drive"]
                if task == "MC":
                    val = mc_task(W, delay, dmax, LEAK, g, in_idx, seed)
                elif task == "NARMA":
                    val = narma_task(W, delay, dmax, LEAK, g, in_idx, seed)
                else:
                    val = pc_hard_task(W, delay, dmax, LEAK, g, in_idx, seed)
                results[cand][task].append(float(val) if val == val else float("nan"))
                print(f"tune {cand} seed={seed} {task}={val} {time.time()-t0:.1f}s", flush=True)
                with open(OUT / "tune_grid.jsonl", "a") as f:
                    f.write(json.dumps(dict(candidate=cand, seed=seed, task=task,
                                            value=val, sec=time.time()-t0)) + "\n")
    scored = {}
    for cand in CANDIDATES:
        S, detail = score_candidate(results[cand])
        scored[cand] = detail
        print(f"tune score {cand}: S={S:.4f} {detail}", flush=True)
    # pick max S; tie-break: prefer fly, then fewer myelin rules
    order = sorted(
        CANDIDATES,
        key=lambda c: (
            -scored[c]["S"],
            0 if c.startswith("fly") else 1,
            0 if "only" not in c and "realmyelin" not in c else (1 if "only" in c else 2),
        ),
    )
    pick = order[0]
    choice = dict(
        candidate=pick,
        scores=scored,
        tune_seeds=TUNE_SEEDS,
        composite="MC/10 - NARMA + PC_HARD",
        locked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    cache.write_text(json.dumps(choice, indent=2))
    print("LOCKED BEST", pick, flush=True)
    return choice


def main():
    mode = os.environ.get("MODE", "full")  # tune | full
    if mode == "tune":
        tune_best()
        (OUT / "tune.done").write_text("ok\n")
        return
    best_choice = tune_best()
    arms = FIXED_ARMS + BEST_ARMS
    only = os.environ.get("ONLY_ARMS", "")
    if only:
        arms = [a.strip() for a in only.split(",") if a.strip()]
    jobs = [(a, s, t, best_choice) for a in arms for s in SEEDS for t in TASKS]
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
    print(f"best_fly jobs={len(jobs)} skip={len(done)} workers={workers} best={best_choice['candidate']}", flush=True)
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
    (OUT / "best_fly.done").write_text("ok\n")
    print("best_fly done", flush=True)


if __name__ == "__main__":
    main()

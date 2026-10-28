"""H01 myelin REAL vs SHUF replication. Prereg: PREREGISTRATION_myelin_real_rep.md."""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
H = Path(__file__).resolve().parent
_ROOT = H.parents[1]
sys.path.insert(0, str(_ROOT / "compress"))
sys.path.insert(0, str(_ROOT / "regions" / "empn"))
sys.path.insert(0, str(_ROOT / "regions" / "hybrid_human"))
sys.path.insert(0, str(_ROOT / "regions" / "myelin_real"))
sys.path.insert(0, str(_ROOT))
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import hybrid_human as HH
import fast_kernels as K
import myelin_real as MR  # DelayReservoirFast, run_delay_reservoir, masked_win

OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
_DATA = _ROOT / "regions" / "hybrid_human" / "data"
MYELIN = _DATA / "human_myelin.npz"
HUMAN = _DATA / "human_module.npz"
SRC_MY = _DATA / "SOURCE_myelin.json"

EVAL_SEEDS = list(range(100, 124))  # 24 new
TUNE_SEEDS = [90, 91, 92]
LEAKS = [0.1, 0.2, 0.3]
PRIMARY_LEAK = 0.2
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
ARMS = ["NODELAY", "REAL_MYELIN", "SHUF_MYELIN", "LAYER_SHUF", "EI_SHUF"]
SHUF_ARMS = {"SHUF_MYELIN", "LAYER_SHUF", "EI_SHUF"}
N_SHUF = 3
TASKS = ["MC", "NARMA", "PC_HARD"]
NET = "HUMAN_LAYERED"

# Short end-to-end smoke (Kaggle/CI): MYELIN_SMOKE=1
if os.environ.get("MYELIN_SMOKE") == "1":
    EVAL_SEEDS = [100]
    TUNE_SEEDS = [90]
    LEAKS = [0.2]
    ARMS = ["NODELAY", "REAL_MYELIN"]
    N_SHUF = 1
    TASKS = ["NARMA"]
    ALPHA_GRID = [1e-4, 1e-2]  # smaller tune grid for smoke


def load_maps():
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf_filled = mf.copy()
    mf_filled[~np.isfinite(mf_filled)] = med
    h = np.load(HUMAN, allow_pickle=True)
    layer = h["layer"].astype(str)
    sign = h["sign"].astype(np.int8)
    assert len(mf_filled) == len(layer) == len(sign)
    return mf_filled, med, layer, sign


def myelin_to_delay(mf, dmax=DMAX):
    mf = np.clip(mf, 0.0, 1.0)
    return (1 + np.rint((1.0 - mf) * (dmax - 1))).astype(np.int32)


def shuffle_mf(arm, mf, layer, sign, seed, shuffle_id):
    """Return shuffled copy of mf for control arms; REAL/NODELAY return copy."""
    out = mf.copy()
    if arm == "NODELAY" or arm == "REAL_MYELIN":
        return out
    g = np.random.default_rng(seed + 909 + 10007 * int(shuffle_id) + {
        "SHUF_MYELIN": 0, "LAYER_SHUF": 17, "EI_SHUF": 41
    }[arm])
    if arm == "SHUF_MYELIN":
        g.shuffle(out)
        return out
    if arm == "LAYER_SHUF":
        for ly in np.unique(layer):
            idx = np.where(layer == ly)[0]
            if len(idx) > 1:
                vals = out[idx].copy()
                g.shuffle(vals)
                out[idx] = vals
        return out
    if arm == "EI_SHUF":
        for s in (+1, -1):
            idx = np.where(sign == s)[0]
            if len(idx) > 1:
                vals = out[idx].copy()
                g.shuffle(vals)
                out[idx] = vals
        return out
    raise ValueError(arm)


def build_net(seed, g):
    W, info = HH.build_arm("HUMAN_LAYERED", seed, g)
    return W, info


def delays_for(arm, W, info, mf_use):
    Wcoo = W.tocoo()
    n_e = Wcoo.nnz
    delay = np.ones(n_e, dtype=np.int32)
    if arm == "NODELAY":
        return delay, DMAX
    h0, h1 = 0, int(info["Nm"])
    assert (h1 - h0) == len(mf_use)
    pre = Wcoo.col
    d_neuron = myelin_to_delay(mf_use)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    return delay, DMAX


def drive_idx(info):
    if "L4" in info and len(info["L4"]):
        return np.asarray(info["L4"])
    return np.arange(info["Nm"])


def states_mc(W, delay, dmax, leak, rng, in_idx, seed):
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, rng)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=False, seed=seed)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    return X, Y, wash, tr, te, "mc"


def states_narma(W, delay, dmax, leak, rng, in_idx, seed):
    wash, tr, te = 200, 1500, 500
    T = wash + tr + te
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    if not np.all(np.isfinite(y)):
        return None, None, wash, tr, te, "narma_bad"
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, rng)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=False, seed=seed)
    if not np.all(np.isfinite(X)):
        return None, None, wash, tr, te, "narma_bad"
    return X, y, wash, tr, te, "narma"


def states_pc(W, delay, dmax, leak, rng, in_idx, seed):
    n = W.shape[0]
    T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = np.asarray(in_idx)
    rng2 = np.random.default_rng(int(rng.integers(1e9)))
    k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
    drive = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
    Win, b = MR.masked_win(n, 2, drive, rng)
    wash, tr = 200, 800
    te = T - tr
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak, adapt=False, seed=seed)
    return X, u_clean, wash, tr, te, "pc"


def score_with_alpha(task, X, Y, wash, tr, te, alpha):
    if task == "MC":
        B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
        P = K.pred(X[wash + tr:], B)
        Yt = Y[wash + tr:]
        D = Yt.shape[1]
        return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
    if task == "NARMA":
        B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
        p = K.pred(X[wash + tr:], B)
        yt = Y[wash + tr:]
        if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
            return float("nan")
        v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
        return v if np.isfinite(v) else float("nan")
    # PC_HARD: lstsq / ridge on 2-d; use ridge with alpha for consistency
    B = K.ridge(X[wash:tr], Y[wash:tr], alpha)
    pred = K.pred(X[tr:], B)
    yt = Y[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))


_MAPS = None
def maps():
    global _MAPS
    if _MAPS is None:
        _MAPS = load_maps()
    return _MAPS


def run_states(arm, seed, task, leak, shuffle_id):
    mf, med, layer, sign = maps()
    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
    W, info = build_net(seed, g)
    mf_use = shuffle_mf(arm, mf, layer, sign, seed, shuffle_id)
    delay, dmax = delays_for(arm, W, info, mf_use)
    in_idx = drive_idx(info)
    if task == "MC":
        return states_mc(W, delay, dmax, leak, g, in_idx, seed) + (med, delay)
    if task == "NARMA":
        return states_narma(W, delay, dmax, leak, g, in_idx, seed) + (med, delay)
    return states_pc(W, delay, dmax, leak, g, in_idx, seed) + (med, delay)


def tune_job(args):
    arm, leak, seed = args
    shuffle_id = 0
    t0 = time.time()
    X, Y, wash, tr, te, kind, med, delay = run_states(arm, seed, "NARMA", leak, shuffle_id)
    if X is None:
        return arm, leak, seed, {a: float("nan") for a in ALPHA_GRID}, time.time() - t0
    scores = {a: score_with_alpha("NARMA", X, Y, wash, tr, te, a) for a in ALPHA_GRID}
    return arm, leak, seed, scores, time.time() - t0


def eval_job(args):
    arm, seed, task, leak, shuffle_id, alpha = args
    t0 = time.time()
    try:
        X, Y, wash, tr, te, kind, med, delay = run_states(arm, seed, task, leak, shuffle_id)
        if X is None:
            val = float("nan")
        else:
            val = score_with_alpha(task, X, Y, wash, tr, te, alpha)
        return dict(
            net=NET, arm=arm, seed=seed, task=task, leak=leak,
            shuffle_id=int(shuffle_id), alpha=float(alpha),
            value=(None if (isinstance(val, float) and (math.isnan(val) or math.isinf(val))) else float(val)),
            n=int(X.shape[1]) if X is not None else None,
            sec=time.time() - t0, source="myelin_real_rep",
            flags={"myelin_median_fill": med, "dmax": DMAX,
                   "delay_mean": float(np.mean(delay)) if delay is not None else None},
        )
    except Exception as e:
        import traceback
        return dict(net=NET, arm=arm, seed=seed, task=task, leak=leak,
                    shuffle_id=int(shuffle_id), alpha=float(alpha),
                    value=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-600:],
                    sec=time.time() - t0, source="myelin_real_rep")


def shuf_ids_for(arm):
    return list(range(N_SHUF)) if arm in SHUF_ARMS else [0]


def main():
    if not MYELIN.exists() or not SRC_MY.exists():
        print("MISSING myelin artifacts", flush=True); sys.exit(2)
    cov = json.loads(SRC_MY.read_text())["coverage"]
    if cov["n_with_axon_annotation"] < 100:
        print("GATE FAIL", cov, flush=True); sys.exit(3)
    (OUT / "coverage_gate.json").write_text(json.dumps(cov, indent=2))

    workers = int(os.environ.get("WORKERS", "2"))
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        locked = json.loads(cache.read_text())
        alphas = {tuple(k.split("|")): v for k, v in locked["alphas"].items()}
        # keys "ARM|LEAK"
        alphas = {(a, float(lk)): v for (a, lk), v in alphas.items()}
        print("LOADED alphas", alphas, flush=True)
    else:
        print("TUNE start", flush=True)
        jobs = [(arm, leak, seed) for arm in ARMS for leak in LEAKS for seed in TUNE_SEEDS]
        acc = {(arm, leak): {a: [] for a in ALPHA_GRID} for arm in ARMS for leak in LEAKS}
        with Pool(workers) as P:
            for arm, leak, seed, scores, sec in P.imap_unordered(tune_job, jobs):
                for a, v in scores.items():
                    if v == v:
                        acc[(arm, leak)][a].append(v)
                print("tune", arm, leak, seed, {k: round(scores[k], 4) for k in ALPHA_GRID}, f"{sec:.1f}s", flush=True)
        alphas = {}
        for arm in ARMS:
            for leak in LEAKS:
                means = {a: (float(np.mean(acc[(arm, leak)][a])) if acc[(arm, leak)][a] else float("inf"))
                         for a in ALPHA_GRID}
                best = min(ALPHA_GRID, key=lambda a: means[a])
                alphas[(arm, leak)] = best
        payload = {
            "alphas": {f"{a}|{lk}": alphas[(a, lk)] for a in ARMS for lk in LEAKS},
            "grid": ALPHA_GRID, "tune_seeds": TUNE_SEEDS,
            "criterion": "argmin mean NARMA on tune seeds; shuffle_id=0 for shuffle arms",
            "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        cache.write_text(json.dumps(payload, indent=2))
        print("LOCKED", payload["alphas"], flush=True)

    done = set()
    if JF.exists():
        for line in open(JF):
            r = json.loads(line)
            done.add((r["arm"], r["seed"], r["task"], float(r["leak"]), int(r.get("shuffle_id", 0))))
    jobs = []
    for arm in ARMS:
        for leak in LEAKS:
            alpha = alphas[(arm, leak)]
            for seed in EVAL_SEEDS:
                for task in TASKS:
                    for sid in shuf_ids_for(arm):
                        key = (arm, seed, task, float(leak), sid)
                        if key not in done:
                            jobs.append((arm, seed, task, leak, sid, alpha))
    print(f"eval jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
    with Pool(workers) as P:
        for r in P.imap_unordered(eval_job, jobs):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r.get("arm"), r.get("leak"), r.get("task"), r.get("seed"),
                  r.get("shuffle_id"), r.get("value"), f"{r.get('sec', 0):.1f}s",
                  r.get("error", ""), flush=True)
    (OUT / "myelin_real_rep.done").write_text("ok\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

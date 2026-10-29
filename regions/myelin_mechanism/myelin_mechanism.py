"""Myelin mechanism rules on HUMAN_LAYERED. Prereg: PREREG_myelin_mechanism.md."""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
H = Path(__file__).resolve().parent
_ROOT = H.parents[1]
sys.path.insert(0, str(_ROOT / "compress"))
sys.path.insert(0, str(_ROOT / "regions" / "hybrid_human"))
sys.path.insert(0, str(_ROOT / "regions" / "myelin_real"))
sys.path.insert(0, str(_ROOT))
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from collections import deque
from scipy import stats
import hybrid_human as HH
import fast_kernels as K
import myelin_real as MR

OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"; DATA.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
_DATA = _ROOT / "regions" / "hybrid_human" / "data"
MYELIN = _DATA / "human_myelin.npz"
HUMAN = _DATA / "human_module.npz"

EVAL_SEEDS = list(range(800, 824))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
RULE_ARMS = ["RULE_LAYER", "RULE_DEGREE", "RULE_EI_LAYER", "RULE_FF_DEPTH"]
ARMS = ["REAL", "SHUF", "NODELAY"] + RULE_ARMS
TASKS = ["NARMA", "MC", "PC_HARD"]
PRIMARY_TASK = "NARMA"
WASH, TR, TE = 200, 1500, 500

_CACHE = {}


def load_maps():
    if "maps" in _CACHE:
        return _CACHE["maps"]
    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf = np.where(np.isfinite(mf), mf, med)
    h = np.load(HUMAN, allow_pickle=True)
    layer = h["layer"].astype(str)
    sign = h["sign"].astype(np.int8)
    W = sp.csr_matrix((h["W_data"], h["W_indices"], h["W_indptr"]), shape=tuple(h["W_shape"]))
    outdeg = np.asarray(W.sum(1)).ravel()
    indeg = np.asarray(W.sum(0)).ravel()
    L4 = np.where(layer == "L4")[0]
    N = W.shape[0]
    depth = np.full(N, -1, dtype=np.int32)
    depth[L4] = 0
    q = deque(L4.tolist())
    while q:
        u = q.popleft()
        for v in W.indices[W.indptr[u]:W.indptr[u + 1]]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                q.append(int(v))
    if (depth >= 0).any():
        depth[depth < 0] = int(depth[depth >= 0].max()) + 1
    else:
        depth[:] = 0
    rho_in = float(stats.spearmanr(mf, indeg).statistic)
    rho_dep = float(stats.spearmanr(mf, depth).statistic)
    maps = dict(mf=mf, med=med, layer=layer, sign=sign, outdeg=outdeg, indeg=indeg,
                depth=depth, rho_indeg=rho_in, rho_depth=rho_dep)
    _CACHE["maps"] = maps
    return maps


def myelin_to_delay(mf, dmax=DMAX):
    mf = np.clip(mf, 0.0, 1.0)
    return (1 + np.rint((1.0 - mf) * (dmax - 1))).astype(np.int32)


def isotonic_assign(rank_key, mf_real, positive_corr, g):
    """Assign sorted mf to neurons sorted by rank_key (or reverse if corr negative)."""
    N = len(mf_real)
    order = np.argsort(rank_key, kind="mergesort")
    # break ties with RNG permutation of equal keys
    # stable sort is fine; add tiny noise
    key = rank_key.astype(float) + g.uniform(-1e-9, 1e-9, N)
    order = np.argsort(key, kind="mergesort")
    sorted_mf = np.sort(mf_real)
    out = np.empty(N, dtype=np.float64)
    if positive_corr:
        out[order] = sorted_mf
    else:
        out[order] = sorted_mf[::-1]
    return out


def assign_mf(arm, seed):
    m = load_maps()
    mf = m["mf"].copy()
    N = len(mf)
    g = np.random.default_rng(seed + 4242 + hash(arm) % 100000)
    if arm == "REAL" or arm == "NODELAY":
        return mf
    if arm == "SHUF":
        g.shuffle(mf)
        return mf
    if arm == "RULE_LAYER":
        out = np.empty(N, dtype=np.float64)
        for ly in np.unique(m["layer"]):
            idx = np.where(m["layer"] == ly)[0]
            pool = mf[idx]
            out[idx] = g.choice(pool, size=len(idx), replace=True)
        return out
    if arm == "RULE_DEGREE":
        return isotonic_assign(m["indeg"], mf, m["rho_indeg"] >= 0, g)
    if arm == "RULE_EI_LAYER":
        out = np.empty(N, dtype=np.float64)
        for ly in np.unique(m["layer"]):
            for s in (+1, -1):
                idx = np.where((m["layer"] == ly) & (m["sign"] == s))[0]
                if len(idx) == 0:
                    continue
                pool = mf[idx]
                if len(pool) == 0:
                    pool = mf[m["layer"] == ly]
                out[idx] = g.choice(pool, size=len(idx), replace=True)
        return out
    if arm == "RULE_FF_DEPTH":
        return isotonic_assign(m["depth"].astype(float), mf, m["rho_depth"] >= 0, g)
    raise KeyError(arm)


def build_net(seed):
    g = np.random.default_rng(seed + 7)
    return HH.build_arm("HUMAN_LAYERED", seed, g)


def delays_for(arm, W, info, mf_use):
    Wcoo = W.tocoo()
    delay = np.ones(Wcoo.nnz, dtype=np.int32)
    if arm == "NODELAY":
        return delay, DMAX
    h0, h1 = 0, int(info["Nm"])
    pre = Wcoo.col
    d_neuron = myelin_to_delay(mf_use)
    mask = (pre >= h0) & (pre < h1)
    delay[mask] = d_neuron[pre[mask] - h0]
    return delay, DMAX


def drive_idx(info):
    # Layered graft: drive L4 only (matches hybrid_human / myelin_real_rep).
    if "L4" in info:
        return np.asarray(info["L4"])
    return np.arange(min(200, int(info.get("N", 400))))


def states_narma(W, delay, dmax, leak, rng, in_idx, seed):
    T = WASH + TR + TE
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, rng)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=False, seed=seed)
    return X, y, WASH, TR, TE


def states_mc(W, delay, dmax, leak, rng, in_idx, seed):
    T = WASH + TR + TE
    u = rng.uniform(-1, 1, T)
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, rng)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u, leak, adapt=False, seed=seed)
    Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
    return X, Y, WASH, TR, TE


def states_pc(W, delay, dmax, leak, rng, in_idx, seed):
    n = W.shape[0]; T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = np.asarray(in_idx)
    rng2 = np.random.default_rng(int(rng.integers(1e9)))
    k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
    drive = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
    Win, b = MR.masked_win(n, 2, drive, rng)
    u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
    X = MR.run_delay_reservoir(W, delay, dmax, Win, b, u_drive, leak, adapt=False, seed=seed)
    return X, u_clean, 200, 600, 400


def score_xy(task, X, Y, wash, tr, te, alpha):
    if X is None or not np.all(np.isfinite(X)):
        return float("nan")
    if task == "MC":
        B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
        P = K.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
        return float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(Yt.shape[1])))
    if task == "PC_HARD":
        B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
        pred = K.pred(X[wash + tr:wash + tr + te] if wash + tr + te <= len(X) else X[wash + tr:], B)
        yt = Y[wash + tr:wash + tr + te] if wash + tr + te <= len(Y) else Y[wash + tr:]
        n = min(len(pred), len(yt))
        pred, yt = pred[:n], yt[:n]
        return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))
    B = K.ridge(X[wash:wash + tr], Y[wash:wash + tr], alpha)
    p = K.pred(X[wash + tr:], B); yt = Y[wash + tr:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    return float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))


def run_states(arm, seed, task):
    W, info = build_net(seed)
    mf = assign_mf(arm, seed)
    delay, dmax = delays_for(arm, W, info, mf)
    rng = np.random.default_rng(seed + 17 + {"NARMA": 2, "MC": 0, "PC_HARD": 4}[task])
    in_idx = drive_idx(info)
    if task == "NARMA":
        return states_narma(W, delay, dmax, LEAK, rng, in_idx, seed) + (delay,)
    if task == "MC":
        return states_mc(W, delay, dmax, LEAK, rng, in_idx, seed) + (delay,)
    return states_pc(W, delay, dmax, LEAK, rng, in_idx, seed) + (delay,)


def write_desc():
    m = load_maps()
    desc = {
        "n": int(len(m["mf"])),
        "mf_mean": float(m["mf"].mean()),
        "spearman_mf_indeg": m["rho_indeg"],
        "spearman_mf_outdeg": float(stats.spearmanr(m["mf"], m["outdeg"]).statistic),
        "spearman_mf_depth": m["rho_depth"],
        "layer_means": {str(ly): float(m["mf"][m["layer"] == ly].mean()) for ly in sorted(set(m["layer"]))},
        "sign_means": {"E": float(m["mf"][m["sign"] == 1].mean()), "I": float(m["mf"][m["sign"] == -1].mean())},
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT / "desc_corr.json").write_text(json.dumps(desc, indent=2))
    (DATA / "desc_corr.json").write_text(json.dumps(desc, indent=2))
    print("DESC", desc, flush=True)
    return desc


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    write_desc()
    choice = {"alphas": {}, "grid": ALPHA_GRID, "tune_seeds": TUNE_SEEDS,
              "leak": LEAK, "prereg": "PREREG_myelin_mechanism.md",
              "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for arm in ARMS:
        choice["alphas"][arm] = {}
        for task in TASKS:
            scores = {a: [] for a in ALPHA_GRID}
            for seed in TUNE_SEEDS:
                t0 = time.time()
                X, Y, wash, tr, te, delay = run_states(arm, seed, task)
                for a in ALPHA_GRID:
                    scores[a].append(score_xy(task, X, Y, wash, tr, te, a))
                print(f"tune {arm} {task} seed={seed} {time.time()-t0:.1f}s", flush=True)
            means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
            if task == "NARMA":
                best = min(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else 1e99, a))
            else:
                best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
            choice["alphas"][arm][task] = {"alpha": best, "means": means}
            print(f"LOCK {arm} {task} -> {best}", flush=True)
            cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_one(arm, seed, task, alpha):
    t0 = time.time()
    X, Y, wash, tr, te, delay = run_states(arm, seed, task)
    val = score_xy(task, X, Y, wash, tr, te, alpha)
    return dict(arm=arm, seed=int(seed), task=task, value=float(val) if val == val else float("nan"),
                alpha=float(alpha), delay_mean=float(np.mean(delay)), sec=time.time() - t0,
                source="myelin_mechanism")


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-600:], source="myelin_mechanism")


def _done():
    s = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    s.add((r["arm"], r["seed"], r["task"]))
            except Exception:
                pass
    return s


def run_eval(alphas):
    jobs = []
    for arm in ARMS:
        for task in TASKS:
            a = alphas["alphas"][arm][task]["alpha"]
            for seed in EVAL_SEEDS:
                jobs.append((arm, seed, task, a))
    done = _done()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"eval jobs={len(jobs)} skip={len(done)} workers={workers}", flush=True)
    with open(JF, "a") as f:
        if workers <= 1:
            for j in jobs:
                r = _job(j); f.write(json.dumps(r) + "\n"); f.flush()
                print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"), round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value"), round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
    (OUT / "eval.done").write_text("ok\n")


def smoke():
    write_desc()
    for arm in ("REAL", "SHUF", "RULE_LAYER", "RULE_DEGREE"):
        r = run_one(arm, 800, "NARMA", 1e-6)
        print("SMOKE", r, flush=True)


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "desc":
        write_desc(); return
    if mode == "tune":
        tune_alphas(); return
    alphas = tune_alphas()
    run_eval(alphas)
    print("myelin_mechanism done", flush=True)


if __name__ == "__main__":
    main()

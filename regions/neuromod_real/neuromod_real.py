"""Real FlyWire neuromodulators as slow gain. Prereg: PREREG_neuromod_real.md."""
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

SEEDS = list(range(1000, 1024))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
DMAX = 5
RHO = 0.9
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
GAMMA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
TAU = 0.1  # low-pass for modulatory activity (fixed)
XOR_D = 5
CTX_K = 40
PC_HARD_CUE = 0.35
PC_HARD_NOISE = 1.5

ARMS = ["NM_REAL", "NM_LABEL_SHUF", "NM_TARGET_SHUF", "NO_NM", "NM_ER"]
PRIMARY_TASK = "CTX_SWITCH"
SECONDARY_TASKS = ["XOR_DELAY", "NARMA", "MC", "PC_HARD"]
ALL_TASKS = [PRIMARY_TASK] + SECONDARY_TASKS
TASK_SEED_OFF = {
    "CTX_SWITCH": 8, "XOR_DELAY": 6, "NARMA": 2, "MC": 0, "PC_HARD": 4,
}


# ---------------------------------------------------------------------------
# NT labels
# ---------------------------------------------------------------------------
_NT = None

def load_nt():
    global _NT
    if _NT is None:
        z = np.load(DATA / "fly_cx_nt.npz", allow_pickle=True)
        _NT = dict(
            root_id=z["root_id"],
            top_nt=z["top_nt"].astype(str),
            is_mod=z["is_mod"].astype(bool),
            conf=z["top_nt_conf"].astype(float),
        )
    return _NT


def load_W0():
    W, sign, layer, ct = F.load_mod("fly_cx")
    return W.tocsr(), sign.astype(np.int8), layer.astype(str), ct


# ---------------------------------------------------------------------------
# Build fast W + modulatory diffusion Â_mod
# ---------------------------------------------------------------------------
def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr().astype(np.float64)
    rs = np.asarray(A.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ A


def zero_mod_columns(W: sp.csr_matrix, mod_mask: np.ndarray) -> sp.csr_matrix:
    """Zero outgoing fast synapses from modulatory neurons (columns = pre)."""
    W = W.tocsc(copy=True)
    idx = np.where(mod_mask)[0]
    for j in idx:
        start, end = W.indptr[j], W.indptr[j + 1]
        W.data[start:end] = 0.0
    W.eliminate_zeros()
    return W.tocsr()


def mod_diffusion(W: sp.csr_matrix, mod_mask: np.ndarray) -> sp.csr_matrix:
    """Absolute outgoing adjacency of modulatory neurons, row-normalized."""
    Wc = W.tocoo()
    keep = mod_mask[Wc.col]
    if not np.any(keep):
        return sp.csr_matrix(W.shape, dtype=np.float64)
    A = sp.csr_matrix(
        (np.abs(Wc.data[keep]), (Wc.row[keep], Wc.col[keep])),
        shape=W.shape, dtype=np.float64,
    )
    return row_normalize(A)


def rewire_mod_targets(W: sp.csr_matrix, mod_mask: np.ndarray, g: np.random.Generator) -> sp.csr_matrix:
    """Preserve each mod neuron's out-degree; randomize posts (no self-loops)."""
    N = W.shape[0]
    Wc = W.tocoo()
    row, col, data = Wc.row.copy(), Wc.col.copy(), Wc.data.copy()
    mod_edge = mod_mask[col]
    idx = np.where(mod_edge)[0]
    # group by pre
    for pre in np.unique(col[idx]):
        eidx = idx[col[idx] == pre]
        posts = g.choice(N, size=len(eidx), replace=True)
        for _ in range(12):
            bad = posts == pre
            if not bad.any():
                break
            posts[bad] = g.integers(0, N, size=int(bad.sum()))
        row[eidx] = posts
    return sp.csr_matrix((data, (row, col)), shape=(N, N))


def build_arm(arm: str, seed: int):
    """Return dict with W_fast (or W_full), A_mod, mod_mask, info, use_nm."""
    W0, sign, layer, ct = load_W0()
    nt = load_nt()
    assert len(nt["is_mod"]) == W0.shape[0]
    g = np.random.default_rng(seed + 13)

    if arm == "NO_NM":
        W = F.scale(W0, RHO)
        L4, _, _ = F.layer_sets(layer, W.shape[0])
        info = dict(names=["fly_cx"], sizes=[W.shape[0]], offsets=[0, W.shape[0]],
                    N=W.shape[0], drive=L4, n_mod=0, arm=arm)
        return dict(W=W.tocsr(), A_mod=None, mod_mask=None, info=info, use_nm=False, sign=sign)

    if arm == "NM_ER":
        Wer = F.er_match(W0, sign, g)
        n_mod = int(nt["is_mod"].sum())
        mod_mask = np.zeros(Wer.shape[0], dtype=bool)
        pick = g.choice(Wer.shape[0], size=n_mod, replace=False)
        mod_mask[pick] = True
        W_fast = zero_mod_columns(Wer, mod_mask)
        A_mod = mod_diffusion(Wer, mod_mask)  # targets from ER edges of chosen mods
        W_fast = F.scale(W_fast, RHO)
        # ER has no layer — drive = random L4-sized set
        n_drive = max(1, int((layer == "L4").sum()) if (layer == "L4").any() else min(200, Wer.shape[0]))
        drive = g.choice(Wer.shape[0], size=min(n_drive, Wer.shape[0]), replace=False)
        info = dict(names=["er"], sizes=[Wer.shape[0]], offsets=[0, Wer.shape[0]],
                    N=Wer.shape[0], drive=drive, n_mod=int(n_mod), arm=arm)
        return dict(W=W_fast.tocsr(), A_mod=A_mod, mod_mask=mod_mask, info=info, use_nm=True, sign=sign)

    # Real topology arms
    mod_mask = nt["is_mod"].copy()
    W_use = W0.tocsr()

    if arm == "NM_LABEL_SHUF":
        g2 = np.random.default_rng(seed + 4242)
        labels = mod_mask.copy()
        g2.shuffle(labels)
        mod_mask = labels
    elif arm == "NM_TARGET_SHUF":
        g2 = np.random.default_rng(seed + 5252)
        W_use = rewire_mod_targets(W0, mod_mask, g2)
    elif arm != "NM_REAL":
        raise KeyError(arm)

    W_fast = zero_mod_columns(W_use, mod_mask)
    A_mod = mod_diffusion(W_use, mod_mask)
    W_fast = F.scale(W_fast, RHO)
    L4, _, _ = F.layer_sets(layer, W_fast.shape[0])
    info = dict(names=["fly_cx"], sizes=[W_fast.shape[0]], offsets=[0, W_fast.shape[0]],
                N=W_fast.shape[0], drive=L4, n_mod=int(mod_mask.sum()), arm=arm)
    return dict(W=W_fast.tocsr(), A_mod=A_mod, mod_mask=mod_mask, info=info, use_nm=True, sign=sign)


def delays_fixed(W, seed):
    g = np.random.default_rng(seed + 909)
    return g.integers(1, DMAX + 1, W.tocoo().nnz).astype(np.int32), DMAX


# ---------------------------------------------------------------------------
# Reservoirs
# ---------------------------------------------------------------------------
class DelayBuckets:
    def __init__(self, W, delay, dmax):
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = delay == d
            if m.any():
                self.buckets.append(sp.csr_matrix(
                    (W.data[m], (W.row[m], W.col[m])), shape=(self.n, self.n)
                ))
            else:
                self.buckets.append(None)
        self.buf = np.zeros((self.dmax + 1, self.n), dtype=np.float64)
        self.t = 0

    def recurrent(self):
        self.t += 1
        h = np.zeros(self.n, dtype=np.float64)
        for di, Wd in enumerate(self.buckets):
            if Wd is None:
                continue
            h += Wd @ self.buf[(self.t - (di + 1)) % (self.dmax + 1)]
        return h

    def store(self, x):
        self.buf[self.t % (self.dmax + 1)] = x


def run_plain(W, delay, dmax, Win, b, U, leak):
    db = DelayBuckets(W, delay, dmax)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W.shape[0]
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        h = db.recurrent()
        y = np.tanh(h + drive)
        x = (1.0 - leak) * x + leak * y
        db.store(x)
        X[i] = x
    return X


def run_nm(W_fast, A_mod, mod_mask, delay, dmax, Win, b, U, leak, gamma, tau=TAU):
    db = DelayBuckets(W_fast, delay, dmax)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W_fast.shape[0]
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    m = np.zeros(n)
    mod = mod_mask.astype(bool)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        h = db.recurrent()
        m = (1.0 - tau) * m + tau * np.abs(x)
        src = m * mod  # only modulatory sources
        s = A_mod @ src
        g = np.clip(1.0 + float(gamma) * s, 0.05, 8.0)
        y = np.tanh(g * (h + drive))
        x = (1.0 - leak) * x + leak * y
        db.store(x)
        X[i] = x
    return X


def masked_win(n, d, in_idx, rng):
    return F.masked_win(n, d, in_idx, rng)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def make_task(task, rng, n, in_idx):
    if task == "XOR_DELAY":
        T = WASH + TR + TE
        a = rng.integers(0, 2, T).astype(np.float64)
        b = rng.integers(0, 2, T).astype(np.float64)
        y = np.zeros(T)
        for t in range(XOR_D, T):
            y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
        U = np.stack([2 * a - 1, 2 * b - 1], 1)
        Win, bias = masked_win(n, 2, in_idx, rng)
        return U, y, Win, bias, WASH, TR, WASH + TR

    if task == "CTX_SWITCH":
        T = WASH + TR + TE
        a = rng.integers(0, 2, T).astype(np.float64)
        b = rng.integers(0, 2, T).astype(np.float64)
        # deterministic cue flip every CTX_K steps; start +1
        c = np.empty(T, dtype=np.float64)
        for t in range(T):
            block = t // CTX_K
            c[t] = 1.0 if (block % 2 == 0) else -1.0
        y = np.zeros(T)
        for t in range(T):
            if c[t] > 0:
                if t >= XOR_D:
                    y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
            else:
                y[t] = 2 * a[t] - 1.0
        U = np.stack([2 * a - 1, 2 * b - 1, c], 1)
        Win, bias = masked_win(n, 3, in_idx, rng)
        return U, y, Win, bias, WASH, TR, WASH + TR

    if task == "NARMA":
        T = WASH + TR + TE
        u = rng.uniform(0, 0.5, T)
        y = np.zeros(T)
        for i in range(10, T - 1):
            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
        Win, bias = masked_win(n, 1, in_idx, rng)
        return u, y, Win, bias, WASH, TR, WASH + TR

    if task == "MC":
        T = WASH + TR + TE
        u = rng.uniform(-1, 1, T)
        Y = np.stack([np.roll(u, k) for k in range(1, 101)], 1)
        Win, bias = masked_win(n, 1, in_idx, rng)
        return u, Y, Win, bias, WASH, TR, WASH + TR

    if task == "PC_HARD":
        T = 1200
        theta = np.linspace(0, 8 * np.pi, T)
        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
        idx = np.asarray(in_idx)
        rng2 = np.random.default_rng(int(rng.integers(1e9)))
        k = max(1, int(round(PC_HARD_CUE * max(1, len(idx)))))
        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(PC_HARD_CUE * n)))
        Win, bias = masked_win(n, 2, drive_idx, rng)
        u_drive = u_clean + PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
        wash, tr = 200, 800
        return u_drive, u_clean, Win, bias, wash, tr - wash, tr

    raise KeyError(task)


def score_pred(task, pred, yt):
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(yt)):
        return float("nan")
    if task == "MC":
        D = yt.shape[1]
        return float(sum(np.nan_to_num(np.corrcoef(pred[:, k], yt[:, k])[0, 1]) ** 2 for k in range(D)))
    if task == "PC_HARD":
        return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12))
    if np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((pred - yt) ** 2)) / np.std(yt))
    if not np.isfinite(nrmse):
        return float("nan")
    if task in ("XOR_DELAY", "CTX_SWITCH"):
        return float(1.0 - nrmse)
    return nrmse  # NARMA


def run_one(arm, seed, task, alpha, gamma):
    t0 = time.time()
    built = build_arm(arm, seed)
    W = built["W"]
    delay, dmax = delays_fixed(W, seed)
    rng = np.random.default_rng(seed + TASK_SEED_OFF[task] + 17)
    U, y, Win, bias, wash, tr, te_start = make_task(task, rng, W.shape[0], built["info"]["drive"])

    if built["use_nm"]:
        X = run_nm(W, built["A_mod"], built["mod_mask"], delay, dmax, Win, bias, U, LEAK, gamma)
    else:
        X = run_plain(W, delay, dmax, Win, bias, U, LEAK)

    if not np.all(np.isfinite(X)):
        return dict(arm=arm, seed=int(seed), task=task, value=float("nan"),
                    n=int(W.shape[0]), alpha=float(alpha), gamma=float(gamma),
                    sec=time.time() - t0, error="nonfinite_X", source="neuromod_real",
                    n_mod=int(built["info"].get("n_mod", 0)))

    if task == "PC_HARD":
        B = K.ridge(X[wash:te_start], y[wash:te_start], alpha)
        pred = K.pred(X[te_start:], B)
        yt = y[te_start:]
    else:
        B = K.ridge(X[wash:wash + tr], y[wash:wash + tr], alpha)
        pred = K.pred(X[wash + tr:], B)
        yt = y[wash + tr:]
    val = score_pred(task, pred, yt)
    return dict(
        arm=arm, seed=int(seed), task=task,
        value=float(val) if val == val else float("nan"),
        n=int(W.shape[0]), alpha=float(alpha), gamma=float(gamma if built["use_nm"] else 0.0),
        n_mod=int(built["info"].get("n_mod", 0)),
        delay_mean=float(np.mean(delay)), tau=TAU, leak=LEAK,
        sec=time.time() - t0, has_fastcore2=_HAS_FC2, source="neuromod_real",
    )


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        arm, seed, task, alpha, gamma = args
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    error=str(e)[:400], traceback=traceback.format_exc()[-800:],
                    source="neuromod_real")


# ---------------------------------------------------------------------------
# Tune / run
# ---------------------------------------------------------------------------
def tune():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    print(f"TUNE seeds={TUNE_SEEDS} arms={ARMS} fc2={_HAS_FC2}", flush=True)
    choice = {
        "alphas": {}, "gammas": {},
        "alpha_grid": ALPHA_GRID, "gamma_grid": GAMMA_GRID,
        "tau": TAU, "criterion": "mean CTX_SWITCH on tune seeds",
        "tune_seeds": TUNE_SEEDS, "has_fastcore2": _HAS_FC2,
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": "PREREG_neuromod_real.md",
        "nt_counts": json.loads((DATA / "NT_SOURCE.json").read_text())["counts"],
    }
    for arm in ARMS:
        best_score = -1e99
        best_a, best_g = ALPHA_GRID[0], (0.0 if arm == "NO_NM" else GAMMA_GRID[0])
        means_table = {}
        gamma_list = [0.0] if arm == "NO_NM" else GAMMA_GRID
        for gamma in gamma_list:
            for alpha in ALPHA_GRID:
                vals = []
                for seed in TUNE_SEEDS:
                    r = run_one(arm, seed, PRIMARY_TASK, alpha, gamma)
                    vals.append(r.get("value", float("nan")))
                m = float(np.nanmean(vals))
                means_table[f"g{gamma}_a{alpha}"] = {"mean": m, "vals": vals}
                score = m if np.isfinite(m) else -1e99
                # prefer higher CTX; tie-break smaller |gamma| then smaller alpha
                key = (score, -abs(gamma), -alpha)
                if key > (best_score, -abs(best_g), -best_a):
                    best_score, best_a, best_g = score, alpha, gamma
                print(f"tune {arm} g={gamma} a={alpha} mean={m:.4f} vals={[round(v,3) if v==v else None for v in vals]}", flush=True)
        choice["alphas"][arm] = {"alpha": best_a, "tune_mean_ctx": best_score}
        choice["gammas"][arm] = {"gamma": best_g}
        choice[f"grid_{arm}"] = means_table
        print(f"LOCK {arm} alpha={best_a} gamma={best_g} ctx={best_score:.4f}", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def _load_done():
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
    return done


def smoke():
    out = []
    for arm in ARMS:
        r = run_one(arm, 1000, PRIMARY_TASK, 1e-6, 1.0)
        out.append(r)
        print("SMOKE", {k: r.get(k) for k in ("arm", "value", "n", "n_mod", "sec", "error")}, flush=True)
    (OUT / "smoke.json").write_text(json.dumps(out, indent=2, default=float))
    (OUT / "smoke.done").write_text("ok\n")


def build_jobs(choice, phase="all"):
    jobs = []
    tasks = [PRIMARY_TASK] if phase == "primary" else (SECONDARY_TASKS if phase == "secondary" else ALL_TASKS)
    for arm in ARMS:
        a = choice["alphas"][arm]["alpha"]
        g = choice["gammas"][arm]["gamma"]
        for seed in SEEDS:
            for task in tasks:
                jobs.append((arm, seed, task, a, g))
    return jobs


def run_jobs(jobs, label="run"):
    done = _load_done()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    workers = int(os.environ.get("WORKERS", "2"))
    print(f"{label} jobs={len(jobs)} skip={len(_load_done())} workers={workers}", flush=True)
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


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "tune":
        tune(); return
    choice = tune()
    phase = os.environ.get("PHASE", "all")
    if phase == "primary":
        run_jobs(build_jobs(choice, "primary"), "primary")
        (OUT / "primary.done").write_text("ok\n"); return
    if phase == "secondary":
        run_jobs(build_jobs(choice, "secondary"), "secondary")
        (OUT / "secondary.done").write_text("ok\n"); return
    run_jobs(build_jobs(choice, "primary"), "primary")
    (OUT / "primary.done").write_text("ok\n")
    run_jobs(build_jobs(choice, "secondary"), "secondary")
    (OUT / "secondary.done").write_text("ok\n")
    (OUT / "neuromod_real.done").write_text("ok\n")
    print("neuromod_real done", flush=True)


if __name__ == "__main__":
    main()

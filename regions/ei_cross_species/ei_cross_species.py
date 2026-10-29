"""Cross-species E/I XOR generalization. Prereg: PREREG_ei_cross_species.md.

Intended for Kaggle CPU (WORKERS default 4). Do not assume GPU.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path

for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"

H = Path(__file__).resolve().parent
_ROOT = H.parents[1]
sys.path.insert(0, str(_ROOT / "compress"))
sys.path.insert(0, str(_ROOT / "regions" / "frankenstein"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from multiprocessing import Pool
import frankenstein as F
import fast_kernels as K

OUT = H / "out"; OUT.mkdir(exist_ok=True)
DATA = H / "data"; DATA.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"

EVAL_SEEDS = list(range(900, 924))
TUNE_SEEDS = [90, 91, 92]
LEAK = 0.2
RHO = 0.9
DMAX = 5
XOR_D = 5
WASH, TR, TE = 200, 1500, 500
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
DESIGN_N = 4000
DESIGN_GRAPH_SEED = 424242

# Species with usable E/I (ELEGAN skipped — see SOURCES.md)
SPECIES = {
    "MOUSE": {"file": "mouse_module.npz", "label": "mouse MICrONS module"},
    "HUMAN": {"file": "human_module.npz", "label": "human H01 patch"},
    "FINCH": {"file": "areax_module.npz", "label": "zebra finch Area X j0251"},
}
SPECIES_ARMS = ["REAL", "ER", "SHUF", "EI_SHUF", "SBM_EI"]
DESIGN_ARMS = ["DESIGN_EI_ONLY", "ER4000"]

_CACHE = {}


def workers() -> int:
    return int(os.environ.get("WORKERS", "4"))


# -------------------- reservoir --------------------
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
            if Wd is None:
                continue
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
    Win = np.zeros((n, d))
    b = rng.uniform(-0.1, 0.1, n)
    idx = np.asarray(in_idx)
    if len(idx) == 0:
        idx = np.arange(n)
    Win[idx] = rng.uniform(-0.5, 0.5, (len(idx), d))
    return Win, b


def delays_fixed(W, seed):
    Wcoo = W.tocoo()
    g = np.random.default_rng(seed + 909)
    return g.integers(1, DMAX + 1, Wcoo.nnz).astype(np.int32), DMAX


def xor_delay_score(W, delay, dmax, leak, rng, in_idx, alpha):
    T = WASH + TR + TE
    a = rng.integers(0, 2, T).astype(np.float64)
    b = rng.integers(0, 2, T).astype(np.float64)
    y = np.zeros(T)
    for t in range(XOR_D, T):
        y[t] = 1.0 if (int(a[t - XOR_D]) ^ int(b[t - XOR_D])) else -1.0
    U = np.stack([2 * a - 1, 2 * b - 1], 1)
    Win, bias = masked_win(W.shape[0], 2, in_idx, rng)
    X = run_delay_reservoir(W, delay, dmax, Win, bias, U, leak)
    if not np.all(np.isfinite(X)):
        return float("nan")
    B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], alpha)
    p = K.pred(X[WASH + TR:], B)
    yt = y[WASH + TR:]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan")


# -------------------- graph helpers --------------------
def ei_class_shuffle(W, sign, g):
    Wc = W.tocoo()
    N = W.shape[0]
    row, col, data = Wc.row.copy(), Wc.col.copy(), np.abs(Wc.data).copy()
    s_pre = sign[row]; s_post = sign[col]
    new_col = col.copy()
    for s_pre_v in (-1, 1):
        for s_post_v in (-1, 1):
            m = (s_pre == s_pre_v) & (s_post == s_post_v)
            idx = np.where(m)[0]
            if len(idx) < 2:
                continue
            cands = np.where(sign == s_post_v)[0]
            if len(cands) == 0:
                continue
            posts = col[idx].copy()
            g.shuffle(posts)
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
        return float((ev > 0.5).mean())
    except Exception:
        return float("nan")


def temper_spectral(W, target_max=0.40, max_iter=8):
    W = W.tocsr()
    for _ in range(max_iter):
        frac = spectral_frac(W)
        if not np.isfinite(frac) or frac <= target_max:
            return W, frac
        mag = np.abs(W.data)
        thr = np.median(mag)
        small = mag <= thr
        new = mag.copy(); new[small] *= 0.7
        Wc = W.tocoo()
        W = sp.csr_matrix((new * np.sign(Wc.data), (Wc.row, Wc.col)), shape=W.shape)
        W = F.scale(W, 1.0)
    return W, spectral_frac(W)


def sbm_from_ei(W_real, sign, g):
    """E/I×E/I density + magnitude SBM; no copied edges."""
    N = W_real.shape[0]
    Wc = W_real.tocoo()
    dens = {}; mag_pools = {}
    for a in (1, -1):
        for b in (1, -1):
            pre = np.where(sign == a)[0]; post = np.where(sign == b)[0]
            n_pos = max(1, len(pre) * len(post) - (len(pre) if a == b else 0))
            m = (sign[Wc.row] == a) & (sign[Wc.col] == b)
            n_e = int(m.sum())
            dens[(a, b)] = n_e / n_pos
            mag_pools[(a, b)] = np.abs(Wc.data[m]) if m.any() else np.array([1.0])
    rows, cols, data = [], [], []
    class_idx = {1: np.where(sign == 1)[0], -1: np.where(sign == -1)[0]}
    for a in (1, -1):
        for b in (1, -1):
            pre = class_idx[a]; post = class_idx[b]
            if len(pre) == 0 or len(post) == 0:
                continue
            n_pos = len(pre) * len(post) - (len(pre) if a == b else 0)
            n_e = int(round(dens[(a, b)] * max(1, n_pos)))
            if n_e <= 0:
                continue
            if a == b:
                got = 0; pr = []; po = []
                chunk = max(n_e * 2, n_e + 1000)
                while got < n_e:
                    rr = g.choice(pre, size=chunk, replace=True)
                    cc = g.choice(post, size=chunk, replace=True)
                    ok = rr != cc
                    rr, cc = rr[ok], cc[ok]
                    take = min(n_e - got, len(rr))
                    pr.append(rr[:take]); po.append(cc[:take]); got += take
                rr = np.concatenate(pr)[:n_e]; cc = np.concatenate(po)[:n_e]
            else:
                rr = g.choice(pre, size=n_e, replace=True)
                cc = g.choice(post, size=n_e, replace=True)
            mags = g.choice(mag_pools[(a, b)], size=n_e, replace=True)
            rows.append(rr); cols.append(cc); data.append(mags * float(a))
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    W = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    W.setdiag(0); W.eliminate_zeros()
    W = F.scale(W, 1.0)
    W, _ = temper_spectral(W, target_max=0.40, max_iter=8)
    return W


def load_species(species):
    if species in _CACHE:
        return _CACHE[species]
    if species == "DESIGN":
        # fly CX template for DESIGN_EI
        path = DATA / "fly_cx_module.npz"
        if not path.exists():
            # fall back to frankenstein data
            path = _ROOT / "regions" / "frankenstein" / "data" / "fly_cx_module.npz"
        z = np.load(path, allow_pickle=True)
        W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
        sign = z["sign"].astype(np.int8)
        layer = z["layer"].astype(str) if "layer" in z.files else np.array(["L4"] * W.shape[0])
        _CACHE[species] = (W, sign, layer)
        return _CACHE[species]
    meta = SPECIES[species]
    path = DATA / meta["file"]
    if not path.exists():
        # try common locations
        for alt in [
            _ROOT / "regions" / "frankenstein" / "data" / meta["file"],
            _ROOT / "regions" / "hybrid" / "data" / meta["file"],
            _ROOT / "regions" / "hybrid_human" / "data" / meta["file"],
            _ROOT / "regions" / "finch" / "data" / meta["file"],
        ]:
            if alt.exists():
                path = alt; break
    z = np.load(path, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    sign = z["sign"].astype(np.int8)
    if "layer" in z.files:
        layer = z["layer"].astype(str)
    else:
        layer = np.array(["NA"] * W.shape[0])
    # require both E and I
    nE = int((sign == 1).sum()); nI = int((sign == -1).sum())
    if nE == 0 or nI == 0:
        raise RuntimeError(f"{species}: E/I unavailable (E={nE} I={nI}) — skip per prereg")
    _CACHE[species] = (W, sign, layer)
    return _CACHE[species]


def drive_idx(layer, N):
    L4 = np.where(layer == "L4")[0]
    if len(L4) >= 8:
        return L4
    # finch / unlabeled: take a fixed fraction
    k = max(32, N // 20)
    return np.arange(min(k, N))


def get_design_ei():
    """Frozen 4000-neuron E/I SBM from fly CX stats (DESIGN_EI_ONLY recipe, fixed delays)."""
    if "design_ei" in _CACHE:
        return _CACHE["design_ei"]
    Wcx, sign_cx, layer_cx = load_species("DESIGN")
    g = np.random.default_rng(DESIGN_GRAPH_SEED)
    n = DESIGN_N
    frac_E = float((sign_cx == 1).mean())
    sign = np.ones(n, dtype=np.int8)
    nE = int(round(frac_E * n))
    sign[nE:] = -1
    g.shuffle(sign)
    # layer proportions from CX if available else all L4
    props = {}
    for L in np.unique(layer_cx):
        props[str(L)] = float((layer_cx == L).mean())
    if not props:
        props = {"L4": 1.0}
    layers = []
    for L, p in props.items():
        layers.extend([L] * int(round(p * n)))
    while len(layers) < n:
        layers.append("L4")
    layers = np.asarray(layers[:n], dtype=str)
    g.shuffle(layers)
    # build SBM using CX dens on synthetic sign
    # map CX dens onto new sign vector via sbm_from_ei on a projected template:
    # construct a temporary CX-sized signed graph then resample dens onto n
    dens = {}; mag_pools = {}
    Wc = Wcx.tocoo()
    for a in (1, -1):
        for b in (1, -1):
            pre = np.where(sign_cx == a)[0]; post = np.where(sign_cx == b)[0]
            n_pos = max(1, len(pre) * len(post) - (len(pre) if a == b else 0))
            m = (sign_cx[Wc.row] == a) & (sign_cx[Wc.col] == b)
            dens[(a, b)] = int(m.sum()) / n_pos
            mag_pools[(a, b)] = np.abs(Wc.data[m]) if m.any() else np.array([1.0])
    rows, cols, data = [], [], []
    class_idx = {1: np.where(sign == 1)[0], -1: np.where(sign == -1)[0]}
    for a in (1, -1):
        for b in (1, -1):
            pre = class_idx[a]; post = class_idx[b]
            n_pos = len(pre) * len(post) - (len(pre) if a == b else 0)
            n_e = int(round(dens[(a, b)] * max(1, n_pos)))
            if n_e <= 0 or len(pre) == 0 or len(post) == 0:
                continue
            if a == b:
                got = 0; pr = []; po = []
                chunk = max(n_e * 2, n_e + 2000)
                while got < n_e:
                    rr = g.choice(pre, size=chunk, replace=True)
                    cc = g.choice(post, size=chunk, replace=True)
                    ok = rr != cc
                    rr, cc = rr[ok], cc[ok]
                    take = min(n_e - got, len(rr))
                    pr.append(rr[:take]); po.append(cc[:take]); got += take
                rr = np.concatenate(pr)[:n_e]; cc = np.concatenate(po)[:n_e]
            else:
                rr = g.choice(pre, size=n_e, replace=True)
                cc = g.choice(post, size=n_e, replace=True)
            mags = g.choice(mag_pools[(a, b)], size=n_e, replace=True)
            rows.append(rr); cols.append(cc); data.append(mags * float(a))
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0); W.eliminate_zeros()
    W = F.scale(W, 1.0)
    W, frac = temper_spectral(W, target_max=0.40, max_iter=8)
    out = dict(W=W.tocsr(), sign=sign, layer=layers, spectral_frac=frac,
               frac_E=float((sign > 0).mean()), nnz=int(W.nnz))
    _CACHE["design_ei"] = out
    return out


def build(species, arm, seed):
    g = np.random.default_rng(seed + 13 + hash(f"{species}:{arm}") % 100000)
    if species == "DESIGN":
        D = get_design_ei()
        if arm == "DESIGN_EI_ONLY":
            W = F.scale(D["W"], RHO)
        elif arm == "ER4000":
            W = F.scale(F.er_match(D["W"], D["sign"], g), RHO)
        else:
            raise KeyError(arm)
        drive = drive_idx(D["layer"], W.shape[0])
        return W.tocsr(), drive, dict(N=W.shape[0], nnz=int(W.nnz), species=species, arm=arm)

    W0, sign, layer = load_species(species)
    if arm == "REAL":
        Wm = W0
    elif arm == "ER":
        Wm = F.er_match(W0, sign, g)
    elif arm == "SHUF":
        Wm = F.degree_shuffle(W0, sign, g)
    elif arm == "EI_SHUF":
        Wm = ei_class_shuffle(W0, sign, g)
    elif arm == "SBM_EI":
        Wm = sbm_from_ei(W0, sign, g)
    else:
        raise KeyError(arm)
    W = F.scale(F.scale(Wm, 1.0), RHO)
    return W.tocsr(), drive_idx(layer, W.shape[0]), dict(N=W.shape[0], nnz=int(W.nnz), species=species, arm=arm)


def run_one(species, arm, seed, alpha):
    t0 = time.time()
    W, drive, meta = build(species, arm, seed)
    delay, dmax = delays_fixed(W, seed)
    rng = np.random.default_rng(seed + 17 + 6)  # XOR_DELAY offset
    val = xor_delay_score(W, delay, dmax, LEAK, rng, drive, alpha)
    return dict(species=species, arm=arm, seed=int(seed), task="XOR_DELAY", xor_d=XOR_D,
                value=float(val) if val == val else float("nan"), alpha=float(alpha),
                N=int(meta["N"]), nnz=int(meta["nnz"]), sec=time.time() - t0,
                source="ei_cross_species")


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:
        import traceback
        species, arm, seed, alpha = args
        return dict(species=species, arm=arm, seed=seed, task="XOR_DELAY", xor_d=XOR_D,
                    value=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-600:], source="ei_cross_species")


def _done():
    s = set()
    if JF.exists():
        for line in open(JF):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    s.add((r["species"], r["arm"], r["seed"]))
            except Exception:
                pass
    return s


def all_units():
    units = []
    for sp in SPECIES:
        for arm in SPECIES_ARMS:
            units.append((sp, arm))
    for arm in DESIGN_ARMS:
        units.append(("DESIGN", arm))
    return units


def tune_alphas():
    cache = OUT / "alpha_choice.json"
    if cache.exists():
        return json.loads(cache.read_text())
    choice = {"alphas": {}, "grid": ALPHA_GRID, "tune_seeds": TUNE_SEEDS,
              "leak": LEAK, "prereg": "PREREG_ei_cross_species.md",
              "skipped": {"ELEGAN": "E/I unavailable (ei_status=unknown); see SOURCES.md"},
              "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for species, arm in all_units():
        key = f"{species}:{arm}"
        scores = {a: [] for a in ALPHA_GRID}
        for seed in TUNE_SEEDS:
            t0 = time.time()
            W, drive, meta = build(species, arm, seed)
            delay, dmax = delays_fixed(W, seed)
            rng = np.random.default_rng(seed + 17 + 6)
            # reuse states across alphas
            T = WASH + TR + TE
            aa = rng.integers(0, 2, T).astype(np.float64)
            bb = rng.integers(0, 2, T).astype(np.float64)
            y = np.zeros(T)
            for t in range(XOR_D, T):
                y[t] = 1.0 if (int(aa[t - XOR_D]) ^ int(bb[t - XOR_D])) else -1.0
            U = np.stack([2 * aa - 1, 2 * bb - 1], 1)
            Win, bias = masked_win(W.shape[0], 2, drive, rng)
            X = run_delay_reservoir(W, delay, dmax, Win, bias, U, LEAK)
            for a in ALPHA_GRID:
                if not np.all(np.isfinite(X)):
                    scores[a].append(float("nan")); continue
                B = K.ridge(X[WASH:WASH + TR], y[WASH:WASH + TR], a)
                p = K.pred(X[WASH + TR:], B); yt = y[WASH + TR:]
                if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
                    scores[a].append(float("nan")); continue
                nrmse = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
                scores[a].append(float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan"))
            print(f"tune {key} seed={seed} N={meta['N']} {time.time()-t0:.1f}s", flush=True)
        means = {str(a): float(np.nanmean(scores[a])) for a in ALPHA_GRID}
        best = max(ALPHA_GRID, key=lambda a: (means[str(a)] if np.isfinite(means[str(a)]) else -1e99, -a))
        choice["alphas"][key] = {"alpha": best, "means": means}
        print(f"LOCK {key} -> {best}", flush=True)
        cache.write_text(json.dumps(choice, indent=2))
    (OUT / "tune.done").write_text("ok\n")
    return choice


def run_eval(alphas):
    jobs = []
    for species, arm in all_units():
        a = alphas["alphas"][f"{species}:{arm}"]["alpha"]
        for seed in EVAL_SEEDS:
            jobs.append((species, arm, seed, a))
    done = _done()
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    w = workers()
    print(f"eval jobs={len(jobs)} skip={len(done)} workers={w}", flush=True)
    with open(JF, "a") as f:
        if w <= 1:
            for j in jobs:
                r = _job(j); f.write(json.dumps(r) + "\n"); f.flush()
                print(r.get("species"), r.get("arm"), r.get("seed"), r.get("value"),
                      round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
        else:
            with Pool(w) as pool:
                for r in pool.imap_unordered(_job, jobs):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    print(r.get("species"), r.get("arm"), r.get("seed"), r.get("value"),
                          round(r.get("sec", 0) or 0, 1), r.get("error", ""), flush=True)
    (OUT / "eval.done").write_text("ok\n")


def smoke():
    print("SMOKE workers", workers(), flush=True)
    for sp, arm in [("MOUSE", "REAL"), ("MOUSE", "ER"), ("HUMAN", "EI_SHUF"), ("DESIGN", "DESIGN_EI_ONLY")]:
        r = run_one(sp, arm, 900, 1e-6)
        print("SMOKE", r, flush=True)


def main():
    mode = os.environ.get("MODE", "full")
    if mode == "smoke":
        smoke(); return
    if mode == "tune":
        tune_alphas(); return
    alphas = tune_alphas()
    run_eval(alphas)
    print("ei_cross_species done", flush=True)


if __name__ == "__main__":
    main()

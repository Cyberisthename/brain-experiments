"""Opt-in speedups on top of compress/fast_kernels.py — do NOT change numerics.

Ideas (parent task):
  1) Batch seeds/inputs: sparse W @ dense multi-column state (same W), or
     block-diagonal stack of different Ws.
  2) Reuse states: simulate once per (arm, seed, input stream); train all
     readouts that share that input on the cached X. Disk-cache random
     control graphs (ER/SHUF/DENS) per (name, kind, seed).

Enable by importing this module; running experiments that do not import it
are untouched. All public simulators fall back to the single-column
fast_kernels path when batch size == 1 so B=1 is bit-identical.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Re-use existing kernels (dual ridge, radius cache, single CSR loop, …)
import sys

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "compress"))
import fast_kernels as K  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path(os.environ.get("FASTCORE2_CACHE", str(_ROOT / ".cache" / "fastcore2")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR = CACHE_DIR / "control_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# In-process state cache: key -> {"X": ndarray, "meta": dict}
_STATE_CACHE: dict[str, dict] = {}
_STATE_CACHE_BYTES = 0
_STATE_CACHE_MAX_BYTES = int(os.environ.get("FASTCORE2_STATE_CACHE_MB", "512")) * 1024 * 1024

# Stats for SPEEDUP2 reporting
STATS = {
    "graph_cache_hit": 0,
    "graph_cache_miss": 0,
    "state_cache_hit": 0,
    "state_cache_miss": 0,
    "batch_calls": 0,
    "batch_columns": 0,
}


def reset_stats():
    for k in STATS:
        STATS[k] = 0
    global _STATE_CACHE, _STATE_CACHE_BYTES
    _STATE_CACHE = {}
    _STATE_CACHE_BYTES = 0


def clear_state_cache():
    global _STATE_CACHE, _STATE_CACHE_BYTES
    _STATE_CACHE = {}
    _STATE_CACHE_BYTES = 0


# ---------------------------------------------------------------------------
# Helpers shared with fast_kernels
# ---------------------------------------------------------------------------
ridge = K.ridge
pred = K.pred
radius_cached = K.radius_cached
run_reservoir_csr = K.run_reservoir_csr  # single-traj reference


def _as_U_list(Us):
    out = []
    for U in Us:
        U = np.asarray(U, dtype=np.float64)
        out.append(U)
    return out


def _drive_at(Win, b, U_t, multi):
    if not multi:
        return Win[:, 0] * U_t + b
    return Win @ U_t + b


# ---------------------------------------------------------------------------
# (1) Batched CSR reservoir — same W, many independent trajectories
# ---------------------------------------------------------------------------
def run_reservoir_csr_batch(W, Wins, bs, Us, leak):
    """Same W; B independent (Win, b, U) streams.

    Returns list of X_b with shape (T, n), float64.
    When B==1, delegates to fast_kernels.run_reservoir_csr (bit-identical).
    """
    Wins = list(Wins)
    bs = [np.asarray(b, dtype=np.float64) for b in bs]
    Us = _as_U_list(Us)
    B = len(Us)
    if B == 0:
        return []
    if B == 1:
        return [K.run_reservoir_csr(W, Wins[0], bs[0], Us[0], leak)]

    assert len(Wins) == B and len(bs) == B
    T = len(Us[0])
    for U in Us:
        assert len(U) == T, "batched Us must share length T"
    multi = Us[0].ndim == 2
    for U in Us:
        assert (U.ndim == 2) == multi

    W = W.tocsr()
    n = W.shape[0]
    # Stack Win columns / biases: for 1-d input, Win_stack[:,b] = Win_b[:,0]
    if not multi:
        Win_stack = np.column_stack([np.asarray(Wins[b], dtype=np.float64)[:, 0] for b in range(B)])
    else:
        # d may vary? require same d
        d = Wins[0].shape[1]
        for Win in Wins:
            assert Win.shape == (n, d)
    b_stack = np.column_stack(bs)  # (n, B)

    X_out = [np.empty((T, n), dtype=np.float64) for _ in range(B)]
    x = np.zeros((n, B), dtype=np.float64)  # multi-column state
    STATS["batch_calls"] += 1
    STATS["batch_columns"] += B

    if not multi:
        U_mat = np.column_stack(Us)  # (T, B)
        for i in range(T):
            drive = Win_stack * U_mat[i] + b_stack
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + drive)
            for b in range(B):
                X_out[b][i] = x[:, b]
    else:
        for i in range(T):
            drive = np.column_stack(
                [Wins[b] @ Us[b][i] + bs[b] for b in range(B)]
            )
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + drive)
            for b in range(B):
                X_out[b][i] = x[:, b]
    return X_out


# ---------------------------------------------------------------------------
# (1b) Block-diagonal batch for different Ws
# ---------------------------------------------------------------------------
def run_reservoir_blockdiag(Ws, Wins, bs, Us, leak):
    """Different Ws; stack as block-diag and simulate as one big reservoir.

    Returns list of X_b (T, n_b). B==1 → single path. Requires equal T.
    """
    Ws = list(Ws)
    Wins = list(Wins)
    bs = [np.asarray(b, dtype=np.float64) for b in bs]
    Us = _as_U_list(Us)
    B = len(Ws)
    if B == 0:
        return []
    if B == 1:
        return [K.run_reservoir_csr(Ws[0], Wins[0], bs[0], Us[0], leak)]

    T = len(Us[0])
    for U in Us:
        assert len(U) == T
    multi = Us[0].ndim == 2
    ns = [W.shape[0] for W in Ws]
    N = sum(ns)

    # Block-diagonal W
    Wbig = sp.block_diag([W.tocsr() for W in Ws], format="csr")

    # Build big Win, b. For 1-d: Win_big (N,1) is wrong for independent U —
    # each block has its own U_b[t], so drive must be assembled per step.
    # We use the multi-column trick on the BIG state with B=1 "fat" vector
    # of size N (concatenated), single trajectory of the block-diag system.
    Win_big_list = []  # not used as single Win; we drive manually
    b_big = np.concatenate(bs)
    X_big = np.empty((T, N), dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    STATS["batch_calls"] += 1
    STATS["batch_columns"] += B

    offs = np.cumsum([0] + ns)
    for i in range(T):
        drives = []
        for b in range(B):
            Ub = Us[b][i]
            if not multi:
                drives.append(np.asarray(Wins[b], dtype=np.float64)[:, 0] * Ub + bs[b])
            else:
                drives.append(np.asarray(Wins[b], dtype=np.float64) @ Ub + bs[b])
        drive = np.concatenate(drives)
        x = (1.0 - leak) * x + leak * np.tanh(Wbig @ x + drive)
        X_big[i] = x

    out = []
    for b in range(B):
        out.append(X_big[:, offs[b] : offs[b + 1]].copy())
    return out


# ---------------------------------------------------------------------------
# Delay-reservoir batch (same W + same delay buckets; many drives)
# Matches myelin_real / best_fly / fewshot_delay leak-integrated loop.
# ---------------------------------------------------------------------------
class DelayReservoirBatched:
    """Bucketed CSR delays with multi-column state (n, B)."""

    def __init__(self, W, delay, dmax, B=1):
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.B = int(B)
        self.row = W.row.copy()
        self.col = W.col.copy()
        self.data = W.data.astype(np.float64).copy()
        self.delay = np.asarray(delay, dtype=np.int32).copy()
        self._rebuild_buckets()
        self.buf = np.zeros((self.dmax + 1, self.n, self.B), dtype=np.float64)
        self.t = 0

    def _rebuild_buckets(self):
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = self.delay == d
            if m.any():
                Wd = sp.csr_matrix(
                    (self.data[m], (self.row[m], self.col[m])),
                    shape=(self.n, self.n),
                )
            else:
                Wd = None
            self.buckets.append(Wd)

    def step(self, x_in):
        """x_in: (n,) or (n, B). Returns tanh(h + x_in) same shape as stacked."""
        self.t += 1
        slot = self.t % (self.dmax + 1)
        x_in = np.asarray(x_in, dtype=np.float64)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        h = np.zeros((self.n, self.B), dtype=np.float64)
        for di, Wd in enumerate(self.buckets):
            d = di + 1
            if Wd is None:
                continue
            src_slot = (self.t - d) % (self.dmax + 1)
            h += Wd @ self.buf[src_slot]
        y = np.tanh(h + x_in)
        self.buf[slot] = y
        return y


def run_delay_reservoir(W, delay, dmax, Win, b, U, leak, seed=0, adapt=False):
    """Single-trajectory delay reservoir — bit-matches experiment loops (no adapt).

    adapt=True falls back to a local copy of the myelin_real adaptive loop so we
    do not claim a batched adaptive path (adaptive rebuilds are serial by nature).
    """
    if adapt:
        return _run_delay_adapt(W, delay, dmax, Win, b, U, leak, seed=seed)

    dr = DelayReservoirBatched(W, delay, dmax, B=1)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = W.shape[0]
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros((n, 1), dtype=np.float64)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = (Win[:, 0] * U[i] + b) if not multi else (Win @ U[i] + b)
        y = dr.step(drive)  # (n, 1)
        x = (1.0 - leak) * x + leak * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = x[:, 0]
    return X


def _run_delay_adapt(W, delay, dmax, Win, b, U, leak, seed=0):
    """Exact copy of myelin_real adaptive delay loop (serial)."""
    g = np.random.default_rng(seed + 11)
    Wcoo = W.tocoo()
    n = W.shape[0]
    row = Wcoo.row.copy()
    col = Wcoo.col.copy()
    data = Wcoo.data.astype(np.float64).copy()
    delay = np.asarray(delay, dtype=np.int32).copy()
    dmax = int(dmax)

    def rebuild():
        buckets = []
        for d in range(1, dmax + 1):
            m = delay == d
            if m.any():
                buckets.append(
                    sp.csr_matrix((data[m], (row[m], col[m])), shape=(n, n))
                )
            else:
                buckets.append(None)
        return buckets

    buckets = rebuild()
    buf = np.zeros((dmax + 1, n), dtype=np.float64)
    t = 0
    pre_trace = np.zeros(n)
    post_trace = np.zeros(n)
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n)
    multi = U.ndim == 2
    for i in range(len(U)):
        drive = Win[:, 0] * U[i] + b if not multi else Win @ U[i] + b
        t += 1
        slot = t % (dmax + 1)
        h = np.zeros(n, dtype=np.float64)
        for di, Wd in enumerate(buckets):
            d = di + 1
            if Wd is None:
                continue
            src_slot = (t - d) % (dmax + 1)
            h += Wd @ buf[src_slot]
        y = np.tanh(h + drive)
        buf[slot] = y
        pre_trace = 0.9 * pre_trace + 0.1 * (buf[(t - 1) % (dmax + 1)] ** 2)
        post_trace = 0.9 * post_trace + 0.1 * (y ** 2)
        co = pre_trace[col] * post_trace[row]
        med = np.median(co)
        shorten = co > med
        delay[shorten] = np.maximum(1, delay[shorten] - 1)
        lengthen = co < np.percentile(co, 20)
        delay[lengthen] = np.minimum(dmax, delay[lengthen] + 1)
        if t % 50 == 0:
            buckets = rebuild()
        x = (1.0 - leak) * x + leak * y
        buf[t % (dmax + 1)] = x
        X[i] = x
    return X


def run_delay_reservoir_batch(W, delay, dmax, Wins, bs, Us, leak):
    """Same W+delay; B independent drives. B==1 → run_delay_reservoir."""
    Wins = list(Wins)
    bs = [np.asarray(b, dtype=np.float64) for b in bs]
    Us = _as_U_list(Us)
    B = len(Us)
    if B == 0:
        return []
    if B == 1:
        return [run_delay_reservoir(W, delay, dmax, Wins[0], bs[0], Us[0], leak)]

    T = len(Us[0])
    for U in Us:
        assert len(U) == T
    multi = Us[0].ndim == 2
    n = W.shape[0]
    dr = DelayReservoirBatched(W, delay, dmax, B=B)
    X_out = [np.empty((T, n), dtype=np.float64) for _ in range(B)]
    x = np.zeros((n, B), dtype=np.float64)
    STATS["batch_calls"] += 1
    STATS["batch_columns"] += B

    if not multi:
        Win_stack = np.column_stack(
            [np.asarray(Wins[b], dtype=np.float64)[:, 0] for b in range(B)]
        )
        U_mat = np.column_stack(Us)
        b_stack = np.column_stack(bs)
        for i in range(T):
            drive = Win_stack * U_mat[i] + b_stack
            y = dr.step(drive)
            x = (1.0 - leak) * x + leak * y
            dr.buf[dr.t % (dr.dmax + 1)] = x
            for b in range(B):
                X_out[b][i] = x[:, b]
    else:
        for i in range(T):
            drive = np.column_stack(
                [np.asarray(Wins[b]) @ Us[b][i] + bs[b] for b in range(B)]
            )
            y = dr.step(drive)
            x = (1.0 - leak) * x + leak * y
            dr.buf[dr.t % (dr.dmax + 1)] = x
            for b in range(B):
                X_out[b][i] = x[:, b]
    return X_out


def run_delay_blockdiag(Ws, delays, dmax, Wins, bs, Us, leak):
    """Different (W, delay) pairs via block-diag. Equal T required.

    Delay buckets are per-block; we implement by concatenating states and
    building block-diag buckets per delay value.
    """
    Ws = list(Ws)
    delays = list(delays)
    Wins = list(Wins)
    bs = [np.asarray(b, dtype=np.float64) for b in bs]
    Us = _as_U_list(Us)
    B = len(Ws)
    if B == 0:
        return []
    if B == 1:
        return [run_delay_reservoir(Ws[0], delays[0], dmax, Wins[0], bs[0], Us[0], leak)]

    T = len(Us[0])
    for U in Us:
        assert len(U) == T
    multi = Us[0].ndim == 2
    ns = [W.shape[0] for W in Ws]
    N = sum(ns)
    offs = np.cumsum([0] + ns)
    dmax = int(dmax)

    # Per-delay block-diagonal Wd
    buckets = []
    for d in range(1, dmax + 1):
        blocks = []
        any_d = False
        for b, W in enumerate(Ws):
            Wcoo = W.tocoo()
            m = np.asarray(delays[b], dtype=np.int32) == d
            if m.any():
                any_d = True
                blocks.append(
                    sp.csr_matrix(
                        (Wcoo.data[m], (Wcoo.row[m], Wcoo.col[m])),
                        shape=(ns[b], ns[b]),
                    )
                )
            else:
                blocks.append(sp.csr_matrix((ns[b], ns[b]), dtype=np.float64))
        buckets.append(sp.block_diag(blocks, format="csr") if any_d else None)

    buf = np.zeros((dmax + 1, N), dtype=np.float64)
    X_big = np.empty((T, N), dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    t = 0
    STATS["batch_calls"] += 1
    STATS["batch_columns"] += B

    for i in range(T):
        drives = []
        for b in range(B):
            Ub = Us[b][i]
            if not multi:
                drives.append(np.asarray(Wins[b], dtype=np.float64)[:, 0] * Ub + bs[b])
            else:
                drives.append(np.asarray(Wins[b], dtype=np.float64) @ Ub + bs[b])
        drive = np.concatenate(drives)
        t += 1
        slot = t % (dmax + 1)
        h = np.zeros(N, dtype=np.float64)
        for di, Wd in enumerate(buckets):
            d = di + 1
            if Wd is None:
                continue
            src_slot = (t - d) % (dmax + 1)
            h += Wd @ buf[src_slot]
        y = np.tanh(h + drive)
        buf[slot] = y
        x = (1.0 - leak) * x + leak * y
        buf[t % (dmax + 1)] = x
        X_big[i] = x

    return [X_big[:, offs[b] : offs[b + 1]].copy() for b in range(B)]


# ---------------------------------------------------------------------------
# (2) State reuse — simulate once, score many readouts
# ---------------------------------------------------------------------------
def _fingerprint(parts: list) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None:
            h.update(b"none")
        elif isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, str):
            h.update(p.encode())
        elif isinstance(p, (int, float)):
            h.update(repr(p).encode())
        elif isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def state_key(arm, seed, input_tag, extra="") -> str:
    return f"{arm}|{seed}|{input_tag}|{extra}"


def cache_states(key: str, X: np.ndarray, meta: dict | None = None):
    global _STATE_CACHE_BYTES
    arr = np.asarray(X, dtype=np.float64)
    nbytes = arr.nbytes
    # Evict arbitrarily if over budget
    while _STATE_CACHE and _STATE_CACHE_BYTES + nbytes > _STATE_CACHE_MAX_BYTES:
        k0 = next(iter(_STATE_CACHE))
        _STATE_CACHE_BYTES -= _STATE_CACHE.pop(k0)["X"].nbytes
    _STATE_CACHE[key] = {"X": arr, "meta": meta or {}}
    _STATE_CACHE_BYTES += nbytes
    STATS["state_cache_miss"] += 1
    return arr


def get_cached_states(key: str):
    hit = _STATE_CACHE.get(key)
    if hit is None:
        return None
    STATS["state_cache_hit"] += 1
    return hit["X"]


def simulate_once_score_many(simulate_fn, score_fns, cache_key: str | None = None):
    """Run simulate_fn() once (or load cache); apply each score_fn(X).

    score_fns: list of callables X -> metric (or dict).
    Returns (X, [scores...]).
    """
    X = get_cached_states(cache_key) if cache_key else None
    if X is None:
        X = simulate_fn()
        if cache_key:
            cache_states(cache_key, X)
    scores = [fn(X) for fn in score_fns]
    return X, scores


# ---------------------------------------------------------------------------
# (2b) Disk cache for random control graphs
# ---------------------------------------------------------------------------
def _graph_path(name: str, kind: str, seed: int) -> Path:
    safe = f"{name}__{kind}__seed{int(seed)}.npz"
    return GRAPH_DIR / safe


def save_sparse(path: Path, W, sign, layer, ct):
    W = W.tocsr()
    np.savez_compressed(
        path,
        data=W.data,
        indices=W.indices,
        indptr=W.indptr,
        shape=np.array(W.shape),
        sign=np.asarray(sign),
        layer=np.asarray(layer),
        ct=np.asarray(ct, dtype=object),
    )


def load_sparse(path: Path):
    z = np.load(path, allow_pickle=True)
    W = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    )
    return W, z["sign"], z["layer"], z["ct"]


def cached_make_block(make_block_fn, name: str, kind: str, seed: int):
    """Disk-cache ER/SHUF/DENS (and optionally real) make_block results.

    Real blocks are also cached — they are deterministic and rebuild cost is
    mostly spectral scaling + load; caching keeps them bit-identical across
    process restarts too.
    """
    path = _graph_path(name, kind, seed)
    if path.exists():
        try:
            STATS["graph_cache_hit"] += 1
            return load_sparse(path)
        except Exception:
            pass
    STATS["graph_cache_miss"] += 1
    Wm, sign, layer, ct = make_block_fn(name, kind, seed)
    try:
        save_sparse(path, Wm, sign, layer, ct)
    except Exception:
        pass
    return Wm, sign, layer, ct


def wrap_frankenstein_make_block(F):
    """Monkeypatch F.make_block with disk-cached version (opt-in, process-local).

    Returns a restore() callable. Safe for verification; do not use on
    currently-running experiment processes (they have their own memory).
    """
    orig = F.make_block

    def cached(name, kind, seed):
        return cached_make_block(orig, name, kind, seed)

    F.make_block = cached
    return lambda: setattr(F, "make_block", orig)


# ---------------------------------------------------------------------------
# Convenience: multi-readout scorers matching experiment metrics
# ---------------------------------------------------------------------------
def score_mc(X, u, wash=200, tr=1500, te=500, D=100, lam=1e-6):
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = ridge(X[wash : wash + tr], Y[wash : wash + tr], lam)
    P = pred(X[wash + tr :], B)
    Yt = Y[wash + tr :]
    return float(
        sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D))
    )


def score_narma(X, y, wash=200, tr=1500, te=500, lam=1e-6):
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return float("nan")
    B = ridge(X[wash : wash + tr], y[wash : wash + tr], lam)
    p = pred(X[wash + tr :], B)
    yt = y[wash + tr :]
    if (not np.all(np.isfinite(p))) or np.std(yt) < 1e-12:
        return float("nan")
    v = float(np.sqrt(np.mean((p - yt) ** 2)) / np.std(yt))
    return v if np.isfinite(v) else float("nan")


def score_pc_hard(X, u_clean, wash=200, tr=800):
    Wout = np.linalg.lstsq(X[wash:tr], u_clean[wash:tr], rcond=None)[0]
    pred_ = X[tr:] @ Wout
    yt = u_clean[tr:]
    return float(
        1
        - ((yt - pred_) ** 2).sum()
        / (((yt - yt.mean(0)) ** 2).sum() + 1e-12)
    )


def score_mc_grid(X, u, n_trains, alpha, wash=200, tr=3000, te=500, D=100):
    """Reuse one X for many n_train values (fewshot pattern)."""
    return [_score_mc_ntrain(X, u, nt, alpha, wash, tr, te, D) for nt in n_trains]


def _score_mc_ntrain(X, u, n_train, alpha, wash, tr, te, D):
    n_train = min(int(n_train), tr)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = ridge(X[wash : wash + n_train], Y[wash : wash + n_train], alpha)
    P = pred(X[wash + tr :], B)
    Yt = Y[wash + tr :]
    return float(
        sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D))
    )

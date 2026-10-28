"""Opt-in CuPy GPU backend next to fastcore2 — does NOT change defaults.

Enable (opt-in):
  export FLYBRAIN_GPU=1
  import gpu_core as GC; GC.enable()

When CuPy is absent, every public function falls back to the CPU path
(fast_kernels / numpy / scipy). float64 by default; target match vs CPU
within 1e-6 relative.

Running experiments that do not import this module are untouched.
Do not import this from mega_franken / trained_bridges while they run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# CuPy detection
# ---------------------------------------------------------------------------
_HAS_CUPY = False
cp = None
csp = None
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp

    # Require a real device; import alone is not enough on CPU-only boxes
    try:
        _cp.cuda.Device(0).use()
        _ = _cp.zeros(1, dtype=_cp.float64) + 1.0
        cp = _cp
        csp = _csp
        _HAS_CUPY = True
    except Exception:
        _HAS_CUPY = False
        cp = None
        csp = None
except Exception:
    _HAS_CUPY = False
    cp = None
    csp = None

# Re-use CPU kernels for fallback + radius cache
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "compress"))
import fast_kernels as K  # noqa: E402

# Bound at import time — safe CPU fallbacks even after enable() patches K.*
_CPU_ridge = K.ridge
_CPU_pred = K.pred
_CPU_run_reservoir_csr = K.run_reservoir_csr

DTYPE = np.float64
GPU_ENV = "FLYBRAIN_GPU"

STATS = {
    "backend": "cpu",
    "cupy_available": _HAS_CUPY,
    "enabled": False,
    "gpu_calls": 0,
    "cpu_fallback_calls": 0,
}

_ORIG_PATCHES: dict = {}
_ENABLED = False


def cupy_available() -> bool:
    return bool(_HAS_CUPY)


def gpu_wanted() -> bool:
    return os.environ.get(GPU_ENV, "").strip() in ("1", "true", "True", "yes", "YES")


def using_gpu() -> bool:
    if os.environ.get("_FLYBRAIN_GPU_FORCE_CPU") == "1":
        return False
    return _ENABLED and _HAS_CUPY


def reset_stats():
    STATS["gpu_calls"] = 0
    STATS["cpu_fallback_calls"] = 0
    STATS["backend"] = "gpu" if using_gpu() else "cpu"
    STATS["cupy_available"] = _HAS_CUPY
    STATS["enabled"] = _ENABLED


# ---------------------------------------------------------------------------
# Array / sparse helpers
# ---------------------------------------------------------------------------
def _as_np64(a):
    return np.asarray(a, dtype=DTYPE)


def _to_gpu_csr(W):
    """SciPy sparse → cupyx CSR float64."""
    W = W.tocsr()
    return csp.csr_matrix(
        (
            cp.asarray(W.data, dtype=cp.float64),
            cp.asarray(W.indices, dtype=cp.int32),
            cp.asarray(W.indptr, dtype=cp.int32),
        ),
        shape=W.shape,
    )


def _to_gpu(a):
    return cp.asarray(np.asarray(a, dtype=DTYPE))


# ---------------------------------------------------------------------------
# Ridge / pred (GPU when enabled)
# ---------------------------------------------------------------------------
def ridge(X, Y, lam):
    """Ridge with bias column. Dual form when n_samples < n_features."""
    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        return _CPU_ridge(X, Y, lam)

    STATS["gpu_calls"] += 1
    Y = _as_np64(Y)
    squeeze = False
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    Xg = _to_gpu(X)
    Yg = _to_gpu(Y)
    n, d = int(Xg.shape[0]), int(Xg.shape[1])
    if n < d:
        G = Xg @ Xg.T
        G += 1.0
        G[cp.diag_indices(n)] += lam
        Z = cp.linalg.solve(G, Yg)
        B = cp.vstack([Xg.T @ Z, Z.sum(axis=0, keepdims=True)])
    else:
        ones = cp.ones((n, 1), dtype=cp.float64)
        Xb = cp.hstack([Xg, ones])
        A = Xb.T @ Xb + lam * cp.eye(Xb.shape[1], dtype=cp.float64)
        B = cp.linalg.solve(A, Xb.T @ Yg)
    Bout = cp.asnumpy(B)
    return Bout[:, 0] if squeeze else Bout


def pred(X, B):
    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        return _CPU_pred(X, B)
    STATS["gpu_calls"] += 1
    Xg = _to_gpu(X)
    Bg = _to_gpu(B)
    ones = cp.ones((Xg.shape[0], 1), dtype=cp.float64)
    return cp.asnumpy(cp.hstack([Xg, ones]) @ Bg)


# ---------------------------------------------------------------------------
# CSR reservoir (single + batched seeds)
# ---------------------------------------------------------------------------
def run_reservoir_csr(W, Win, b, U, leak):
    """Reservoir rollout; W CSR. U (T,) or (T,d). float64.
    GPU path when enabled+CuPy; else fast_kernels (bit-identical CPU).
    """
    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        return _CPU_run_reservoir_csr(W, Win, b, U, leak)

    STATS["gpu_calls"] += 1
    Wg = _to_gpu_csr(W)
    n = Wg.shape[0]
    U = _as_np64(U)
    Win = _as_np64(Win)
    b = _as_np64(b)
    T = len(U)
    multi = U.ndim == 2
    Ug = _to_gpu(U)
    bg = _to_gpu(b)
    x = cp.zeros(n, dtype=cp.float64)
    X = cp.empty((T, n), dtype=cp.float64)
    if not multi:
        win0 = _to_gpu(Win[:, 0])
        for i in range(T):
            x = (1.0 - leak) * x + leak * cp.tanh(Wg @ x + win0 * Ug[i] + bg)
            X[i] = x
    else:
        Wing = _to_gpu(Win)
        for i in range(T):
            x = (1.0 - leak) * x + leak * cp.tanh(Wg @ x + Wing @ Ug[i] + bg)
            X[i] = x
    return cp.asnumpy(X)


def run_reservoir_csr_batch(W, Wins, bs, Us, leak):
    """Same W; B independent (Win, b, U) streams. Returns list of (T, n)."""
    Wins = list(Wins)
    bs = [_as_np64(b) for b in bs]
    Us = [_as_np64(U) for U in Us]
    B = len(Us)
    if B == 0:
        return []
    if B == 1:
        return [run_reservoir_csr(W, Wins[0], bs[0], Us[0], leak)]

    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        # Prefer fastcore2 batch if present, else serial CPU
        try:
            import fastcore2 as FC2

            return FC2.run_reservoir_csr_batch(W, Wins, bs, Us, leak)
        except Exception:
            return [_CPU_run_reservoir_csr(W, Wins[b], bs[b], Us[b], leak) for b in range(B)]

    STATS["gpu_calls"] += 1
    T = len(Us[0])
    for U in Us:
        assert len(U) == T
    multi = Us[0].ndim == 2
    Wg = _to_gpu_csr(W)
    n = Wg.shape[0]
    b_stack = _to_gpu(np.column_stack(bs))
    x = cp.zeros((n, B), dtype=cp.float64)
    X_out = [np.empty((T, n), dtype=DTYPE) for _ in range(B)]

    if not multi:
        Win_stack = _to_gpu(
            np.column_stack([_as_np64(Wins[b])[:, 0] for b in range(B)])
        )
        U_mat = _to_gpu(np.column_stack(Us))
        for i in range(T):
            drive = Win_stack * U_mat[i] + b_stack
            x = (1.0 - leak) * x + leak * cp.tanh(Wg @ x + drive)
            xn = cp.asnumpy(x)
            for b in range(B):
                X_out[b][i] = xn[:, b]
    else:
        Wins_g = [_to_gpu(Wins[b]) for b in range(B)]
        Us_g = [_to_gpu(Us[b]) for b in range(B)]
        for i in range(T):
            drive = cp.column_stack(
                [Wins_g[b] @ Us_g[b][i] + b_stack[:, b] for b in range(B)]
            )
            x = (1.0 - leak) * x + leak * cp.tanh(Wg @ x + drive)
            xn = cp.asnumpy(x)
            for b in range(B):
                X_out[b][i] = xn[:, b]
    return X_out


# ---------------------------------------------------------------------------
# Delay reservoir (bucketed CSR) — used by myelin / best_fly
# ---------------------------------------------------------------------------
class DelayReservoirGPU:
    """Bucketed CSR delays; state on device when GPU enabled."""

    def __init__(self, W, delay, dmax, B=1, use_gpu=None):
        if use_gpu is None:
            use_gpu = using_gpu()
        self.use_gpu = bool(use_gpu) and _HAS_CUPY
        W = W.tocoo()
        self.n = W.shape[0]
        self.dmax = int(dmax)
        self.B = int(B)
        self.row = W.row.copy()
        self.col = W.col.copy()
        self.data = W.data.astype(DTYPE).copy()
        self.delay = np.asarray(delay, dtype=np.int32).copy()
        self.t = 0
        self._rebuild_buckets()
        if self.use_gpu:
            self.buf = cp.zeros((self.dmax + 1, self.n, self.B), dtype=cp.float64)
        else:
            self.buf = np.zeros((self.dmax + 1, self.n, self.B), dtype=DTYPE)

    def _rebuild_buckets(self):
        self.buckets = []
        for d in range(1, self.dmax + 1):
            m = self.delay == d
            if not m.any():
                self.buckets.append(None)
                continue
            Wd_cpu = sp.csr_matrix(
                (self.data[m], (self.row[m], self.col[m])),
                shape=(self.n, self.n),
            )
            if self.use_gpu:
                self.buckets.append(_to_gpu_csr(Wd_cpu))
            else:
                self.buckets.append(Wd_cpu)

    def step(self, x_in):
        self.t += 1
        slot = self.t % (self.dmax + 1)
        if self.use_gpu:
            x_in = cp.asarray(x_in, dtype=cp.float64)
            if x_in.ndim == 1:
                x_in = x_in[:, None]
            h = cp.zeros((self.n, self.B), dtype=cp.float64)
            for di, Wd in enumerate(self.buckets):
                d = di + 1
                if Wd is None:
                    continue
                src_slot = (self.t - d) % (self.dmax + 1)
                h += Wd @ self.buf[src_slot]
            y = cp.tanh(h + x_in)
            self.buf[slot] = y
            return y
        # CPU
        x_in = np.asarray(x_in, dtype=DTYPE)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        h = np.zeros((self.n, self.B), dtype=DTYPE)
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
    """Leak-integrated delay reservoir. Matches myelin_real / fastcore2 (no adapt)."""
    if adapt:
        # Adaptive rebuilds are serial by nature — always CPU reference
        STATS["cpu_fallback_calls"] += 1
        try:
            import fastcore2 as FC2

            return FC2.run_delay_reservoir(
                W, delay, dmax, Win, b, U, leak, seed=seed, adapt=True
            )
        except Exception:
            sys.path.insert(0, str(_ROOT / "regions" / "myelin_real"))
            import myelin_real as MR

            return MR.run_delay_reservoir(
                W, delay, dmax, Win, b, U, leak, adapt=True, seed=seed
            )

    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        try:
            import fastcore2 as FC2

            return FC2.run_delay_reservoir(W, delay, dmax, Win, b, U, leak, seed=seed)
        except Exception:
            sys.path.insert(0, str(_ROOT / "regions" / "myelin_real"))
            import myelin_real as MR

            return MR.run_delay_reservoir(
                W, delay, dmax, Win, b, U, leak, adapt=False, seed=seed
            )

    STATS["gpu_calls"] += 1
    dr = DelayReservoirGPU(W, delay, dmax, B=1, use_gpu=True)
    U = _as_np64(U)
    Win = _as_np64(Win)
    b = _as_np64(b)
    n = W.shape[0]
    X = np.empty((len(U), n), dtype=DTYPE)
    x = cp.zeros((n, 1), dtype=cp.float64)
    multi = U.ndim == 2
    Wing = _to_gpu(Win)
    bg = _to_gpu(b)
    Ug = _to_gpu(U)
    for i in range(len(U)):
        if not multi:
            drive = Wing[:, 0] * Ug[i] + bg
        else:
            drive = Wing @ Ug[i] + bg
        y = dr.step(drive)  # (n, 1) on device
        x = (1.0 - leak) * x + leak * y
        dr.buf[dr.t % (dr.dmax + 1)] = x
        X[i] = cp.asnumpy(x[:, 0])
    return X


def run_delay_reservoir_batch(W, delay, dmax, Wins, bs, Us, leak):
    Wins = list(Wins)
    bs = [_as_np64(b) for b in bs]
    Us = [_as_np64(U) for U in Us]
    B = len(Us)
    if B == 0:
        return []
    if B == 1:
        return [run_delay_reservoir(W, delay, dmax, Wins[0], bs[0], Us[0], leak)]

    if not using_gpu():
        STATS["cpu_fallback_calls"] += 1
        try:
            import fastcore2 as FC2

            return FC2.run_delay_reservoir_batch(W, delay, dmax, Wins, bs, Us, leak)
        except Exception:
            return [
                run_delay_reservoir(W, delay, dmax, Wins[b], bs[b], Us[b], leak)
                for b in range(B)
            ]

    STATS["gpu_calls"] += 1
    T = len(Us[0])
    multi = Us[0].ndim == 2
    n = W.shape[0]
    dr = DelayReservoirGPU(W, delay, dmax, B=B, use_gpu=True)
    X_out = [np.empty((T, n), dtype=DTYPE) for _ in range(B)]
    x = cp.zeros((n, B), dtype=cp.float64)
    b_stack = _to_gpu(np.column_stack(bs))

    if not multi:
        Win_stack = _to_gpu(
            np.column_stack([_as_np64(Wins[b])[:, 0] for b in range(B)])
        )
        U_mat = _to_gpu(np.column_stack(Us))
        for i in range(T):
            drive = Win_stack * U_mat[i] + b_stack
            y = dr.step(drive)
            x = (1.0 - leak) * x + leak * y
            dr.buf[dr.t % (dr.dmax + 1)] = x
            xn = cp.asnumpy(x)
            for b in range(B):
                X_out[b][i] = xn[:, b]
    else:
        Wins_g = [_to_gpu(Wins[b]) for b in range(B)]
        Us_g = [_to_gpu(Us[b]) for b in range(B)]
        for i in range(T):
            drive = cp.column_stack(
                [Wins_g[b] @ Us_g[b][i] + b_stack[:, b] for b in range(B)]
            )
            y = dr.step(drive)
            x = (1.0 - leak) * x + leak * y
            dr.buf[dr.t % (dr.dmax + 1)] = x
            xn = cp.asnumpy(x)
            for b in range(B):
                X_out[b][i] = xn[:, b]
    return X_out


# ---------------------------------------------------------------------------
# Opt-in enable / disable — monkeypatch fast_kernels (process-local)
# ---------------------------------------------------------------------------
def enable(force_cpu: bool = False):
    """Opt-in: patch fast_kernels ridge/pred/run_reservoir_csr to gpu_core.

    If CuPy is missing or force_cpu=True, patches still install but functions
    use the CPU fallback (so callers can always `enable()` safely).
    """
    global _ENABLED
    if _ENABLED:
        return using_gpu()

    _ORIG_PATCHES["ridge"] = K.ridge
    _ORIG_PATCHES["pred"] = K.pred
    _ORIG_PATCHES["run_reservoir_csr"] = K.run_reservoir_csr

    K.ridge = ridge
    K.pred = pred
    K.run_reservoir_csr = run_reservoir_csr

    # Soft-patch myelin_real delay path if already imported
    for mod_name in ("myelin_real", "fastcore2"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        if hasattr(mod, "run_delay_reservoir"):
            key = f"{mod_name}.run_delay_reservoir"
            if key not in _ORIG_PATCHES:
                _ORIG_PATCHES[key] = mod.run_delay_reservoir
            mod.run_delay_reservoir = run_delay_reservoir
        if hasattr(mod, "ridge"):
            key = f"{mod_name}.ridge"
            if key not in _ORIG_PATCHES:
                _ORIG_PATCHES[key] = mod.ridge
            mod.ridge = ridge

    _ENABLED = True
    if force_cpu or not _HAS_CUPY:
        # Keep enabled flag for API, but using_gpu() stays False because we
        # temporarily clear the "want gpu" by not having cupy — using_gpu
        # already requires _HAS_CUPY. force_cpu: set a sentinel.
        if force_cpu:
            os.environ["_FLYBRAIN_GPU_FORCE_CPU"] = "1"
        STATS["backend"] = "cpu"
    else:
        os.environ.pop("_FLYBRAIN_GPU_FORCE_CPU", None)
        # Mark wanted so using_gpu is True even if env was unset
        if not gpu_wanted():
            os.environ[GPU_ENV] = "1"
        STATS["backend"] = "gpu"
    STATS["enabled"] = True
    STATS["cupy_available"] = _HAS_CUPY
    return using_gpu()


def disable():
    """Restore fast_kernels (and any soft-patched modules) to originals."""
    global _ENABLED
    if not _ENABLED:
        return
    if "ridge" in _ORIG_PATCHES:
        K.ridge = _ORIG_PATCHES["ridge"]
        K.pred = _ORIG_PATCHES["pred"]
        K.run_reservoir_csr = _ORIG_PATCHES["run_reservoir_csr"]
    for key, orig in list(_ORIG_PATCHES.items()):
        if "." in key:
            mod_name, attr = key.split(".", 1)
            mod = sys.modules.get(mod_name)
            if mod is not None:
                setattr(mod, attr, orig)
    _ORIG_PATCHES.clear()
    _ENABLED = False
    os.environ.pop("_FLYBRAIN_GPU_FORCE_CPU", None)
    STATS["enabled"] = False
    STATS["backend"] = "cpu"


radius_cached = K.radius_cached

__all__ = [
    "cupy_available",
    "gpu_wanted",
    "using_gpu",
    "enable",
    "disable",
    "ridge",
    "pred",
    "run_reservoir_csr",
    "run_reservoir_csr_batch",
    "run_delay_reservoir",
    "run_delay_reservoir_batch",
    "DelayReservoirGPU",
    "STATS",
    "reset_stats",
    "radius_cached",
    "DTYPE",
    "GPU_ENV",
]

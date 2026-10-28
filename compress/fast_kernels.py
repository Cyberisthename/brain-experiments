"""Fast numerically-near-equivalent kernels for hybrid/frankenstein.
Mouse hybrid does NOT import this module (copy-on-write isolation).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

# ---- spectral radius cache ----
_RADIUS_CACHE: dict[int, float] = {}

def radius_cached(W, tol=1e-6, maxiter=20000):
    """Cache by (shape, nnz, data id) — safe within one process/job."""
    key = (W.shape[0], W.shape[1], W.nnz, int(W.data[: min(8, W.nnz)].sum() * 1e6) if W.nnz else 0)
    if key in _RADIUS_CACHE:
        return _RADIUS_CACHE[key]
    from scipy.sparse.linalg import eigs
    if sp.issparse(W):
        r = float(abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=maxiter, tol=tol)[0]))
    else:
        r = float(np.max(np.abs(np.linalg.eigvals(W))))
    _RADIUS_CACHE[key] = r
    return r


def ridge(X, Y, lam):
    """Ridge with bias column. Uses dual form when n_samples < n_features (same math)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    else:
        squeeze = False
    n, d = X.shape
    if n < d:
        # Dual: W = Φ.T (ΦΦ.T + λI)^{-1} Y, Φ=[X,1]
        G = X @ X.T
        G += 1.0  # ones @ ones.T
        G[np.diag_indices(n)] += lam
        Z = np.linalg.solve(G, Y)
        B = np.vstack([X.T @ Z, Z.sum(axis=0, keepdims=True)])
    else:
        Xb = np.hstack([X, np.ones((n, 1))])
        B = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return B[:, 0] if squeeze else B


def pred(X, B):
    return np.hstack([X, np.ones((len(X), 1))]) @ B


def run_reservoir_csr(W, Win, b, U, leak):
    """Reservoir rollout; W CSR. U (T,) or (T,d). float64.
    Uses SciPy CSR matvec (faster than pure-numba on this box); bit-matches old loop.
    """
    W = W.tocsr()
    n = W.shape[0]
    U = np.asarray(U, dtype=np.float64)
    Win = np.asarray(Win, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    X = np.empty((len(U), n), dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    multi = U.ndim == 2
    if not multi:
        win0 = Win[:, 0]
        for i in range(len(U)):
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + win0 * U[i] + b)
            X[i] = x
    else:
        for i in range(len(U)):
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win @ U[i] + b)
            X[i] = x
    return X


def oja_update_csr_fast(Wm, h_pre, h_post, eta):
    """In-place Oja; precomputes row index once via indptr (no np.repeat each call)."""
    if eta <= 0 or Wm.nnz == 0:
        return
    try:
        from numba import njit
        indptr = Wm.indptr
        indices = Wm.indices
        data = Wm.data

        @njit(cache=True)
        def _oja(indptr, indices, data, h_pre, h_post, eta):
            n = indptr.shape[0] - 1
            for i in range(n):
                hi = h_post[i]
                for jj in range(indptr[i], indptr[i + 1]):
                    j = indices[jj]
                    w = data[jj]
                    data[jj] = w + eta * hi * (h_pre[j] - hi * w)

        _oja(indptr, indices, data, h_pre, h_post, float(eta))
    except Exception:
        rows = np.repeat(np.arange(Wm.shape[0]), np.diff(Wm.indptr))
        cols = Wm.indices
        data = Wm.data
        data += eta * h_post[rows] * (h_pre[cols] - h_post[rows] * data)


def put_Wm_fast(W, Wm):
    """Replace top-left block without full tolil of entire W."""
    Nm = Wm.shape[0]
    W = W.tocsr()
    # zero out old block via subtract, then add new — use block assemble
    top_right = W[:Nm, Nm:]
    bot = W[Nm:, :]
    top = sp.hstack([Wm.tocsr(), top_right], format="csr")
    return sp.vstack([top, bot], format="csr")

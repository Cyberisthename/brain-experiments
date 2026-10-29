"""Filtered spike traces -> ridge readout (same math as fast_kernels)."""
from __future__ import annotations
import numpy as np


def filter_spikes(spikes: np.ndarray, dt_ms: float, tau_ms: float = 20.0) -> np.ndarray:
    """Causal exp filter. spikes (T,N) or (T,N,B) -> same shape float32."""
    spikes = np.asarray(spikes, dtype=np.float32)
    exp_f = np.float32(np.exp(-dt_ms / tau_ms))
    out = np.empty_like(spikes)
    if spikes.ndim == 2:
        f = np.zeros(spikes.shape[1], dtype=np.float32)
        for t in range(len(spikes)):
            f = f * exp_f + spikes[t]
            out[t] = f
    else:
        f = np.zeros(spikes.shape[1:], dtype=np.float32)
        for t in range(len(spikes)):
            f = f * exp_f + spikes[t]
            out[t] = f
    return out


def ridge(X, Y, lam: float = 1e-4):
    """Ridge with bias column. Dual form when n_samples < n_features."""
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    else:
        squeeze = False
    n, d = X.shape
    if n < d:
        G = X @ X.T
        G += 1.0
        G[np.diag_indices(n)] += lam
        Z = np.linalg.solve(G, Y)
        B = np.vstack([X.T @ Z, Z.sum(axis=0, keepdims=True)])
    else:
        Xb = np.hstack([X, np.ones((n, 1))])
        B = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return B[:, 0] if squeeze else B


def pred(X, B):
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((len(X), 1))]) @ B

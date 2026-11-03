"""Membrane Phase-Sliding: virtual 3D coords, warp bridges, phase wave."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


def spectral_embed(mask: np.ndarray, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Laplacian eigenmap on undirected mask graph. Returns (N, dim)."""
    A = ((mask | mask.T) & ~np.eye(mask.shape[0], dtype=bool)).astype(np.float64)
    d = A.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    # eigh ascending; skip trivial
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)
    # take dim eigenvectors after the smallest
    comps = []
    for i in idx:
        if w[i] < 1e-8:
            continue
        comps.append(v[:, i])
        if len(comps) >= dim:
            break
    while len(comps) < dim:
        rng = np.random.default_rng(seed + len(comps))
        comps.append(rng.standard_normal(A.shape[0]))
    Z = np.stack(comps[:dim], 1)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    return Z.astype(np.float64)


def init_coords_from_griffa(sc: np.ndarray, n_local: int, hemis=None, seed: int = 0) -> np.ndarray:
    """Broadcast region coords to units. x from hemisphere, yz from SC spectral embed on 68."""
    R = sc.shape[0]
    # region embedding
    A = (sc > 0).astype(np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    # spectral on 68
    d = A.sum(1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    L = np.eye(R) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    w, v = np.linalg.eigh(L)
    idx = np.argsort(w)
    comps = []
    for i in idx:
        if abs(w[i]) < 1e-8:
            continue
        comps.append(v[:, i])
        if len(comps) >= 2:
            break
    while len(comps) < 2:
        comps.append(np.random.default_rng(seed).standard_normal(R))
    y = comps[0]; z = comps[1]
    y = (y - y.mean()) / (y.std() + 1e-8)
    z = (z - z.mean()) / (z.std() + 1e-8)
    if hemis is None:
        x = np.concatenate([np.full(R // 2, -1.0), np.full(R - R // 2, 1.0)])
    else:
        hemis = np.asarray(hemis).astype(str)
        x = np.array([ -1.0 if h.upper().startswith("L") else 1.0 for h in hemis ])
    Z_r = np.stack([x, y, z], 1)  # (68,3)
    N = R * n_local
    Z = np.zeros((N, 3), dtype=np.float64)
    for r in range(R):
        Z[r * n_local:(r + 1) * n_local] = Z_r[r]
        # tiny jitter so units aren't identical
        rng = np.random.default_rng(seed + r)
        Z[r * n_local:(r + 1) * n_local] += 0.01 * rng.standard_normal((n_local, 3))
    return Z


@dataclass
class MPSState:
    Z: np.ndarray          # (N,3) current
    Z0: np.ndarray         # init / spring target
    R_max: float           # sprout radius
    n_local: int
    warp_pairs: List[Tuple[int, int]] = None  # parcel pairs currently warping
    bridges_added: int = 0
    t: int = 0
    omega: float = 0.35
    kappa: float = 0.8
    use_warp: bool = False
    use_wave: bool = False

    def __post_init__(self):
        if self.warp_pairs is None:
            self.warp_pairs = []

    def phase(self) -> np.ndarray:
        """φ_i = sin(ω t + κ z_x) for each node."""
        return np.sin(self.omega * self.t + self.kappa * self.Z[:, 0])

    def sprout_gain(self) -> np.ndarray:
        """Per-node multiplier in [1, 10]."""
        if not self.use_wave:
            return np.ones(self.Z.shape[0])
        phi = self.phase()
        return 1.0 + 9.0 * np.maximum(phi, 0.0)

    def famine_gain(self) -> np.ndarray:
        if not self.use_wave:
            return np.ones(self.Z.shape[0])
        phi = self.phase()
        return 1.0 + 9.0 * np.maximum(-phi, 0.0)

    def allowed_pair(self, post: int, pre: int, mask_hop: bool) -> bool:
        if mask_hop:
            return True
        return float(np.linalg.norm(self.Z[post] - self.Z[pre])) <= self.R_max

    def maybe_warp(self, scent: np.ndarray, tau: float = 0.35, lam: float = 0.4):
        """scent: (R,R) parcel correlation. Pull high-scent far parcels together."""
        if not self.use_warp:
            return
        R = scent.shape[0]
        nl = self.n_local
        cents = self.Z.reshape(R, nl, 3).mean(1)
        # find best far pair
        best = None
        best_s = tau
        for i in range(R):
            for j in range(i + 1, R):
                if scent[i, j] < best_s:
                    continue
                dist = np.linalg.norm(cents[i] - cents[j])
                if dist <= self.R_max:
                    continue
                if scent[i, j] > best_s:
                    best_s = scent[i, j]
                    best = (i, j, dist)
        if best is None:
            return
        i, j, _ = best
        mid = 0.5 * (cents[i] + cents[j])
        for r, c in ((i, cents[i]), (j, cents[j])):
            new_c = (1 - lam) * c + lam * mid
            sl = slice(r * nl, (r + 1) * nl)
            self.Z[sl] += (new_c - c)
        self.warp_pairs.append((i, j))

    def relax(self, mu: float = 0.15):
        self.Z = (1 - mu) * self.Z + mu * self.Z0

    def step_time(self):
        self.t += 1


def parcel_scent_from_states(states: np.ndarray, n_local: int) -> np.ndarray:
    """states (T,N) -> (R,R) abs corr of parcel-mean activity."""
    T, N = states.shape
    R = N // n_local
    P = states.reshape(T, R, n_local).mean(-1)  # (T,R)
    P = P - P.mean(0, keepdims=True)
    std = P.std(0, keepdims=True) + 1e-8
    Pn = P / std
    C = (Pn.T @ Pn) / max(T - 1, 1)
    np.fill_diagonal(C, 0.0)
    return np.abs(C)


def default_R_max(Z: np.ndarray, mask_edges: np.ndarray) -> float:
    if len(mask_edges) == 0:
        return 1.0
    # sample up to 2000 mask edges
    idx = np.linspace(0, len(mask_edges) - 1, num=min(2000, len(mask_edges)), dtype=int)
    d = np.linalg.norm(Z[mask_edges[idx, 0]] - Z[mask_edges[idx, 1]], axis=1)
    return float(np.median(d) * 1.25)

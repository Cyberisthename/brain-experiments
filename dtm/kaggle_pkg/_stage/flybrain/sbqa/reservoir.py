"""Masked rate reservoir (torch, device-auto)."""
from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(device: Optional[str] = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class MaskedReservoir(nn.Module):
    """x <- (1-a)x + a tanh((W*mask) x + Win u + b). Optional trainable W/readout."""

    def __init__(
        self,
        mask: np.ndarray,
        n_in: int,
        out_dim: int = 1,
        leak: float = 0.4,
        rho: float = 0.9,
        device: Optional[str] = None,
        trainable_W: bool = False,
    ):
        super().__init__()
        self.device = pick_device(device)
        self.leak = leak
        self.rho = rho
        self.N = int(mask.shape[0])
        m = torch.from_numpy(mask.astype(np.float32))
        self.register_buffer("mask", m)
        # free parameters under mask
        W0 = 0.08 * torch.randn(self.N, self.N)
        W0 = W0 * m
        self.W_raw = nn.Parameter(W0, requires_grad=trainable_W)
        self.Win = nn.Parameter(0.5 * torch.randn(self.N, n_in), requires_grad=trainable_W)
        self.b = nn.Parameter(0.05 * torch.randn(self.N), requires_grad=trainable_W)
        self.readout = nn.Linear(self.N, out_dim)
        self.to(self.device)
        with torch.no_grad():
            self.rescale_rho_()

    def active_W(self) -> torch.Tensor:
        return self.W_raw * self.mask

    @torch.no_grad()
    def rescale_rho_(self):
        W = self.active_W()
        # power iteration for spectral radius
        v = torch.randn(self.N, device=W.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(20):
            v = W @ v
            n = v.norm() + 1e-12
            v = v / n
        r = float(n)
        if r > 1e-8:
            self.W_raw.mul_(self.rho / r)

    @torch.no_grad()
    def set_topology_from_edges(self, edge_idx: np.ndarray, all_edges: np.ndarray, weight_scale: float = 0.08):
        """edge_idx: indices into all_edges that are ON. Rebuild W_raw under mask."""
        W = torch.zeros(self.N, self.N, device=self.device)
        if len(edge_idx):
            ee = all_edges[edge_idx]
            vals = weight_scale * torch.randn(len(ee), device=self.device)
            W[ee[:, 0], ee[:, 1]] = vals
        W = W * self.mask
        self.W_raw.copy_(W)
        self.rescale_rho_()

    def forward_sequence(self, u: torch.Tensor) -> torch.Tensor:
        # u: (T,B,d)
        T, B, _ = u.shape
        x = torch.zeros(B, self.N, device=u.device, dtype=u.dtype)
        W = self.active_W()
        outs = []
        for t in range(T):
            drive = u[t] @ self.Win.t() + self.b
            rec = x @ W.t()
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            outs.append(self.readout(x))
        return torch.stack(outs, 0)  # (T,B,out)

    @torch.no_grad()
    def states(self, u: torch.Tensor) -> torch.Tensor:
        T, B, _ = u.shape
        x = torch.zeros(B, self.N, device=u.device, dtype=u.dtype)
        W = self.active_W()
        xs = []
        for t in range(T):
            drive = u[t] @ self.Win.t() + self.b
            rec = x @ W.t()
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            xs.append(x)
        return torch.stack(xs, 0)  # (T,B,N)


@torch.no_grad()
def ridge_readout(X: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor, lam: float = 1e-4) -> torch.Tensor:
    """X (T,B,N), Y (T,B,1), mask (T,B) -> weight (N+1, out)."""
    m = mask.bool()
    Xf = X[m]  # (P,N)
    Yf = Y[m]  # (P,out)
    ones = torch.ones(Xf.shape[0], 1, device=Xf.device, dtype=Xf.dtype)
    Xa = torch.cat([Xf, ones], 1)
    out = Yf.shape[1]
    A = Xa.T @ Xa + lam * torch.eye(Xa.shape[1], device=Xa.device, dtype=Xa.dtype)
    B = Xa.T @ Yf
    try:
        W = torch.linalg.solve(A, B)
    except Exception:
        W = torch.linalg.lstsq(A, B).solution
    return W


@torch.no_grad()
def pred_ridge(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*X.shape[:-1], 1, device=X.device, dtype=X.dtype)
    Xa = torch.cat([X, ones], -1)
    return Xa @ W

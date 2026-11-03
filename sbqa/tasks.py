"""Lightweight NARMA-10 and XOR_HARD batches (torch)."""
from __future__ import annotations

import numpy as np
import torch


def make_narma_batch(seq_len: int, batch: int, rng: np.random.Generator, max_abs: float = 50.0, max_tries: int = 40):
    T = seq_len
    u = rng.uniform(0, 0.5, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for b in range(batch):
        for attempt in range(max_tries):
            if attempt > 0:
                u[:, b, 0] = rng.uniform(0, 0.5, size=(T,)).astype(np.float32)
            y[:, b, 0] = 0.0
            ok = True
            for t in range(9, T - 1):
                yt = (
                    0.3 * y[t, b, 0]
                    + 0.05 * y[t, b, 0] * y[t - 9 : t + 1, b, 0].sum()
                    + 1.5 * u[t - 9, b, 0] * u[t, b, 0]
                    + 0.1
                )
                y[t + 1, b, 0] = yt
                if (not np.isfinite(yt)) or abs(yt) > max_abs:
                    ok = False
                    break
            if ok:
                break
        else:
            y[:, b, 0] = 0.0
            u[:, b, 0] = 0.25
    mask = np.ones((T, batch), dtype=np.float32)
    mask[:10] = 0.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_xor_hard_batch(d: int, seq_len: int, batch: int, rng: np.random.Generator):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def nrmse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    # pred,y: (T,B,1); mask (T,B)
    m = mask.unsqueeze(-1)
    diff = (pred - y) * m
    num = (diff ** 2).sum()
    den = m.sum().clamp_min(1.0)
    mse = num / den
    # std of y under mask
    yflat = y[mask.bool()]
    if yflat.numel() < 2:
        return float("nan")
    s = yflat.std().clamp_min(1e-8)
    return float(torch.sqrt(mse) / s)


def xor_acc(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.bool()
    if m.sum() < 1:
        return float("nan")
    ph = (pred[..., 0] > 0.5).float()
    yh = (y[..., 0] > 0.5).float()
    return float((ph[m] == yh[m]).float().mean())

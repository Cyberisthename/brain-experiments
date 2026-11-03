"""SBQA tunneling operator + classical baselines (discrete & continuous)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch


def wkb_accept(delta_E: float, width: float, rng: np.random.Generator) -> bool:
    """P ≈ exp(-w * sqrt(max(ΔE,0))). Always accept improvements (ΔE<=0)."""
    if delta_E <= 0:
        return True
    p = float(np.exp(-width * np.sqrt(delta_E + 1e-12)))
    return bool(rng.random() < p)


def metropolis_accept(delta_E: float, temp: float, rng: np.random.Generator) -> bool:
    if delta_E <= 0:
        return True
    if temp <= 1e-12:
        return False
    return bool(rng.random() < np.exp(-delta_E / temp))


@dataclass
class PlateauMonitor:
    window: int = 5
    var_thresh: float = 1e-4
    scores: List[float] = None

    def __post_init__(self):
        if self.scores is None:
            self.scores = []

    def push(self, s: float) -> bool:
        self.scores.append(float(s))
        if len(self.scores) < self.window:
            return False
        w = np.asarray(self.scores[-self.window:], dtype=float)
        return float(np.var(w)) < self.var_thresh


def propose_edge_flip(
    active: np.ndarray,
    n_mask: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip k memberships in binary active mask over [0, n_mask). Keep |active| constant via swap."""
    on = np.flatnonzero(active)
    off = np.flatnonzero(~active)
    if len(on) == 0 or len(off) == 0:
        return active.copy()
    k = int(min(k, len(on), len(off)))
    drop = rng.choice(on, size=k, replace=False)
    add = rng.choice(off, size=k, replace=False)
    new = active.copy()
    new[drop] = False
    new[add] = True
    return new


def hamming_midpoints(a: np.ndarray, b: np.ndarray, n_mid: int = 2) -> List[np.ndarray]:
    """Sample configs along Hamming path from a -> b (boolean vectors)."""
    diff = np.flatnonzero(a != b)
    if len(diff) == 0:
        return []
    out = []
    for frac in np.linspace(0.0, 1.0, n_mid + 2)[1:-1]:
        take = int(round(frac * len(diff)))
        mid = a.copy()
        mid[diff[:take]] = b[diff[:take]]
        out.append(mid)
    return out


def estimate_barrier(score_fn: Callable[[np.ndarray], float], a: np.ndarray, b: np.ndarray, score_a: float, score_b: float) -> Tuple[float, float]:
    """Barrier height ΔE (higher score_fn = worse for NARMA) and width w (Hamming/2).

    Energy E := score_fn (NRMSE). Barrier = max(mid scores) - min(E_a, E_b), floored at 0.
    """
    mids = hamming_midpoints(a, b, n_mid=2)
    scores = [score_a, score_b] + [score_fn(m) for m in mids]
    base = min(score_a, score_b)
    barrier = max(0.0, float(max(scores) - base))
    width = 0.5 * float(np.sum(a != b))  # Hamming distance / 2
    return barrier, max(width, 1.0)

"""Wiring helpers: edge expand, myelin/FA delay priors, Dale sign assignment."""
from __future__ import annotations

from typing import Optional

import numpy as np


def delay_from_length(
    length: float,
    lengths_pos: np.ndarray,
    dmax: int = 5,
    speed_scale: float = 1.0,
) -> int:
    """Quantile-bin fiber length → delay in 1..dmax, then scale by 1/speed."""
    if lengths_pos.size == 0 or not np.isfinite(length) or length <= 0:
        return 1
    qs = np.quantile(lengths_pos, np.linspace(0, 1, dmax + 1)[1:-1])
    d = 1
    for q in qs:
        if length > q:
            d += 1
    if speed_scale > 0:
        d = int(round(d / speed_scale))
    return int(min(dmax, max(1, d)))


def myelin_speed_scale(fa: Optional[float] = None, sector: Optional[str] = None) -> float:
    """Map FA / callosal sector → relative conduction speed (V / V_ref).

    Caminiti-ish defaults; FA callosal ~1.15× mild boost (ENGINE_V3_PLAN Q8).
    """
    scale = 1.0
    if fa is not None and np.isfinite(fa):
        # FA typical WM ~0.4–0.6; higher FA → faster
        scale *= float(np.clip(0.7 + 0.8 * fa, 0.5, 1.6))
    if sector:
        # relative velocities (m/s) / 8.5 ref — Caminiti Method-3 style
        table = {
            "genu": 6.0 / 8.5,
            "anterior_body": 7.5 / 8.5,
            "mid_body": 8.5 / 8.5,
            "posterior_body": 9.5 / 8.5,
            "splenium": 11.0 / 8.5,
        }
        scale *= table.get(str(sector).lower(), 1.0)
    return float(scale)


def assign_dale_signs(n: int, p_inhibit: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = np.ones(n, dtype=np.float32)
    n_inh = max(1, int(round(p_inhibit * n)))
    inh = rng.choice(n, size=n_inh, replace=False)
    signs[inh] = -1.0
    return signs


def masks_to_edges(mask_intra, mask_inter, delays, esign_mat=None):
    """Pack dense masks into edge lists (src, dst, delay, sign)."""
    mi = np.asarray(mask_intra).astype(bool)
    mj = np.asarray(mask_inter).astype(bool)
    delays = np.asarray(delays, dtype=np.int64)
    if esign_mat is None:
        esign_mat = np.ones_like(delays, dtype=np.float32)
    else:
        esign_mat = np.asarray(esign_mat, dtype=np.float32)
    ii, jj = np.where(mi)
    intra = dict(src=ii.astype(np.int64), dst=jj.astype(np.int64))
    ii, jj = np.where(mj)
    inter = dict(
        src=ii.astype(np.int64),
        dst=jj.astype(np.int64),
        delay=delays[ii, jj].astype(np.int64),
        sign=esign_mat[ii, jj].astype(np.float32),
    )
    return intra, inter


def apply_myelin_delays(
    delays: np.ndarray,
    mask_inter: np.ndarray,
    lengths: Optional[np.ndarray] = None,
    fa: Optional[np.ndarray] = None,
    dmax: int = 5,
) -> np.ndarray:
    """Rebuild inter delays from lengths + optional FA map (quality prior)."""
    out = np.array(delays, copy=True, dtype=np.int64)
    mj = np.asarray(mask_inter).astype(bool)
    if lengths is None:
        return out
    lengths = np.asarray(lengths, dtype=np.float64)
    pos = lengths[lengths > 0]
    ii, jj = np.where(mj)
    for i, j in zip(ii, jj):
        ss = 1.0
        if fa is not None:
            ss = myelin_speed_scale(fa=float(fa[i, j]) if np.ndim(fa) == 2 else float(fa[i]))
        out[i, j] = delay_from_length(float(lengths[i, j]), pos, dmax=dmax, speed_scale=ss)
    return out

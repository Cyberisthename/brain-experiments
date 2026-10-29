"""Task definitions matching existing reservoir experiments (MC, NARMA-10, XOR_DELAY, PC_HARD)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp

from .network import SpikingNetwork, NetworkConfig
from .calibrate import calibrate_ei
from .readout import ridge, pred, filter_spikes

TASKS = ("MC", "NARMA", "NARMA-10", "XOR_DELAY", "PC_HARD")

# Defaults aligned with xor_cx / neuromod_real scales but shorter for spiking smoke
WASH, TR, TE = 50, 200, 100
DT_MS_DEFAULT = 0.5  # production dt for tasks (speed); accuracy tests use finer


def _drive_current(
    U: np.ndarray,
    N: int,
    in_idx: np.ndarray,
    rng: np.random.Generator,
    amp_pA: float = 350.0,
    baseline_pA: float = 260.0,
) -> np.ndarray:
    """Map low-dim input U (T,) or (T,d) to Iext (T,N) via sparse random Win onto in_idx.

    Adds a population baseline near rheobase so the network stays in the calibrated
    active regime; task modulation rides on top of that baseline.
    """
    T = len(U)
    multi = U.ndim == 2
    d = U.shape[1] if multi else 1
    Win = rng.uniform(-1, 1, (len(in_idx), d)).astype(np.float32) * amp_pA
    I = np.full((T, N), baseline_pA, dtype=np.float32)
    if multi:
        I[:, in_idx] = I[:, in_idx] + (U.astype(np.float32) @ Win.T)
    else:
        I[:, in_idx] = I[:, in_idx] + (U.astype(np.float32)[:, None] * Win[:, 0])
    # tiny private noise breaks perfect synchrony (not task data — unit noise)
    I += rng.normal(0.0, 15.0, size=I.shape).astype(np.float32)
    return I


def _score_nrmse(p, y) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    s = np.std(y)
    if s < 1e-12 or not np.all(np.isfinite(p)):
        return float("nan")
    return float(np.sqrt(np.mean((p - y) ** 2)) / s)


def run_xor_delay(
    net: SpikingNetwork,
    rng: np.random.Generator,
    in_idx: np.ndarray,
    d: int = 5,
    alpha: float = 10.0,
    wash: int = WASH,
    tr: int = TR,
    te: int = TE,
    steps_per_symbol: int = 40,
) -> Dict:
    """Delayed XOR: score = 1 - NRMSE (higher better)."""
    Tsym = wash + tr + te
    a = rng.integers(0, 2, Tsym).astype(np.float64)
    b = rng.integers(0, 2, Tsym).astype(np.float64)
    y_sym = np.zeros(Tsym)
    for t in range(d, Tsym):
        y_sym[t] = 1.0 if (int(a[t - d]) ^ int(b[t - d])) else -1.0
    U_sym = np.stack([2 * a - 1, 2 * b - 1], 1)

    # Expand symbols to simulation steps
    U = np.repeat(U_sym, steps_per_symbol, axis=0)
    y = np.repeat(y_sym, steps_per_symbol)
    T = len(U)
    I = _drive_current(U, net.N, in_idx, rng)
    net.reset(batch=1)
    out = net.run(T, Iext_t=I, record_spikes=False, record_filtered=True)
    if out["aborted"] or out["filtered"] is None:
        return {"score": float("nan"), "nrmse": float("nan"), "aborted": True, "reason": out["abort_reason"]}
    X = out["filtered"][:, :, 0]  # (T,N)
    # pool within each symbol window (matches old reservoir's per-symbol state)
    T_pool = (len(X) // steps_per_symbol) * steps_per_symbol
    Xp = X[:T_pool].reshape(-1, steps_per_symbol, X.shape[1]).mean(axis=1)
    yp = y_sym[: len(Xp)]
    w0, w1 = wash, wash + tr
    var = Xp[w0:w1].var(axis=0)
    keep = int(min(400, Xp.shape[1]))
    idx = np.argsort(var)[-keep:]
    Xs = Xp[:, idx]
    B = ridge(Xs[w0:w1], yp[w0:w1], alpha)
    p = pred(Xs[w1:], B)
    nrmse = _score_nrmse(p, yp[w1:])
    return {"score": float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan"), "nrmse": nrmse, "aborted": False}


def run_narma(
    net: SpikingNetwork,
    rng: np.random.Generator,
    in_idx: np.ndarray,
    alpha: float = 1e-4,
    wash: int = WASH,
    tr: int = TR,
    te: int = TE,
    steps_per: int = 20,
) -> Dict:
    Tsym = wash + tr + te
    u = rng.uniform(0, 0.5, Tsym)
    y = np.zeros(Tsym)
    for t in range(10, Tsym - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9 : t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    U = np.repeat(u, steps_per)
    Y = np.repeat(y, steps_per)
    I = _drive_current(U, net.N, in_idx, rng, amp_pA=60.0)
    net.reset()
    out = net.run(len(U), Iext_t=I, record_filtered=True, record_spikes=False)
    if out["aborted"]:
        return {"nrmse": float("nan"), "aborted": True}
    X = out["filtered"][:, :, 0]
    w0, w1 = wash * steps_per, (wash + tr) * steps_per
    B = ridge(X[w0:w1], Y[w0:w1], alpha)
    p = pred(X[w1:], B)
    return {"nrmse": _score_nrmse(p, Y[w1:]), "aborted": False}


def run_mc(
    net: SpikingNetwork,
    rng: np.random.Generator,
    in_idx: np.ndarray,
    alpha: float = 1e-4,
    wash: int = WASH,
    tr: int = TR,
    te: int = TE,
    D: int = 50,
    steps_per: int = 20,
) -> Dict:
    Tsym = wash + tr + te
    u = rng.uniform(-1, 1, Tsym)
    U = np.repeat(u, steps_per)
    I = _drive_current(U, net.N, in_idx, rng, amp_pA=60.0)
    net.reset()
    out = net.run(len(U), Iext_t=I, record_filtered=True, record_spikes=False)
    if out["aborted"]:
        return {"mc": float("nan"), "aborted": True}
    X = out["filtered"][:, :, 0]
    # targets: delayed u
    Y = np.stack([np.roll(U, k) for k in range(1, D + 1)], 1)
    w0, w1 = wash * steps_per, (wash + tr) * steps_per
    B = ridge(X[w0:w1], Y[w0:w1], alpha)
    P = pred(X[w1:], B)
    Yt = Y[w1:]
    mc = 0.0
    for k in range(D):
        if np.std(Yt[:, k]) < 1e-12:
            continue
        c = np.corrcoef(P[:, k], Yt[:, k])[0, 1]
        if np.isfinite(c):
            mc += float(c) ** 2
    return {"mc": float(mc), "aborted": False}


def run_pc_hard(
    net: SpikingNetwork,
    rng: np.random.Generator,
    in_idx: np.ndarray,
    alpha: float = 1e-2,
    wash: int = WASH,
    tr: int = TR,
    te: int = TE,
    cue_frac: float = 0.35,
    noise: float = 1.5,
    steps_per: int = 20,
) -> Dict:
    """Noisy cue reconstruction — 1-NRMSE higher better."""
    Tsym = wash + tr + te
    u_clean = rng.uniform(-1, 1, Tsym)
    u_drive = u_clean + noise * rng.standard_normal(Tsym)
    U = np.repeat(u_drive, steps_per)
    Y = np.repeat(u_clean, steps_per)
    # drive only cue_frac of in_idx
    k = max(1, int(round(cue_frac * len(in_idx))))
    drive_idx = rng.choice(in_idx, size=min(k, len(in_idx)), replace=False)
    I = _drive_current(U, net.N, drive_idx, rng, amp_pA=70.0)
    net.reset()
    out = net.run(len(U), Iext_t=I, record_filtered=True, record_spikes=False)
    if out["aborted"]:
        return {"score": float("nan"), "aborted": True}
    X = out["filtered"][:, :, 0]
    w0, w1 = wash * steps_per, (wash + tr) * steps_per
    B = ridge(X[w0:w1], Y[w0:w1], alpha)
    p = pred(X[w1:], B)
    nrmse = _score_nrmse(p, Y[w1:])
    return {"score": float(1.0 - nrmse) if np.isfinite(nrmse) else float("nan"), "nrmse": nrmse, "aborted": False}


def prepare_network(
    W, sign, nt_labels=None, delays_ms=None, seed=0, dt_ms=DT_MS_DEFAULT, calibrate=True,
) -> Tuple[SpikingNetwork, Dict]:
    cfg = NetworkConfig()
    cfg.sim.dt_ms = dt_ms
    cfg.use_numba = True
    meta = {}
    if calibrate:
        cal = calibrate_ei(W, sign, nt_labels=nt_labels, delays_ms=delays_ms, config=cfg, seed=seed)
        meta["cal"] = {k: v for k, v in cal.items() if k != "network"}
        net = cal["network"]
    else:
        net = SpikingNetwork(W, sign, delays_ms=delays_ms, nt_labels=nt_labels, config=cfg, weight_scale=150.0)
        meta["cal"] = {"scale": 0.08, "skipped": True}
    return net, meta


def run_task(task: str, net: SpikingNetwork, seed: int, in_idx: Optional[np.ndarray] = None) -> Dict:
    rng = np.random.default_rng(seed)
    if in_idx is None:
        in_idx = rng.choice(net.N, size=min(200, net.N), replace=False)
    task_u = task.upper().replace("-", "_")
    if task_u in ("XOR_DELAY", "XOR"):
        return run_xor_delay(net, rng, in_idx)
    if task_u in ("NARMA", "NARMA_10", "NARMA10"):
        return run_narma(net, rng, in_idx)
    if task_u == "MC":
        return run_mc(net, rng, in_idx)
    if task_u == "PC_HARD":
        return run_pc_hard(net, rng, in_idx)
    raise KeyError(task)

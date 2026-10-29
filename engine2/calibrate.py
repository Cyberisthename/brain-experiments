"""Automatic E/I balance calibration so networks settle at biological rates."""
from __future__ import annotations

from typing import Optional, Dict, Tuple
import numpy as np
import scipy.sparse as sp

from .network import SpikingNetwork, NetworkConfig
from .params import NeuronParams, SynapseParams, SimParams


def _isi_cv(spikes_TB: np.ndarray, dt_ms: float) -> float:
    """CV of ISIs pooled over neurons (batch 0). spikes: (T,N)."""
    T, N = spikes_TB.shape
    cvs = []
    for i in range(N):
        t_idx = np.flatnonzero(spikes_TB[:, i] > 0)
        if len(t_idx) < 5:
            continue
        isi = np.diff(t_idx.astype(np.float64)) * dt_ms
        if isi.mean() <= 0:
            continue
        cvs.append(isi.std() / isi.mean())
    if not cvs:
        return float("nan")
    return float(np.median(cvs))


def calibrate_ei(
    W: sp.spmatrix,
    sign: np.ndarray,
    nt_labels: Optional[np.ndarray] = None,
    delays_ms: Optional[np.ndarray] = None,
    config: Optional[NetworkConfig] = None,
    target_rate_hz: Tuple[float, float] = (2.0, 10.0),
    target_cv: float = 1.0,
    warm_ms: float = 1200.0,
    noise_pA: float = 50.0,
    scales: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict:
    """Find a global weight scale so spontaneous activity sits in target band.

    Uses only spontaneous Poisson-like noise current (no task data).
    Returns best scale, achieved rate/CV, and a ready SpikingNetwork.
    """
    config = config or NetworkConfig()
    dt = config.sim.dt_ms
    T = int(round(warm_ms / dt))
    lo, hi = target_rate_hz
    if scales is None:
        scales = np.array([0.3, 1, 3, 5, 10, 20, 40, 80, 150, 300, 600, 1200, 2500], dtype=np.float64)

    rng = np.random.default_rng(seed)
    best = None
    results = []

    for s in scales:
        net = SpikingNetwork(
            W, sign, delays_ms=delays_ms, nt_labels=nt_labels,
            config=config, weight_scale=float(s), batch=1,
        )
        # spontaneous drive: DC near rheobase + noise (no task data)
        rheobase = net.np_.g_L * (net.np_.V_th - net.np_.E_L)
        Iext = (rheobase * 1.15 + rng.standard_normal((T, net.N)) * noise_pA).astype(np.float32)
        out = net.run(T, Iext_t=Iext, record_spikes=True, record_filtered=False)
        if out["aborted"]:
            results.append({"scale": float(s), "rate": float("nan"), "cv": float("nan"), "aborted": True})
            continue
        # discard first half as wash
        wash = T // 2
        spk = out["spikes"][wash:, :, 0]  # (T',N)
        rate = float(spk.mean() / (dt * 1e-3))  # population mean Hz
        cv = _isi_cv(spk, dt)
        rec = {"scale": float(s), "rate": rate, "cv": cv, "aborted": False}
        results.append(rec)
        # score: in-band rate, CV near target
        if lo <= rate <= hi:
            cv_term = abs(cv - target_cv) if np.isfinite(cv) else 0.5
            score = cv_term + 0.1 * abs(rate - 0.5 * (lo + hi))
            if best is None or score < best["score"]:
                best = {**rec, "score": score}

    if best is None:
        # pick closest rate to mid-band among non-aborted
        ok = [r for r in results if not r["aborted"] and np.isfinite(r["rate"])]
        if not ok:
            # ultra-weak fallback
            best = {"scale": float(scales[0]), "rate": 0.0, "cv": float("nan"), "score": 1e9, "fallback": True}
        else:
            mid = 0.5 * (lo + hi)
            ok.sort(key=lambda r: abs(r["rate"] - mid))
            best = {**ok[0], "score": abs(ok[0]["rate"] - mid), "fallback": True}

    # Build calibrated network
    net = SpikingNetwork(
        W, sign, delays_ms=delays_ms, nt_labels=nt_labels,
        config=config, weight_scale=float(best["scale"]), batch=1,
    )
    return {
        "scale": float(best["scale"]),
        "rate_hz": float(best.get("rate", float("nan"))),
        "cv": float(best.get("cv", float("nan"))),
        "grid": results,
        "network": net,
        "fallback": bool(best.get("fallback", False)),
    }

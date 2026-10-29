"""Unit test: single-neuron f-I curve vs analytic LIF (no adaptation, no syn)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine2.network import SpikingNetwork, NetworkConfig
from engine2.params import NeuronParams, SynapseParams, SimParams


def analytic_lif_rate(I_pA, C_m, g_L, E_L, V_th, V_reset, t_ref_ms):
    rheobase = g_L * (V_th - E_L)
    if I_pA <= rheobase:
        return 0.0
    tau_ms = C_m / g_L
    V_inf = E_L + I_pA / g_L
    if V_inf <= V_th:
        return 0.0
    isi_ms = t_ref_ms + tau_ms * np.log((V_inf - V_reset) / (V_inf - V_th))
    return 1000.0 / isi_ms


def make_isolated(n_params: NeuronParams, dt_ms=0.05) -> SpikingNetwork:
    W = sp.csr_matrix((1, 1), dtype=np.float32)
    cfg = NetworkConfig(
        neuron=n_params,
        syn=SynapseParams(),
        sim=SimParams(dt_ms=dt_ms, max_delay_ms=1.0, abort_rate_hz=1000.0),
        use_neuromod=False,
        use_stp=False,
        use_numba=True,
    )
    cfg.neuron.a = 0.0
    cfg.neuron.b = 0.0
    cfg.neuron.homeo_rate = 0.0
    return SpikingNetwork(W, sign=np.array([1], dtype=np.int8), config=cfg, weight_scale=0.0, batch=1)


def test_fi_curve(tol_rel=0.08):
    dt = 0.05
    np_ = NeuronParams(a=0.0, b=0.0, homeo_rate=0.0)
    net = make_isolated(np_, dt_ms=dt)
    currents = [0.0, 150.0, 200.0, 250.0, 350.0, 500.0]
    T_ms = 1000.0
    T = int(T_ms / dt)
    wash = int(200 / dt)
    errs = []
    for I in currents:
        net.reset()
        Iext = np.full((T, 1), I, dtype=np.float32)
        out = net.run(T, Iext_t=Iext, record_spikes=True, record_filtered=False)
        assert not out["aborted"], out["abort_reason"]
        spk = out["spikes"][wash:, 0, 0]
        rate = float(spk.sum() / ((T - wash) * dt * 1e-3))
        pred = analytic_lif_rate(I, np_.C_m, np_.g_L, np_.E_L, np_.V_th, np_.V_reset, np_.t_ref_ms)
        err = abs(rate - pred) if pred < 1e-6 else abs(rate - pred) / pred
        errs.append((I, rate, pred, err))
        print(f"I={I:6.1f} pA  sim={rate:7.2f} Hz  analytic={pred:7.2f} Hz  rel_err={err:.4f}")
        assert err < tol_rel or (pred < 1e-6 and rate < 0.5), f"f-I mismatch I={I}: {rate} vs {pred}"
    return errs


if __name__ == "__main__":
    test_fi_curve()
    print("PASS test_lif_fi")

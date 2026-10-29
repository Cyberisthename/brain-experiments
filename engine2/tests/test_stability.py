"""Stability: calibrated fly CX should not blow up on spontaneous input."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine2.builders import build_arm
from engine2.calibrate import calibrate_ei
from engine2.network import NetworkConfig
from engine2.params import SimParams


def test_fly_calibrated_stable():
    b = build_arm("FLY_CX", seed=0)
    cfg = NetworkConfig(sim=SimParams(dt_ms=0.5), use_neuromod=True)
    cal = calibrate_ei(
        b["W"], b["sign"], nt_labels=b["nt_labels"], delays_ms=b["delays_ms"],
        config=cfg, seed=0, warm_ms=300.0,
    )
    net = cal["network"]
    print(f"cal scale={cal['scale']:.4f} rate={cal['rate_hz']:.2f} cv={cal['cv']}")
    assert cal["scale"] > 0
    # run longer spontaneous
    T = int(1000 / 0.5)
    rng = np.random.default_rng(42)
    rheo = net.np_.g_L * (net.np_.V_th - net.np_.E_L)
    I = (rheo * 1.15 + rng.standard_normal((T, net.N)) * 80.0).astype(np.float32)
    out = net.run(T, Iext_t=I, record_spikes=True, record_filtered=False)
    assert not out["aborted"], out["abort_reason"]
    rate = float(out["rates"][0])
    print(f"long-run mean neuron rate ~ {rate:.2f} Hz (from spike_count/T)")
    # population mean from spikes
    pop = out["spikes"].mean() / (0.5 * 1e-3)
    print(f"pop mean rate={pop:.2f} Hz")
    assert pop < 150.0, "rate too high — unstable"
    assert pop > 0.1, "no activity after calibration"
    assert np.all(np.isfinite(net.V))
    return cal, pop


if __name__ == "__main__":
    test_fly_calibrated_stable()
    print("PASS test_stability")

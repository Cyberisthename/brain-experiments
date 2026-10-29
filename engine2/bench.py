"""Speed / stability / accuracy benchmarks -> BENCH.md numbers."""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine2.network import SpikingNetwork, NetworkConfig, HAS_NUMBA, HAS_CUPY
from engine2.params import NeuronParams, SynapseParams, SimParams
from engine2.builders import build_arm
from engine2.calibrate import calibrate_ei
from engine2.tests.test_lif_fi import test_fi_curve, analytic_lif_rate
from engine2.tests.test_integrator import test_integrator


def er_graph(N, nnz, seed=0):
    g = np.random.default_rng(seed)
    rows = g.integers(0, N, nnz)
    cols = g.integers(0, N, nnz)
    m = rows != cols
    rows, cols = rows[m], cols[m]
    sign = np.ones(N, dtype=np.int8)
    sign[N // 2 :] = -1
    data = np.abs(g.standard_normal(len(rows))).astype(np.float32) * sign[rows].astype(np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N)), sign


def bench_steps(N, nnz_per=100, dt_ms=0.25, T_ms=200.0, batch=1, scale=5.0, seed=0):
    nnz = N * nnz_per
    W, sign = er_graph(N, nnz, seed=seed)
    cfg = NetworkConfig(
        sim=SimParams(dt_ms=dt_ms),
        use_neuromod=False,
        use_numba=True,
    )
    net = SpikingNetwork(W, sign, config=cfg, weight_scale=scale, batch=batch)
    T = int(round(T_ms / dt_ms))
    rng = np.random.default_rng(seed + 1)
    I = (210.0 + rng.standard_normal((T, N, batch)) * 25.0).astype(np.float32) if batch > 1 else (210.0 + rng.standard_normal((T, N)) * 25.0).astype(np.float32)
    # warmup
    net.reset(batch=batch)
    net.run(min(50, T), Iext_t=I[: min(50, T)], record_spikes=False, record_filtered=False)
    net.reset(batch=batch)
    t0 = time.perf_counter()
    out = net.run(T, Iext_t=I, record_spikes=False, record_filtered=False)
    elapsed = time.perf_counter() - t0
    steps = out["steps"]
    steps_per_sec = steps / max(elapsed, 1e-12)
    neuron_steps_per_sec = steps_per_sec * N * batch
    return {
        "N": N,
        "batch": batch,
        "dt_ms": dt_ms,
        "T_ms": T_ms,
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps_per_sec,
        "neuron_steps_per_sec": neuron_steps_per_sec,
        "aborted": out["aborted"],
        "numba": HAS_NUMBA,
        "cupy": HAS_CUPY,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    report = {"numba": HAS_NUMBA, "cupy": HAS_CUPY, "speed": [], "accuracy": {}, "stability": {}}

    print("=== f-I accuracy ===", flush=True)
    try:
        fi = test_fi_curve(tol_rel=0.08)
        report["accuracy"]["fi_curve"] = [
            {"I": a, "sim": b, "analytic": c, "rel_err": d} for a, b, c, d in fi
        ]
        report["accuracy"]["fi_pass"] = True
    except Exception as e:
        report["accuracy"]["fi_pass"] = False
        report["accuracy"]["fi_error"] = str(e)
        print("f-I FAIL", e)

    print("=== integrator accuracy ===", flush=True)
    try:
        integ = test_integrator()
        report["accuracy"]["integrator"] = integ
        report["accuracy"]["integrator_pass"] = True
    except Exception as e:
        report["accuracy"]["integrator_pass"] = False
        report["accuracy"]["integrator_error"] = str(e)
        print("integrator FAIL", e)

    print("=== stability fly CX ===", flush=True)
    try:
        b = build_arm("FLY_CX", seed=0)
        cfg = NetworkConfig(sim=SimParams(dt_ms=0.5), use_neuromod=True)
        cal = calibrate_ei(
            b["W"], b["sign"], nt_labels=b["nt_labels"], delays_ms=b["delays_ms"],
            config=cfg, seed=0, warm_ms=400.0,
        )
        report["stability"]["fly_cx_cal"] = {
            "scale": cal["scale"], "rate_hz": cal["rate_hz"], "cv": cal["cv"],
            "fallback": cal["fallback"],
        }
        net = cal["network"]
        T = int(800 / 0.5)
        rng = np.random.default_rng(7)
        rheo = net.np_.g_L * (net.np_.V_th - net.np_.E_L)
        I = (rheo * 1.15 + rng.standard_normal((T, net.N)) * 80).astype(np.float32)
        out = net.run(T, Iext_t=I, record_spikes=True, record_filtered=False)
        pop = float(out["spikes"].mean() / (0.5e-3))
        report["stability"]["fly_cx_long"] = {
            "aborted": out["aborted"], "reason": out["abort_reason"], "pop_rate_hz": pop,
        }
        print(f"fly cal scale={cal['scale']:.4f} rate={cal['rate_hz']:.2f} long_pop={pop:.2f} abort={out['aborted']}")
    except Exception as e:
        report["stability"]["error"] = str(e)
        print("stability FAIL", e)

    print("=== speed N=2875 ===", flush=True)
    # Use real fly connectivity density ~300k nnz / 2875 ≈ 104
    s1 = bench_steps(2875, nnz_per=100, dt_ms=0.25, T_ms=100.0, batch=1)
    report["speed"].append(s1)
    print(s1, flush=True)
    s1b = bench_steps(2875, nnz_per=100, dt_ms=0.25, T_ms=100.0, batch=4)
    report["speed"].append(s1b)
    print(s1b, flush=True)

    print("=== speed N=20000 ===", flush=True)
    s2 = bench_steps(20000, nnz_per=50, dt_ms=0.25, T_ms=50.0, batch=1)
    report["speed"].append(s2)
    print(s2, flush=True)

    (out_dir / "bench.json").write_text(json.dumps(report, indent=2, default=str))
    _write_md(report)
    print("Wrote", out_dir / "bench.json", "and BENCH.md")


def _write_md(report):
    lines = []
    lines.append("# engine2 BENCH\n")
    lines.append(f"Generated by `python -m engine2.bench`. numba={report['numba']} cupy={report['cupy']}.\n")
    lines.append("## Speed\n")
    lines.append("| N | batch | dt (ms) | steps/s | neuron-steps/s | elapsed (s) | aborted |")
    lines.append("|---:|---:|---:|---:|---:|---:|:---|")
    for s in report["speed"]:
        lines.append(
            f"| {s['N']} | {s['batch']} | {s['dt_ms']} | {s['steps_per_sec']:.0f} | "
            f"{s['neuron_steps_per_sec']:.2e} | {s['elapsed_s']:.3f} | {s['aborted']} |"
        )
    lines.append("\n## Accuracy\n")
    acc = report.get("accuracy", {})
    lines.append(f"- Single-neuron f-I vs analytic LIF: **{'PASS' if acc.get('fi_pass') else 'FAIL'}**")
    if acc.get("fi_curve"):
        lines.append("  - sample points:")
        for row in acc["fi_curve"][:6]:
            lines.append(
                f"    - I={row['I']} pA → sim {row['sim']:.2f} Hz, analytic {row['analytic']:.2f} Hz "
                f"(rel err {row['rel_err']:.3f})"
            )
    lines.append(
        f"- Integrator dt=0.01 ms vs dt=0.25 ms: **{'PASS' if acc.get('integrator_pass') else 'FAIL'}**"
    )
    if acc.get("integrator"):
        i = acc["integrator"]
        lines.append(
            f"  - rates {i['rate_ref']:.3f} vs {i['rate_prod']:.3f} Hz (rel err {i['rel_err']:.4f}); "
            f"spike-time agreement {i['agree']:.3f}"
        )
    lines.append("\n## Stability\n")
    st = report.get("stability", {})
    if "fly_cx_cal" in st:
        c = st["fly_cx_cal"]
        lines.append(
            f"- Fly CX auto E/I calibration: scale={c['scale']:.4f}, "
            f"spontaneous rate={c['rate_hz']:.2f} Hz, CV≈{c['cv']}"
        )
    if "fly_cx_long" in st:
        L = st["fly_cx_long"]
        lines.append(
            f"- 800 ms spontaneous after calibration: aborted={L['aborted']}, "
            f"pop rate={L['pop_rate_hz']:.2f} Hz"
        )
    if "error" in st:
        lines.append(f"- ERROR: {st['error']}")
    lines.append(
        "\n## Notes\n"
        "- Production task dt is typically 0.25–0.5 ms; accuracy reference uses 0.01 ms.\n"
        "- CuPy path is optional and not installed on this box; CPU+numba is the default fast path.\n"
        "- Old engine (leaky tanh reservoir) did ~whole-trajectory dense/CSR matvecs per symbol step; "
        "engine2 advances real milliseconds with spikes + delay ring.\n"
    )
    Path(__file__).resolve().parent.joinpath("BENCH.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

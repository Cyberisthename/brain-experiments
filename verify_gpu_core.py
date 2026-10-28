"""CPU-box + GPU self-check for gpu_core.py.

On this CPU-only box:
  - Importing gpu_core must not alter fast_kernels until enable().
  - After enable() with no CuPy, functions fall back to CPU and match
    fast_kernels within 1e-6 relative (usually bit-identical).
  - fastcore2.py / fast_kernels.py hashes must stay unchanged.

On a GPU (Colab T4): also compare GPU vs CPU on 1 seed of
myelin_real_rep and best_fly and print max abs/rel diffs.

Usage:
  python3 verify_gpu_core.py
  FLYBRAIN_GPU=1 python3 verify_gpu_core.py   # attempt GPU path
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compress"))
sys.path.insert(0, str(ROOT / "regions" / "myelin_real"))
sys.path.insert(0, str(ROOT / "regions" / "hybrid_human"))
sys.path.insert(0, str(ROOT / "regions" / "frankenstein"))
sys.path.insert(0, str(ROOT / "regions" / "empn"))

OUT = ROOT / "out_gpu_core"
OUT.mkdir(exist_ok=True)
TOL_REL = 1e-6
# Expected hashes of untouched CPU kernels (snapshot at gpu_core authoring)
EXPECTED_HASH = {
    "fastcore2.py": "e213e52c2816492368117981592a1f98989d1f16edb11b10a7325fbdf1610ff5",
    "compress/fast_kernels.py": "93435ed23590beff0137c66ae1d80c7620526825a0b1f4b288051548ac76b438",
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def max_diffs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    d = np.abs(a - b)
    abs_max = float(np.max(d)) if d.size else 0.0
    denom = np.maximum(np.abs(b), 1e-30)
    rel_max = float(np.max(d / denom)) if d.size else 0.0
    return abs_max, rel_max


def synthetic_W(n=64, nnz=400, seed=0):
    g = np.random.default_rng(seed)
    rows = g.integers(0, n, nnz)
    cols = g.integers(0, n, nnz)
    data = g.normal(size=nnz)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.setdiag(0)
    W.eliminate_zeros()
    # scale roughly
    from scipy.sparse.linalg import eigs

    try:
        r = float(abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=5000)[0]))
    except Exception:
        r = 1.0
    return (W * (0.9 / (r + 1e-12))).tocsr()


def check_hashes(report):
    ok = True
    for rel, expect in EXPECTED_HASH.items():
        got = file_sha(ROOT / rel)
        match = got == expect
        report["file_hashes"][rel] = {"sha256": got, "matches_snapshot": match}
        if not match:
            ok = False
            # Allow if file was intentionally updated later — still record
    report["cpu_kernels_untouched"] = ok
    return ok


def check_import_does_not_patch(report):
    import fast_kernels as K

    # Capture identities before import gpu_core
    before = (K.ridge, K.pred, K.run_reservoir_csr)
    import gpu_core as GC  # noqa: F401

    after = (K.ridge, K.pred, K.run_reservoir_csr)
    ok = before == after
    report["import_without_enable_leaves_kernels"] = ok
    return ok


def check_cpu_fallback_numerics(report):
    import fast_kernels as K
    import gpu_core as GC

    # Force CPU path even if a GPU were present
    GC.disable()
    GC.enable(force_cpu=True)
    assert not GC.using_gpu()

    g = np.random.default_rng(7)
    W = synthetic_W(80, 600, seed=3)
    n = W.shape[0]
    Win = g.uniform(-0.5, 0.5, (n, 1))
    b = g.uniform(-0.1, 0.1, n)
    U = g.uniform(-1, 1, 120)
    leak = 0.2

    # Disable patch temporarily to get pure CPU ref via originals
    GC.disable()
    X_ref = K.run_reservoir_csr(W, Win, b, U, leak)
    B_ref = K.ridge(X_ref[20:80], np.roll(U, 1)[20:80], 1e-6)
    P_ref = K.pred(X_ref[80:], B_ref)

    # GPU-module CPU fallback (patched but force_cpu)
    GC.enable(force_cpu=True)
    X_fb = GC.run_reservoir_csr(W, Win, b, U, leak)
    B_fb = GC.ridge(X_fb[20:80], np.roll(U, 1)[20:80], 1e-6)
    P_fb = GC.pred(X_fb[80:], B_fb)

    rows = []
    for name, a, b_ in [
        ("X", X_ref, X_fb),
        ("B", B_ref, B_fb),
        ("P", P_ref, P_fb),
    ]:
        abs_m, rel_m = max_diffs(a, b_)
        rows.append({"name": name, "max_abs": abs_m, "max_rel": rel_m})
    report["cpu_fallback_vs_fast_kernels"] = rows
    ok = all(r["max_rel"] <= TOL_REL for r in rows)
    report["cpu_fallback_ok"] = ok

    # Delay path
    delay = g.integers(1, 6, size=W.nnz).astype(np.int32)
    Xd_fb = GC.run_delay_reservoir(W, delay, 5, Win, b, U, leak)
    GC.disable()
    import myelin_real as MR

    Xd_ref = MR.run_delay_reservoir(W, delay, 5, Win, b, U, leak, adapt=False, seed=0)
    abs_m, rel_m = max_diffs(Xd_ref, Xd_fb)
    report["delay_cpu_fallback"] = {"max_abs": abs_m, "max_rel": rel_m}
    report["delay_cpu_fallback_ok"] = rel_m <= TOL_REL
    GC.disable()
    return ok and report["delay_cpu_fallback_ok"]


def _one_seed_myelin(seed, use_gpu_module: bool):
    """Minimal 1-seed myelin HUMAN REAL_MYELIN NARMA+MC+PC slice."""
    import gpu_core as GC
    import myelin_real as MR
    import hybrid_human as HH
    import fast_kernels as K

    MYELIN = ROOT / "regions/hybrid_human/data/human_myelin.npz"
    if not MYELIN.exists():
        return None, "missing human_myelin.npz"

    z = np.load(MYELIN, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf = mf.copy()
    mf[~np.isfinite(mf)] = med

    g = np.random.default_rng(seed)
    W, info = HH.build_arm("HUMAN_LAYERED", seed, g)
    Wcoo = W.tocoo()
    delay = np.ones(Wcoo.nnz, dtype=np.int32)
    pre = Wcoo.col
    d_neuron = (1 + np.rint((1.0 - np.clip(mf, 0, 1)) * 4)).astype(np.int32)
    delay[:] = d_neuron[pre]  # human-only net: all pre are human
    dmax = 5
    leak = 0.2
    in_idx = np.arange(W.shape[0])

    if use_gpu_module:
        run_delay = GC.run_delay_reservoir
        ridge = GC.ridge
        pred = GC.pred
    else:
        run_delay = lambda W, delay, dmax, Win, b, U, leak: MR.run_delay_reservoir(
            W, delay, dmax, Win, b, U, leak, adapt=False, seed=seed
        )
        ridge = K.ridge
        pred = K.pred

    metrics = {}
    # MC
    wash, tr, te, D = 200, 400, 200, 40  # shortened for self-check speed
    T = wash + tr + te
    u = g.uniform(-1, 1, T)
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, g)
    X = run_delay(W, delay, dmax, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = ridge(X[wash : wash + tr], Y[wash : wash + tr], 1e-6)
    P = pred(X[wash + tr :], B)
    Yt = Y[wash + tr :]
    metrics["MC"] = float(
        sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D))
    )
    # NARMA
    u = g.uniform(0, 0.5, T)
    y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9 : i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = MR.masked_win(W.shape[0], 1, in_idx, g)
    X = run_delay(W, delay, dmax, Win, b, u, leak)
    B = ridge(X[wash : wash + tr], y[wash : wash + tr], 1e-6)
    p = pred(X[wash + tr :], B)
    yt = y[wash + tr :]
    metrics["NARMA"] = float(np.sqrt(np.mean((p - yt) ** 2)) / (np.std(yt) + 1e-12))
    # states fingerprint
    metrics["_X_tail_mean"] = float(np.mean(X[-10:]))
    metrics["_X_tail_std"] = float(np.std(X[-10:]))
    return metrics, None


def _one_seed_best_fly(seed, use_gpu_module: bool):
    import gpu_core as GC
    import frankenstein as F
    import fast_kernels as K
    import myelin_real as MR

    fly = ROOT / "regions/frankenstein/data/fly_cx_module.npz"
    if not fly.exists():
        return None, "missing fly_cx_module.npz"

    # patch scale to cached radius
    _orig = F.scale

    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))

    F.scale = _scale
    try:
        g = np.random.default_rng(seed)
        W, info = F.build_arm("FLY_CX", seed, g)
    finally:
        F.scale = _orig

    n = W.shape[0]
    leak = 0.2
    g = np.random.default_rng(seed + 99)
    Win = g.uniform(-0.5, 0.5, (n, 1))
    b = g.uniform(-0.1, 0.1, n)
    wash, tr, te, D = 200, 400, 200, 40
    T = wash + tr + te
    u = g.uniform(-1, 1, T)

    if use_gpu_module:
        run = GC.run_reservoir_csr
        ridge = GC.ridge
        pred = GC.pred
    else:
        run = K.run_reservoir_csr
        ridge = K.ridge
        pred = K.pred

    X = run(W, Win, b, u, leak)
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = ridge(X[wash : wash + tr], Y[wash : wash + tr], 1e-6)
    P = pred(X[wash + tr :], B)
    Yt = Y[wash + tr :]
    mc = float(
        sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D))
    )
    return {
        "MC": mc,
        "_X_tail_mean": float(np.mean(X[-10:])),
        "_X_tail_std": float(np.std(X[-10:])),
    }, None


def check_gpu_vs_cpu_experiments(report):
    import gpu_core as GC

    section = {
        "ran": False,
        "reason": None,
        "myelin_real_rep": None,
        "best_fly": None,
        "max_abs": None,
        "max_rel": None,
        "pass": None,
    }
    if not GC.cupy_available():
        section["reason"] = "no CuPy / no GPU device — skip GPU vs CPU experiment compare"
        report["gpu_vs_cpu_experiments"] = section
        return True  # not a failure on CPU box

    GC.disable()
    # CPU reference
    m_cpu, err = _one_seed_myelin(0, use_gpu_module=False)
    if err:
        section["reason"] = err
        report["gpu_vs_cpu_experiments"] = section
        return False
    b_cpu, err = _one_seed_best_fly(0, use_gpu_module=False)
    if err:
        section["reason"] = err
        report["gpu_vs_cpu_experiments"] = section
        return False

    # GPU path
    os.environ["FLYBRAIN_GPU"] = "1"
    GC.enable(force_cpu=False)
    assert GC.using_gpu(), "CuPy present but using_gpu() False after enable"
    m_gpu, err = _one_seed_myelin(0, use_gpu_module=True)
    b_gpu, err2 = _one_seed_best_fly(0, use_gpu_module=True)
    GC.disable()

    diffs = []
    myelin_rows = {}
    for k in m_cpu:
        a, r = max_diffs(m_cpu[k], m_gpu[k])
        myelin_rows[k] = {"cpu": m_cpu[k], "gpu": m_gpu[k], "max_abs": a, "max_rel": r}
        diffs.append((a, r))
    best_rows = {}
    for k in b_cpu:
        a, r = max_diffs(b_cpu[k], b_gpu[k])
        best_rows[k] = {"cpu": b_cpu[k], "gpu": b_gpu[k], "max_abs": a, "max_rel": r}
        diffs.append((a, r))

    max_abs = max(d[0] for d in diffs)
    max_rel = max(d[1] for d in diffs)
    section.update(
        {
            "ran": True,
            "myelin_real_rep": myelin_rows,
            "best_fly": best_rows,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "pass": max_rel <= TOL_REL,
            "seed": 0,
            "note": "shortened wash/train for self-check; same seed & inputs on both backends",
        }
    )
    report["gpu_vs_cpu_experiments"] = section
    print(f"GPU vs CPU max abs={max_abs:.6e}  max rel={max_rel:.6e}  pass={section['pass']}")
    return section["pass"]


def main():
    t0 = time.time()
    report = {
        "tol_rel": TOL_REL,
        "cupy_available": False,
        "file_hashes": {},
        "wall_sec": None,
    }
    # Import-safety check MUST run before any other gpu_core use.
    ok_import = check_import_does_not_patch(report)
    import gpu_core as GC

    report["cupy_available"] = GC.cupy_available()
    report["gpu_env"] = os.environ.get("FLYBRAIN_GPU", "")

    ok_hash = check_hashes(report)
    ok_fb = check_cpu_fallback_numerics(report)
    ok_gpu = check_gpu_vs_cpu_experiments(report)

    report["wall_sec"] = time.time() - t0
    report["all_ok"] = bool(ok_hash and ok_import and ok_fb and ok_gpu)
    report["cpu_box_checks_ok"] = bool(ok_hash and ok_import and ok_fb)

    out = OUT / "verify_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "cupy_available", "cpu_kernels_untouched",
        "import_without_enable_leaves_kernels", "cpu_fallback_ok",
        "delay_cpu_fallback_ok", "cpu_box_checks_ok", "all_ok", "wall_sec"
    ) if k in report}, indent=2))
    print(f"wrote {out}")
    if not report["cpu_box_checks_ok"]:
        sys.exit(2)
    if report["cupy_available"] and not report["all_ok"]:
        sys.exit(3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

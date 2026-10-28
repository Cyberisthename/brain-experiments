"""A/B verify fastcore2 vs existing paths. Does NOT touch running experiment outs.

Runs 2 seeds × all arms for myelin_real, best_fly, fewshot_delay (fewshot test only).
Reports max abs/rel metric diffs and wall-clock speedup.
"""
from __future__ import annotations
import os, sys, json, time, copy
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"

sys.path.insert(0, "/workspace/flybrain/compress")
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/regions/frankenstein")
sys.path.insert(0, "/workspace/flybrain/regions/hybrid_human")
sys.path.insert(0, "/workspace/flybrain/regions/myelin_real")
sys.path.insert(0, "/workspace/flybrain/regions/best_fly")
sys.path.insert(0, "/workspace/flybrain/regions/fewshot_delay")
sys.path.insert(0, "/workspace/flybrain")

import numpy as np
import scipy.sparse as sp
from pathlib import Path
import fast_kernels as K
import fastcore2 as F2
import frankenstein as F

OUT = Path("/workspace/flybrain/out_speedup2")
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1]
REL_TOL = 1e-6


def patch_scale():
    def _scale(W, rho=F.RHO):
        r = float(K.radius_cached(W))
        return W * (rho / (r + 1e-12))
    F.scale = _scale


patch_scale()


def rel_diff(a, b):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return float("inf")
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def abs_diff(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("inf")
    return abs(a - b)


# -------------------- numerical unit checks --------------------
def unit_batch_csr():
    rng = np.random.default_rng(0)
    n, T, B = 200, 300, 4
    W = sp.random(n, n, density=0.05, format="csr", random_state=0)
    W = W - sp.diags(W.diagonal())
    r = float(K.radius_cached(W))
    W = W * (0.9 / (r + 1e-12))
    Wins, bs, Us = [], [], []
    for b in range(B):
        Wins.append(rng.uniform(-0.5, 0.5, (n, 1)))
        bs.append(rng.uniform(-0.1, 0.1, n))
        Us.append(rng.uniform(-1, 1, T))
    Xs_old = [K.run_reservoir_csr(W, Wins[b], bs[b], Us[b], 0.2) for b in range(B)]
    Xs_new = F2.run_reservoir_csr_batch(W, Wins, bs, Us, 0.2)
    mx = max(float(np.max(np.abs(Xs_old[b] - Xs_new[b]))) for b in range(B))
    # blockdiag with different Ws
    Ws = []
    for b in range(B):
        Wb = sp.random(n, n, density=0.05, format="csr", random_state=10 + b)
        Wb = Wb - sp.diags(Wb.diagonal())
        rr = float(K.radius_cached(Wb))
        Ws.append(Wb * (0.9 / (rr + 1e-12)))
    Xs_old2 = [K.run_reservoir_csr(Ws[b], Wins[b], bs[b], Us[b], 0.2) for b in range(B)]
    Xs_new2 = F2.run_reservoir_blockdiag(Ws, Wins, bs, Us, 0.2)
    mx2 = max(float(np.max(np.abs(Xs_old2[b] - Xs_new2[b]))) for b in range(B))
    return {"csr_batch_max_abs": mx, "blockdiag_max_abs": mx2}


def unit_batch_delay():
    rng = np.random.default_rng(1)
    n, T, B, dmax = 150, 250, 3, 5
    W = sp.random(n, n, density=0.08, format="csr", random_state=1)
    W = W - sp.diags(W.diagonal())
    r = float(K.radius_cached(W))
    W = W * (0.9 / (r + 1e-12))
    delay = rng.integers(1, dmax + 1, W.nnz).astype(np.int32)
    Wins, bs, Us = [], [], []
    for b in range(B):
        Wins.append(rng.uniform(-0.5, 0.5, (n, 1)))
        bs.append(rng.uniform(-0.1, 0.1, n))
        Us.append(rng.uniform(-1, 1, T))
    # old: sequential single via F2.run_delay_reservoir
    Xs_old = [F2.run_delay_reservoir(W, delay, dmax, Wins[b], bs[b], Us[b], 0.2) for b in range(B)]
    Xs_new = F2.run_delay_reservoir_batch(W, delay, dmax, Wins, bs, Us, 0.2)
    mx = max(float(np.max(np.abs(Xs_old[b] - Xs_new[b]))) for b in range(B))
    # different delays/Ws
    Ws, delays = [], []
    for b in range(B):
        Wb = sp.random(n, n, density=0.08, format="csr", random_state=20 + b)
        Wb = Wb - sp.diags(Wb.diagonal())
        rr = float(K.radius_cached(Wb))
        Wb = Wb * (0.9 / (rr + 1e-12))
        Ws.append(Wb)
        delays.append(rng.integers(1, dmax + 1, Wb.nnz).astype(np.int32))
    Xs_old2 = [F2.run_delay_reservoir(Ws[b], delays[b], dmax, Wins[b], bs[b], Us[b], 0.2) for b in range(B)]
    Xs_new2 = F2.run_delay_blockdiag(Ws, delays, dmax, Wins, bs, Us, 0.2)
    mx2 = max(float(np.max(np.abs(Xs_old2[b] - Xs_new2[b]))) for b in range(B))
    return {"delay_batch_max_abs": mx, "delay_blockdiag_max_abs": mx2}


# -------------------- myelin_real --------------------
def run_myelin(path: str):
    """path: 'old' | 'new'. new uses F2 delay + graph cache + batched same-seed groups where possible."""
    import myelin_real as M
    # Ensure myelin_real uses patched scale already on F / HH via its build
    results = []
    restore = None
    if path == "new":
        restore = F2.wrap_frankenstein_make_block(F)
        # also patch myelin_real.run_delay_reservoir to F2
        M_run_delay_orig = M.run_delay_reservoir
        M.run_delay_reservoir = lambda W, delay, dmax, Win, b, U, leak, adapt=False, seed=0: (
            F2.run_delay_reservoir(W, delay, dmax, Win, b, U, leak, seed=seed, adapt=adapt)
        )
    t0 = time.time()
    try:
        for net in M.NETS:
            for arm in M.ARMS:
                for seed in SEEDS:
                    for task in M.TASKS:
                        r = M.run_one(net, arm, seed, task)
                        results.append(dict(
                            exp="myelin_real", path=path, net=r.get("net"), arm=r.get("arm"),
                            seed=r.get("seed"), task=r.get("task"), value=r.get("value"),
                            sec=r.get("sec"), error=r.get("error"),
                        ))
    finally:
        if path == "new":
            M.run_delay_reservoir = M_run_delay_orig
            if restore:
                restore()
    wall = time.time() - t0
    return results, wall


def run_myelin_batched_new():
    """New path that also batches the 2 seeds when W+delay match is impossible —
    instead: for FLY-less HUMAN / FRANKEN we batch tasks that share W? They don't.
    Batch seeds for HUMAN_LAYERED NODELAY/etc via blockdiag after building each W.
    """
    import myelin_real as M
    restore = F2.wrap_frankenstein_make_block(F)
    results = []
    t0 = time.time()
    try:
        mf_raw, mf_filled, med, _ = M._mf()
        for net in M.NETS:
            for arm in M.ARMS:
                adapt = arm == "ADAPT"
                # Build per seed (W differs); for each task, batch the seed rollouts via blockdiag
                for task in M.TASKS:
                    built = []
                    for seed in SEEDS:
                        g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
                        W, info = M.build_net(net, seed, g)
                        delay, dmax = M.delays_for_arm(arm, W, net, info, seed, mf_filled)
                        in_idx = M.drive_idx_for(net, info)
                        # generate inputs with same rng sequence as task fns
                        if task == "MC":
                            wash, tr, te, D = 200, 1500, 500, 100
                            T = wash + tr + te
                            u = g.uniform(-1, 1, T)
                            Win, b = M.masked_win(W.shape[0], 1, in_idx, g)
                            built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax,
                                              Win=Win, b=b, U=u, kind="MC", D=D,
                                              wash=wash, tr=tr, te=te, info=info))
                        elif task == "NARMA":
                            wash, tr, te = 200, 1500, 500
                            T = wash + tr + te
                            u = g.uniform(0, 0.5, T)
                            y = np.zeros(T)
                            for i in range(10, T - 1):
                                y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
                            Win, b = M.masked_win(W.shape[0], 1, in_idx, g)
                            built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax,
                                              Win=Win, b=b, U=u, y=y, kind="NARMA",
                                              wash=wash, tr=tr, te=te, info=info))
                        else:
                            n = W.shape[0]
                            T = 1200
                            theta = np.linspace(0, 8 * np.pi, T)
                            u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
                            idx = np.asarray(in_idx)
                            rng2 = np.random.default_rng(int(g.integers(1e9)))
                            k = max(1, int(round(M.PC_HARD_CUE * max(1, len(idx)))))
                            drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(M.PC_HARD_CUE * n)))
                            Win, b = M.masked_win(n, 2, drive_idx, g)
                            wash, tr = 200, 800
                            u_drive = u_clean + M.PC_HARD_NOISE * g.standard_normal(u_clean.shape)
                            built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax,
                                              Win=Win, b=b, U=u_drive, u_clean=u_clean,
                                              kind="PC_HARD", wash=wash, tr=tr, info=info))

                    # Simulate: ADAPT can't batch safely → serial F2
                    t_sim = time.time()
                    if adapt:
                        Xs = []
                        for item in built:
                            Xs.append(F2.run_delay_reservoir(
                                item["W"], item["delay"], item["dmax"], item["Win"], item["b"],
                                item["U"], M.LEAK, seed=item["seed"], adapt=True))
                    else:
                        # Check if all W identical (bit) — then sameW batch; else blockdiag
                        same = all(
                            built[0]["W"].nnz == it["W"].nnz
                            and built[0]["W"].shape == it["W"].shape
                            and np.array_equal(built[0]["W"].data, it["W"].data)
                            and np.array_equal(built[0]["W"].indices, it["W"].indices)
                            and np.array_equal(built[0]["delay"], it["delay"])
                            for it in built[1:]
                        )
                        if same:
                            Xs = F2.run_delay_reservoir_batch(
                                built[0]["W"], built[0]["delay"], built[0]["dmax"],
                                [it["Win"] for it in built],
                                [it["b"] for it in built],
                                [it["U"] for it in built],
                                M.LEAK,
                            )
                        else:
                            Xs = F2.run_delay_blockdiag(
                                [it["W"] for it in built],
                                [it["delay"] for it in built],
                                built[0]["dmax"],
                                [it["Win"] for it in built],
                                [it["b"] for it in built],
                                [it["U"] for it in built],
                                M.LEAK,
                            )
                    # Score (reuse X — one sim per seed/task already; here batch was across seeds)
                    for item, X in zip(built, Xs):
                        if item["kind"] == "MC":
                            val = F2.score_mc(X, item["U"], wash=item["wash"], tr=item["tr"], te=item["te"], D=item["D"])
                        elif item["kind"] == "NARMA":
                            val = F2.score_narma(X, item["y"], wash=item["wash"], tr=item["tr"], te=item["te"])
                        else:
                            val = F2.score_pc_hard(X, item["u_clean"], wash=item["wash"], tr=item["tr"])
                        results.append(dict(
                            exp="myelin_real", path="new_batch", net=net, arm=arm,
                            seed=item["seed"], task=task, value=val,
                            sec=time.time() - t_sim, error=None,
                        ))
    finally:
        restore()
    wall = time.time() - t0
    return results, wall


# -------------------- best_fly --------------------
def run_best_fly(path: str):
    import best_fly as BF
    best_choice = json.loads((BF.OUT / "best_choice.json").read_text())
    arms = BF.FIXED_ARMS + BF.BEST_ARMS
    results = []
    restore = None
    if path == "new":
        restore = F2.wrap_frankenstein_make_block(F)
        BF_run_delay_orig = BF.run_delay_reservoir
        BF.run_delay_reservoir = lambda W, delay, dmax, Win, b, U, leak, seed=0: (
            F2.run_delay_reservoir(W, delay, dmax, Win, b, U, leak, seed=seed, adapt=False)
        )
    t0 = time.time()
    try:
        for arm in arms:
            for seed in SEEDS:
                for task in BF.TASKS:
                    r = BF.run_one(arm, seed, task, best_choice)
                    results.append(dict(
                        exp="best_fly", path=path, arm=r.get("arm"), seed=r.get("seed"),
                        task=r.get("task"), value=r.get("value"), sec=r.get("sec"),
                        error=r.get("error"),
                    ))
    finally:
        if path == "new":
            BF.run_delay_reservoir = BF_run_delay_orig
            if restore:
                restore()
    return results, time.time() - t0


def run_best_fly_batched_new():
    import best_fly as BF
    best_choice = json.loads((BF.OUT / "best_choice.json").read_text())
    arms = BF.FIXED_ARMS + BF.BEST_ARMS
    restore = F2.wrap_frankenstein_make_block(F)
    results = []
    t0 = time.time()
    try:
        mf, med = BF.mf_filled()
        for arm in arms:
            for task in BF.TASKS:
                built = []
                for seed in SEEDS:
                    g = np.random.default_rng(seed + {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task] + 17)
                    wiring, recipe = BF.arm_to_spec(arm, best_choice)
                    if wiring.startswith("fly_") and wiring != "fly_cx":
                        ctrl = wiring.split("_", 1)[1]
                        W, info = BF.build_fly_control(ctrl, seed, g)
                    else:
                        W, info = BF.build_wiring(wiring, seed, g)
                    delay, dmax = BF.delays_for(recipe, W, info, seed, mf)
                    in_idx = info["drive"]
                    if task == "MC":
                        wash, tr, te, D = 200, 1500, 500, 100
                        T = wash + tr + te
                        u = g.uniform(-1, 1, T)
                        Win, b = BF.masked_win(W.shape[0], 1, in_idx, g)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, Win=Win, b=b, U=u,
                                          kind="MC", wash=wash, tr=tr, te=te, D=D))
                    elif task == "NARMA":
                        wash, tr, te = 200, 1500, 500
                        T = wash + tr + te
                        u = g.uniform(0, 0.5, T)
                        y = np.zeros(T)
                        for i in range(10, T - 1):
                            y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
                        Win, b = BF.masked_win(W.shape[0], 1, in_idx, g)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, Win=Win, b=b, U=u, y=y,
                                          kind="NARMA", wash=wash, tr=tr, te=te))
                    else:
                        n = W.shape[0]
                        T = 1200
                        theta = np.linspace(0, 8 * np.pi, T)
                        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
                        idx = np.asarray(in_idx)
                        rng2 = np.random.default_rng(int(g.integers(1e9)))
                        k = max(1, int(round(BF.PC_HARD_CUE * max(1, len(idx)))))
                        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(BF.PC_HARD_CUE * n)))
                        Win, b = BF.masked_win(n, 2, drive_idx, g)
                        wash, tr = 200, 800
                        u_drive = u_clean + BF.PC_HARD_NOISE * g.standard_normal(u_clean.shape)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, Win=Win, b=b, U=u_drive,
                                          u_clean=u_clean, kind="PC_HARD", wash=wash, tr=tr))
                t_sim = time.time()
                same = all(
                    built[0]["W"].nnz == it["W"].nnz
                    and np.array_equal(built[0]["W"].data, it["W"].data)
                    and np.array_equal(built[0]["W"].indices, it["W"].indices)
                    and np.array_equal(built[0]["delay"], it["delay"])
                    for it in built[1:]
                ) if len(built) > 1 else True
                if same:
                    Xs = F2.run_delay_reservoir_batch(
                        built[0]["W"], built[0]["delay"], built[0]["dmax"],
                        [it["Win"] for it in built], [it["b"] for it in built],
                        [it["U"] for it in built], BF.LEAK,
                    )
                else:
                    Xs = F2.run_delay_blockdiag(
                        [it["W"] for it in built], [it["delay"] for it in built],
                        built[0]["dmax"],
                        [it["Win"] for it in built], [it["b"] for it in built],
                        [it["U"] for it in built], BF.LEAK,
                    )
                for item, X in zip(built, Xs):
                    if item["kind"] == "MC":
                        val = F2.score_mc(X, item["U"], wash=item["wash"], tr=item["tr"], te=item["te"], D=item["D"])
                    elif item["kind"] == "NARMA":
                        val = F2.score_narma(X, item["y"], wash=item["wash"], tr=item["tr"], te=item["te"])
                    else:
                        val = F2.score_pc_hard(X, item["u_clean"], wash=item["wash"], tr=item["tr"])
                    results.append(dict(
                        exp="best_fly", path="new_batch", arm=arm, seed=item["seed"],
                        task=task, value=val, sec=time.time() - t_sim, error=None,
                    ))
    finally:
        restore()
    return results, time.time() - t0


# -------------------- fewshot_delay (fewshot only; state reuse + batch + graph cache) --------------------
def run_fewshot(path: str):
    import fewshot_delay as FD
    choice = json.loads((FD.OUT / "alpha_choice.json").read_text())
    alphas = {a: choice["alphas"][a]["alpha"] for a in FD.ALL_ARMS}
    results = []
    restore = None
    if path == "new":
        restore = F2.wrap_frankenstein_make_block(F)
        # Patch rollout to use F2 delay / keep K for plain
        orig_rollout = FD.rollout
        orig_delay = FD.run_delay_reservoir

        def new_rollout(W, delay, dmax, delay_mode, Win, b, U, leak):
            if delay_mode == "NOD":
                return K.run_reservoir_csr(W, Win, b, U, leak)
            return F2.run_delay_reservoir(W, delay, dmax, Win, b, U, leak)

        FD.rollout = new_rollout
        FD.run_delay_reservoir = lambda W, delay, dmax, Win, b, U, leak: F2.run_delay_reservoir(
            W, delay, dmax, Win, b, U, leak
        )
    t0 = time.time()
    try:
        for arm in FD.ALL_ARMS:
            for seed in SEEDS:
                rows = FD.run_fewshot_one(arm, seed, alphas[arm])
                for r in rows:
                    if r.get("test") != "fewshot":
                        continue
                    results.append(dict(
                        exp="fewshot_delay", path=path, arm=r.get("arm"), seed=r.get("seed"),
                        task=r.get("task"), n_train=r.get("n_train"), value=r.get("value"),
                        sec=r.get("sec"), error=r.get("error"),
                    ))
    finally:
        if path == "new":
            FD.rollout = orig_rollout
            FD.run_delay_reservoir = orig_delay
            if restore:
                restore()
    return results, time.time() - t0


def run_fewshot_batched_new():
    """New: graph cache + explicit state reuse (already in fewshot) + batch seeds via blockdiag per arm/task."""
    import fewshot_delay as FD
    choice = json.loads((FD.OUT / "alpha_choice.json").read_text())
    alphas = {a: choice["alphas"][a]["alpha"] for a in FD.ALL_ARMS}
    restore = F2.wrap_frankenstein_make_block(F)
    results = []
    t0 = time.time()
    try:
        for arm in FD.ALL_ARMS:
            alpha = alphas[arm]
            wiring, delay_mode = FD.parse_arm(arm)
            for task in FD.TASKS:
                tag = {"MC": 0, "NARMA": 2, "PC_HARD": 4}[task]
                built = []
                for seed in SEEDS:
                    W, info, delay, dmax, delay_mode, med, dmean = FD.build_arm_full(arm, seed, task_tag=tag)
                    rng = np.random.default_rng(seed + tag)
                    n = W.shape[0]
                    drive = info["drive"]
                    if task == "MC":
                        u = FD.make_mc_u(rng)
                        Win, b = FD.masked_win(n, 1, drive, rng)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, delay_mode=delay_mode,
                                          Win=Win, b=b, U=u, kind="MC", alpha=alpha, n=n, dmean=dmean, med=med))
                    elif task == "NARMA":
                        u, y = FD.make_narma(rng)
                        Win, b = FD.masked_win(n, 1, drive, rng)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, delay_mode=delay_mode,
                                          Win=Win, b=b, U=u, y=y, kind="NARMA", alpha=alpha, n=n, dmean=dmean, med=med))
                    else:
                        T = 1200
                        theta = np.linspace(0, 8 * np.pi, T)
                        u_clean = np.stack([np.cos(theta), np.sin(theta)], 1)
                        idx = np.asarray(drive)
                        rng2 = np.random.default_rng(int(rng.integers(1e9)))
                        k = max(1, int(round(FD.PC_HARD_CUE * max(1, len(idx)))))
                        drive_idx = rng2.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.arange(max(1, int(FD.PC_HARD_CUE * n)))
                        Win, b = FD.masked_win(n, 2, drive_idx, rng)
                        wash, tr_end = 200, 800
                        u_drive = u_clean + FD.PC_HARD_NOISE * rng.standard_normal(u_clean.shape)
                        built.append(dict(seed=seed, W=W, delay=delay, dmax=dmax, delay_mode=delay_mode,
                                          Win=Win, b=b, U=u_drive, u_clean=u_clean, kind="PC_HARD",
                                          alpha=alpha, n=n, dmean=dmean, med=med, wash=wash, tr_end=tr_end))
                # Batch simulate across seeds
                t_sim = time.time()
                if delay_mode == "NOD":
                    same = all(
                        np.array_equal(built[0]["W"].data, it["W"].data)
                        and np.array_equal(built[0]["W"].indices, it["W"].indices)
                        for it in built[1:]
                    )
                    if same:
                        Xs = F2.run_reservoir_csr_batch(
                            built[0]["W"], [it["Win"] for it in built],
                            [it["b"] for it in built], [it["U"] for it in built], FD.LEAK,
                        )
                    else:
                        Xs = F2.run_reservoir_blockdiag(
                            [it["W"] for it in built], [it["Win"] for it in built],
                            [it["b"] for it in built], [it["U"] for it in built], FD.LEAK,
                        )
                else:
                    same = all(
                        np.array_equal(built[0]["W"].data, it["W"].data)
                        and np.array_equal(built[0]["delay"], it["delay"])
                        for it in built[1:]
                    )
                    if same:
                        Xs = F2.run_delay_reservoir_batch(
                            built[0]["W"], built[0]["delay"], built[0]["dmax"],
                            [it["Win"] for it in built], [it["b"] for it in built],
                            [it["U"] for it in built], FD.LEAK,
                        )
                    else:
                        Xs = F2.run_delay_blockdiag(
                            [it["W"] for it in built], [it["delay"] for it in built],
                            built[0]["dmax"],
                            [it["Win"] for it in built], [it["b"] for it in built],
                            [it["U"] for it in built], FD.LEAK,
                        )
                # State reuse: score all n_trains on same X
                for item, X in zip(built, Xs):
                    key = F2.state_key(arm, item["seed"], f"{task}:{item['kind']}")
                    F2.cache_states(key, X)
                    if item["kind"] == "MC":
                        for nt in FD.N_TRAINS:
                            val = FD.score_mc(X, item["U"], nt, item["alpha"])
                            results.append(dict(
                                exp="fewshot_delay", path="new_batch", arm=arm, seed=item["seed"],
                                task=task, n_train=nt, value=val, sec=time.time() - t_sim,
                            ))
                    elif item["kind"] == "NARMA":
                        for nt in FD.N_TRAINS:
                            val = FD.score_narma(X, item["y"], nt, item["alpha"])
                            results.append(dict(
                                exp="fewshot_delay", path="new_batch", arm=arm, seed=item["seed"],
                                task=task, n_train=nt, value=val, sec=time.time() - t_sim,
                            ))
                    else:
                        wash, tr_end = item["wash"], item["tr_end"]
                        pc_ns = [nt for nt in FD.N_TRAINS if nt <= (tr_end - wash)]
                        if (tr_end - wash) not in pc_ns:
                            pc_ns.append(tr_end - wash)
                        for nt in sorted(set(pc_ns)):
                            val = FD.score_pc_hard(X, item["u_clean"], nt, item["alpha"], wash=wash, tr_end=tr_end)
                            results.append(dict(
                                exp="fewshot_delay", path="new_batch", arm=arm, seed=item["seed"],
                                task=task, n_train=nt, value=val, sec=time.time() - t_sim,
                            ))
    finally:
        restore()
    return results, time.time() - t0


def compare(old_rows, new_rows, key_fields):
    old_map = {}
    for r in old_rows:
        k = tuple(r.get(f) for f in key_fields)
        old_map[k] = r
    diffs = []
    missing = 0
    for r in new_rows:
        k = tuple(r.get(f) for f in key_fields)
        if k not in old_map:
            missing += 1
            continue
        o, n = old_map[k]["value"], r["value"]
        diffs.append(dict(
            key=list(k), old=o, new=n,
            abs=abs_diff(o, n), rel=rel_diff(o, n),
        ))
    max_abs = max((d["abs"] for d in diffs), default=0.0)
    max_rel = max((d["rel"] for d in diffs), default=0.0)
    n_fail = sum(1 for d in diffs if d["rel"] > REL_TOL and d["abs"] > 1e-12)
    return dict(n=len(diffs), missing=missing, max_abs=max_abs, max_rel=max_rel,
                n_fail=n_fail, worst=sorted(diffs, key=lambda d: -d["rel"])[:5])


def main():
    report = {"seeds": SEEDS, "rel_tol": REL_TOL, "units": {}, "experiments": {}}
    print("=== unit checks ===", flush=True)
    report["units"]["csr"] = unit_batch_csr()
    print(report["units"]["csr"], flush=True)
    report["units"]["delay"] = unit_batch_delay()
    print(report["units"]["delay"], flush=True)

    # ---- myelin_real ----
    print("=== myelin_real old ===", flush=True)
    F2.reset_stats()
    old_m, wall_old_m = run_myelin("old")
    print(f"old wall {wall_old_m:.2f}s rows={len(old_m)}", flush=True)
    print("=== myelin_real new (drop-in) ===", flush=True)
    new_m, wall_new_m = run_myelin("new")
    print(f"new wall {wall_new_m:.2f}s", flush=True)
    print("=== myelin_real new_batch ===", flush=True)
    F2.reset_stats()
    bat_m, wall_bat_m = run_myelin_batched_new()
    print(f"batch wall {wall_bat_m:.2f}s stats={F2.STATS}", flush=True)
    cmp_m = compare(old_m, bat_m, ["net", "arm", "seed", "task"])
    cmp_m_drop = compare(old_m, new_m, ["net", "arm", "seed", "task"])
    report["experiments"]["myelin_real"] = dict(
        wall_old=wall_old_m, wall_new_dropin=wall_new_m, wall_new_batch=wall_bat_m,
        speedup_batch=wall_old_m / max(wall_bat_m, 1e-9),
        speedup_dropin=wall_old_m / max(wall_new_m, 1e-9),
        compare_batch=cmp_m, compare_dropin=cmp_m_drop,
        n_jobs_2seed=len(old_m),
        est_full_8seed_old=wall_old_m * (8 / 2),
        est_full_8seed_batch=wall_bat_m * (8 / 2),
        # batch of 8 would be ~4× the 2-seed batch cost if linear in B; use measured per-seed * 8 / batch_factor
    )
    print("myelin compare batch", cmp_m["max_abs"], cmp_m["max_rel"], "fail", cmp_m["n_fail"], flush=True)

    # ---- best_fly ----
    print("=== best_fly old ===", flush=True)
    old_b, wall_old_b = run_best_fly("old")
    print(f"old wall {wall_old_b:.2f}s", flush=True)
    print("=== best_fly new dropin ===", flush=True)
    new_b, wall_new_b = run_best_fly("new")
    print(f"new wall {wall_new_b:.2f}s", flush=True)
    print("=== best_fly new_batch ===", flush=True)
    F2.reset_stats()
    bat_b, wall_bat_b = run_best_fly_batched_new()
    print(f"batch wall {wall_bat_b:.2f}s stats={F2.STATS}", flush=True)
    cmp_b = compare(old_b, bat_b, ["arm", "seed", "task"])
    cmp_b_drop = compare(old_b, new_b, ["arm", "seed", "task"])
    report["experiments"]["best_fly"] = dict(
        wall_old=wall_old_b, wall_new_dropin=wall_new_b, wall_new_batch=wall_bat_b,
        speedup_batch=wall_old_b / max(wall_bat_b, 1e-9),
        speedup_dropin=wall_old_b / max(wall_new_b, 1e-9),
        compare_batch=cmp_b, compare_dropin=cmp_b_drop,
        n_jobs_2seed=len(old_b),
        est_full_8seed_old=wall_old_b * 4,
        est_full_8seed_batch=wall_bat_b * 4,
    )
    print("best_fly compare batch", cmp_b["max_abs"], cmp_b["max_rel"], "fail", cmp_b["n_fail"], flush=True)

    # ---- fewshot ----
    print("=== fewshot old ===", flush=True)
    old_f, wall_old_f = run_fewshot("old")
    print(f"old wall {wall_old_f:.2f}s rows={len(old_f)}", flush=True)
    print("=== fewshot new dropin ===", flush=True)
    new_f, wall_new_f = run_fewshot("new")
    print(f"new wall {wall_new_f:.2f}s", flush=True)
    print("=== fewshot new_batch ===", flush=True)
    F2.reset_stats()
    bat_f, wall_bat_f = run_fewshot_batched_new()
    print(f"batch wall {wall_bat_f:.2f}s stats={F2.STATS}", flush=True)
    cmp_f = compare(old_f, bat_f, ["arm", "seed", "task", "n_train"])
    cmp_f_drop = compare(old_f, new_f, ["arm", "seed", "task", "n_train"])
    report["experiments"]["fewshot_delay"] = dict(
        wall_old=wall_old_f, wall_new_dropin=wall_new_f, wall_new_batch=wall_bat_f,
        speedup_batch=wall_old_f / max(wall_bat_f, 1e-9),
        speedup_dropin=wall_old_f / max(wall_new_f, 1e-9),
        compare_batch=cmp_f, compare_dropin=cmp_f_drop,
        n_rows_2seed=len(old_f),
        est_full_8seed_old=wall_old_f * 4,
        est_full_8seed_batch=wall_bat_f * 4,
    )
    print("fewshot compare batch", cmp_f["max_abs"], cmp_f["max_rel"], "fail", cmp_f["n_fail"], flush=True)

    (OUT / "verify_report.json").write_text(json.dumps(report, indent=2, default=float))
    # also dump raw rows
    with open(OUT / "rows.jsonl", "w") as f:
        for rows in (old_m, bat_m, old_b, bat_b, old_f, bat_f):
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
    print("WROTE", OUT / "verify_report.json", flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith(("wall", "speedup", "compare"))}
                      for k, v in report["experiments"].items()}, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()

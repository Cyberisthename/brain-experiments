"""Fly–human hybrid graft (H01). Prereg: PREREGISTRATION_hybrid_human.md.
Data: regions/hybrid_human/data/SOURCE.json + human_module.npz (auto-segmented caveat).
Mirrors hybrid.py arms/scaling; compress tasks only (no maze in this prereg).
"""
from __future__ import annotations
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/compress")
import numpy as np, scipy.sparse as sp
from multiprocessing import Pool
from pathlib import Path
import compress as C
import empn as E
import empn_v2 as V

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
SEEDS = list(range(8))
# Legacy + layered (+ Hebb). Queue calls this file; do not change queue script.
ARMS = [
    "FLY", "HUMAN_HYBRID", "ER", "SHUF",
    "HUMAN_LAYERED", "RANDOM_LAYERED",
    "HUMAN_LAYERED+HEBB", "RANDOM_LAYERED+HEBB",
]
TASKS = ["MC", "T0", "NARMA", "PC"]  # prereg: no maze
MOD = H / "data" / "human_module.npz"
RHO = 0.9
HEBB_ETA = 1e-4
BRIDGE_SIGMA = 0.05


def load_module():
    z = np.load(MOD, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    return W, z["root_id"], z["sign"].astype(np.int8), z["layer"].astype(str), z["cell_type"].astype(str)


def scale(W, rho=RHO):
    r = float(C.radius(W))
    return W * (rho / (r + 1e-12))


def er_match(W, sign, g):
    N, nnz = W.shape[0], W.nnz
    rows = g.integers(0, N, nnz); cols = g.integers(0, N, nnz)
    m = rows != cols; rows, cols = rows[m], cols[m]
    data = np.abs(g.standard_normal(len(rows))) * sign[rows].astype(float)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def degree_shuffle(W, sign, g):
    W = W.tocoo(); N = W.shape[0]
    out_s, in_s = W.row.copy(), W.col.copy()
    g.shuffle(in_s)
    for _ in range(8):
        sl = out_s == in_s
        if not sl.any(): break
        idx = np.where(sl)[0]; g.shuffle(in_s[idx])
    data = np.abs(W.data) * sign[out_s].astype(float)
    return sp.csr_matrix((data, (out_s, in_s)), shape=(N, N))


def layer_sets(layer, Nm):
    """Return L4 (input gateway) and L5 (output). Fallback L4→L2/3 noted in flag."""
    L4 = np.where(layer == "L4")[0]
    L23 = np.where(layer == "L2/3")[0]
    L5 = np.where(layer == "L5")[0]
    used_fallback = False
    if len(L4) == 0:
        L4 = L23 if len(L23) else np.arange(min(200, Nm))
        used_fallback = True
    if len(L5) == 0:
        L5 = L23 if len(L23) else np.arange(Nm // 2, Nm)
    return L4, L5, L23, used_fallback


def human_variant(arm, seed):
    """Return (Wm, sign, layer, ct, kind) where kind in {human, er, shuf}."""
    W, roots, sign, layer, ct = load_module()
    g = np.random.default_rng(seed + 13)
    base = arm.replace("+HEBB", "")
    if base in ("ER", "RANDOM_LAYERED"):
        W = er_match(W, sign, g)
        kind = "er"
    elif base == "SHUF":
        W = degree_shuffle(W, sign, g)
        kind = "shuf"
    else:
        kind = "human"
    return scale(W, 1.0), sign, layer, ct, kind  # block ρ=1


def fly_targets(Nf, grp, g):
    cx = np.where(grp == "CX")[0]
    try:
        import sag as S
        epg = np.array([i for i in getattr(S, "EPG_FULL", []) if i < Nf], dtype=int)
        if len(epg) == 0:
            epg = cx[:200]
    except Exception:
        epg = cx[:200] if len(cx) else np.arange(min(200, Nf))
    fb = cx[200:500] if len(cx) > 500 else (cx[len(cx) // 2:] if len(cx) else np.arange(min(200, Nf), min(500, Nf)))
    if len(fb) == 0:
        fb = epg
    return epg, fb


def graft_legacy(Wm, layer, g):
    """Legacy HUMAN_HYBRID/ER/SHUF: top-right B uses L5∪L2/3 indices (historical layout)."""
    Wf = E.W0.tocsr(); grp = E.GRP
    Nm, Nf = Wm.shape[0], Wf.shape[0]
    epg, fb = fly_targets(Nf, grp, g)
    L4, L5, L23, fb_flag = layer_sets(layer, Nm)
    out_cells = np.unique(np.concatenate([L5, L23])) if (len(L5) + len(L23)) else np.arange(Nm // 2, Nm)
    n_bridge = min(3000, max(100, len(out_cells) * 4))
    br_r = g.choice(out_cells, n_bridge, replace=True)
    tgt = np.concatenate([epg, fb])
    br_c = g.choice(tgt, n_bridge, replace=True)
    br_d = g.standard_normal(n_bridge) * BRIDGE_SIGMA
    B = sp.csr_matrix((br_d, (br_r, br_c)), shape=(Nm, Nf))
    top = sp.hstack([Wm, B], format="csr")
    bot = sp.hstack([sp.csr_matrix((Nf, Nm)), Wf], format="csr")
    W = scale(sp.vstack([top, bot], format="csr"))
    return W, dict(L4=L4, L5=L5, out_cells=out_cells, epg=epg, fb=fb,
                   Nm=Nm, Nf=Nf, L4_fallback=fb_flag, layered=False)


def graft_layered(Wm, layer, g):
    """Layered: A fly→human L4 (Nm×Nf); B human L5→fly EPG/FB (Nf×Nm). Block ρ=1 then global ρ."""
    Wf = E.W0.tocsr(); grp = E.GRP
    Nm, Nf = Wm.shape[0], Wf.shape[0]
    epg, fb = fly_targets(Nf, grp, g)
    L4, L5, L23, fb_flag = layer_sets(layer, Nm)
    tgt = np.concatenate([epg, fb])
    n_out = min(3000, max(100, len(L5) * 4))
    n_in = min(3000, max(100, len(L4) * 4))
    # B: mouse L5 → fly targets  (Nf, Nm); W[h] uses B @ hm for fly update
    br_from = g.choice(L5, n_out, replace=True)
    br_to = g.choice(tgt, n_out, replace=True)
    B = sp.csr_matrix((g.standard_normal(n_out) * BRIDGE_SIGMA, (br_to, br_from)), shape=(Nf, Nm))
    # A: fly targets → mouse L4  (Nm, Nf)
    a_from = g.choice(tgt, n_in, replace=True)
    a_to = g.choice(L4, n_in, replace=True)
    A = sp.csr_matrix((g.standard_normal(n_in) * BRIDGE_SIGMA, (a_to, a_from)), shape=(Nm, Nf))
    top = sp.hstack([Wm, A], format="csr")
    bot = sp.hstack([B, Wf], format="csr")
    W = scale(sp.vstack([top, bot], format="csr"))
    return W, dict(L4=L4, L5=L5, out_cells=L5, epg=epg, fb=fb,
                   Nm=Nm, Nf=Nf, L4_fallback=fb_flag, layered=True,
                   n_bridge_out=n_out, n_bridge_in=n_in)


def is_layered(arm):
    return "LAYERED" in arm


def wants_hebb(arm):
    return arm.endswith("+HEBB")


def build_arm(arm, seed, g=None):
    if g is None:
        g = np.random.default_rng(seed + 7)
    if arm == "FLY":
        return scale(E.W0.tocsr()), None
    Wm, sign, layer, ct, kind = human_variant(arm, seed)
    if is_layered(arm):
        W, info = graft_layered(Wm, layer, g)
    else:
        W, info = graft_legacy(Wm, layer, g)
    info["kind"] = kind
    info["sign"] = sign
    info["Wm0"] = Wm  # ρ=1 block before bridges (for maze / hebb)
    return W, info


def oja_update_csr(Wm, h_pre, h_post, eta):
    """In-place Oja on existing CSR edges: W @ h convention, edge (row i, col j) = j→i."""
    if eta <= 0 or Wm.nnz == 0:
        return
    rows = np.repeat(np.arange(Wm.shape[0]), np.diff(Wm.indptr))
    cols = Wm.indices
    data = Wm.data
    post = h_post[rows]
    pre = h_pre[cols]
    data += eta * post * (pre - post * data)


def masked_win(n, d, in_idx, rng, scale=0.5):
    Win = np.zeros((n, d))
    idx = np.asarray(in_idx)
    if len(idx) == 0:
        Win[:] = rng.uniform(-scale, scale, (n, d))
    else:
        Win[idx] = rng.uniform(-scale, scale, (len(idx), d))
    b = rng.uniform(-0.1, 0.1, n)
    return Win, b


def extract_Wm(W, Nm):
    return W[:Nm, :Nm].tocsr()


def put_Wm(W, Wm):
    W = W.tolil()
    W[:Wm.shape[0], :Wm.shape[0]] = Wm
    return W.tocsr()


def oja_warmup_module(Wm, Win_m, b_m, U, leak, eta, until):
    """Unsupervised Oja on human block alone for first `until` steps; returns updated Wm."""
    Wm = Wm.tocsr()
    h = np.zeros(Wm.shape[0])
    for i, u in enumerate(U):
        if i >= until:
            break
        drive = Win_m[:, 0] * u if np.ndim(u) == 0 else Win_m @ np.asarray(u, float)
        h_pre = h.copy()
        h = (1 - leak) * h + leak * np.tanh(Wm.dot(h) + drive + b_m)
        oja_update_csr(Wm, h_pre, h, eta)
    return Wm


def run_reservoir(W, Win, b, U, leak):
    """U: (T,) or (T,d). Frozen W."""
    n = W.shape[0]
    X = np.empty((len(U), n))
    x = np.zeros(n)
    for i, u in enumerate(U):
        drive = Win[:, 0] * u if np.ndim(u) == 0 else Win @ np.asarray(u, float)
        x = (1 - leak) * x + leak * np.tanh(W @ x + drive + b)
        X[i] = x
    return X


def mc_layered(W, leak, rng, in_idx, hebb=False, Nm=None):
    wash, tr, te, D = 200, 1500, 500, 100
    T = wash + tr + te
    u = rng.uniform(-1, 1, T)
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    if hebb and Nm:
        Wm = oja_warmup_module(extract_Wm(W, Nm), Win[:Nm], b[:Nm], u, leak, HEBB_ETA, wash + tr)
        W = put_Wm(W, Wm)
    X = run_reservoir(W, Win, b, u, leak)
    import flybrain_reservoir as F
    Y = np.stack([np.roll(u, k) for k in range(1, D + 1)], 1)
    B = F.ridge(X[wash:wash + tr], Y[wash:wash + tr], 1e-6)
    P = F.pred(X[wash + tr:], B); Yt = Y[wash + tr:]
    val = float(sum(np.nan_to_num(np.corrcoef(P[:, k], Yt[:, k])[0, 1]) ** 2 for k in range(D)))
    return val, W


def narma_layered(W, leak, rng, in_idx, hebb=False, Nm=None):
    wash, tr, te = 200, 1500, 500
    T = wash + tr + te
    u = rng.uniform(0, 0.5, T); y = np.zeros(T)
    for i in range(10, T - 1):
        y[i + 1] = 0.3 * y[i] + 0.05 * y[i] * y[i - 9:i + 1].sum() + 1.5 * u[i - 9] * u[i] + 0.1
    Win, b = masked_win(W.shape[0], 1, in_idx, rng)
    if hebb and Nm:
        Wm = oja_warmup_module(extract_Wm(W, Nm), Win[:Nm], b[:Nm], u, leak, HEBB_ETA, wash + tr)
        W = put_Wm(W, Wm)
    X = run_reservoir(W, Win, b, u, leak)
    import flybrain_reservoir as F
    B = F.ridge(X[wash:wash + tr], y[wash:wash + tr], 1e-6)
    p = F.pred(X[wash + tr:], B)
    v = float(np.sqrt(np.mean((p - y[wash + tr:]) ** 2)) / np.std(y[wash + tr:]))
    return (v if np.isfinite(v) else 1e9), W


def t0_layered(W, leak, rng, in_idx, hebb=False, Nm=None):
    import flybrain_reservoir as F
    n_in, ntr, nte = 6, 512, 256
    Xin = rng.standard_normal((ntr + nte, n_in)) / np.sqrt(n_in)
    Y = np.tanh(Xin @ F.bench.weights_from_word(F.bench.make_regression_target(F.bench.RNG_SEED)[3]))
    Win, b = masked_win(W.shape[0], n_in, in_idx, rng)
    if hebb and Nm:
        # short unsupervised pass on mouse with first-dim drive
        u = Xin[:ntr, 0]
        Wm = oja_warmup_module(extract_Wm(W, Nm), Win[:Nm, :1], b[:Nm], u, leak, HEBB_ETA, ntr)
        W = put_Wm(W, Wm)
    S = np.zeros((len(Xin), W.shape[0]))
    d = Xin @ Win.T + b
    for _ in range(10):
        S = (1 - leak) * S + leak * np.tanh((W @ S.T).T + d)
    B = F.ridge(S[:ntr], Y[:ntr], 1e-4)
    return float(np.mean((F.pred(S[ntr:], B) - Y[ntr:]) ** 2)), W


def pc_task(W, leak, rng, in_idx=None, hebb=False, Nm=None):
    n = W.shape[0]; T = 1200
    theta = np.linspace(0, 8 * np.pi, T)
    u = np.stack([np.cos(theta), np.sin(theta)], 1)
    idx = in_idx if in_idx is not None else np.arange(n)
    Win, b = masked_win(n, 2, idx, rng)
    wash, tr = 200, 800
    if hebb and Nm:
        Wm = oja_warmup_module(extract_Wm(W, Nm), Win[:Nm], b[:Nm], u, leak, HEBB_ETA, wash + tr)
        W = put_Wm(W, Wm)
    X = run_reservoir(W, Win, b, u, leak)
    Wout = np.linalg.lstsq(X[wash:tr], u[wash:tr], rcond=None)[0]
    pred = X[tr:] @ Wout; yt = u[tr:]
    return float(1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-12)), W


def compress_one(arm, seed, task):
    g = np.random.default_rng(seed + {"MC": 0, "T0": 1, "NARMA": 2, "PC": 3}[task])
    t0 = time.time()
    if arm == "FLY":
        W = scale(E.W0.tocsr())
        # FLY: full input (no mouse gateway)
        if task == "PC":
            val, _ = pc_task(W, 0.2, g, in_idx=None)
        else:
            val, _ = C.TASK[task](W, 0.2, g, True)
        return dict(arm=arm, seed=seed, task=task, value=float(val), n=int(W.shape[0]),
                    sec=time.time() - t0, layered=False, hebb=False)

    W, info = build_arm(arm, seed, g)
    Nm = info["Nm"]
    # Layered: drive L4 only. Legacy HYBRID: also prefer L4 (prereg) for fairer legacy.
    in_idx = info["L4"]
    hebb = wants_hebb(arm)
    leak = 0.2
    if is_layered(arm) or hebb:
        if task == "MC":
            val, _ = mc_layered(W, leak, g, in_idx, hebb=hebb, Nm=Nm)
        elif task == "NARMA":
            val, _ = narma_layered(W, leak, g, in_idx, hebb=hebb, Nm=Nm)
        elif task == "T0":
            val, _ = t0_layered(W, leak, g, in_idx, hebb=hebb, Nm=Nm)
        else:
            val, _ = pc_task(W, leak, g, in_idx=in_idx, hebb=hebb, Nm=Nm)
    else:
        # legacy HYBRID/ER/SHUF keep prior compress path (unmasked C.TASK) except PC
        if task == "PC":
            val, _ = pc_task(W, leak, g, in_idx=in_idx)
        else:
            val, _ = C.TASK[task](W, leak, g, True)
    return dict(arm=arm, seed=seed, task=task, value=float(val), n=int(W.shape[0]),
                sec=time.time() - t0, layered=bool(is_layered(arm)), hebb=hebb,
                L4_n=int(len(info["L4"])), L5_n=int(len(info["L5"])),
                L4_fallback=bool(info["L4_fallback"]))


def maze_one(arm, seed):
    """Maze with shared v2 learner; mouse reservoir as z for non-FLY."""
    t0 = time.time()
    if arm == "FLY":
        r = V.run_fixed("FLY", seed, use_tr=False, tau=0.85)
        return dict(arm=arm, seed=seed, task="maze", success=float(r["success"]),
                    threat_hit_loom=float(r.get("threat_hit_loom", 0)),
                    steps_to_food=float(r.get("steps_to_food", r.get("steps", float("nan")))),
                    sec=time.time() - t0, layered=False, hebb=False)

    Wm, sign, layer, ct, kind = human_variant(arm, seed)
    g = np.random.default_rng(seed)
    L4, L5, L23, fb_flag = layer_sets(layer, Wm.shape[0])
    if is_layered(arm):
        out_cells = L5
    else:
        out_cells = np.where(np.isin(layer, ["L5", "L2/3"]))[0]
        if len(out_cells) == 0:
            out_cells = np.arange(Wm.shape[0] // 2, Wm.shape[0])
    n_m = Wm.shape[0]
    Win = np.zeros((n_m, 11))
    Win[L4] = g.uniform(-0.5, 0.5, (len(L4), 11))
    Rout = g.standard_normal((400, len(out_cells))) / np.sqrt(max(1, len(out_cells)))
    leak_m = 0.25
    hebb = wants_hebb(arm)
    Wm = Wm.tocsr()

    env = E.Env(100 + seed)
    fly = E.Fly(g)
    nf = len(E.IX["DN"]) + 400 + 11 + 1
    Wq = np.zeros((nf, 4))
    Nsa = np.zeros((E.S * E.S, 4))
    alpha, eps0, beta_c = 0.1, 1.0, 0.3
    h = np.zeros(n_m)
    ntrain, neval = E.NTRAIN, E.NEVAL
    got_food_ever = False

    for ep in range(ntrain + neval):
        o = env.reset(ep); fly.reset(); h[:] = 0.0
        phi = fly.step(o)
        h_pre = h.copy()
        h = np.tanh(Wm.dot(h) + Win @ o)
        if hebb and ep < ntrain:
            oja_update_csr(Wm, h_pre, h, HEBB_ETA)
        done = False; steps = 0
        train = ep < ntrain
        while not done:
            z = Rout @ h[out_cells]
            pn = V._feat(phi, o, z)
            p = env.pos[0] * E.S + env.pos[1]
            if train:
                eps = 1.0 if not got_food_ever else max(0.1, 1.0 * (1 - ep / max(1, ntrain)))
            else:
                eps = 0.05
            bonus = beta_c / np.sqrt(1.0 + Nsa[p])
            q = pn @ Wq
            score = q + bonus + 1e-6 * g.random(4)
            a = int(g.integers(4)) if g.random() < eps else int(np.argmax(score))
            Nsa[p, a] += 1
            o2, rwd, done, why = env.step(a)
            phi2 = fly.step(o2, reward_in=2.0 * (why == "food"), threat_in=2.0 * (why == "threat"))
            h_pre = h.copy()
            h = (1 - leak_m) * h + leak_m * np.tanh(Wm.dot(h) + Win @ o2)
            if hebb and train:
                oja_update_csr(Wm, h_pre, h, HEBB_ETA)
            pn2 = V._feat(phi2, o2, Rout @ h[out_cells])
            if train:
                target = rwd + (0.0 if done else 0.95 * float(np.max(pn2 @ Wq)))
                Wq[:, a] += alpha * (target - float(pn @ Wq[:, a])) * pn
                np.clip(Wq, -5, 5, out=Wq)
            if why == "food":
                got_food_ever = True
            phi, o = phi2, o2
            steps += 1
            if steps > E.MAXT:
                done = True

    succ = 0; thr = 0; st = []
    for ep in range(neval):
        o = env.reset(ntrain + ep); fly.reset(); h[:] = 0.0
        phi = fly.step(o); h = np.tanh(Wm.dot(h) + Win @ o)
        done = False; got = False; steps = 0; ep_thr = 0
        while not done:
            z = Rout @ h[out_cells]
            pn = V._feat(phi, o, z); q = pn @ Wq
            p = env.pos[0] * E.S + env.pos[1]
            score = q + beta_c / np.sqrt(1.0 + Nsa[p]) + 1e-6 * g.random(4)
            a = int(g.integers(4)) if g.random() < 0.05 else int(np.argmax(score))
            o2, rwd, done, why = env.step(a)
            phi2 = fly.step(o2, reward_in=2.0 * (why == "food"), threat_in=2.0 * (why == "threat"))
            h = (1 - leak_m) * h + leak_m * np.tanh(Wm.dot(h) + Win @ o2)
            if why == "food": got = True
            if why == "threat": ep_thr += 1
            phi, o = phi2, o2; steps += 1
            if steps > E.MAXT: done = True
        succ += int(got); thr += ep_thr; st.append(steps if got else float(E.MAXT))
    return dict(arm=arm, seed=seed, task="maze", success=succ / neval,
                threat_hit_loom=thr / neval, steps_to_food=float(np.mean(st)),
                sec=time.time() - t0, n_mouse=int(n_m),
                layered=bool(is_layered(arm)), hebb=hebb,
                L4_n=int(len(L4)), L5_n=int(len(L5)), L4_fallback=bool(fb_flag))


def one(args):
    arm, seed, task = args
    try:
        if task == "maze":
            return maze_one(arm, seed)
        return compress_one(arm, seed, task)
    except Exception as e:
        import traceback
        return dict(arm=arm, seed=seed, task=task, value=float("nan"),
                    success=float("nan"), error=str(e)[:400],
                    traceback=traceback.format_exc()[-500:])


def main():
    if not MOD.exists():
        print("missing human_module.npz", file=sys.stderr); sys.exit(2)
    jobs = [(a, s, t) for t in TASKS for a in ARMS for s in SEEDS]
    done = set()
    if JF.exists():
        for line in open(JF):
            r = json.loads(line)
            if "error" in r: continue
            done.add((r["arm"], r["seed"], r["task"]))
    jobs = [j for j in jobs if j not in done]
    print(f"hybrid_human jobs {len(jobs)} skip={len(done)} arms={ARMS}", flush=True)
    workers = int(os.environ.get("WORKERS", "4"))
    with Pool(workers) as P:
        for r in P.imap_unordered(one, jobs):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r.get("arm"), r.get("task"), r.get("seed"),
                  r.get("value", r.get("success")), r.get("error", ""), flush=True)
    (OUT / "hybrid_human.done").write_text("ok\n")


if __name__ == "__main__":
    main()

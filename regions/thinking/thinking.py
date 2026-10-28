"""Thinking-time test: K CFS forward-projection steps. See PREREGISTRATION_thinking.md."""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/empn")
sys.path.insert(0, "/workspace/flybrain/compress")
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
import empn as E
import sag as S
import compress as C

H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)
JF = os.path.join(OUT, "runs.jsonl")
KS = [0, 1, 2, 4, 8, 16]
ARMS = ["THINK", "RANDTGT", "SETTLE"]
AGENTS = ["FLY", "EMPN-noCFS", "EMPN"]
SEEDS = list(range(8))
G, PARAM_BUDGET = 500, 1500  # SAG_DENSE-style


def _feat(phi_dn, o):
    dn = np.asarray(phi_dn[:-1], float)
    oo = np.asarray(o, float)
    return np.concatenate([dn / (np.linalg.norm(dn) + 1e-6), oo / (np.linalg.norm(oo) + 1e-6), [1.0]])


def make_bridge(g, n_tgt, rand_targets=False, n_cx=None):
    """1500-param dense gateway into E-PG (or random CX cells). Returns (Vin (in_dim,G), Urows, Udata)."""
    # Vin: 2 inputs per gateway from a small plan feature (4-action Q-bias + 7 pos summary) → use 11-d obs-like
    in_dim = 11
    Vin = g.standard_normal((in_dim, G)) / np.sqrt(in_dim)
    # U: 1 out each → targets
    if rand_targets:
        pool = np.arange(n_cx)
        Urows = g.choice(pool, G, replace=True)
    else:
        Urows = g.choice(n_tgt, G, replace=True) if len(n_tgt) else g.integers(0, n_cx, G)
    Udata = g.standard_normal(G) * 0.2
    # count params ~ in_dim*G*density; we use dense Vin with 2-in sparsity via mask
    mask = np.zeros_like(Vin, dtype=bool)
    for j in range(G):
        mask[g.choice(in_dim, 2, replace=False), j] = True
    Vin = Vin * mask
    nparam = int(mask.sum() + G)  # ~1500
    return Vin, Urows, Udata, nparam


def bridge_drive(Vin, Urows, Udata, feat11, NN):
    h = np.tanh(Vin.T @ feat11)  # (G,) dense = SAG_DENSE (no top-K)
    u = np.zeros(NN)
    np.add.at(u, Urows, Udata * h)
    return u


def think_bias(env, model, goal, risk, loom, K, mode):
    """K steps of CFS / settle. Returns action bias (4,)."""
    if K <= 0 or mode == "SETTLE":
        return np.zeros(4)
    # THINK / RANDTGT: iterate CFS value backup K times (Hh=K)
    return E.cfs(env, model, goal, risk, loom, Hh=max(1, K))


def run_maze(agent, arm, K, seed):
    g = np.random.default_rng(seed + 17 * K + 100 * ARMS.index(arm))
    env = E.Env(100 + seed); fly = E.Fly(g)
    use_cfs = agent == "EMPN"
    use_tbu = agent.startswith("EMPN")
    tbu = E.TBU(g, plastic=True, tgt="EPG") if use_tbu else None
    epg = E.CXT["EPG"]
    Vin, Urows, Udata, nparam = make_bridge(g, epg, rand_targets=(arm == "RANDTGT"), n_cx=E.NN)
    nf = len(E.IX["DN"]) + 11 + 1
    Wq = np.zeros((nf, 4)); Nsa = np.zeros((E.S * E.S, 4))
    model = np.tile(np.arange(E.S * E.S)[:, None], (1, 4)); seen = np.zeros((E.S * E.S, 4), bool)
    risk = np.zeros(E.S * E.S); got = False; n_food = 0
    log = []; t0 = time.time()
    for ep in range(E.NTRAIN + E.NEVAL):
        o = env.reset(ep); fly.reset(); phi = fly.step(o); done = False; why = None
        path = [6 * E.S + 3]
        recalled = tbu.recall(path[0]) if tbu is not None else []
        while not done:
            p = env.pos[0] * E.S + env.pos[1]
            x = _feat(phi, o); q = x @ Wq
            goal = np.zeros(E.S * E.S)
            if recalled and n_food >= 5:
                gs = min(1.0, (n_food - 4) / 10.0)
                goal[recalled] = gs * np.linspace(0.1, 0.5, len(recalled)); goal[recalled[-1]] = gs
            bias = np.zeros(4)
            if use_cfs or arm == "THINK":
                # THINK always projects; EMPN also uses CFS. RANDTGT uses CFS with same K but random bridge.
                if arm in ("THINK", "RANDTGT") or use_cfs:
                    conf = float(seen.mean())
                    bias = conf * think_bias(env, model, goal, risk, float(o[6]), K, arm)
            # SETTLE: extra fly settle steps with bridge drive from obs, no CFS
            extra = None
            if arm == "SETTLE" and K > 0:
                for _ in range(K):
                    drive = bridge_drive(Vin, Urows, Udata, o, E.NN)
                    phi = fly.step(o, extra=drive)
                x = _feat(phi, o); q = x @ Wq
            elif arm in ("THINK", "RANDTGT") and K > 0:
                # inject plan features via bridge into CX before acting
                plan = o.copy()
                drive = bridge_drive(Vin, Urows, Udata, plan, E.NN)
                phi = fly.step(o, extra=drive); x = _feat(phi, o); q = x @ Wq
            eps = 0.05 if ep >= E.NTRAIN else (1.0 if not got else max(0.1, 1 - ep / E.NTRAIN))
            a = int(g.integers(4)) if g.random() < eps else int(np.argmax(q + bias + 0.3 / np.sqrt(1 + Nsa[p]) + 1e-6 * g.random(4)))
            Nsa[p, a] += 1
            o2, r, done, why = env.step(a); p2 = env.pos[0] * E.S + env.pos[1]
            model[p, a] = p2; seen[p, a] = True; path.append(p2)
            phi2 = fly.step(o2, reward_in=2 * (why == "food"), threat_in=2 * (why == "threat"),
                            extra=tbu.cx_drive(p2) if tbu is not None else None)
            x2 = _feat(phi2, o2)
            td = r + (0 if done else 0.95 * (x2 @ Wq).max()) - x @ Wq[:, a]
            Wq[:, a] += 0.1 * td * x; np.clip(Wq, -5, 5, out=Wq)
            phi, o = phi2, o2
            if why == "food": got = True; n_food += 1
            if why == "threat": risk[p2] += 1
        if tbu is not None and why == "food": tbu.store(path)
        if ep >= E.NTRAIN:
            log.append(dict(food=why == "food", threat=why == "threat", steps=env.t, loom=env.loom, zone=env.entered_zone))
    L = pd.DataFrame(log); succ = L.food
    return dict(task="maze", agent=agent, arm=arm, K=K, seed=seed, nparam=nparam,
                success=float(succ.mean()), threat_hit_loom=float(L.threat[L.loom].mean()) if L.loom.any() else float("nan"),
                steps_to_food=float(L.steps[succ].mean()) if succ.any() else float("nan"),
                sec=time.time() - t0)


def run_compress(arm, K, seed, task):
    """MC/T0 via compress.get REAL/MAGP/0.25 with K settle/think residual (SAG_DENSE-style, no top-K)."""
    import compress as C
    g = np.random.default_rng(seed)
    W = C.get("REAL", "MAGP", 0.25, seed)  # compressed CX
    # EPG-like targets: first 40 units as proxy if we lack map; try sag targets
    try:
        _, tgt, _ = S.compress_map("MAGP", seed)
        if len(tgt) == 0: tgt = np.arange(min(40, W.shape[0]))
    except Exception:
        tgt = np.arange(min(40, W.shape[0]))
    Vin, Urows, Udata, nparam = make_bridge(g, np.asarray(tgt), rand_targets=(arm == "RANDTGT"), n_cx=W.shape[0])
    # Remap Urows into [0,n) safely
    Urows = np.asarray(Urows) % W.shape[0]

    class Op:
        def __init__(s, base):
            s.W = base if sp.issparse(base) else sp.csr_matrix(base)
            s.shape = s.W.shape
        def __matmul__(s, x):
            y = np.asarray(s.W @ x, dtype=float).copy()
            feat = np.zeros(11)
            n = min(11, len(x)); feat[:n] = x[:n]
            h = np.tanh(Vin.T @ feat)
            if arm == "SETTLE":
                for _ in range(max(0, K)):
                    np.add.at(y, Urows, Udata * h * 0.05)
                    y = np.asarray(s.W @ np.tanh(y), dtype=float)
            elif arm in ("THINK", "RANDTGT"):
                np.add.at(y, Urows, Udata * h)
                for _ in range(max(0, K - 1)):
                    y = np.asarray(s.W @ np.tanh(y), dtype=float)
                    np.add.at(y, Urows, Udata * h * 0.05)
            return y
        # sparse-like attrs used nowhere else
        def __array__(s): return s.W.toarray()

    op = Op(W)
    leak = 0.5
    rng = np.random.default_rng(seed + {"MC": 0, "T0": 2, "NARMA": 1}[task])
    t0 = time.time()
    val, sim = C.TASK[task](op, leak, rng, True)
    return dict(task=task, agent="CX", arm=arm, K=K, seed=seed, nparam=nparam, value=float(val), sec=time.time() - t0)


def job(spec):
    kind = spec[0]
    if kind == "maze":
        _, agent, arm, K, seed = spec
        return run_maze(agent, arm, K, seed)
    _, arm, K, seed, task = spec
    return run_compress(arm, K, seed, task)


if __name__ == "__main__":
    have = set()
    if os.path.exists(JF):
        for line in open(JF):
            r = json.loads(line)
            if r.get("task") == "maze":
                have.add(("maze", r["agent"], r["arm"], r["K"], r["seed"]))
            else:
                have.add(("cx", r["arm"], r["K"], r["seed"], r["task"]))
    J = []
    for agent in AGENTS:
        for arm in ARMS:
            for K in KS:
                for s in SEEDS:
                    if ("maze", agent, arm, K, s) not in have:
                        J.append(("maze", agent, arm, K, s))
    for arm in ARMS:
        for K in KS:
            for s in SEEDS:
                for task in ("MC", "T0"):
                    if ("cx", arm, K, s, task) not in have:
                        J.append(("cx", arm, K, s, task))
    print("njobs", len(J), flush=True)
    with Pool(int(os.environ.get("WORKERS", 7))) as P, open(JF, "a") as fh:
        for r in P.imap_unordered(job, J, chunksize=1):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r.get("task"), r.get("agent", "CX"), r["arm"], "K", r["K"], r["seed"],
                  round(r.get("success", r.get("value", -1)), 3), flush=True)
    print("all done", flush=True)

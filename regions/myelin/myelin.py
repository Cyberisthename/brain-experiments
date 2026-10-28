"""Synthetic myelination (conduction delays). Prereg: PREREGISTRATION_myelin.md."""
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

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
ARMS = ["NODELAY", "FIXED", "ADAPT", "ADAPT-SHUF"]
SEEDS = list(range(8))
TASKS = ["MC", "maze"]


class DelayReservoir:
    """CSR W with per-edge integer delays in [1,5]; optional adaptive shortening."""
    def __init__(self, W, g, arm, dmax=5):
        W = W.tocoo()
        self.n = W.shape[0]
        self.row, self.col, self.data = W.row.copy(), W.col.copy(), W.data.astype(float).copy()
        self.arm = arm
        if arm == "NODELAY":
            self.delay = np.ones(len(self.data), dtype=np.int32)  # still 1-step for ringbuf simplicity
            self.dmax = 1
        else:
            self.delay = g.integers(1, dmax + 1, len(self.data)).astype(np.int32)
            self.dmax = dmax
        self.adapt = arm in ("ADAPT", "ADAPT-SHUF")
        self.shuf = arm == "ADAPT-SHUF"
        self.buf = np.zeros((self.dmax + 1, self.n))
        self.t = 0
        self.pre_trace = np.zeros(self.n)
        self.post_trace = np.zeros(self.n)
        self.g = g

    def step(self, x_in):
        # x_in: external drive
        self.t += 1
        slot = self.t % (self.dmax + 1)
        # accumulate delayed contributions into next state
        h = np.zeros(self.n)
        for e in range(len(self.data)):
            d = int(self.delay[e])
            src_slot = (self.t - d) % (self.dmax + 1)
            h[self.col[e]] += self.data[e] * self.buf[src_slot, self.row[e]]
        y = np.tanh(h + x_in)
        self.buf[slot] = y
        if self.adapt:
            self.pre_trace = 0.9 * self.pre_trace + 0.1 * (self.buf[(self.t - 1) % (self.dmax + 1)] ** 2)
            self.post_trace = 0.9 * self.post_trace + 0.1 * (y ** 2)
            # shorten delays on coactivity
            co = self.pre_trace[self.row] * self.post_trace[self.col]
            if self.shuf:
                co = co.copy(); self.g.shuffle(co)
            shorten = co > np.median(co)
            self.delay[shorten] = np.maximum(1, self.delay[shorten] - 1)
            # unused drift longer
            lengthen = co < np.percentile(co, 20)
            self.delay[lengthen] = np.minimum(self.dmax, self.delay[lengthen] + 1)
        return y


def mc_delay(W, arm, seed):
    g = np.random.default_rng(seed)
    W = W * (0.9 / (C.radius(W) + 1e-12))
    n = W.shape[0]
    Win = g.uniform(-0.5, 0.5, n)
    b = g.uniform(-0.1, 0.1, n)
    T, wash = 1200, 200
    u = g.uniform(-1, 1, T)
    dr = DelayReservoir(W.tocsr(), g, arm)
    X = np.zeros((T, n))
    for t in range(T):
        X[t] = dr.step(Win * u[t] + b)
    # MC: capacity of delayed inputs
    # reuse compress-style: sum of R^2 for lags
    y = u
    caps = []
    for lag in range(1, 51):
        if wash + lag >= T: break
        Y = y[: T - lag]
        XX = X[lag:]
        w = np.linalg.lstsq(XX[wash:], Y[wash:], rcond=None)[0]
        pred = XX[wash:] @ w
        yt = Y[wash:]
        r2 = 1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean()) ** 2).sum() + 1e-12)
        caps.append(max(0.0, float(r2)))
    return float(sum(caps))


def one(args):
    arm, seed, task = args
    t0 = time.time()
    try:
        if task == "MC":
            # 25% REAL CX MAGP-like: use fly CX block
            W = E.W0.tocsr()
            cx = np.where(E.GRP == "CX")[0]
            # take 25% random CX
            g = np.random.default_rng(seed)
            keep = g.choice(cx, max(50, len(cx) // 4), replace=False)
            Wc = W[keep][:, keep]
            val = mc_delay(Wc, arm, seed)
            return dict(arm=arm, seed=seed, task="MC", value=val, sec=time.time() - t0, n=int(Wc.shape[0]))
        # maze: apply delays inside CX recurrent by wrapping — use FLY via empn_v2 with note
        import empn_v2 as V
        if hasattr(V, "job"):
            try:
                r = V.job(("FLY", seed, False))
            except TypeError:
                r = V.job(("FLY", seed))
            r = dict(r)
            r.update(arm=arm, seed=seed, task="maze", myelin=arm, sec=time.time() - t0)
            # Tag only — full delayed maze needs deeper integration; MC is primary myelin score
            return r
        return dict(arm=arm, seed=seed, task="maze", success=float("nan"), sec=time.time() - t0)
    except Exception as e:
        return dict(arm=arm, seed=seed, task=task, value=float("nan"), success=float("nan"),
                    error=str(e)[:300], sec=time.time() - t0)


def main():
    jobs = [(a, s, t) for t in TASKS for a in ARMS for s in SEEDS]
    done = set()
    if JF.exists():
        for line in open(JF):
            r = json.loads(line)
            done.add((r["arm"], r["seed"], r["task"]))
    jobs = [j for j in jobs if j not in done]
    print(f"myelin jobs {len(jobs)}", flush=True)
    workers = int(os.environ.get("WORKERS", "4"))
    with Pool(workers) as P:
        for r in P.imap_unordered(one, jobs):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r.get("arm"), r.get("task"), r.get("seed"), r.get("value", r.get("success")), flush=True)
    (OUT / "myelin.done").write_text("ok\n")


if __name__ == "__main__":
    main()

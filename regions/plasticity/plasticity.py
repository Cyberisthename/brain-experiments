"""Plasticity boost sweep. Prereg: PREREGISTRATION_plasticity.md."""
from __future__ import annotations
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/empn")
import numpy as np
from multiprocessing import Pool
from pathlib import Path

H = Path(__file__).resolve().parent
OUT = H / "out"; OUT.mkdir(exist_ok=True)
JF = OUT / "runs.jsonl"
# Tune grid on 900/901 then full seeds at winners — here we run sparse full grid as preregged
ETAS = [0.5, 1.0, 2.0, 4.0, 8.0]
TAUS = [0.85, 0.9, 0.95]
ARMS = ["EMPN+TR", "EMPN-noCFS+TR"]  # organ arms added after v2b lands
SEEDS_TUNE = [900, 901]
SEEDS_FULL = list(range(8))


def one(args):
    arm, eta_mul, tau, seed = args
    import empn_v2 as V
    t0 = time.time()
    # Prefer a hook; else monkeypatch TBU eta / TR tau
    try:
        if hasattr(V, "run_with_plasticity"):
            r = V.run_with_plasticity(arm=arm, seed=seed, eta_mul=eta_mul, tau=tau)
        else:
            # Patch module-level defaults if present
            old_tau = getattr(V, "TR_TAU", None)
            old_eta = getattr(V, "ETA_MUL", None)
            V.TR_TAU = tau
            V.ETA_MUL = eta_mul
            if hasattr(V, "job"):
                # job signature varies
                try:
                    r = V.job((arm, seed, True))  # force TR
                except TypeError:
                    r = V.job((arm, seed))
            else:
                r = {"success": float("nan"), "error": "no job entry"}
            if old_tau is not None: V.TR_TAU = old_tau
            if old_eta is not None: V.ETA_MUL = old_eta
        if not isinstance(r, dict):
            r = {"success": float(r)}
        r = dict(r)
        r.update(arm=arm, eta_mul=eta_mul, tau=tau, seed=seed, task="maze",
                 sec=time.time() - t0)
        return r
    except Exception as e:
        return dict(arm=arm, eta_mul=eta_mul, tau=tau, seed=seed, task="maze",
                    success=float("nan"), error=str(e)[:300], sec=time.time() - t0)


def main():
    # Phase 1: tune
    jobs = [(a, e, t, s) for a in ARMS for e in ETAS for t in TAUS for s in SEEDS_TUNE]
    done = set()
    if JF.exists():
        for line in open(JF):
            r = json.loads(line)
            done.add((r["arm"], r["eta_mul"], r["tau"], r["seed"]))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"plasticity tune jobs {len(jobs)}", flush=True)
    workers = int(os.environ.get("WORKERS", "4"))
    with Pool(workers) as P:
        for r in P.imap_unordered(one, jobs):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print("tune", r.get("arm"), r.get("eta_mul"), r.get("tau"), r.get("seed"), r.get("success"), flush=True)
    # pick best tau per arm by mean success on tune seeds; then eta sweep at that tau on full seeds
    import pandas as pd
    d = pd.read_json(JF, lines=True)
    d = d[d.seed.isin(SEEDS_TUNE) & d.success.notna()]
    picks = {}
    for a, g in d.groupby("arm"):
        # best (eta,tau) by mean success
        m = g.groupby(["eta_mul", "tau"]).success.mean().reset_index()
        if len(m) == 0: continue
        i = m.success.idxmax()
        picks[a] = (float(m.loc[i, "eta_mul"]), float(m.loc[i, "tau"]))
    (OUT / "picks.json").write_text(json.dumps(picks, indent=2))
    print("picks", picks, flush=True)
    jobs2 = []
    for a, (e_best, t_best) in picks.items():
        # sparse grid: all etas at best tau + best eta at all taus, full seeds
        for e in ETAS:
            for s in SEEDS_FULL:
                jobs2.append((a, e, t_best, s))
        for t in TAUS:
            if t == t_best: continue
            for s in SEEDS_FULL:
                jobs2.append((a, e_best, t, s))
    jobs2 = [j for j in jobs2 if (j[0], j[1], j[2], j[3]) not in done]
    # also skip ones we just wrote
    cur = {(r["arm"], r["eta_mul"], r["tau"], r["seed"]) for r in map(json.loads, open(JF))}
    jobs2 = [j for j in jobs2 if (j[0], j[1], j[2], j[3]) not in cur]
    print(f"plasticity full jobs {len(jobs2)}", flush=True)
    with Pool(workers) as P:
        for r in P.imap_unordered(one, jobs2):
            with open(JF, "a") as f:
                f.write(json.dumps(r) + "\n")
            print("full", r.get("arm"), r.get("eta_mul"), r.get("tau"), r.get("seed"), r.get("success"), flush=True)
    (OUT / "plasticity.done").write_text("ok\n")


if __name__ == "__main__":
    main()

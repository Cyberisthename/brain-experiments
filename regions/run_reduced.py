"""Reduced-scope regions run (deviation documented in RESULTS_regions.md).
S0-S2: full prereg (2 tune seeds, 9 grid cells, 3 tasks, 8 test seeds).
S3-S4: MC + T0 only, 1 tune seed (900), 4 test seeds (0-3). Results checkpointed per job to jobs.jsonl."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regions as R, pandas as pd
from multiprocessing import Pool
OUT = R.OUT; JF = os.path.join(OUT, "jobs.jsonl")
def plan(k):
    big = k >= 3
    return dict(tasks=["MC", "T0"] if big else ["MC", "NARMA", "T0"], tune=[900] if big else R.TUNE, test=[0, 1, 2, 3] if big else R.TEST)
def done():
    d = {}
    if os.path.exists(JF):
        for l in open(JF):
            r = json.loads(l); d[(r["step"], r["cond"], r["seed"], r["task"], r["leak"], r["rho"], r["val"])] = r
    return d
def job(a):
    r = R.job(a); r["val"] = a[-1]; return r
def runall(P, J):
    have = done(); J = [j for j in J if tuple(j) not in have]
    with open(JF, "a") as f:
        for r in P.imap_unordered(job, J, chunksize=1): f.write(json.dumps(r) + "\n"); f.flush()
if __name__ == "__main__":
    P = Pool(int(os.environ.get("WORKERS", 3)))
    for k in range(5):
        p = plan(k)
        J = sorted([(k, c, s, t, l, r, True) for c in R.CONDS for s in p["tune"] for t in p["tasks"] for l in R.LEAKS for r in R.RHOS], key=lambda j: (j[1], j[2]))
        runall(P, J)
        sw = pd.DataFrame([v for v in done().values() if v["step"] == k and v["val"]])
        m = sw.groupby(["cond", "task", "leak", "rho"]).value.mean().reset_index(); best = {}
        for (c, t), mm in m.groupby(["cond", "task"]):
            i = mm.value.idxmax() if t == "MC" else mm.value.idxmin(); best[(c, t)] = (float(mm.loc[i, "leak"]), float(mm.loc[i, "rho"]))
        J = sorted([(k, c, s, t, *best[(c, t)], False) for c in R.CONDS for t in p["tasks"] for s in p["test"]], key=lambda j: (j[1], j[2]))
        runall(P, J); print("step", k, "done", best, flush=True)

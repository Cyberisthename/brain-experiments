"""Round-3 sanity variant: per-condition tuned theta learning rate on T0 (tuning seed 100), then 8 seeds at the chosen LR."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
import round3 as R
from scipy.stats import spearmanr
OUT = os.path.join(R.OUT, "lr_sanity"); os.makedirs(OUT, exist_ok=True)
LRS = [0.001, 0.003, 0.01, 0.03, 0.1]

def train_lr(task, N, row, col, w0, s, lr):
    orig = torch.optim.Adam
    def patched(groups, **kw):
        groups[0]["lr"] = lr; return orig(groups, **kw)
    torch.optim.Adam = patched
    try: return R.train(task, N, row, col, w0, s)
    finally: torch.optim.Adam = orig

def tune(a):
    cond, lr = a; N, row, col, w0 = R.mask(cond, 100); w, l = train_lr("T0", N, row, col, w0, 100, lr)
    return dict(cond=cond, lr=lr, val=R.evaluate("T0", N, row, col, w, 100, 0.5, 1.0), loss_last=float(np.mean(l[-10:])))

def test(a):
    cond, s, lr = a; fp = os.path.join(OUT, f"{cond}_{s}.json")
    if os.path.exists(fp): return json.load(open(fp))
    N, row, col, w0 = R.mask(cond, s); w, l = train_lr("T0", N, row, col, w0, s, lr)
    vals = [R.evaluate("T0", N, row, col, w, s, le, m) for le, m in R.GRID]
    out = dict(cond=cond, seed=s, lr=lr, trained_test=vals[4], trained_gridmean=float(np.mean(vals)), trained_worst=float(max(vals)),
               rel_L2=float(np.linalg.norm(w - w0) / np.linalg.norm(w0)))
    json.dump(out, open(fp, "w")); print(out, flush=True); return out

if __name__ == "__main__":
    with Pool(4) as p:
        tf = os.path.join(OUT, "tuning.csv")
        if not os.path.exists(tf): pd.DataFrame(p.map(tune, [(c, lr) for c in ["REAL", "ER", "DEG"] for lr in LRS])).to_csv(tf, index=False)
        tu = pd.read_csv(tf); print(tu)
        best = {c: float(tu[tu.cond == c].sort_values("val").lr.iloc[0]) for c in ["REAL", "ER", "DEG"]}
        print("best", best, flush=True)
        rows = p.map(test, [(c, s, best[c]) for c in best for s in R.SEEDS])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "test.csv"), index=False)

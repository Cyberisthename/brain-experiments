"""Discrete wiring: DTM(+MPS+SKF) vs SBQA/SA/random/greedy/SET/RigL/LONG_RANDOM/PSO."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_SBQA = HERE.parent / "sbqa"
sys.path[:0] = [str(HERE), str(_SBQA)]

from mask import make_mask
from reservoir import MaskedReservoir, pick_device
from tasks import make_narma_batch, make_xor_hard_batch
from baselines import init_active, run_random, run_greedy, run_sa, run_sbqa, run_set, run_rigl
from dtm_core import score_dense_active, dtm_run, hebb_from_states
from vines import VineGraph
from util_mem import track
from skf import skf_run, pso_run
from dtm_core import score_vines

# import hebb helper used by rigl
from dtm_core import apply_vine_to_model  # noqa: F401


def make_data(task, seq, batch, rng, device):
    if task == "narma":
        u, y, m = make_narma_batch(seq, batch, rng)
    else:
        u, y, m = make_xor_hard_batch(8, seq, batch, rng)
    return u.to(device), y.to(device), m.to(device)


def run_seed(seed, task, methods, budget, n_local, seq, batch, wash, device):
    rng = np.random.default_rng(seed)
    info = make_mask(n_local=n_local)
    # point mask loader at dtm data if env unset
    os.environ.setdefault("SBQA_GRIFFA_NPZ", str(HERE / "data" / "griffa70_scale68_group_minimal.npz"))
    info = make_mask(n_local=n_local)  # reload with env
    n_in = 1 if task == "narma" else 2
    model = MaskedReservoir(info["mask"], n_in=n_in, out_dim=1, leak=0.4, rho=0.9,
                            device=device, trainable_W=False)
    with torch.no_grad():
        g = torch.Generator(device=model.device); g.manual_seed(int(seed) + 7)
        model.Win.copy_(0.5 * torch.randn(model.N, n_in, generator=g, device=model.device))
        model.b.copy_(0.05 * torch.randn(model.N, generator=g, device=model.device))
    u, y, msk = make_data(task, seq, batch, rng, model.device)
    E_budget = max(8, int(round(0.35 * info["E_mask"])))
    active0 = init_active(info["E_mask"], E_budget, rng)
    weights0 = 0.08 * rng.standard_normal(info["E_mask"])

    def score_fn(active):
        sc, _ = score_dense_active(model, active, info["edges"], u, y, msk, wash, task)
        return sc

    def hebb_fn(active):
        sc, X = score_dense_active(model, active, info["edges"], u, y, msk, wash, task)
        on = np.flatnonzero(active); off = np.flatnonzero(~active)
        # hebb for on edges
        ee = info["edges"]
        hebb_on = hebb_from_states(X, ee[on, 0], ee[on, 1]) if len(on) else np.zeros(0)
        # for off: use |mean x_i x_j| as grow proxy
        if len(off):
            hebb_off = hebb_from_states(X, ee[off, 0], ee[off, 1])
        else:
            hebb_off = np.zeros(0)
        return hebb_on, hebb_off

    rows = []
    for method in methods:
        rlocal = np.random.default_rng(seed + (abs(hash(method)) % 100000))
        with track() as mem:
            nnz_init = nnz_final = bridges = None
            if method == "dtm":
                out = dtm_run(model, info, u, y, msk, wash, task, budget, rlocal)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out["bridges"]
            elif method == "dtm_mps_warp":
                out = dtm_run(model, info, u, y, msk, wash, task, budget, rlocal, use_warp=True)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out["bridges"]
            elif method == "dtm_wave":
                out = dtm_run(model, info, u, y, msk, wash, task, budget, rlocal, use_wave=True)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out["bridges"]
            elif method == "dtm_both":
                out = dtm_run(model, info, u, y, msk, wash, task, budget, rlocal, use_warp=True, use_wave=True)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out["bridges"]
            elif method == "long_random":
                out = dtm_run(model, info, u, y, msk, wash, task, budget, rlocal, long_random=True)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out["bridges"]

            elif method == "skf":
                out = skf_run(model, info, u, y, msk, wash, task, budget, rlocal, K=4,
                              score_vines_fn=score_vines)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out.get("bridges", 0)
            elif method == "skf_k1":
                out = skf_run(model, info, u, y, msk, wash, task, budget, rlocal, K=1,
                              score_vines_fn=score_vines)
                best, hist = out["best_score"], out["history"]
                nnz_init, nnz_final, bridges = out["nnz_init"], out["nnz_final"], out.get("bridges", 0)
            elif method == "pso":
                best, hist = pso_run(score_fn, active0, budget, rlocal)
                nnz_init = nnz_final = bridges = None
            elif method == "sbqa":
                best, hist = run_sbqa(score_fn, active0, budget, rlocal)
            elif method == "sa":
                best, hist = run_sa(score_fn, active0, budget, rlocal)
            elif method == "random":
                best, hist = run_random(score_fn, active0, budget, rlocal)
            elif method == "greedy":
                best, hist = run_greedy(score_fn, active0, budget, rlocal)
            elif method == "set":
                best, hist = run_set(score_fn, active0, info["edges"], weights0, budget, rlocal)
            elif method == "rigl":
                best, hist = run_rigl(score_fn, active0, hebb_fn, budget, rlocal)
            else:
                raise KeyError(method)
        row = {
            "seed": seed, "task": task, "method": method,
            "best_score": float(best), "evals": len(hist),
            "seconds": mem["seconds"], "peak_rss_mb": mem["peak_rss_mb"],
            "N": info["N"], "E_budget": E_budget, "budget": budget,
            "nnz_init": nnz_init, "nnz_final": nnz_final, "bridges": bridges,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="narma", choices=["narma", "xor"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--methods", default=None, help="comma list override")
    args = ap.parse_args()
    os.environ.setdefault("SBQA_GRIFFA_NPZ", str(HERE / "data" / "griffa70_scale68_group_minimal.npz"))
    out_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or (HERE / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pick_device(args.device))

    if args.smoke:
        seeds = [91]
        n_local, budget, seq, batch, wash = 2, 12, 40, 4, 8
        methods = ["dtm", "dtm_mps_warp", "dtm_wave", "dtm_both", "long_random",
                   "skf", "skf_k1", "pso",
                   "sbqa", "sa", "random", "greedy", "set", "rigl"]
    else:
        seeds = list(range(5100, 5108))
        n_local, budget, seq, batch, wash = 4, 80, 56, 8, 10
        methods = ["dtm", "dtm_mps_warp", "dtm_wave", "dtm_both", "long_random",
                   "skf", "skf_k1", "pso",
                   "sbqa", "sa", "random", "greedy", "set", "rigl"]
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    jsonl = out_dir / "discrete_runs.jsonl"
    with open(jsonl, "a") as f:
        for s in seeds:
            for row in run_seed(s, args.task, methods, budget, n_local, seq, batch, wash, device):
                f.write(json.dumps(row) + "\n"); f.flush()
    print("wrote", jsonl)


if __name__ == "__main__":
    main()

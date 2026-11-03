"""Discrete wiring search: SBQA vs SA vs RANDOM vs GREEDY (+ ablations)."""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

from mask import make_mask
from reservoir import MaskedReservoir, ridge_readout, pred_ridge, pick_device
from tasks import make_narma_batch, make_xor_hard_batch, nrmse, xor_acc
from sbqa_ops import (
    PlateauMonitor, propose_edge_flip, estimate_barrier,
    wkb_accept, metropolis_accept,
)

HERE = Path(__file__).resolve().parent


def score_wiring(model, active, all_edges, u, y, mask, wash, task: str):
    idx = np.flatnonzero(active)
    model.set_topology_from_edges(idx, all_edges)
    X = model.states(u)
    # train ridge on wash:wash+tr portion — use first 60% after wash as train
    T = X.shape[0]
    tr_end = wash + max(8, int(0.6 * (T - wash)))
    B = ridge_readout(X[wash:tr_end], y[wash:tr_end], mask[wash:tr_end])
    P = pred_ridge(X[tr_end:], B)
    if task == "narma":
        return nrmse(P, y[tr_end:], mask[tr_end:])
    return 1.0 - xor_acc(P, y[tr_end:], mask[tr_end:])  # lower better unified


def init_active(E_mask, E_budget, rng):
    active = np.zeros(E_mask, dtype=bool)
    pick = rng.choice(E_mask, size=min(E_budget, E_mask), replace=False)
    active[pick] = True
    return active


def run_method(method, model, all_edges, active0, score_fn, budget, rng, temp0=1.0, accept="wkb", use_mask=True):
    active = active0.copy()
    best = active.copy()
    s = score_fn(active)
    best_s = s
    hist = [s]
    mon = PlateauMonitor(window=5, var_thresh=1e-5)
    temp = temp0
    E_mask = len(active)
    for step in range(1, budget):
        stuck = mon.push(s)
        k = 3 if stuck or method == "sbqa" else 1
        if method == "random":
            cand = init_active(E_mask, int(active.sum()), rng)
            sc = score_fn(cand)
            if sc < best_s:
                best_s, best = sc, cand.copy()
            s = sc
            hist.append(best_s)
            continue
        if method == "greedy":
            cand = propose_edge_flip(active, E_mask, k=1, rng=rng)
            sc = score_fn(cand)
            if sc < s - 1e-12:
                active, s = cand, sc
            elif stuck:
                # restart
                active = init_active(E_mask, int(active0.sum()), rng)
                s = score_fn(active)
            if s < best_s:
                best_s, best = s, active.copy()
            hist.append(best_s)
            continue
        # SA or SBQA
        cand = propose_edge_flip(active, E_mask, k=k if method == "sbqa" else 1, rng=rng)
        if method == "sbqa" and not use_mask:
            # ignore biological mask: random dense pairs (still scored via set_topology which *applies* mask)
            # Approximate "no mask" by allowing proposals over a virtual full index set then projecting:
            # flip random ON edges off and sample random mask-legal OFF — same support. Real no-mask =
            # expand proposal pool: we re-draw cand from full mask but with larger k and no plateau gate.
            cand = propose_edge_flip(active, E_mask, k=max(k, 5), rng=rng)
        sc = score_fn(cand)
        dE = sc - s
        if method == "sa":
            ok = metropolis_accept(dE, temp, rng)
            temp *= 0.97
        else:  # sbqa
            if stuck or dE > 0:
                # estimate barrier along path (costs 2 extra evals — count them against budget via hist padding)
                barrier, width = estimate_barrier(score_fn, active, cand, s, sc)
                if accept == "wkb":
                    ok = wkb_accept(barrier if dE > 0 else dE, width, rng)
                else:
                    ok = metropolis_accept(dE, max(temp, 0.05), rng)
                # account for midpoint evals
                hist.append(best_s)
                hist.append(best_s)
            else:
                ok = True
        if ok:
            active, s = cand, sc
        if s < best_s:
            best_s, best = s, active.copy()
        hist.append(best_s)
        if len(hist) >= budget:
            break
    return {"best_score": float(best_s), "history": hist[:budget], "evals": len(hist[:budget])}


def run_seed(seed, task, methods, budget, n_local, seq, batch, wash, device, ablation=False):
    rng = np.random.default_rng(seed)
    info = make_mask(n_local=n_local)
    M, edges = info["mask"], info["edges"]
    E_budget = max(8, int(round(0.35 * info["E_mask"])))
    n_in = 1 if task == "narma" else 2
    model = MaskedReservoir(M, n_in=n_in, out_dim=1, leak=0.4, rho=0.9, device=device, trainable_W=False)
    # freeze Win/b from seed
    with torch.no_grad():
        g = torch.Generator(device=model.device)
        g.manual_seed(int(seed) + 7)
        model.Win.copy_(0.5 * torch.randn(model.N, n_in, generator=g, device=model.device))
        model.b.copy_(0.05 * torch.randn(model.N, generator=g, device=model.device))
    if task == "narma":
        u, y, msk = make_narma_batch(seq, batch, rng)
    else:
        u, y, msk = make_xor_hard_batch(8, seq, batch, rng)
    u, y, msk = u.to(model.device), y.to(model.device), msk.to(model.device)

    def score_fn(active):
        return score_wiring(model, active, edges, u, y, msk, wash, task)

    active0 = init_active(info["E_mask"], E_budget, rng)
    rows = []
    for method in methods:
        accept = "wkb"
        use_mask = True
        label = method
        if ablation:
            if method == "sbqa_metropolis":
                method_k, accept, label = "sbqa", "metropolis", "sbqa_metropolis"
            elif method == "sbqa_nomask":
                method_k, use_mask, label = "sbqa", False, "sbqa_nomask"
            else:
                method_k = method
        else:
            method_k = method
        t0 = time.time()
        # re-init active per method from same active0 for fairness
        out = run_method(method_k, model, edges, active0, score_fn, budget, np.random.default_rng(seed + hash(label) % 10000),
                         accept=accept, use_mask=use_mask)
        rows.append({
            "seed": seed, "task": task, "method": label,
            "best_score": out["best_score"], "evals": out["evals"],
            "seconds": time.time() - t0, "N": info["N"], "E_budget": E_budget,
            "E_mask": info["E_mask"], "budget": budget,
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--task", default="narma", choices=["narma", "xor"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or (HERE / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pick_device(args.device))

    if args.smoke:
        seeds = [91]
        n_local, budget, seq, batch, wash = 2, 12, 40, 4, 8
    else:
        seeds = list(range(4100, 4108)) if args.seeds is None else [int(x) for x in args.seeds.split(",")]
        n_local, budget, seq, batch, wash = 4, 80, 56, 8, 10

    if args.ablation:
        methods = ["sbqa", "sbqa_metropolis", "sbqa_nomask"]
        jsonl = out_dir / "ablation_runs.jsonl"
    else:
        methods = ["sbqa", "sa", "random", "greedy"]
        jsonl = out_dir / "discrete_runs.jsonl"

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]

    with open(jsonl, "a") as f:
        for s in seeds:
            for row in run_seed(s, args.task, methods, budget, n_local, seq, batch, wash, device, ablation=args.ablation):
                f.write(json.dumps(row) + "\n")
                f.flush()
    print("wrote", jsonl)


if __name__ == "__main__":
    main()

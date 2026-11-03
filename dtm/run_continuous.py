"""Continuous arm: DTM nutrient weight updates vs Adam / SBQA-tunnel / SET / SGLD."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "sbqa")]

from mask import make_mask
from reservoir import MaskedReservoir, pick_device
from tasks import make_narma_batch, make_xor_hard_batch, nrmse, xor_acc
from vines import VineGraph, hebb_from_states, rescale_rho_sparse
from util_mem import track

os.environ.setdefault("SBQA_GRIFFA_NPZ", str(HERE / "data" / "griffa70_scale68_group_minimal.npz"))


def batch_for(task, seq, batch, rng):
    return make_narma_batch(seq, batch, rng) if task == "narma" else make_xor_hard_batch(8, seq, batch, rng)


def eval_model(model, task, seq, batch, seed_eval, device):
    rng = np.random.default_rng(seed_eval + 999)
    u, y, m = batch_for(task, seq, batch, rng)
    u, y, m = u.to(device), y.to(device), m.to(device)
    with torch.no_grad():
        p = model.forward_sequence(u)
        if task == "narma":
            return nrmse(p, y, m)
        return 1.0 - xor_acc(p, y, m)


def train_adam_family(method, model, task, epochs, steps, seq, batch, lr, seed, device, noise=0.0):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    evals = 0
    for ep in range(epochs):
        for st in range(steps):
            u, y, m = batch_for(task, seq, batch, rng)
            u, y, m = u.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            p = model.forward_sequence(u)
            diff = (p - y).squeeze(-1)
            loss = ((diff ** 2) * m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            if method == "sgld":
                with torch.no_grad():
                    for p_ in model.parameters():
                        if p_.grad is not None:
                            p_.grad.add_(torch.randn_like(p_.grad) * noise)
            opt.step()
            with torch.no_grad():
                model.W_raw.mul_(model.mask)
            evals += 1
        sc = eval_model(model, task, seq, batch * 2, seed + ep * 17, device)
        evals += 1
        if sc < best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if method == "set" and ep % 2 == 1:
            # magnitude prune + random regrow under mask
            with torch.no_grad():
                W = model.W_raw.data * model.mask
                flat = W.abs().flatten()
                nnz = int((model.mask > 0).sum().item())
                k = max(1, nnz // 20)
                # zero smallest nonzero
                vals = W[model.mask.bool()].abs()
                if vals.numel() > k:
                    thr = torch.kthvalue(vals, k).values
                    drop = (W.abs() <= thr) & model.mask.bool()
                    model.W_raw.data[drop] = 0
                    # regrow random
                    empty = model.mask.bool() & (model.W_raw.data.abs() < 1e-12)
                    idx = empty.nonzero(as_tuple=False)
                    if len(idx) > 0:
                        pick = idx[torch.randint(0, len(idx), (min(k, len(idx)),), device=device)]
                        model.W_raw.data[pick[:, 0], pick[:, 1]] = 0.08 * torch.randn(len(pick), device=device)
                model.rescale_rho_()
        if method == "sbqa" and ep > 0 and ep % 3 == 0:
            with torch.no_grad():
                jump = 0.2 * torch.randn_like(model.W_raw) * model.mask
                before = model.W_raw.data.clone()
                model.W_raw.add_(jump); model.rescale_rho_()
            sc2 = eval_model(model, task, seq, batch, seed + 1000 + ep, device)
            evals += 1
            if sc2 > sc:  # worse — revert (simple accept-if-better; WKB skipped for speed)
                with torch.no_grad():
                    model.W_raw.copy_(before)
            elif sc2 < best_score:
                best_score = sc2
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final = eval_model(model, task, seq, batch * 2, seed + 4242, device)
    return float(min(best_score, final)), evals


def train_dtm_continuous(model, info, task, epochs, steps, seq, batch, seed, device):
    rng = np.random.default_rng(seed)
    E_budget = max(8, int(round(0.35 * info["E_mask"])))
    vines = VineGraph.seed(info["N"], info["edges"], max(8, E_budget // 5), rng)
    best = float("inf"); evals = 0; nnz_trace = []
    baseline = None
    for ep in range(epochs):
        for st in range(steps):
            u, y, m = batch_for(task, seq, batch, rng)
            u, y, m = u.to(device), y.to(device), m.to(device)
            with torch.no_grad():
                W = vines.to_dense_W(device=device)
                W = rescale_rho_sparse(W)
                model.W_raw.copy_(W)
            # forward states
            X = model.states(u)
            # ridge energy
            from reservoir import ridge_readout, pred_ridge
            T = X.shape[0]; wash = 8
            tr_end = wash + max(8, int(0.6 * (T - wash)))
            B = ridge_readout(X[wash:tr_end], y[wash:tr_end], m[wash:tr_end])
            P = pred_ridge(X[tr_end:], B)
            if task == "narma":
                sc = nrmse(P, y[tr_end:], m[tr_end:])
            else:
                sc = 1.0 - xor_acc(P, y[tr_end:], m[tr_end:])
            evals += 1
            if baseline is None:
                baseline = sc
            reward = float(baseline - sc)
            baseline = 0.9 * baseline + 0.1 * sc
            hebb = hebb_from_states(X, vines.post, vines.pre)
            vines.update_nutrients(hebb, reward)
            vines.sprout(rng, k=4, soft_cap=int(1.2 * E_budget))
            vines.prune_nonpositive()
            vines.enforce_budget(E_budget, rng)
            best = min(best, sc)
            nnz_trace.append(vines.nnz)
        sc = eval_model(model, task, seq, batch * 2, seed + ep * 17, device)
        evals += 1
        best = min(best, sc)
    return float(best), evals, nnz_trace


def run_seed(seed, task, methods, n_local, epochs, steps, seq, batch, lr, device):
    info = make_mask(n_local=n_local)
    n_in = 1 if task == "narma" else 2
    rows = []
    for method in methods:
        torch.manual_seed(seed + 13)
        model = MaskedReservoir(info["mask"], n_in=n_in, out_dim=1, leak=0.4, rho=0.9,
                                device=device, trainable_W=True)
        with track() as mem:
            nnz_i = nnz_f = None
            if method == "dtm":
                best, evals, nz = train_dtm_continuous(model, info, task, epochs, steps, seq, batch, seed, device)
                nnz_i, nnz_f = (nz[0], nz[-1]) if nz else (None, None)
            else:
                noise = 1e-3 if method == "sgld" else 0.0
                best, evals = train_adam_family(method, model, task, epochs, steps, seq, batch, lr, seed, device, noise)
        row = {
            "seed": seed, "task": task, "method": method, "best_score": best,
            "evals": evals, "seconds": mem["seconds"], "peak_rss_mb": mem["peak_rss_mb"],
            "N": info["N"], "epochs": epochs, "nnz_init": nnz_i, "nnz_final": nnz_f,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--task", default="narma")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or (HERE / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pick_device(args.device))
    methods = ["dtm", "adam", "sbqa", "set", "sgld"]
    if args.smoke:
        seeds = [91]; n_local, epochs, steps, seq, batch, lr = 2, 4, 3, 24, 4, 5e-3
    else:
        seeds = list(range(5200, 5208)); n_local, epochs, steps, seq, batch, lr = 4, 40, 8, 32, 16, 5e-3
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
    jsonl = out_dir / "continuous_runs.jsonl"
    with open(jsonl, "a") as f:
        for s in seeds:
            for row in run_seed(s, args.task, methods, n_local, epochs, steps, seq, batch, lr, device):
                f.write(json.dumps(row) + "\n"); f.flush()
    print("wrote", jsonl)


if __name__ == "__main__":
    main()

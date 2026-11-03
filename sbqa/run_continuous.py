"""Continuous weight training: SBQA (Adam+tunnel) vs Adam vs restarts vs SGLD."""
from __future__ import annotations

import argparse, json, os, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from mask import make_mask
from reservoir import MaskedReservoir, pick_device
from tasks import make_narma_batch, make_xor_hard_batch, nrmse, xor_acc
from sbqa_ops import PlateauMonitor, wkb_accept

HERE = Path(__file__).resolve().parent


def batch_for(task, seq, batch, rng):
    if task == "narma":
        return make_narma_batch(seq, batch, rng)
    return make_xor_hard_batch(8, seq, batch, rng)


def eval_model(model, task, seq, batch, seed_eval, device):
    rng = np.random.default_rng(seed_eval + 999)
    u, y, m = batch_for(task, seq, batch, rng)
    u, y, m = u.to(device), y.to(device), m.to(device)
    with torch.no_grad():
        p = model.forward_sequence(u)
        if task == "narma":
            return nrmse(p, y, m)
        return 1.0 - xor_acc(p, y, m)


def train_loop(method, model, task, epochs, steps_per, seq, batch, lr, seed, device, noise_std=0.0):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    mon = PlateauMonitor(window=4, var_thresh=1e-6)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_score = float("inf")
    evals = 0
    plateau_hits = 0
    for ep in range(epochs):
        model.train()
        losses = []
        grad_norms = []
        for st in range(steps_per):
            u, y, m = batch_for(task, seq, batch, rng)
            u, y, m = u.to(device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            p = model.forward_sequence(u)
            # masked MSE
            diff = (p - y).squeeze(-1)
            loss = ((diff ** 2) * m).sum() / m.sum().clamp_min(1.0)
            if task != "narma":
                # also nudge toward 0/1
                loss = loss + 0.1 * F.binary_cross_entropy_with_logits(p.squeeze(-1), y.squeeze(-1), reduction="none").mul(m).sum() / m.sum().clamp_min(1.0)
            loss.backward()
            if method == "sgld" and noise_std > 0:
                with torch.no_grad():
                    for p_ in model.parameters():
                        if p_.grad is not None:
                            p_.grad.add_(noise_std, alpha=1.0) if False else p_.grad.add_(torch.randn_like(p_.grad) * noise_std)
            gn = 0.0
            for p_ in model.parameters():
                if p_.grad is not None:
                    gn += float(p_.grad.detach().norm().cpu())
            grad_norms.append(gn)
            opt.step()
            # keep W under mask
            with torch.no_grad():
                model.W_raw.mul_(model.mask)
            losses.append(float(loss.detach().cpu()))
            evals += 1
        # held-out-ish eval on fresh batch
        sc = eval_model(model, task, seq, batch * 2, seed + ep * 17, device)
        evals += 1
        if sc < best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        stuck = mon.push(float(np.mean(losses[-4:]))) if losses else False
        # also use grad norm variance
        if len(grad_norms) >= 3 and float(np.var(grad_norms[-3:])) < 1e-8:
            stuck = True

        if method == "sbqa" and stuck:
            plateau_hits += 1
            # non-local jump on W_raw under mask
            with torch.no_grad():
                W_before = model.W_raw.data.clone()
                jump = 0.25 * torch.randn_like(model.W_raw) * model.mask
                model.W_raw.add_(jump)
                model.rescale_rho_()
            sc2 = eval_model(model, task, seq, batch, seed + 1000 + ep, device)
            evals += 1
            dE = sc2 - sc
            # width ~ jump L2
            width = float(jump.norm().cpu()) + 1.0
            barrier = max(0.0, dE)
            if wkb_accept(barrier, width * 0.1, rng):
                if sc2 < best_score:
                    best_score = sc2
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                with torch.no_grad():
                    model.W_raw.copy_(W_before)

        if method == "restarts" and stuck:
            plateau_hits += 1
            # reload best + noise
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            with torch.no_grad():
                model.W_raw.add_(0.05 * torch.randn_like(model.W_raw) * model.mask)
                model.rescale_rho_()
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final = eval_model(model, task, seq, batch * 2, seed + 4242, device)
    return {"best_score": float(min(best_score, final)), "final_score": float(final),
            "evals": evals, "plateau_hits": plateau_hits}


def run_seed(seed, task, methods, n_local, epochs, steps, seq, batch, lr, device):
    info = make_mask(n_local=n_local)
    n_in = 1 if task == "narma" else 2
    rows = []
    for method in methods:
        torch.manual_seed(seed + 13)
        np.random.seed(seed + 13)
        model = MaskedReservoir(info["mask"], n_in=n_in, out_dim=1, leak=0.4, rho=0.9,
                                device=device, trainable_W=True)
        # map method names
        m = method
        noise = 0.0
        if method == "adam":
            mkey = "adam"
        elif method == "restarts":
            mkey = "restarts"
        elif method == "sgld":
            mkey = "sgld"
            noise = 1e-3
        else:
            mkey = "sbqa"
        t0 = time.time()
        out = train_loop(mkey, model, task, epochs, steps, seq, batch, lr, seed, device, noise_std=noise)
        row = {
            "seed": seed, "task": task, "method": method,
            "best_score": out["best_score"], "final_score": out["final_score"],
            "evals": out["evals"], "plateau_hits": out["plateau_hits"],
            "seconds": time.time() - t0, "N": info["N"],
            "epochs": epochs, "steps_per_epoch": steps,
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
    args = ap.parse_args()
    out_dir = Path(args.out_dir or os.environ.get("OUT_DIR") or (HERE / "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = str(pick_device(args.device))
    methods = ["sbqa", "adam", "restarts", "sgld"]

    if args.smoke:
        seeds = [91]
        n_local, epochs, steps, seq, batch, lr = 2, 4, 3, 24, 4, 5e-3
    else:
        seeds = list(range(4200, 4208))
        n_local, epochs, steps, seq, batch, lr = 4, 40, 8, 32, 16, 5e-3
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]

    jsonl = out_dir / "continuous_runs.jsonl"
    with open(jsonl, "a") as f:
        for s in seeds:
            for row in run_seed(s, args.task, methods, n_local, epochs, steps, seq, batch, lr, device):
                f.write(json.dumps(row) + "\n")
                f.flush()
    print("wrote", jsonl)


if __name__ == "__main__":
    main()

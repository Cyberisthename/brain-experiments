"""DTM / Neuro-Vines training step + MPS variants + LONG_RANDOM control."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_SBQA = HERE.parent / "sbqa"
if str(_SBQA) not in sys.path:
    sys.path.insert(0, str(_SBQA))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mask import make_mask  # noqa: E402
from reservoir import MaskedReservoir, ridge_readout, pred_ridge, pick_device  # noqa: E402
from tasks import make_narma_batch, make_xor_hard_batch, nrmse, xor_acc  # noqa: E402
from vines import VineGraph, hebb_from_states, rescale_rho_sparse  # noqa: E402
from mps import (  # noqa: E402
    MPSState, init_coords_from_griffa, spectral_embed,
    default_R_max, parcel_scent_from_states,
)


def energy(pred, y, mask, task: str) -> float:
    if task == "narma":
        return nrmse(pred, y, mask)
    return 1.0 - xor_acc(pred, y, mask)


def score_dense_active(model, active, all_edges, u, y, msk, wash, task):
    idx = np.flatnonzero(active)
    model.set_topology_from_edges(idx, all_edges)
    X = model.states(u)
    T = X.shape[0]
    tr_end = wash + max(8, int(0.6 * (T - wash)))
    B = ridge_readout(X[wash:tr_end], y[wash:tr_end], msk[wash:tr_end])
    P = pred_ridge(X[tr_end:], B)
    return energy(P, y[tr_end:], msk[tr_end:], task), X


def apply_vine_to_model(model, vines: VineGraph, rho: float = 0.9):
    with torch.no_grad():
        W = vines.to_dense_W(device=model.device)
        W = rescale_rho_sparse(W, rho=rho)
        model.W_raw.copy_(W)
        # mask already registered; keep Win/b


def score_vines(model, vines, u, y, msk, wash, task):
    apply_vine_to_model(model, vines)
    X = model.states(u)
    T = X.shape[0]
    tr_end = wash + max(8, int(0.6 * (T - wash)))
    B = ridge_readout(X[wash:tr_end], y[wash:tr_end], msk[wash:tr_end])
    P = pred_ridge(X[tr_end:], B)
    return energy(P, y[tr_end:], msk[tr_end:], task), X


def build_mps(info, seed, use_warp, use_wave) -> MPSState:
    sc = info["sc"]
    n_local = info["n_local"]
    # try hemis from npz
    hemis = None
    try:
        from mask import _npz_path
        z = np.load(_npz_path(), allow_pickle=True)
        if "hemis" in z.files:
            hemis = z["hemis"]
    except Exception:
        pass
    Z = init_coords_from_griffa(sc, n_local, hemis=hemis, seed=seed)
    R_max = default_R_max(Z, info["edges"])
    return MPSState(Z=Z.copy(), Z0=Z.copy(), R_max=R_max, n_local=n_local,
                    use_warp=use_warp, use_wave=use_wave)


def dtm_run(
    model,
    info,
    u, y, msk,
    wash: int,
    task: str,
    budget: int,
    rng: np.random.Generator,
    use_warp: bool = False,
    use_wave: bool = False,
    long_random: bool = False,
    eta: float = 0.5,
    famine: float = 0.02,
) -> Dict:
    E_mask = info["E_mask"]
    E_budget = max(8, int(round(0.35 * E_mask)))
    E_seed = max(8, E_budget // 5)
    vines = VineGraph.seed(info["N"], info["edges"], E_seed, rng)
    mps = build_mps(info, seed=int(rng.integers(1e9)), use_warp=use_warp, use_wave=use_wave)
    soft_cap = int(1.2 * E_budget)

    # LONG_RANDOM: after seed, add fixed number of random long mask edges once
    long_added = 0
    if long_random:
        # estimate bridge count as ~n_parcels/4
        n_long = max(4, info["N"] // info["n_local"] // 4)
        active = vines.active_set()
        # prefer currently far pairs under init coords
        Z = mps.Z0
        dists = []
        for i, (p, q) in enumerate(info["edges"]):
            if (int(p), int(q)) in active:
                continue
            dists.append((float(np.linalg.norm(Z[p] - Z[q])), i))
        dists.sort(reverse=True)
        for _, i in dists[:n_long]:
            p, q = map(int, info["edges"][i])
            if (p, q) in vines.active_set():
                continue
            vines._append(p, q, 1.0, 0.08 * rng.standard_normal())
            long_added += 1
        vines.enforce_budget(E_budget, rng)

    s, X = score_vines(model, vines, u, y, msk, wash, task)
    baseline = s
    best_s = s
    hist = [s]
    nnz_trace = [vines.nnz]
    bridges = 0

    for step in range(1, budget):
        # phase / sprout gains
        if use_wave or use_warp:
            mps.step_time()
        gain = mps.sprout_gain() if (use_wave or use_warp) else np.ones(info["N"])
        # mean gain on random node → k_sprout
        k = 4
        if use_wave:
            node = int(rng.integers(0, info["N"]))
            k = int(max(1, round(4 * float(gain[node]))))
            k = min(k, 40)  # cap 10x * 4
        vines.sprout(rng, k=k, n0=0.5, soft_cap=soft_cap)

        # distance filter: drop brand-new edges that violate radius unless mask 1-hop
        # (sprout already only uses mask neighbors; MPS radius additionally allows
        # warp-bridged pairs — handle via explicit bridge attempt below)

        if use_warp:
            # scent from last states
            Xs = X.mean(1).detach().cpu().numpy()  # (T,N)
            scent = parcel_scent_from_states(Xs, info["n_local"])
            before = set(mps.warp_pairs)
            mps.maybe_warp(scent, tau=0.35, lam=0.45)
            # try add bridge edges between warped parcels
            if len(mps.warp_pairs) > len(before):
                i, j = mps.warp_pairs[-1]
                nl = info["n_local"]
                # sample a few cross edges in mask between parcels
                for _ in range(4):
                    a = int(rng.integers(i * nl, (i + 1) * nl))
                    b = int(rng.integers(j * nl, (j + 1) * nl))
                    for post, pre in ((a, b), (b, a)):
                        if info["mask"][post, pre] and (post, pre) not in vines.active_set():
                            if vines.nnz < soft_cap:
                                vines._append(post, pre, 1.0, 0.08 * rng.standard_normal())
                                bridges += 1
            mps.relax(mu=0.12)

        s, X = score_vines(model, vines, u, y, msk, wash, task)
        reward = float(baseline - s)
        baseline = 0.9 * baseline + 0.1 * s

        hebb = hebb_from_states(X, vines.post, vines.pre)
        # wave-modulated nutrient / famine
        if use_wave and vines.nnz:
            fg = mps.famine_gain()
            # edge famine gain = mean of endpoints
            edge_fg = 0.5 * (fg[vines.post] + fg[vines.pre])
            sg = mps.sprout_gain()
            edge_sg = 0.5 * (sg[vines.post] + sg[vines.pre])
            vines.nutrient += eta * hebb * reward * edge_sg - famine * edge_fg
            vines.weight += 0.01 * reward * hebb * np.sign(vines.weight + 1e-12)
        else:
            vines.update_nutrients(hebb, reward, eta=eta, famine=famine)

        vines.prune_nonpositive()
        vines.enforce_budget(E_budget, rng)

        best_s = min(best_s, s)
        hist.append(best_s)
        nnz_trace.append(vines.nnz)

    return {
        "best_score": float(best_s),
        "history": hist,
        "nnz_trace": nnz_trace,
        "nnz_init": int(nnz_trace[0]),
        "nnz_final": int(nnz_trace[-1]),
        "bridges": int(bridges + long_added),
        "E_budget": E_budget,
        "evals": len(hist),
    }

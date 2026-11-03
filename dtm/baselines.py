"""Discrete baselines: SA, random, greedy, SET, RigL, thin SBQA wrapper."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_SBQA = Path(__file__).resolve().parents[1] / "sbqa"
if str(_SBQA) not in sys.path:
    sys.path.insert(0, str(_SBQA))

from sbqa_ops import propose_edge_flip, metropolis_accept, PlateauMonitor, estimate_barrier, wkb_accept  # noqa: E402


def init_active(E_mask: int, E_budget: int, rng: np.random.Generator) -> np.ndarray:
    active = np.zeros(E_mask, dtype=bool)
    pick = rng.choice(E_mask, size=min(E_budget, E_mask), replace=False)
    active[pick] = True
    return active


def run_random(score_fn, active0, budget, rng):
    best = active0.copy(); best_s = score_fn(active0); hist = [best_s]
    for _ in range(1, budget):
        cand = init_active(len(active0), int(active0.sum()), rng)
        sc = score_fn(cand)
        if sc < best_s:
            best_s, best = sc, cand.copy()
        hist.append(best_s)
    return best_s, hist


def run_greedy(score_fn, active0, budget, rng):
    active = active0.copy(); s = score_fn(active); best_s, best = s, active.copy(); hist = [s]
    mon = PlateauMonitor(window=5, var_thresh=1e-5)
    for _ in range(1, budget):
        stuck = mon.push(s)
        cand = propose_edge_flip(active, len(active), 1, rng)
        sc = score_fn(cand)
        if sc < s - 1e-12:
            active, s = cand, sc
        elif stuck:
            active = init_active(len(active0), int(active0.sum()), rng)
            s = score_fn(active)
        if s < best_s:
            best_s, best = s, active.copy()
        hist.append(best_s)
    return best_s, hist


def run_sa(score_fn, active0, budget, rng, temp0=1.0):
    active = active0.copy(); s = score_fn(active); best_s = s; hist = [s]; temp = temp0
    for _ in range(1, budget):
        cand = propose_edge_flip(active, len(active), 1, rng)
        sc = score_fn(cand)
        if metropolis_accept(sc - s, temp, rng):
            active, s = cand, sc
        temp *= 0.97
        best_s = min(best_s, s)
        hist.append(best_s)
    return best_s, hist


def run_sbqa(score_fn, active0, budget, rng):
    active = active0.copy(); s = score_fn(active); best_s = s; hist = [s]
    mon = PlateauMonitor(window=5, var_thresh=1e-5)
    for step in range(1, budget):
        stuck = mon.push(s)
        k = 3 if stuck else 1
        cand = propose_edge_flip(active, len(active), k, rng)
        sc = score_fn(cand)
        dE = sc - s
        if stuck or dE > 0:
            barrier, width = estimate_barrier(score_fn, active, cand, s, sc)
            ok = wkb_accept(barrier if dE > 0 else dE, width, rng)
            hist.append(best_s); hist.append(best_s)  # midpoint cost
        else:
            ok = True
        if ok:
            active, s = cand, sc
        best_s = min(best_s, s)
        hist.append(best_s)
        if len(hist) >= budget:
            break
    return best_s, hist[:budget]


def run_set(score_fn, active0, all_edges, weights0, budget, rng, drop_frac=0.2):
    """SET: prune lowest-|w|, random regrow in mask, keep budget."""
    active = active0.copy()
    w = weights0.copy()
    s = score_fn(active); best_s = s; hist = [s]
    E_budget = int(active.sum())
    for _ in range(1, budget):
        on = np.flatnonzero(active)
        off = np.flatnonzero(~active)
        if len(on) == 0 or len(off) == 0:
            hist.append(best_s); continue
        k = max(1, int(round(drop_frac * E_budget)))
        k = min(k, len(on), len(off))
        # prune smallest |w|
        mag = np.abs(w[on])
        drop = on[np.argsort(mag)[:k]]
        add = rng.choice(off, size=k, replace=False)
        active[drop] = False
        active[add] = True
        w[drop] = 0.0
        w[add] = 0.08 * rng.standard_normal(k)
        sc = score_fn(active)
        best_s = min(best_s, sc)
        hist.append(best_s)
        s = sc
    return best_s, hist


def run_rigl(score_fn, active0, hebb_fn, budget, rng, drop_frac=0.2):
    """RigL-style: prune low |w|/hebb, regrow high hebb candidates among OFF edges."""
    active = active0.copy()
    s = score_fn(active); best_s = s; hist = [s]
    E_budget = int(active.sum())
    for _ in range(1, budget):
        on = np.flatnonzero(active)
        off = np.flatnonzero(~active)
        if len(on) == 0 or len(off) == 0:
            hist.append(best_s); continue
        k = max(1, int(round(drop_frac * E_budget)))
        k = min(k, len(on), len(off))
        hebb_on, hebb_off = hebb_fn(active)  # arrays aligned to on/off
        drop = on[np.argsort(hebb_on)[:k]]
        add = off[np.argsort(-hebb_off)[:k]]
        active[drop] = False
        active[add] = True
        sc = score_fn(active)
        best_s = min(best_s, sc)
        hist.append(best_s)
    return best_s, hist

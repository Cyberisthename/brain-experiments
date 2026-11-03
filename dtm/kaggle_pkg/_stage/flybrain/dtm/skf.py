"""Spatial Kinetic Folding: 4D positions/velocities, clouds, friction pruning, PSO control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from vines import VineGraph, hebb_from_states, rescale_rho_sparse
from mps import init_coords_from_griffa, default_R_max


@dataclass
class CloudNode:
    """K particles for one neuron."""
    z: np.ndarray   # (K, 4)
    v: np.ndarray   # (K, 4)
    score: np.ndarray  # (K,) local score proxy (higher better)


@dataclass
class SKFState:
    N: int
    K: int
    clouds: List[CloudNode]
    R_max: float
    gamma: float = 0.15          # friction on velocity
    dt: float = 0.2
    freeze_every: int = 4
    t: int = 0
    edge_var: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    frozen_edges: set = field(default_factory=set)
    var_tau: float = 0.05

    @staticmethod
    def create(N: int, Z3: np.ndarray, R_max: float, K: int, rng: np.random.Generator) -> "SKFState":
        clouds = []
        for i in range(N):
            z0 = np.zeros((K, 4), dtype=np.float64)
            z0[:, :3] = Z3[i]
            z0[:, 3] = rng.normal(0, 0.1, size=K)
            if K > 1:
                z0 += 0.05 * rng.standard_normal((K, 4))
            v = 0.01 * rng.standard_normal((K, 4))
            clouds.append(CloudNode(z=z0, v=v, score=np.zeros(K)))
        return SKFState(N=N, K=K, clouds=clouds, R_max=R_max)

    def positions(self) -> np.ndarray:
        """Representative pos = particle 0 (after freeze = best). (N,4)"""
        return np.stack([c.z[0] for c in self.clouds], 0)

    def step_dynamics(self, affinity: np.ndarray, rng: np.random.Generator):
        """affinity (N,N) symmetric nonnegative. Update all particles."""
        pos = self.positions()  # use current reps for force field
        # soft forces from affinity to representatives
        # F_i = sum_j a_ij (z_j - z_i) - soft repulsion
        for i, c in enumerate(self.clouds):
            for k in range(self.K):
                zi = c.z[k]
                F = np.zeros(4)
                # sample subset of partners for speed
                partners = rng.choice(self.N, size=min(32, self.N), replace=False)
                for j in partners:
                    if j == i:
                        continue
                    a = float(affinity[i, j])
                    if a < 1e-6:
                        continue
                    delta = pos[j] - zi
                    dist = np.linalg.norm(delta) + 1e-8
                    F += a * delta
                    if dist < 0.05:
                        F -= 0.5 * delta / (dist ** 2)
                c.v[k] = (1.0 - self.gamma) * c.v[k] + self.dt * F
                c.z[k] = c.z[k] + self.dt * c.v[k]

    def phase_freeze(self):
        """Snap each node to its best-scoring particle; prune others by copying best into slot 0 and re-seeding weak ones lightly."""
        for c in self.clouds:
            k_best = int(np.argmax(c.score))
            # also prefer higher |v| killed — low velocity & low score die
            # score already encodes that if we subtract |v| penalty when updating
            best_z = c.z[k_best].copy()
            best_v = c.v[k_best].copy()
            c.z[:] = best_z + 0.01 * np.random.randn(*c.z.shape) * (self.K > 1)
            c.z[0] = best_z
            c.v[:] = 0.0
            c.v[0] = best_v * 0.5
            c.score[:] = 0.0
            c.score[0] = 1.0

    def update_particle_scores(self, hebb_node: np.ndarray, reward: float):
        """hebb_node (N,) mean coactivity; reward>0 good. Add |v| penalty for pruning low-score high-speed noise."""
        for i, c in enumerate(self.clouds):
            base = float(hebb_node[i]) * reward
            speed = np.linalg.norm(c.v, axis=1)
            c.score += base - 0.01 * speed

    def form_edges_from_proximity(self, vines: VineGraph, mask: np.ndarray, rng: np.random.Generator,
                                  soft_cap: int, n0: float = 0.8) -> int:
        pos = self.positions()
        added = 0
        # sample random node pairs near each other
        for _ in range(min(64, self.N)):
            i = int(rng.integers(0, self.N))
            # check a few random j
            for j in rng.choice(self.N, size=min(8, self.N), replace=False):
                j = int(j)
                if i == j:
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) > self.R_max:
                    continue
                for post, pre in ((i, j), (j, i)):
                    if not mask[post, pre]:
                        continue
                    if (post, pre) in vines.active_set():
                        continue
                    if vines.nnz >= soft_cap:
                        return added
                    vines._append(post, pre, n0, 0.08 * rng.standard_normal())
                    added += 1
        return added

    def friction_prune_update(self, edge_key: Tuple[int, int], contrib: float):
        lst = self.edge_var.setdefault(edge_key, [])
        lst.append(float(contrib))
        if len(lst) > 8:
            lst.pop(0)
        if len(lst) >= 4:
            if float(np.var(lst)) > self.var_tau:
                self.frozen_edges.add(edge_key)

    def apply_frozen_to_nutrients(self, vines: VineGraph):
        """Zero nutrient delta later — mark frozen by boosting nutrient slightly and skipping in caller."""
        pass


def node_hebb(states: torch.Tensor) -> np.ndarray:
    """(T,B,N) -> (N,) mean |x| activity."""
    X = states.mean(1).abs().mean(0)  # (N,)
    return X.detach().cpu().numpy()


def affinity_from_states(states: torch.Tensor) -> np.ndarray:
    """(T,B,N) -> (N,N) abs corr (clamped)."""
    X = states.mean(1).detach().cpu().numpy()  # (T,N)
    X = X - X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + 1e-8
    Xn = X / std
    C = (Xn.T @ Xn) / max(X.shape[0] - 1, 1)
    np.fill_diagonal(C, 0.0)
    C = np.abs(C)
    C = np.maximum(C, 0.0)
    return C


def skf_run(model, info, u, y, msk, wash, task, budget, rng, K: int = 4,
            score_vines_fn=None, eta: float = 0.5, famine: float = 0.02) -> dict:
    """Full SKF loop. score_vines_fn(model, vines, ...) -> (energy, X)."""
    assert score_vines_fn is not None
    E_budget = max(8, int(round(0.35 * info["E_mask"])))
    E_seed = max(8, E_budget // 5)
    vines = VineGraph.seed(info["N"], info["edges"], E_seed, rng)
    # coords from Griffa 3D
    hemis = None
    try:
        z = np.load(__import__("mask")._npz_path(), allow_pickle=True)
        if "hemis" in z.files:
            hemis = z["hemis"]
    except Exception:
        pass
    Z3 = init_coords_from_griffa(info["sc"], info["n_local"], hemis=hemis, seed=int(rng.integers(1e9)))
    R_max = default_R_max(Z3, info["edges"]) * 1.1
    skf = SKFState.create(info["N"], Z3, R_max, K=K, rng=rng)
    soft_cap = int(1.2 * E_budget)

    s, X = score_vines_fn(model, vines, u, y, msk, wash, task)
    baseline = s
    best_s = s
    hist = [s]
    nnz_trace = [vines.nnz]
    freezes = 0

    for step in range(1, budget):
        skf.t += 1
        aff = affinity_from_states(X)
        # reward modulates affinity
        reward_proxy = max(0.0, float(baseline - s))
        aff = aff * (0.5 + reward_proxy)
        skf.step_dynamics(aff, rng)
        skf.form_edges_from_proximity(vines, info["mask"], rng, soft_cap)

        s, X = score_vines_fn(model, vines, u, y, msk, wash, task)
        reward = float(baseline - s)
        baseline = 0.9 * baseline + 0.1 * s

        hebb = hebb_from_states(X, vines.post, vines.pre)
        # friction: track variance of hebb*reward per edge
        for ei in range(vines.nnz):
            key = (int(vines.post[ei]), int(vines.pre[ei]))
            skf.friction_prune_update(key, float(hebb[ei] * reward))

        # nutrient update; skip frozen edges (cool — no nutrient change, weight locked)
        frozen_mask = np.array([
            (int(vines.post[i]), int(vines.pre[i])) in skf.frozen_edges
            for i in range(vines.nnz)
        ], dtype=bool) if vines.nnz else np.zeros(0, dtype=bool)
        if vines.nnz:
            delta = eta * hebb * reward - famine
            delta = np.where(frozen_mask, 0.0, delta)
            vines.nutrient += delta
            wdelta = 0.01 * reward * hebb * np.sign(vines.weight + 1e-12)
            wdelta = np.where(frozen_mask, 0.0, wdelta)
            vines.weight += wdelta

        skf.update_particle_scores(node_hebb(X), reward)
        if skf.t % skf.freeze_every == 0 and K > 1:
            skf.phase_freeze()
            freezes += 1

        vines.prune_nonpositive()
        vines.enforce_budget(E_budget, rng)
        # also drop edges that are far again? keep — folding may reopen

        best_s = min(best_s, s)
        hist.append(best_s)
        nnz_trace.append(vines.nnz)

    return {
        "best_score": float(best_s),
        "history": hist,
        "nnz_trace": nnz_trace,
        "nnz_init": int(nnz_trace[0]),
        "nnz_final": int(nnz_trace[-1]),
        "bridges": 0,
        "freezes": freezes,
        "frozen_edges": len(skf.frozen_edges),
        "E_budget": E_budget,
        "evals": len(hist),
        "K": K,
    }


def pso_run(score_fn, active0, budget, rng, swarm: int = 8, inertia: float = 0.7, c1: float = 1.4, c2: float = 1.4) -> Tuple[float, list]:
    """Binary PSO / ES-style over edge membership (fixed cardinality via repair)."""
    E = len(active0)
    E_budget = int(active0.sum())
    # particles as bool arrays
    X = np.stack([active0.copy() for _ in range(swarm)])
    for p in range(1, swarm):
        X[p] = np.zeros(E, dtype=bool)
        X[p][rng.choice(E, size=E_budget, replace=False)] = True
    V = rng.normal(0, 0.1, size=(swarm, E))
    scores = np.array([score_fn(X[p]) for p in range(swarm)])
    pbest = X.copy()
    pbest_s = scores.copy()
    g = int(np.argmin(scores))
    gbest = X[g].copy()
    gbest_s = float(scores[g])
    hist = [gbest_s]
    evals = swarm

    def repair(active):
        on = np.flatnonzero(active)
        if len(on) > E_budget:
            drop = rng.choice(on, size=len(on) - E_budget, replace=False)
            active[drop] = False
        elif len(on) < E_budget:
            off = np.flatnonzero(~active)
            add = rng.choice(off, size=E_budget - len(on), replace=False)
            active[add] = True
        return active

    while evals < budget:
        for p in range(swarm):
            if evals >= budget:
                break
            r1, r2 = rng.random(E), rng.random(E)
            V[p] = inertia * V[p] + c1 * r1 * (pbest[p].astype(float) - X[p].astype(float)) + c2 * r2 * (gbest.astype(float) - X[p].astype(float))
            # sigmoid bit flip prob
            prob = 1.0 / (1.0 + np.exp(-V[p]))
            X[p] = rng.random(E) < prob
            X[p] = repair(X[p])
            sc = score_fn(X[p])
            evals += 1
            if sc < pbest_s[p]:
                pbest_s[p] = sc
                pbest[p] = X[p].copy()
            if sc < gbest_s:
                gbest_s = float(sc)
                gbest = X[p].copy()
            hist.append(gbest_s)
    return gbest_s, hist[:budget]

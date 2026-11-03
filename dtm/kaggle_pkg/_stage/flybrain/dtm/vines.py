"""Neuro-Vines sparse grow/prune (true list storage, not dense ghost edges)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch


@dataclass
class VineGraph:
    """Sparse directed vines: parallel arrays (true sparse; prune frees slots)."""
    N: int
    post: np.ndarray  # int32
    pre: np.ndarray
    nutrient: np.ndarray  # float64
    weight: np.ndarray
    mask_edges: np.ndarray  # (E_mask, 2) legal (post, pre)
    edge_to_idx: dict = field(default_factory=dict)  # (post,pre) -> position in arrays

    @staticmethod
    def seed(N: int, mask_edges: np.ndarray, n_seed: int, rng: np.random.Generator, w_scale: float = 0.08) -> "VineGraph":
        n_seed = int(min(n_seed, len(mask_edges)))
        pick = rng.choice(len(mask_edges), size=n_seed, replace=False)
        ee = mask_edges[pick]
        g = VineGraph(
            N=N,
            post=ee[:, 0].astype(np.int32).copy(),
            pre=ee[:, 1].astype(np.int32).copy(),
            nutrient=np.ones(n_seed, dtype=np.float64),
            weight=(w_scale * rng.standard_normal(n_seed)).astype(np.float64),
            mask_edges=mask_edges,
            edge_to_idx={},
        )
        g._reindex()
        return g

    def _reindex(self):
        self.edge_to_idx = {(int(self.post[i]), int(self.pre[i])): i for i in range(len(self.post))}

    @property
    def nnz(self) -> int:
        return int(len(self.post))

    def to_dense_W(self, device=None, dtype=torch.float32) -> torch.Tensor:
        W = torch.zeros(self.N, self.N, device=device, dtype=dtype)
        if self.nnz == 0:
            return W
        idx = torch.as_tensor(np.stack([self.post, self.pre], 1), device=device)
        val = torch.as_tensor(self.weight, device=device, dtype=dtype)
        W[idx[:, 0], idx[:, 1]] = val
        return W

    def active_set(self) -> set:
        return set(self.edge_to_idx.keys())

    def neighbors_in_mask(self, node: int) -> np.ndarray:
        """Return mask-legal (post,pre) rows where post==node or pre==node (local soil)."""
        # candidate outgoing from node as pre, or incoming as post — sprout as post<-pre with pre=node (axon out)
        m = self.mask_edges[:, 1] == node
        return self.mask_edges[m]

    def sprout(self, rng: np.random.Generator, k: int = 4, n0: float = 0.5, w_scale: float = 0.08, soft_cap: Optional[int] = None) -> int:
        """Grow up to k new vines from random active endpoints into unused mask neighbors. Returns #added."""
        if self.nnz == 0:
            # revive from mask
            return self._add_random(rng, k, n0, w_scale)
        if soft_cap is not None and self.nnz >= soft_cap:
            return 0
        nodes = np.unique(np.concatenate([self.post, self.pre]))
        node = int(rng.choice(nodes))
        cands = self.neighbors_in_mask(node)
        if len(cands) == 0:
            return self._add_random(rng, 1, n0, w_scale)
        active = self.active_set()
        free = [tuple(map(int, row)) for row in cands if tuple(map(int, row)) not in active]
        if not free:
            return self._add_random(rng, 1, n0, w_scale)
        k = int(min(k, len(free)))
        if soft_cap is not None:
            k = int(min(k, max(0, soft_cap - self.nnz)))
        if k <= 0:
            return 0
        pick = rng.choice(len(free), size=k, replace=False)
        added = 0
        for j in pick:
            p, q = free[j]
            self._append(p, q, n0, w_scale * rng.standard_normal())
            added += 1
        return added

    def _add_random(self, rng, k, n0, w_scale) -> int:
        active = self.active_set()
        free_idx = [i for i in range(len(self.mask_edges)) if tuple(map(int, self.mask_edges[i])) not in active]
        if not free_idx:
            return 0
        k = int(min(k, len(free_idx)))
        for i in rng.choice(free_idx, size=k, replace=False):
            p, q = map(int, self.mask_edges[i])
            self._append(p, q, n0, w_scale * rng.standard_normal())
        return k

    def _append(self, post: int, pre: int, nutrient: float, weight: float):
        self.post = np.append(self.post, np.int32(post))
        self.pre = np.append(self.pre, np.int32(pre))
        self.nutrient = np.append(self.nutrient, float(nutrient))
        self.weight = np.append(self.weight, float(weight))
        self.edge_to_idx[(post, pre)] = len(self.post) - 1

    def prune_nonpositive(self) -> int:
        keep = self.nutrient > 0
        dropped = int((~keep).sum())
        if dropped == 0:
            return 0
        self.post = self.post[keep]
        self.pre = self.pre[keep]
        self.nutrient = self.nutrient[keep]
        self.weight = self.weight[keep]
        self._reindex()
        return dropped

    def enforce_budget(self, budget: int, rng: np.random.Generator, n0: float = 0.5, w_scale: float = 0.08) -> Tuple[int, int]:
        """Prune lowest nutrient if over budget; sprout if under. Returns (pruned, added)."""
        pruned = 0
        if self.nnz > budget:
            order = np.argsort(self.nutrient)
            drop_n = self.nnz - budget
            keep = np.ones(self.nnz, dtype=bool)
            keep[order[:drop_n]] = False
            pruned = int(drop_n)
            self.post = self.post[keep]
            self.pre = self.pre[keep]
            self.nutrient = self.nutrient[keep]
            self.weight = self.weight[keep]
            self._reindex()
        added = 0
        while self.nnz < budget:
            a = self.sprout(rng, k=min(8, budget - self.nnz), n0=n0, w_scale=w_scale, soft_cap=budget)
            if a == 0:
                break
            added += a
        return pruned, added

    def update_nutrients(self, hebb: np.ndarray, reward: float, eta: float = 0.5, famine: float = 0.02):
        """hebb: length nnz, mean |x_pre * x_post| over time."""
        assert len(hebb) == self.nnz
        self.nutrient += eta * hebb * reward - famine
        # also nudge weights toward useful co-activity (local, forward-only)
        self.weight += 0.01 * reward * hebb * np.sign(self.weight + 1e-12)


def hebb_from_states(states: torch.Tensor, post: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """states (T,B,N) -> mean |x_post * x_pre| per edge."""
    # average over batch and time
    X = states.mean(dim=1)  # (T,N)
    Xp = X[:, torch.as_tensor(post, device=X.device)]
    Xq = X[:, torch.as_tensor(pre, device=X.device)]
    return (Xp * Xq).abs().mean(dim=0).detach().cpu().numpy()


def rescale_rho_sparse(W: torch.Tensor, rho: float = 0.9, iters: int = 20) -> torch.Tensor:
    v = torch.randn(W.shape[0], device=W.device)
    v = v / (v.norm() + 1e-12)
    n = torch.tensor(1.0, device=W.device)
    for _ in range(iters):
        v = W @ v
        n = v.norm() + 1e-12
        v = v / n
    r = float(n)
    if r > 1e-8:
        W = W * (rho / r)
    return W

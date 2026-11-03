"""Deterministic convergent graph primitive (G-graph).

Propagation conserves a bounded, geometrically decaying mass budget. The
output pair is the fraction of propagated mass landing on even and odd node
indices, avoiding saturation from exponentially growing path counts.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import List
import numpy as np

__all__ = ["GGraph"]

@dataclass
class GGraph:
    n: int = 4096
    out_degree: int = 3
    gamma: float = 0.87
    seed: int = 424242

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.out_degree <= 0:
            raise ValueError("out_degree must be positive")
        if not np.isfinite(self.gamma) or not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be finite and in (0,1)")
        rng = np.random.default_rng(self.seed)
        self._edges: List[List[int]] = []
        for node in range(self.n):
            remaining = self.n - node - 1
            if remaining <= 0:
                self._edges.append([])
                continue
            degree = min(self.out_degree, remaining)
            targets = rng.choice(np.arange(node + 1, self.n), size=degree, replace=False)
            self._edges.append(sorted(int(t) for t in targets))

    def influence(self, seed_id: str) -> tuple[float, float]:
        """Return normalized even/odd influence mass for a stable string key."""
        if not isinstance(seed_id, str):
            raise TypeError("seed_id must be a string")
        digest = hashlib.sha256(seed_id.encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "big") % self.n
        weights = np.zeros(self.n, dtype=np.float64)
        weights[start] = 1.0
        for node in range(start, self.n):
            weight = weights[node]
            if weight == 0.0:
                continue
            targets = self._edges[node]
            if targets:
                share = weight * self.gamma / len(targets)
                for target in targets:
                    weights[target] += share
        even_mass, odd_mass = float(weights[::2].sum()), float(weights[1::2].sum())
        total_mass = even_mass + odd_mass
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise ArithmeticError("G-graph propagation produced invalid total mass")
        return (float(even_mass / total_mass), float(odd_mass / total_mass))

    def summary(self) -> dict:
        seeds = ["demo.ybit", "quantacap", "bell", "fringe"]
        return {
            "n": self.n,
            "out_degree": self.out_degree,
            "gamma": self.gamma,
            "influences": {sid: self.influence(sid) for sid in seeds},
        }

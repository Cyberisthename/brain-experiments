"""Y-bit hybrid primitive combining a two-level amplitude pair and Z-bit bias.

This is a classical numerical primitive, not a physical qubit. Its probability
interpolation is a design choice, not a quantum measurement rule.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .zbit import ZBit

__all__ = ["YBit"]

@dataclass
class YBit:
    alpha_beta: Tuple[complex, complex]
    zbit: ZBit
    lam: float = 0.85
    eps_phase: float = 0.02
    bias_k: float = 1.0

    def __post_init__(self) -> None:
        lam, eps, bias_k = float(self.lam), float(self.eps_phase), float(self.bias_k)
        if not math.isfinite(lam):
            raise ValueError("lam must be finite")
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError("eps_phase must be finite and non-negative")
        if not math.isfinite(bias_k) or bias_k < 0.0:
            raise ValueError("bias_k must be finite and non-negative")
        self.lam = float(np.clip(lam, 0.0, 1.0))
        self.eps_phase, self.bias_k = eps, bias_k
        alpha, beta = complex(self.alpha_beta[0]), complex(self.alpha_beta[1])
        if not all(math.isfinite(v) for v in (alpha.real, alpha.imag, beta.real, beta.imag)):
            raise ValueError("alpha_beta components must be finite")
        norm = abs(alpha) ** 2 + abs(beta) ** 2
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("alpha_beta must define a finite, non-zero state")
        scale = 1.0 / math.sqrt(norm)
        self._alpha, self._beta = alpha * scale, beta * scale
        self._p1 = float(abs(self._beta) ** 2)
        sigma_b = float(self.zbit.bias_sigma(self.bias_k))
        if not math.isfinite(sigma_b):
            raise ValueError("Z-bit bias must be finite")
        self._adjusted = float(np.clip(self.lam * self._p1 + (1.0 - self.lam) * sigma_b, 0.0, 1.0))

    def adjusted_prob_1(self) -> float:
        """Return the Z-bit-regularized probability assigned to outcome 1."""
        return self._adjusted

    def base_prob_1(self) -> float:
        """Return normalized amplitude-pair probability before regularization."""
        return self._p1

    def phase_nudge(self) -> float:
        """Return the bounded classical phase offset supplied by the Z-bit."""
        return float(self.eps_phase * self.zbit.phase())

    def info(self) -> dict:
        return {
            "alpha": [float(self._alpha.real), float(self._alpha.imag)],
            "beta": [float(self._beta.real), float(self._beta.imag)],
            "p1_base": self._p1,
            "p1_adjusted": self._adjusted,
            "lam": self.lam,
            "eps_phase": self.eps_phase,
            "bias_k": self.bias_k,
            "zbit_value": self.zbit.value(),
        }

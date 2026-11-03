"""Z-bit scalar primitive with a forbidden band between 1 and 2.

Z-bit is a deterministic seeded classical scalar encoding; it is not a qubit.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

__all__ = ["ZBit"]
_EPS = 1e-12

def _reflect(value: float) -> float:
    """Ensure value stays outside the forbidden band [1, 2]."""
    if 1.0 <= value <= 2.0:
        return 1.0 - _EPS if value <= 1.5 else 2.0 + _EPS
    return value

@dataclass
class ZBit:
    """Deterministic scalar on the real line excluding the interval [1, 2]."""
    seed: int = 424242
    side: str = "auto"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.side not in {"auto", "left", "right"}:
            raise ValueError("side must be 'auto', 'left', or 'right'")
        side = self.side
        if side == "auto":
            side = "left" if rng.random() < 0.5 else "right"
        sample = abs(float(rng.normal()))
        value = 1.0 - sample - _EPS if side == "left" else 2.0 + sample + _EPS
        self._value = _reflect(value)

    def value(self) -> float:
        return float(self._value)

    def bias_sigma(self, k: float = 1.0) -> float:
        """Map scalar to a finite logistic bias without exponential overflow."""
        k = float(k)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("k must be finite and non-negative")
        delta = abs(self._value - 1.5)
        signed = math.copysign(math.log1p(delta), self._value)
        exponent = min(700.0, max(-700.0, -k * signed))
        return float(1.0 / (1.0 + math.exp(exponent)))

    def phase(self) -> float:
        """Map the absolute scalar into [0, 2*pi)."""
        phi = math.fmod(abs(self._value), 2.0 * math.pi)
        return float(phi if phi >= 0.0 else phi + 2.0 * math.pi)

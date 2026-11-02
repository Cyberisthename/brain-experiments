"""Multi-seed same-W helpers + optional compile."""
from __future__ import annotations

from typing import List, Optional

import torch

from .rate_network import OscRNN, OscRNNConfig


def stack_inputs(us: List[torch.Tensor]) -> torch.Tensor:
    """Stack list of (T,B,…) into (T, B*n_seeds, …) for shared-W multi-seed."""
    return torch.cat(us, dim=1)


def maybe_compile(model: OscRNN, enable: bool = True) -> OscRNN:
    if not enable:
        return model
    try:
        model.forward_sequence = torch.compile(model.forward_sequence)  # type: ignore
    except Exception:
        pass
    return model


def clone_cfg(**kwargs) -> OscRNNConfig:
    return OscRNNConfig(**kwargs)

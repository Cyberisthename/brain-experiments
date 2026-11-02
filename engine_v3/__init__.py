"""Engine v3 — shared sparse delayed rate OscRNN for connectome experiments.

Features (all opt-in; defaults match v2 callosum_scaled / wholebrain dynamics):
  - edge-list sparse recurrent weights + ring-buffer delays
  - Tsodyks–Markram STP, dual fast/slow synaptic filters, Dale E/I, homeostasis
  - float32, NaN guards, resume-safe jsonl, deterministic seeds
"""
from .rate_network import OscRNN, OscRNNConfig
from . import tasks, train, wiring

__version__ = "3.0.0"
__all__ = ["OscRNN", "OscRNNConfig", "tasks", "train", "wiring"]

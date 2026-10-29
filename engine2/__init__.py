"""engine2: adaptive LIF (AdEx-lite) spiking brain simulator.

More brain-like than the leaky-tanh reservoir in flybrain_reservoir /
fast_kernels: spikes, Dale's law, synaptic time constants by transmitter,
refractory + adaptation, axonal delays, optional STP and neuromod hooks,
with automatic E/I calibration so networks do not blow up.
"""
from .network import SpikingNetwork, NetworkConfig
from .calibrate import calibrate_ei
from .readout import ridge, pred, filter_spikes
from .tasks import run_task, TASKS
from .builders import build_arm

__all__ = [
    "SpikingNetwork",
    "NetworkConfig",
    "calibrate_ei",
    "ridge",
    "pred",
    "filter_spikes",
    "run_task",
    "TASKS",
    "build_arm",
]
__version__ = "0.1.0"

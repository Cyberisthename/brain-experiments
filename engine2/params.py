"""Biophysical defaults for AdEx-lite / current-based adaptive LIF."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

# Fly transmitter -> (E_or_I, tau_syn_ms). Glu inhibitory via GluCl in fly.
NT_TAU_MS: Dict[str, float] = {
    "ACh": 5.0,   # fast excitatory
    "Glu": 5.0,   # fly: mostly inhibitory (GluCl)
    "GABA": 10.0, # inhibitory
    "DA": 50.0,   # neuromod (slow; used as hook, not fast synapse by default)
    "5HT": 50.0,
    "OA": 50.0,
    "UNK": 7.0,
}

# Sign convention used when NT label present
NT_SIGN: Dict[str, int] = {
    "ACh": +1,
    "Glu": -1,  # fly GluCl
    "GABA": -1,
    "DA": 0,
    "5HT": 0,
    "OA": 0,
    "UNK": +1,
}

MOD_NTS = frozenset({"DA", "5HT", "OA"})


@dataclass
class NeuronParams:
    C_m: float = 200.0          # pF
    g_L: float = 10.0           # nS  -> tau_m = C/g = 20 ms
    E_L: float = -70.0          # mV
    V_th: float = -50.0         # mV
    V_reset: float = -65.0      # mV
    V_peak: float = 20.0        # clamp / spike peak
    t_ref_ms: float = 2.0       # refractory
    # Adaptation (AdEx-lite / aLIF)
    a: float = 2.0              # nS  subthreshold adaptation
    b: float = 40.0             # pA  spike-triggered adaptation
    tau_w_ms: float = 100.0
    # Homeostatic threshold (slow)
    homeo_rate: float = 0.01    # mV per (Hz above target) per second
    target_rate_hz: float = 5.0


@dataclass
class SynapseParams:
    tau_e_ms: float = 5.0
    tau_i_ms: float = 8.0
    # Tsodyks-Markram (optional)
    U: float = 0.4
    tau_rec_ms: float = 200.0
    tau_fac_ms: float = 0.0     # 0 = no facilitation
    # Neuromod hooks: scale adaptation / threshold, not raw gain
    da_adapt_scale: float = 0.5   # DA reduces adaptation (disinhibition-like)
    ht_thresh_shift: float = 2.0  # 5HT raises threshold (mV at full mod)
    oa_adapt_scale: float = 0.7


@dataclass
class SimParams:
    dt_ms: float = 0.25
    max_delay_ms: float = 10.0
    v_min: float = -90.0
    v_max: float = 40.0
    abort_rate_hz: float = 200.0   # mean rate => abort (blow-up)
    abort_nan: bool = True
    dtype: str = "float32"

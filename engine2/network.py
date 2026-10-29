"""Sparse adaptive LIF network with delays and Dale's law."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import numpy as np
import scipy.sparse as sp

from .params import NeuronParams, SynapseParams, SimParams, NT_SIGN, MOD_NTS

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    import cupy as cp  # noqa: F401
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False


@dataclass
class NetworkConfig:
    neuron: NeuronParams = field(default_factory=NeuronParams)
    syn: SynapseParams = field(default_factory=SynapseParams)
    sim: SimParams = field(default_factory=SimParams)
    use_stp: bool = False
    use_neuromod: bool = True
    use_numba: bool = True
    use_cupy: bool = False
    conductance_based: bool = False


def _exp_decay(tau_ms: float, dt_ms: float) -> float:
    if tau_ms <= 0:
        return 0.0
    return float(np.exp(-dt_ms / tau_ms))


if _HAS_NUMBA:
    @njit(cache=True)
    def _step_neurons_numba(
        V, w, ref, thr, Ie, Ii, Iext,
        exp_m, exp_w, E_L, inv_g, a, b, V_reset,
        t_ref_steps, v_min, v_max, thr_mod, adapt_scale, B,
    ):
        N = V.shape[0]
        spikes = np.zeros((N, B), dtype=np.uint8)
        for j in range(B):
            for i in range(N):
                if ref[i, j] > 0:
                    ref[i, j] -= 1
                    V[i, j] = V_reset
                    continue
                I = Ie[i, j] + Ii[i, j] + Iext[i, j] - w[i, j] * adapt_scale[i, j]
                V[i, j] = E_L + (V[i, j] - E_L) * exp_m + (I * inv_g) * (1.0 - exp_m)
                w[i, j] = w[i, j] * exp_w + a * (V[i, j] - E_L) * (1.0 - exp_w)
                if V[i, j] < v_min:
                    V[i, j] = v_min
                elif V[i, j] > v_max:
                    V[i, j] = v_max
                th = thr[i, j] + thr_mod[i, j]
                if V[i, j] >= th:
                    spikes[i, j] = 1
                    V[i, j] = V_reset
                    w[i, j] = w[i, j] + b
                    ref[i, j] = t_ref_steps
        return spikes


class SpikingNetwork:
    """Adaptive LIF population with sparse CSR synapses and axonal delay ring.

    State is (N, B) float32 for batched seeds/trials.
    Positive weights -> excitatory current Ie; negative -> Ii (Dale).
    """

    def __init__(
        self,
        W: sp.spmatrix,
        sign: np.ndarray,
        delays_ms: Optional[np.ndarray] = None,
        nt_labels: Optional[np.ndarray] = None,
        config: Optional[NetworkConfig] = None,
        weight_scale: float = 1.0,
        batch: int = 1,
    ):
        self.cfg = config or NetworkConfig()
        self.np_ = self.cfg.neuron
        self.sp_ = self.cfg.syn
        self.sim = self.cfg.sim
        dt = self.sim.dt_ms
        dtype = np.float32 if self.sim.dtype == "float32" else np.float64

        W = W.tocsr().astype(np.float32)
        if weight_scale != 1.0:
            W = W.copy()
            W.data = (W.data * np.float32(weight_scale)).astype(np.float32)
        self.W = W
        self.N = int(W.shape[0])
        self.B = int(batch)
        self.sign = np.asarray(sign, dtype=np.int8).copy()
        assert self.sign.shape[0] == self.N

        if nt_labels is None:
            self.nt = np.array(["ACh" if s > 0 else "GABA" for s in self.sign], dtype=object)
        else:
            self.nt = np.asarray(nt_labels).astype(str)
            for i, lab in enumerate(self.nt):
                s = NT_SIGN.get(lab, int(self.sign[i]))
                if s != 0:
                    self.sign[i] = s
        self.is_mod = np.array([lab in MOD_NTS for lab in self.nt], dtype=bool)

        Wc = W.tocoo()
        self._rows = Wc.row.astype(np.int32)
        self._cols = Wc.col.astype(np.int32)
        self._wdata = Wc.data.astype(np.float32)
        self._wdata0 = self._wdata.copy()
        self._has_syn = bool(Wc.nnz > 0)

        if Wc.nnz == 0:
            d_steps = np.zeros(0, dtype=np.int32)
            dmax_edge = 1
        else:
            if delays_ms is None:
                d_steps = np.ones(Wc.nnz, dtype=np.int32)
            else:
                delays_ms = np.asarray(delays_ms, dtype=np.float64)
                assert delays_ms.shape[0] == Wc.nnz
                d_steps = np.clip(np.rint(delays_ms / dt), 1, None).astype(np.int32)
            dmax_edge = int(d_steps.max())

        dmax_cap = int(max(1, int(np.ceil(self.sim.max_delay_ms / dt))))
        self.max_delay = int(min(dmax_cap, max(dmax_edge, 1)))
        self.d_steps = (
            np.clip(d_steps, 1, self.max_delay).astype(np.int32) if d_steps.size else d_steps
        )

        self._build_delay_buckets()

        self.exp_m = np.float32(_exp_decay(self.np_.C_m / self.np_.g_L, dt))
        self.exp_w = np.float32(_exp_decay(self.np_.tau_w_ms, dt))
        self.exp_e = np.float32(_exp_decay(self.sp_.tau_e_ms, dt))
        self.exp_i = np.float32(_exp_decay(self.sp_.tau_i_ms, dt))
        self.inv_g = np.float32(1.0 / self.np_.g_L)
        self.t_ref_steps = max(1, int(round(self.np_.t_ref_ms / dt)))

        self.use_stp = bool(self.cfg.use_stp)
        if self.use_stp and Wc.nnz:
            self.u_stp = np.full(Wc.nnz, self.sp_.U, dtype=np.float32)
            self.x_stp = np.ones(Wc.nnz, dtype=np.float32)
            self.exp_rec = np.float32(_exp_decay(self.sp_.tau_rec_ms, dt))

        self.use_nm = bool(self.cfg.use_neuromod)
        self.exp_mod = np.float32(_exp_decay(80.0, dt))

        self.dtype = dtype
        self._use_numba = bool(self.cfg.use_numba and _HAS_NUMBA)
        self._rate_win_steps = max(50, int(round(50.0 / max(dt, 1e-6))))
        self.reset()

    def _build_delay_buckets(self):
        self.buckets_e = []
        self.buckets_i = []
        for d in range(1, self.max_delay + 1):
            if self.d_steps.size == 0:
                self.buckets_e.append(None)
                self.buckets_i.append(None)
                continue
            m = self.d_steps == d
            if not np.any(m):
                self.buckets_e.append(None)
                self.buckets_i.append(None)
                continue
            rows = self._rows[m]
            cols = self._cols[m]
            data = self._wdata[m]
            pos = data > 0
            neg = data < 0
            We = (
                sp.csr_matrix((data[pos], (rows[pos], cols[pos])), shape=(self.N, self.N), dtype=np.float32)
                if np.any(pos) else None
            )
            Wi = (
                sp.csr_matrix(
                    (np.abs(data[neg]), (rows[neg], cols[neg])), shape=(self.N, self.N), dtype=np.float32
                )
                if np.any(neg) else None
            )
            self.buckets_e.append(We)
            self.buckets_i.append(Wi)
        self.W_csc = self.W.tocsc()

    def reset(self, batch: Optional[int] = None):
        if batch is not None:
            self.B = int(batch)
        N, B = self.N, self.B
        d = self.dtype
        self.V = np.full((N, B), self.np_.E_L, dtype=d)
        self.w = np.zeros((N, B), dtype=d)
        self.ref = np.zeros((N, B), dtype=np.int32)
        self.thr = np.full((N, B), self.np_.V_th, dtype=d)
        self.Ie = np.zeros((N, B), dtype=d)
        self.Ii = np.zeros((N, B), dtype=d)
        self.spike_ring = np.zeros((self.max_delay + 1, N, B), dtype=np.float32)
        self.mod_da = np.zeros((N, B), dtype=d)
        self.mod_ht = np.zeros((N, B), dtype=d)
        self.mod_oa = np.zeros((N, B), dtype=d)
        self.spike_count = np.zeros(B, dtype=np.int64)
        self._rate_win_spikes = np.zeros(B, dtype=np.float64)
        self._t = 0
        self._aborted = False
        self.abort_reason = None
        if self.use_stp and hasattr(self, "u_stp"):
            self.u_stp[:] = self.sp_.U
            self.x_stp[:] = 1.0

    def set_weight_scale(self, scale: float):
        scale = float(scale)
        self._wdata = (self._wdata0 * np.float32(scale)).astype(np.float32)
        self.W = sp.csr_matrix(
            (self._wdata, (self._rows, self._cols)), shape=(self.N, self.N), dtype=np.float32
        )
        self._has_syn = self.W.nnz > 0
        self._build_delay_buckets()

    def _neuromod_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_nm:
            ones = np.ones((self.N, self.B), dtype=self.dtype)
            zeros = np.zeros((self.N, self.B), dtype=self.dtype)
            return ones, zeros
        adapt = (
            1.0
            - self.sp_.da_adapt_scale * np.clip(self.mod_da, 0, 1)
            - (1.0 - self.sp_.oa_adapt_scale) * np.clip(self.mod_oa, 0, 1)
        )
        adapt = np.clip(adapt, 0.1, 2.0).astype(self.dtype)
        thr_mod = (self.sp_.ht_thresh_shift * np.clip(self.mod_ht, 0, 1)).astype(self.dtype)
        return adapt, thr_mod

    def _deliver_delayed(self):
        if not self._has_syn:
            return
        t = self._t
        for di, (We, Wi) in enumerate(zip(self.buckets_e, self.buckets_i)):
            if We is None and Wi is None:
                continue
            src = (t - (di + 1)) % (self.max_delay + 1)
            S = self.spike_ring[src]
            if not S.any():
                continue
            if We is not None:
                self.Ie += We @ S
            if Wi is not None:
                self.Ii += Wi @ S

    def _update_neuromod(self, spikes: np.ndarray):
        if not self.use_nm:
            return
        def mean_spk(mask):
            if not np.any(mask):
                return np.zeros(self.B, dtype=self.dtype)
            return spikes[mask].mean(axis=0).astype(self.dtype)

        g_da = mean_spk(self.nt == "DA")
        g_ht = mean_spk(self.nt == "5HT")
        g_oa = mean_spk(self.nt == "OA")
        self.mod_da = self.mod_da * self.exp_mod + g_da[None, :]
        self.mod_ht = self.mod_ht * self.exp_mod + g_ht[None, :]
        self.mod_oa = self.mod_oa * self.exp_mod + g_oa[None, :]

    def _homeostasis(self, spikes: np.ndarray):
        if self.np_.homeo_rate <= 0:
            return
        rate = spikes.astype(np.float32) / (self.sim.dt_ms * 1e-3)
        err = rate - self.np_.target_rate_hz
        self.thr += np.float32(self.np_.homeo_rate * self.sim.dt_ms * 1e-3) * err
        self.thr = np.clip(self.thr, self.np_.V_reset + 5.0, self.np_.V_peak - 5.0)

    def step(self, Iext: Optional[np.ndarray] = None) -> np.ndarray:
        if self._aborted:
            return np.zeros((self.N, self.B), dtype=np.float32)

        B = self.B
        if Iext is None:
            Iext = np.zeros((self.N, B), dtype=self.dtype)
        else:
            Iext = np.asarray(Iext, dtype=self.dtype)
            if Iext.ndim == 1:
                Iext = np.broadcast_to(Iext[:, None], (self.N, B)).copy()
            elif Iext.shape != (self.N, B):
                raise ValueError(f"Iext shape {Iext.shape} != {(self.N, B)}")

        self.Ie *= self.exp_e
        self.Ii *= self.exp_i
        self._deliver_delayed()
        adapt_scale, thr_mod = self._neuromod_factors()

        if self._use_numba:
            spikes_u8 = _step_neurons_numba(
                self.V, self.w, self.ref, self.thr, self.Ie, self.Ii, Iext,
                float(self.exp_m), float(self.exp_w), float(self.np_.E_L),
                float(self.inv_g), float(self.np_.a), float(self.np_.b),
                float(self.np_.V_reset),
                int(self.t_ref_steps), float(self.sim.v_min), float(self.sim.v_max),
                thr_mod.astype(np.float32), adapt_scale.astype(np.float32), int(B),
            )
            spikes = spikes_u8.astype(np.float32)
        else:
            spikes = self._step_neurons_py(Iext, adapt_scale, thr_mod)

        if self.sim.abort_nan and not np.all(np.isfinite(self.V)):
            self._aborted = True
            self.abort_reason = "NaN in V"
            spikes[:] = 0
            return spikes

        self.spike_count += spikes.sum(axis=0).astype(np.int64)
        self._t += 1
        self.spike_ring[self._t % (self.max_delay + 1)] = spikes

        self._rate_win_spikes += spikes.sum(axis=0)
        if self._t % self._rate_win_steps == 0:
            window_s = self._rate_win_steps * self.sim.dt_ms * 1e-3
            recent = (self._rate_win_spikes / max(self.N, 1)) / max(window_s, 1e-12)
            self._rate_win_spikes[:] = 0
            if np.any(recent > self.sim.abort_rate_hz):
                self._aborted = True
                self.abort_reason = f"rate>{self.sim.abort_rate_hz}Hz (got {float(np.max(recent)):.1f})"
                spikes[:] = 0
                return spikes

        self._update_neuromod(spikes)
        self._homeostasis(spikes)
        return spikes

    def _step_neurons_py(self, Iext, adapt_scale, thr_mod) -> np.ndarray:
        active = self.ref <= 0
        I = self.Ie + self.Ii + Iext - self.w * adapt_scale
        self.V = self.np_.E_L + (self.V - self.np_.E_L) * self.exp_m + (I * self.inv_g) * (1.0 - self.exp_m)
        self.w = self.w * self.exp_w + self.np_.a * (self.V - self.np_.E_L) * (1.0 - self.exp_w)
        self.V = np.where(active, self.V, self.np_.V_reset)
        self.ref = np.maximum(self.ref - 1, 0)
        np.clip(self.V, self.sim.v_min, self.sim.v_max, out=self.V)
        th = self.thr + thr_mod
        fired = (self.V >= th) & active
        spikes = fired.astype(np.float32)
        self.V[fired] = self.np_.V_reset
        self.w[fired] += self.np_.b
        self.ref[fired] = self.t_ref_steps
        return spikes

    def run(
        self,
        T_steps: int,
        Iext_t: Optional[np.ndarray] = None,
        record_spikes: bool = True,
        record_filtered: bool = True,
        filter_tau_ms: float = 20.0,
    ) -> Dict[str, np.ndarray]:
        T = int(T_steps)
        N, B = self.N, self.B
        spikes_out = np.zeros((T, N, B), dtype=np.float32) if record_spikes else None
        filt = np.zeros((N, B), dtype=self.dtype)
        exp_f = np.float32(_exp_decay(filter_tau_ms, self.sim.dt_ms))
        filtered_out = np.zeros((T, N, B), dtype=np.float32) if record_filtered else None

        for t in range(T):
            I = None if Iext_t is None else Iext_t[t]
            spk = self.step(I)
            if spikes_out is not None:
                spikes_out[t] = spk
            if record_filtered:
                filt = filt * exp_f + spk
                filtered_out[t] = filt
            if self._aborted:
                break

        rates = self.spike_count / max(self._t * self.sim.dt_ms * 1e-3, 1e-12)
        return {
            "spikes": spikes_out,
            "filtered": filtered_out,
            "rates": rates,
            "aborted": self._aborted,
            "abort_reason": self.abort_reason,
            "steps": self._t,
        }


HAS_NUMBA = _HAS_NUMBA
HAS_CUPY = _HAS_CUPY

"""Sparse delayed rate OscRNN with opt-in STP / dual-τ / Dale / homeostasis.

Default (all flags off) matches regions/callosum_scaled.OscRNN dynamics:
  leaky tanh, softplus on inter edges × esign, integer delays 1..dmax,
  left-hemi / lh-motor readout modes. Ring buffer replaces torch.roll.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OscRNNConfig:
    leak: float = 0.4
    dmax: int = 5
    # STP (Tsodyks–Markram rate proxy)
    use_stp: bool = False
    stp_U: float = 0.2
    stp_tau_rec: float = 100.0  # steps
    stp_tau_fac: float = 500.0  # steps
    stp_on_inter_only: bool = True
    # Dual synaptic filters
    use_dual_tau: bool = False
    alpha_fast: float = 0.5   # ~AMPA (larger = faster)
    alpha_slow: float = 0.05  # ~NMDA
    kappa_slow: float = 0.5
    # Dale's law + E/I dial
    use_dale: bool = False
    p_inhibit: float = 0.2
    ei_beta: float = 0.0  # I strength dial; 0 ≈ |w_I|/|w_E|≈4 via softplus scale
    # Homeostasis
    use_homeostasis: bool = False
    homeo_target: float = 0.3
    homeo_rate: float = 1e-3
    homeo_every: int = 1
    # Numerics
    nan_guard: bool = True
    compile_forward: bool = False


class OscRNN(nn.Module):
    """Shared sparse OscRNN used by callosum / wholebrain wrappers."""

    def __init__(
        self,
        wiring: dict,
        n_in: int,
        out_dim: int,
        cfg: Optional[OscRNNConfig] = None,
        emb_dim: int = 0,
        vocab: int = 0,
        readout_mode: str = "left_hemi",
        input_mode: str = "hemi_split",
        *,
        # convenience overrides (also accepted as kwargs from train)
        leak: Optional[float] = None,
        use_stp: Optional[bool] = None,
        use_dual_tau: Optional[bool] = None,
        use_dale: Optional[bool] = None,
        use_homeostasis: Optional[bool] = None,
        neuron_signs: Optional[np.ndarray] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg or OscRNNConfig()
        if leak is not None:
            self.cfg.leak = float(leak)
        if use_stp is not None:
            self.cfg.use_stp = bool(use_stp)
        if use_dual_tau is not None:
            self.cfg.use_dual_tau = bool(use_dual_tau)
        if use_dale is not None:
            self.cfg.use_dale = bool(use_dale)
        if use_homeostasis is not None:
            self.cfg.use_homeostasis = bool(use_homeostasis)

        self.leak = float(self.cfg.leak)
        self.n = int(wiring["n_hidden"])
        self.n_hemi = int(wiring.get("n_hemi_units", self.n // 2))
        self.dmax = int(wiring.get("dmax", self.cfg.dmax))
        self.use_emb = emb_dim > 0 and vocab > 0
        self.readout_mode = readout_mode
        self.input_mode = input_mode
        self._step_count = 0

        if self.use_emb:
            self.emb = nn.Embedding(vocab, emb_dim)
            n_in_eff = emb_dim
        else:
            self.emb = None
            n_in_eff = n_in

        mi = np.asarray(wiring["mask_intra"]).astype(bool)
        mj = np.asarray(wiring["mask_inter"]).astype(bool)
        delays = np.asarray(wiring["delays"], dtype=np.int64)
        esign = np.asarray(
            wiring.get("esign_mat", np.ones((self.n, self.n), dtype=np.float32)),
            dtype=np.float32,
        )

        # Intra (instant)
        ii, jj = np.where(mi)
        self.register_buffer("intra_src", torch.from_numpy(ii.astype(np.int64)))
        self.register_buffer("intra_dst", torch.from_numpy(jj.astype(np.int64)))
        self.w_intra = nn.Parameter(0.08 * torch.randn(len(ii)))

        # Inter (delayed)
        ii, jj = np.where(mj)
        self.register_buffer("inter_src", torch.from_numpy(ii.astype(np.int64)))
        self.register_buffer("inter_dst", torch.from_numpy(jj.astype(np.int64)))
        inter_delay = delays[ii, jj] if len(ii) else np.zeros(0, dtype=np.int64)
        inter_delay = np.clip(inter_delay, 1, self.dmax).astype(np.int64)
        self.register_buffer("inter_delay", torch.from_numpy(inter_delay))
        self.register_buffer("inter_sign", torch.from_numpy(esign[ii, jj] if len(ii) else np.zeros(0, np.float32)))
        self.w_inter = nn.Parameter(0.08 * torch.randn(len(ii)))

        # Optional myelin/FA calibrated delays already baked into wiring["delays"]
        self.register_buffer("mask_intra", torch.from_numpy(mi.astype(np.float32)))
        self.register_buffer("mask_inter", torch.from_numpy(mj.astype(np.float32)))

        vis = wiring.get("vis_rh_units", np.arange(self.n_hemi, min(self.n, self.n_hemi + 8)))
        mot = wiring.get("mot_lh_units", np.arange(0, min(8, self.n_hemi)))
        self.register_buffer("vis_rh", torch.as_tensor(np.asarray(vis), dtype=torch.int64))
        self.register_buffer("mot_lh", torch.as_tensor(np.asarray(mot), dtype=torch.int64))

        # Dale neuron signs
        if self.cfg.use_dale:
            if neuron_signs is None:
                rng = np.random.default_rng(seed + 91)
                signs = np.ones(self.n, dtype=np.float32)
                n_inh = max(1, int(round(self.cfg.p_inhibit * self.n)))
                inh = rng.choice(self.n, size=n_inh, replace=False)
                signs[inh] = -1.0
            else:
                signs = np.asarray(neuron_signs, dtype=np.float32)
            self.register_buffer("neuron_sign", torch.from_numpy(signs))
        else:
            self.register_buffer("neuron_sign", torch.ones(self.n))

        # Win init (match callosum_scaled modes)
        self.Win = nn.Parameter(torch.zeros(self.n, n_in_eff))
        with torch.no_grad():
            if self.use_emb or input_mode == "left_drive":
                self.Win[: self.n_hemi, :] = 0.6 * torch.randn(self.n_hemi, n_in_eff)
            elif input_mode == "hemi_split":
                for c in range(n_in_eff):
                    if c % 2 == 0:
                        self.Win[: self.n_hemi, c] = 0.6 * torch.randn(self.n_hemi)
                    else:
                        self.Win[self.n_hemi :, c] = 0.6 * torch.randn(self.n - self.n_hemi)
            elif input_mode == "transfer_rh_visual":
                idx = self.vis_rh
                self.Win[idx, :] = 0.8 * torch.randn(len(idx), n_in_eff)
            elif input_mode == "delayed_match":
                idx = self.vis_rh
                self.Win[idx, 0] = 0.8 * torch.randn(len(idx))
                self.Win[: self.n_hemi, 1] = 0.6 * torch.randn(self.n_hemi)
            else:
                self.Win[: self.n_hemi, :] = 0.6 * torch.randn(self.n_hemi, n_in_eff)

        self.b = nn.Parameter(torch.zeros(self.n))
        if readout_mode == "lh_motor":
            read_in = int(self.mot_lh.numel())
        elif readout_mode == "left_hemi":
            read_in = self.n_hemi
        else:
            read_in = self.n
        self.readout = nn.Linear(read_in, out_dim)

        for d in range(1, self.dmax + 1):
            self.register_buffer(f"sel_d{d}", self.inter_delay == d)

        # STP buffers allocated lazily on first forward (need B)
        self._stp_ready = False

        if self.cfg.compile_forward:
            try:
                self.forward_sequence = torch.compile(self.forward_sequence)  # type: ignore
            except Exception:
                pass

    # ----- helpers -----
    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "lh_motor":
            return x[:, self.mot_lh]
        if self.readout_mode == "left_hemi":
            return x[:, : self.n_hemi]
        return x

    def _scatter(self, src_states, src_idx, dst_idx, weights):
        if src_idx.numel() == 0:
            return torch.zeros_like(src_states)
        msgs = src_states[:, src_idx] * weights.unsqueeze(0)
        out = torch.zeros_like(src_states)
        out.index_add_(1, dst_idx, msgs)
        return out

    def _inter_weights(self) -> torch.Tensor:
        """Effective inter edge magnitudes with sign."""
        if self.cfg.use_dale:
            # softplus magnitude × source neuron Dale sign; ei_beta boosts |I|
            mag = F.softplus(self.w_inter)
            src_sign = self.neuron_sign[self.inter_src]
            # I neurons: scale by (4 * exp(ei_beta)) so β=0 → |w_I|/|w_E|≈4 after softplus mean≈1
            i_scale = torch.where(src_sign < 0, 4.0 * torch.exp(torch.tensor(self.cfg.ei_beta, device=mag.device)), torch.ones((), device=mag.device))
            return mag * src_sign * i_scale
        return F.softplus(self.w_inter) * self.inter_sign

    def _intra_weights(self) -> torch.Tensor:
        if self.cfg.use_dale:
            mag = F.softplus(self.w_intra)
            src_sign = self.neuron_sign[self.intra_src]
            i_scale = torch.where(src_sign < 0, 4.0 * torch.exp(torch.tensor(self.cfg.ei_beta, device=mag.device)), torch.ones((), device=mag.device))
            return mag * src_sign * i_scale
        return self.w_intra

    def _ensure_stp(self, B: int, device, dtype):
        E = int(self.inter_src.numel())
        if (not self.cfg.use_stp) or E == 0:
            self._stp_ready = False
            return
        if (
            self._stp_ready
            and hasattr(self, "stp_u")
            and self.stp_u.shape[0] == B
            and self.stp_u.device == device
        ):
            return
        U = float(self.cfg.stp_U)
        self.stp_u = torch.full((B, E), U, device=device, dtype=dtype)
        self.stp_x = torch.ones(B, E, device=device, dtype=dtype)
        self._stp_ready = True

    def _stp_modulate(self, pre_states: torch.Tensor, w_mag: torch.Tensor) -> torch.Tensor:
        """Update STP and return effective weights (B, E). pre_states: (B, N)."""
        if not self.cfg.use_stp or not self._stp_ready:
            return w_mag.unsqueeze(0).expand(pre_states.shape[0], -1)
        pre = pre_states[:, self.inter_src].abs().clamp(0, 1)  # (B, E)
        U = float(self.cfg.stp_U)
        # facilitate then consume (rate proxy)
        self.stp_u = self.stp_u + U * (1.0 - self.stp_u) * pre
        release = self.stp_u * pre
        self.stp_x = self.stp_x * (1.0 - release).clamp(min=0.0)
        eff = w_mag.unsqueeze(0) * self.stp_u * self.stp_x
        # recover
        a_rec = float(np.exp(-1.0 / max(self.cfg.stp_tau_rec, 1e-6)))
        a_fac = float(np.exp(-1.0 / max(self.cfg.stp_tau_fac, 1e-6)))
        self.stp_x = 1.0 - (1.0 - self.stp_x) * a_rec
        self.stp_u = U + (self.stp_u - U) * a_fac
        return eff

    def _recurrent(self, x, hist, head):
        """Ring-buffer recurrent drive. hist: (D+1, B, N), head: int write index."""
        rec = self._scatter(x, self.intra_src, self.intra_dst, self._intra_weights())
        w_mag = self._inter_weights()
        D = self.dmax
        size = D + 1
        # STP uses current x as pre for inter (same timing as instantaneous view)
        if self.cfg.use_stp:
            # build per-delay using modulated weights; update STP once on current x
            w_eff_BE = self._stp_modulate(x, w_mag)  # (B, E)
            for d in range(1, D + 1):
                sel = getattr(self, f"sel_d{d}")
                if not bool(sel.any()):
                    continue
                src_t = (head - d) % size
                delayed = hist[src_t]  # (B, N)
                # scatter with per-batch weights
                src_idx = self.inter_src[sel]
                dst_idx = self.inter_dst[sel]
                w_sel = w_eff_BE[:, sel]  # (B, e)
                msgs = delayed[:, src_idx] * w_sel
                add = torch.zeros_like(x)
                add.index_add_(1, dst_idx, msgs)
                rec = rec + add
        else:
            for d in range(1, D + 1):
                sel = getattr(self, f"sel_d{d}")
                if not bool(sel.any()):
                    continue
                src_t = (head - d) % size
                delayed = hist[src_t]
                rec = rec + self._scatter(
                    delayed, self.inter_src[sel], self.inter_dst[sel], w_mag[sel]
                )
        return rec

    def forward_sequence(self, u: torch.Tensor) -> torch.Tensor:
        if self.use_emb:
            T, B = u.shape
            device = u.device
            dtype = torch.float32
        else:
            T, B, _ = u.shape
            device = u.device
            dtype = u.dtype

        x = torch.zeros(B, self.n, device=device, dtype=dtype)
        hist = torch.zeros(self.dmax + 1, B, self.n, device=device, dtype=dtype)
        head = 0  # after first write will land at 0 following advance
        # Match torch.roll semantics: start with zeros; each step advance then write.
        # We treat head as the index of the most recently written state (buf[0]).
        # Initialize: nothing written; set head=-1 conceptually, first step → head=0.
        head = -1

        g_fast = torch.zeros(B, self.n, device=device, dtype=dtype)
        g_slow = torch.zeros(B, self.n, device=device, dtype=dtype)

        self._ensure_stp(B, device, dtype)
        if self.cfg.use_stp and self._stp_ready:
            self.stp_u.fill_(float(self.cfg.stp_U))
            self.stp_x.fill_(1.0)

        outs = []
        for t in range(T):
            # ring advance ≡ torch.roll(..., 1) then buf[0]=x
            head = (head + 1) % (self.dmax + 1)
            hist[head] = x

            raw = self._recurrent(x, hist, head)

            if self.cfg.use_dual_tau:
                af = float(self.cfg.alpha_fast)
                as_ = float(self.cfg.alpha_slow)
                g_fast = (1 - af) * g_fast + af * raw
                g_slow = (1 - as_) * g_slow + as_ * raw
                rec = g_fast + float(self.cfg.kappa_slow) * g_slow
            else:
                rec = raw

            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b
            pre = rec + drive
            if self.cfg.nan_guard:
                pre = torch.nan_to_num(pre, nan=0.0, posinf=20.0, neginf=-20.0)
            x = (1 - self.leak) * x + self.leak * torch.tanh(pre)
            if self.cfg.nan_guard:
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            if self.cfg.use_homeostasis and (t % max(1, self.cfg.homeo_every) == 0):
                # slow bias toward target |activity|
                with torch.no_grad():
                    act = x.detach().abs().mean(dim=0)
                    self.b.add_(self.cfg.homeo_rate * (self.cfg.homeo_target - act))

            outs.append(self.readout(self._feat(x)))
            self._step_count += 1

        return torch.stack(outs, 0)

    def forward_sequence_roll_reference(self, u: torch.Tensor) -> torch.Tensor:
        """v2-style torch.roll path for unit tests (features off only)."""
        assert not self.cfg.use_stp and not self.cfg.use_dual_tau and not self.cfg.use_dale
        if self.use_emb:
            T, B = u.shape
            device = u.device
            dtype = torch.float32
        else:
            T, B, _ = u.shape
            device = u.device
            dtype = u.dtype
        x = torch.zeros(B, self.n, device=device, dtype=dtype)
        buf = torch.zeros(self.dmax + 1, B, self.n, device=device, dtype=dtype)
        outs = []
        w_intra = self.w_intra
        w_mag = F.softplus(self.w_inter) * self.inter_sign
        for t in range(T):
            buf = torch.roll(buf, 1, dims=0)
            buf[0] = x
            rec = self._scatter(x, self.intra_src, self.intra_dst, w_intra)
            for d in range(1, self.dmax + 1):
                sel = getattr(self, f"sel_d{d}")
                if not bool(sel.any()):
                    continue
                rec = rec + self._scatter(buf[d], self.inter_src[sel], self.inter_dst[sel], w_mag[sel])
            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            outs.append(self.readout(self._feat(x)))
        return torch.stack(outs, 0)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def synthetic_wiring(
    n: int = 64,
    p_intra: float = 0.08,
    p_inter: float = 0.02,
    seed: int = 0,
    dmax: int = 5,
    mean_degree: Optional[float] = None,
) -> dict:
    """ER two-hemi wiring for unit tests / ablations without connectome I/O."""
    rng = np.random.default_rng(seed)
    n_hemi = n // 2
    mi = np.zeros((n, n), dtype=bool)
    mj = np.zeros((n, n), dtype=bool)
    delays = np.ones((n, n), dtype=np.int64)
    esign = np.ones((n, n), dtype=np.float32)
    # intra within each hemi
    for hemi in (range(0, n_hemi), range(n_hemi, n)):
        idx = list(hemi)
        for i in idx:
            for j in idx:
                if i == j:
                    continue
                if rng.random() < p_intra:
                    mi[i, j] = True
    # callosal inter
    for i in range(0, n_hemi):
        for j in range(n_hemi, n):
            if rng.random() < p_inter:
                mj[i, j] = True
                delays[i, j] = int(rng.integers(1, dmax + 1))
                esign[i, j] = 1.0 if rng.random() < 0.9 else -1.0
            if rng.random() < p_inter:
                mj[j, i] = True
                delays[j, i] = int(rng.integers(1, dmax + 1))
                esign[j, i] = 1.0 if rng.random() < 0.9 else -1.0
    # visual RH / motor LH parcels (fake units)
    vis_rh = np.arange(n_hemi, min(n_hemi + max(4, n // 16), n), dtype=np.int64)
    mot_lh = np.arange(0, max(4, n // 16), dtype=np.int64)
    nnz = int(mi.sum() + mj.sum())
    return dict(
        n_hidden=n,
        n_hemi_units=n_hemi,
        n_local=1,
        mask_intra=mi,
        mask_inter=mj,
        delays=delays,
        esign_mat=esign,
        vis_rh_units=vis_rh,
        mot_lh_units=mot_lh,
        nnz=nnz,
        n_bridge=int(mj.sum()),
        dmax=dmax,
        meta={"synthetic": True, "seed": seed},
    )


def dense_masked_forward_once(wiring, w_intra_dense, w_inter_dense, x, buf, leak, Win, b, esign):
    """Reference dense matmul used in unit tests."""
    mi = torch.as_tensor(wiring["mask_intra"].astype(np.float32))
    N = wiring["n_hidden"]
    rec = x @ (w_intra_dense * mi).t()
    dmax = int(wiring.get("dmax", 5))
    delays = wiring["delays"]
    mj = wiring["mask_inter"].astype(bool)
    for d in range(1, dmax + 1):
        md = torch.as_tensor(((delays == d) & mj).astype(np.float32))
        if float(md.sum()) < 1:
            continue
        mag = F.softplus(w_inter_dense) * md * torch.as_tensor(esign)
        rec = rec + buf[d] @ mag.t()
    return rec

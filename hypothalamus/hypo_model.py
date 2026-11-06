"""Hypothalamus motif modulators on engine_v3 OscRNN (flags off).

Does not modify engine_v3 defaults; subclasses OscRNN and overrides forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_v3.rate_network import OscRNN, OscRNNConfig


@dataclass
class HypoModConfig:
    arm: str = "BASE"
    beta: float = 0.05
    kappa: float = 0.5
    s_star: float = 0.3
    inhibit: float = 0.5
    temperature: float = 1.0
    n_pi: int = 8
    k_syn: int = 32
    seed: int = 0


class HypoOscRNN(OscRNN):
    """OscRNN + A/B/C modulators. All engine feature flags stay False."""

    def __init__(self, wiring: dict, n_in: int, out_dim: int, hypo: HypoModConfig, **kwargs):
        cfg = kwargs.pop("cfg", None) or OscRNNConfig(leak=kwargs.get("leak", 0.4))
        # Force biology feature flags off — motifs only.
        cfg.use_stp = False
        cfg.use_dual_tau = False
        cfg.use_dale = False
        cfg.use_homeostasis = False
        kwargs["cfg"] = cfg
        super().__init__(wiring, n_in, out_dim, **kwargs)
        self.hypo = hypo
        arm = hypo.arm.upper()
        self.arm = arm

        n = self.n
        n_hemi = self.n_hemi
        rng = np.random.default_rng(hypo.seed + 4242)

        # Partitions for Test B
        left = np.arange(0, n_hemi, dtype=np.int64)
        right = np.arange(n_hemi, n, dtype=np.int64)
        if arm == "SHUF_RIVAL":
            perm = rng.permutation(n)
            half = n // 2
            part0 = perm[:half]
            part1 = perm[half:]
        else:
            part0, part1 = left, right
        self.register_buffer("part0", torch.from_numpy(np.asarray(part0, dtype=np.int64)))
        self.register_buffer("part1", torch.from_numpy(np.asarray(part1, dtype=np.int64)))

        # PI stub for Test C: first n_pi left-hemi units
        pi = left[: max(1, min(hypo.n_pi, len(left)))]
        self.register_buffer("pi_idx", torch.from_numpy(np.asarray(pi, dtype=np.int64)))

        # SYN edges: K random existing edges (intra ∪ inter)
        intra_s = self.intra_src.detach().cpu().numpy()
        intra_d = self.intra_dst.detach().cpu().numpy()
        inter_s = self.inter_src.detach().cpu().numpy()
        inter_d = self.inter_dst.detach().cpu().numpy()
        if len(intra_s) or len(inter_s):
            all_s = np.concatenate([intra_s, inter_s]) if len(intra_s) and len(inter_s) else (
                intra_s if len(intra_s) else inter_s
            )
            all_d = np.concatenate([intra_d, inter_d]) if len(intra_d) and len(inter_d) else (
                intra_d if len(intra_d) else inter_d
            )
        else:
            k0 = int(min(hypo.k_syn, n))
            all_s = np.arange(k0, dtype=np.int64)
            all_d = (all_s + 1) % n
        K = int(min(hypo.k_syn, len(all_s)))
        if K < 1:
            K = 1
            all_s = np.array([0], dtype=np.int64)
            all_d = np.array([min(1, n - 1)], dtype=np.int64)
        pick = rng.choice(len(all_s), size=K, replace=False)
        self.register_buffer("syn_src", torch.from_numpy(all_s[pick].astype(np.int64)))
        self.register_buffer("syn_dst", torch.from_numpy(all_d[pick].astype(np.int64)))
        self.k_syn_eff = K

        # Slow states (non-learned)
        self.register_buffer("_e", torch.zeros(1))
        self.register_buffer("_v", torch.zeros(1))  # err variance EMA for RAND_MOD
        self.register_buffer("_d0", torch.zeros(1))
        self.register_buffer("_d1", torch.zeros(1))
        self.register_buffer("_g0", torch.ones(1))
        self.register_buffer("_g1", torch.ones(1))
        self._prev_loss = None

    def reset_modulator_state(self, batch_size: int = 1, device=None, dtype=None):
        """Reset per-sequence activity filters; preserve Test-B drive gains."""
        device = device or self._e.device
        dtype = dtype or torch.float32
        self._e = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        self._v = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        # Preserve d0/d1/g0/g1 across forwards (updated from train loss).
        self._d0 = self._d0.detach().to(device=device, dtype=dtype).reshape(1)
        self._d1 = self._d1.detach().to(device=device, dtype=dtype).reshape(1)
        self._g0 = self._g0.detach().to(device=device, dtype=dtype).reshape(1)
        self._g1 = self._g1.detach().to(device=device, dtype=dtype).reshape(1)

    def update_drives_from_loss(self, loss_value: float):
        """Test B: update rivalry / single drives from train loss (outside forward)."""
        if self.arm not in ("RIVAL", "SINGLE", "SHUF_RIVAL"):
            return
        beta = float(self.hypo.beta)
        gamma = float(self.hypo.inhibit)
        kappa = float(self.hypo.kappa)
        tau = float(self.hypo.temperature)
        loss = float(loss_value)
        prev = self._prev_loss
        hunger = loss
        explore = max(0.0, (prev - loss) if prev is not None else 0.0)
        self._prev_loss = loss

        if self.arm == "SINGLE":
            d0 = (1 - beta) * float(self._d0.item()) + beta * hunger
            d0 = float(np.clip(d0, -10, 10))
            self._d0.fill_(d0)
            self._g0.fill_(1.0 + kappa * float(np.tanh(d0)))
            self._g1.fill_(1.0)
            return

        d0 = float(self._d0.item())
        d1 = float(self._d1.item())
        d0 = (1 - beta) * d0 + beta * (hunger - gamma * d1)
        d1 = (1 - beta) * d1 + beta * (explore - gamma * d0)
        d0 = float(np.clip(d0, -10, 10))
        d1 = float(np.clip(d1, -10, 10))
        self._d0.fill_(d0)
        self._d1.fill_(d1)
        # softmax rivalry → affine gains mean≈1
        logits = np.array([d0, d1], dtype=np.float64) / max(tau, 1e-6)
        logits -= logits.max()
        ex = np.exp(logits)
        sm = ex / ex.sum()
        g0 = 1.0 + kappa * (2.0 * sm[0] - 1.0)
        g1 = 1.0 + kappa * (2.0 * sm[1] - 1.0)
        self._g0.fill_(float(g0))
        self._g1.fill_(float(g1))

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
        head = -1
        self.reset_modulator_state(B, device, dtype)
        # keep drive gains from train loop (scalars) — re-broadcast
        g0 = self._g0.to(device=device, dtype=dtype)
        g1 = self._g1.to(device=device, dtype=dtype)

        beta = float(self.hypo.beta)
        kappa = float(self.hypo.kappa)
        s_star = float(self.hypo.s_star)
        arm = self.arm

        outs = []
        for t in range(T):
            head = (head + 1) % (self.dmax + 1)
            hist[head] = x
            rec = self._recurrent(x, hist, head)

            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b

            # ---- apply modulators to drive / pre ----
            if arm in ("HOME_O_ERR", "RAND_MOD"):
                # g from previous step's e (init g=1)
                g = 1.0 + kappa * torch.tanh(self._e)  # (B,1)
                drive = drive * g
            elif arm in ("RIVAL", "SHUF_RIVAL"):
                # partition gains from train-updated buffers
                gain = torch.ones(B, self.n, device=device, dtype=dtype)
                gain[:, self.part0] = g0
                gain[:, self.part1] = g1
                drive = drive * gain
            elif arm == "SINGLE":
                drive = drive * g0

            pre = rec + drive

            if arm == "VOL":
                delta = kappa * torch.tanh(self._e)  # (B,1)
                pre = pre + delta
            elif arm == "SYN":
                delta = kappa * torch.tanh(self._e)  # (B,1)
                # volumetric L2 target: ||delta * 1_N||_2 = |delta| * sqrt(N)
                # synaptic: messages along K edges, scale to match mean L2
                src_act = x[:, self.syn_src]  # (B, K)
                # raw msgs proportional to src * delta
                raw = src_act * delta  # (B, K)
                add = torch.zeros_like(pre)
                add.index_add_(1, self.syn_dst, raw)
                # match L2: scale so ||add||_2 ≈ ||delta||_2 * sqrt(N)
                vol_l2 = delta.abs().squeeze(-1) * (self.n ** 0.5) + 1e-8  # (B,)
                syn_l2 = add.norm(dim=1) + 1e-8
                scale = (vol_l2 / syn_l2).unsqueeze(-1)  # (B,1)
                # detach scale to avoid fighting the graph oddly; still applies matched amp
                pre = pre + add * scale.detach()

            if self.cfg.nan_guard:
                pre = torch.nan_to_num(pre, nan=0.0, posinf=20.0, neginf=-20.0)
            x = (1 - self.leak) * x + self.leak * torch.tanh(pre)
            if self.cfg.nan_guard:
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

            # ---- update slow states from activity ----
            if arm in ("HOME_O_ERR", "RAND_MOD"):
                # ||h||_1 / N = mean(|x|) over units; keep per-batch
                mean_abs = x.abs().mean(dim=1, keepdim=True)  # (B,1)
                err = mean_abs - s_star
                self._v = (1 - beta) * self._v + beta * (err.detach() ** 2)
                if arm == "HOME_O_ERR":
                    self._e = (1 - beta) * self._e + beta * err.detach()
                else:
                    eps = torch.randn_like(err) * torch.sqrt(self._v + 1e-8)
                    self._e = (1 - beta) * self._e + beta * eps
                self._e = self._e.clamp(-10, 10)
            elif arm in ("VOL", "SYN"):
                m = x[:, self.pi_idx].mean(dim=1, keepdim=True)
                self._e = (1 - beta) * self._e + beta * m.detach()
                self._e = self._e.clamp(-10, 10)

            outs.append(self.readout(self._feat(x)))
            self._step_count += 1

        return torch.stack(outs, 0)

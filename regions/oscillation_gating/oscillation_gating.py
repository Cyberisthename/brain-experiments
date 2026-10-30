#!/usr/bin/env python3
"""Full powered CTC / oscillation-gating + myelin + corpus-callosum rate RNN.
Prereg: PREREG_oscillation_gating.md (FULL protocol; supersedes fast pilot).
"""
from __future__ import annotations
import os, json, time, math, argparse
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
DATA = HERE / "data"
MYELIN_NPZ = ROOT / "regions" / "hybrid_human" / "data" / "human_myelin.npz"

# ---- architecture (full protocol) ----
N_REG = 4
N_LOCAL = 40
N = N_REG * N_LOCAL  # 160
N_HEMI = N // 2
INTRA_DENS = 0.12
INTER_DENS = 0.02
PERIOD = 8
INTER_DELAY_FIXED = 2  # uniform delay for non-myelin modular arms
DMAX = 5
GRAPH_SEED = 7
XOR_D = 5
NARMA_ORDER = 10

TUNE_SEEDS = [90, 91]
EVAL_SEEDS = list(range(1100, 1112))  # n=12

CTC_ARMS = ["COHERENT", "INCOHERENT", "NO_GATE", "HUB", "RAND_MATCH"]
MYELIN_ARMS = ["MYELIN_ONLY", "MYELIN_COHERENT"]
CALLOSUM_ARMS = ["CALLOSUM", "RAND_BRIDGE", "NO_BRIDGE", "DENSE_BRIDGE"]
ARMS = CTC_ARMS + MYELIN_ARMS + CALLOSUM_ARMS
# Secondary follow-up (NOT in primary ARMS / not in main eval):
FLY_ARMS = ["FLY"]
FOLLOWUP_COMPARE_ARMS = ["NO_GATE", "MYELIN_ONLY", "CALLOSUM"]  # fair-compare baselines
PRIMARY_TASKS = ["XOR", "CHAR", "NARMA"]

DEFAULT_HP = dict(
    lr=3e-3, leak=0.4, epochs=45, steps_per_epoch=20,
    batch_xor=48, batch_char=40, batch_narma=32,
    seq_xor=36, seq_char=48, seq_narma=64,
    char_train_chars=30000, wd=0.0, emb_dim=32,
)

# Callosum bridge edge budget (homotopic pairs * 2 directions)
CALLOSUM_BUDGET = N_HEMI  # one directed pair per homotopic index → 2*N_HEMI directed edges if reciprocal
# We use reciprocal homotopic: left[i]->right[i] and right[i]->left[i] => 2*N_HEMI edges.


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def load_h01_delay_pool(dmax=DMAX):
    """Return delay samples (1..dmax) from real H01 myelin_frac → delay map."""
    z = np.load(MYELIN_NPZ, allow_pickle=True)
    mf = z["myelin_frac"].astype(np.float64)
    med = float(np.nanmedian(mf))
    if not np.isfinite(med):
        med = 0.0
    mf = np.where(np.isfinite(mf), mf, med)
    mf = np.clip(mf, 0.0, 1.0)
    delays = (1 + np.rint((1.0 - mf) * (dmax - 1))).astype(np.int32)
    return delays


_DELAY_POOL = None
def delay_pool():
    global _DELAY_POOL
    if _DELAY_POOL is None:
        _DELAY_POOL = load_h01_delay_pool()
    return _DELAY_POOL


def build_modular_masks(hub=False, seed=GRAPH_SEED):
    g = np.random.default_rng(seed)
    mask_intra = np.zeros((N, N), dtype=bool)
    mask_inter = np.zeros((N, N), dtype=bool)
    for r in range(N_REG):
        sl = slice(r * N_LOCAL, (r + 1) * N_LOCAL)
        block = g.random((N_LOCAL, N_LOCAL)) < INTRA_DENS
        np.fill_diagonal(block, False)
        mask_intra[sl, sl] = block
    pairs = [(i, j) for i in range(N_REG) for j in range(N_REG) if i != j]
    seen = set(); uniq = []
    for i, j in pairs:
        pre = np.arange(i * N_LOCAL, (i + 1) * N_LOCAL)
        post = np.arange(j * N_LOCAL, (j + 1) * N_LOCAL)
        n_e = max(1, int(round(INTER_DENS * N_LOCAL * N_LOCAL)))
        rr = g.choice(pre, size=n_e * 4, replace=True)
        cc = g.choice(post, size=n_e * 4, replace=True)
        got = 0
        for r, c in zip(rr, cc):
            r, c = int(r), int(c)
            k = r * N + c
            if k in seen or r == c:
                continue
            seen.add(k); uniq.append((r, c)); got += 1
            if got >= n_e:
                break
    target = len(uniq)
    if hub:
        g2 = np.random.default_rng(seed + 4242)
        uniq = []; seen = set()
        quotas = {
            "h2o": int(round(0.60 * target)),
            "o2h": int(round(0.25 * target)),
        }
        quotas["oo"] = max(0, target - quotas["h2o"] - quotas["o2h"])
        def add_edge(r, c):
            k = r * N + c
            if k in seen or r == c:
                return False
            seen.add(k); uniq.append((r, c)); return True
        hub_pre = np.arange(0, N_LOCAL)
        others = np.arange(N_LOCAL, N)
        for key, pre_pool, post_pool in (
            ("h2o", hub_pre, others),
            ("o2h", others, hub_pre),
        ):
            got = 0
            while got < quotas[key]:
                r = int(g2.choice(pre_pool)); c = int(g2.choice(post_pool))
                if add_edge(r, c):
                    got += 1
        nonhub_pairs = [(i, j) for i in range(1, N_REG) for j in range(1, N_REG) if i != j]
        got = 0
        while got < quotas["oo"] and nonhub_pairs:
            i, j = nonhub_pairs[int(g2.integers(0, len(nonhub_pairs)))]
            r = int(g2.integers(i * N_LOCAL, (i + 1) * N_LOCAL))
            c = int(g2.integers(j * N_LOCAL, (j + 1) * N_LOCAL))
            if add_edge(r, c):
                got += 1
        while len(uniq) < target:
            r = int(g2.integers(0, N)); c = int(g2.integers(0, N))
            add_edge(r, c)
        uniq = uniq[:target]
    for r, c in uniq:
        mask_inter[r, c] = True
    return mask_intra, mask_inter


def build_rand_mask(nnz_target, seed):
    g = np.random.default_rng(seed + 12345)
    mask = np.zeros((N, N), dtype=bool)
    got = 0; seen = set()
    while got < nnz_target:
        r = int(g.integers(0, N)); c = int(g.integers(0, N))
        if r == c:
            continue
        k = r * N + c
        if k in seen:
            continue
        seen.add(k); mask[r, c] = True; got += 1
    return mask


def assign_myelin_delays(mask_inter, seed, rule="RULE_FF_DEPTH"):
    """Per-edge delays on inter mask. RULE_FF_DEPTH: isotonic assign by |reg_pre-reg_post|
    using the real H01 delay multiset (same mapping as myelin_real_rep)."""
    pool = delay_pool().copy()
    g = np.random.default_rng(seed + 77701)
    pre, post = np.where(mask_inter)
    n_e = len(pre)
    delays = np.ones((N, N), dtype=np.int32)  # unused entries ignored
    if n_e == 0:
        return delays
    # sample a multiset of size n_e from real delays
    if len(pool) >= n_e:
        chosen = g.choice(pool, size=n_e, replace=False)
    else:
        chosen = g.choice(pool, size=n_e, replace=True)
    if rule == "RULE_FF_DEPTH":
        # depth proxy = region index; |Δreg| ranks long-range; H01 rho(mf,depth) was ≈0
        # but RULE_FF_DEPTH still recovered 60% of REAL−SHUF — use |Δreg| as depth key.
        # Positive corr with mf means deeper → higher mf → lower delay. rho_depth was slightly
        # negative for mf vs depth, so deeper → lower mf → higher delay.
        reg_pre = pre // N_LOCAL
        reg_post = post // N_LOCAL
        depth_key = np.abs(reg_pre - reg_post).astype(float) + g.uniform(-1e-6, 1e-6, n_e)
        order = np.argsort(depth_key, kind="mergesort")
        sorted_d = np.sort(chosen)  # small delays first
        # deeper (larger |Δ|) → larger delay (more demyelinated / slower), matching
        # negative mf–depth Spearman in H01 (deeper → less myelin → larger delay)
        assigned = np.empty(n_e, dtype=np.int32)
        assigned[order] = sorted_d  # small depth → small delay; large depth → large delay
    else:  # RANDOM from real pool
        assigned = chosen.astype(np.int32)
        g.shuffle(assigned)
    for k in range(n_e):
        delays[pre[k], post[k]] = int(assigned[k])
    return delays


def phases_for(arm):
    if arm == "INCOHERENT":
        return np.array([2 * math.pi * r / N_REG for r in range(N_REG)], dtype=np.float64)
    return np.zeros(N_REG, dtype=np.float64)


def precompute_gates(phases, period=PERIOD):
    t = np.arange(period, dtype=np.float64)
    env = 0.5 * (1.0 + np.cos(2 * math.pi * t[:, None] / period + phases[None, :]))
    gates = np.zeros((period, N_REG, N_REG), dtype=np.float64)
    for i in range(N_REG):
        for j in range(N_REG):
            if i == j:
                gates[:, i, j] = 1.0
                continue
            raw = env[:, i] * env[:, j]
            m = raw.mean()
            gates[:, i, j] = raw / m if m > 1e-8 else 1.0
    return gates.astype(np.float32)


def build_callosum_masks(arm, seed=GRAPH_SEED):
    """Two hemispheres with intra-hemi recurrent + cross bridge."""
    g = np.random.default_rng(seed + 909)
    mask_intra = np.zeros((N, N), dtype=bool)
    mask_inter = np.zeros((N, N), dtype=bool)  # bridge = "inter"
    # denser within each hemisphere
    for hemi, sl in enumerate((slice(0, N_HEMI), slice(N_HEMI, N))):
        block = g.random((N_HEMI, N_HEMI)) < INTRA_DENS
        np.fill_diagonal(block, False)
        mask_intra[sl, sl] = block

    budget = 2 * N_HEMI  # reciprocal homotopic count

    def add(r, c, seen, lst):
        k = r * N + c
        if k in seen or r == c:
            return False
        seen.add(k); lst.append((r, c)); return True

    if arm == "NO_BRIDGE":
        edges = []
    elif arm == "CALLOSUM":
        edges = []
        seen = set()
        for i in range(N_HEMI):
            add(i, N_HEMI + i, seen, edges)
            add(N_HEMI + i, i, seen, edges)
    elif arm == "RAND_BRIDGE":
        edges = []; seen = set()
        g2 = np.random.default_rng(seed + 4242)
        # same count as CALLOSUM, random cross-hemi only
        while len(edges) < budget:
            left = int(g2.integers(0, N_HEMI))
            right = int(g2.integers(N_HEMI, N))
            if g2.random() < 0.5:
                add(left, right, seen, edges)
            else:
                add(right, left, seen, edges)
    elif arm == "DENSE_BRIDGE":
        edges = []
        for i in range(N_HEMI):
            for j in range(N_HEMI):
                edges.append((i, N_HEMI + j))
                edges.append((N_HEMI + j, i))
    else:
        raise KeyError(arm)

    for r, c in edges:
        mask_inter[r, c] = True
    return mask_intra, mask_inter, len(edges)


def _modular_target_nnz():
    """nnz of the modular NO_GATE graph (size/param match target for FLY)."""
    mi, mj = build_modular_masks(hub=False, seed=GRAPH_SEED)
    return int(mi.sum() + mj.sum())


def build_fly_ei_sbm(n=N, seed=GRAPH_SEED, target_nnz=None):
    """Fly CX / EI_SBM wiring (FRAC_E + EI_DENS from bio_wired_net), dens-scaled to target_nnz.

    Same N as modular harness; Dale-signed edges; delays fixed to 1 (no gate).
    """
    import sys
    bio = str((HERE.parent / "bio_wired_net").resolve())
    if bio not in sys.path:
        sys.path.insert(0, bio)
    from bio_wired_net import FRAC_E, EI_DENS, _sample_block_edges

    if target_nnz is None:
        target_nnz = _modular_target_nnz()

    def _build(scale):
        g = np.random.default_rng(seed)
        n_E = int(round(FRAC_E * n))
        sign = np.ones(n, dtype=np.int8)
        sign[n_E:] = -1
        g.shuffle(sign)
        idx_E = np.where(sign == 1)[0]
        idx_I = np.where(sign == -1)[0]
        class_idx = {1: idx_E, -1: idx_I}
        rows_l, cols_l, esign_l = [], [], []
        for a in (1, -1):
            for b in (1, -1):
                dens = min(0.95, float(EI_DENS[(a, b)]) * scale)
                rr, cc = _sample_block_edges(g, class_idx[a], class_idx[b], dens)
                rows_l.append(rr); cols_l.append(cc)
                esign_l.append(np.full(len(rr), a, dtype=np.int8))
        rows = np.concatenate(rows_l); cols = np.concatenate(cols_l)
        esign = np.concatenate(esign_l)
        key = rows.astype(np.int64) * n + cols
        _, keep = np.unique(key, return_index=True)
        return rows[keep], cols[keep], esign[keep], sign

    # binary-search dens scale so nnz matches modular
    lo, hi = 0.5, 3.0
    best = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        rows, cols, esign, sign = _build(mid)
        nnz = len(rows)
        if best is None or abs(nnz - target_nnz) < abs(best[0] - target_nnz):
            best = (nnz, mid, rows, cols, esign, sign)
        if nnz == target_nnz:
            break
        if nnz < target_nnz:
            lo = mid
        else:
            hi = mid
    nnz, scale, rows, cols, esign, sign = best
    # If still off by a few, top-up / trim while preserving Dale signs
    if nnz < target_nnz:
        g = np.random.default_rng(seed + 777)
        seen = set(int(r) * n + int(c) for r, c in zip(rows, cols))
        extra_r, extra_c, extra_e = [], [], []
        while len(rows) + len(extra_r) < target_nnz:
            r = int(g.integers(0, n)); c = int(g.integers(0, n))
            if r == c:
                continue
            k = r * n + c
            if k in seen:
                continue
            seen.add(k); extra_r.append(r); extra_c.append(c); extra_e.append(int(sign[r]))
        if extra_r:
            rows = np.concatenate([rows, np.asarray(extra_r, dtype=np.int64)])
            cols = np.concatenate([cols, np.asarray(extra_c, dtype=np.int64)])
            esign = np.concatenate([esign, np.asarray(extra_e, dtype=np.int8)])
    elif nnz > target_nnz:
        g = np.random.default_rng(seed + 777)
        keep = g.choice(nnz, size=target_nnz, replace=False)
        keep.sort()
        rows, cols, esign = rows[keep], cols[keep], esign[keep]

    mask = np.zeros((n, n), dtype=bool)
    mask[rows, cols] = True
    esign_mat = np.zeros((n, n), dtype=np.float32)
    esign_mat[rows, cols] = esign.astype(np.float32)
    delays = np.ones((n, n), dtype=np.int32)  # FLY: unit delay (no myelin / no gate)
    return dict(
        rows=rows, cols=cols, esign=esign, sign=sign, mask=mask,
        esign_mat=esign_mat, delays=delays, nnz=int(mask.sum()), dens_scale=float(scale),
    )


def make_wiring(arm, seed_eval):
    if arm == "FLY":
        fly = build_fly_ei_sbm(n=N, seed=GRAPH_SEED, target_nnz=_modular_target_nnz())
        mi = np.zeros((N, N), dtype=bool)
        mj = fly["mask"]
        return dict(
            arm=arm, mask_intra=mi, mask_inter=mj, use_gate=False,
            phases=np.zeros(N_REG), nnz=int(fly["nnz"]), kind="fly",
            delays=fly["delays"], split_input=False,
            esign_mat=fly["esign_mat"], dens_scale=fly["dens_scale"],
            frac_E=float((fly["sign"] == 1).mean()),
        )
    if arm in CALLOSUM_ARMS:
        mi, mj, n_bridge = build_callosum_masks(arm, seed=GRAPH_SEED)
        # callosum arms: no oscillation gate; fixed delay on bridge
        delays = np.full((N, N), INTER_DELAY_FIXED, dtype=np.int32)
        return dict(
            arm=arm, mask_intra=mi, mask_inter=mj, use_gate=False,
            phases=np.zeros(N_REG), nnz=int(mi.sum() + mj.sum()),
            kind="callosum", delays=delays, n_bridge=n_bridge,
            split_input=True,
        )
    if arm == "RAND_MATCH":
        mi, mj = build_modular_masks(hub=False, seed=GRAPH_SEED)
        nnz = int(mi.sum() + mj.sum())
        mask = build_rand_mask(nnz, seed_eval)
        delays = np.full((N, N), INTER_DELAY_FIXED, dtype=np.int32)
        return dict(
            arm=arm, mask_intra=np.zeros((N, N), dtype=bool),
            mask_inter=mask, use_gate=False, phases=np.zeros(N_REG),
            nnz=nnz, kind="rand", delays=delays, split_input=False,
        )
    hub = arm == "HUB"
    mi, mj = build_modular_masks(hub=hub, seed=GRAPH_SEED)
    use_gate = arm in ("COHERENT", "INCOHERENT", "HUB", "MYELIN_COHERENT")
    ph = phases_for(arm if arm != "MYELIN_COHERENT" else "COHERENT")
    if arm in MYELIN_ARMS:
        delays = assign_myelin_delays(mj, seed_eval, rule="RULE_FF_DEPTH")
    else:
        delays = np.full((N, N), INTER_DELAY_FIXED, dtype=np.int32)
        # still only apply delay on inter edges
        delays = delays * mj.astype(np.int32)
        delays[delays == 0] = 1  # placeholder; masked out
        delays = np.where(mj, INTER_DELAY_FIXED, 1).astype(np.int32)
    return dict(
        arm=arm, mask_intra=mi, mask_inter=mj, use_gate=use_gate,
        phases=ph, nnz=int(mi.sum() + mj.sum()), kind="modular",
        delays=delays, split_input=False,
    )


class OscRNN(nn.Module):
    def __init__(self, wiring, n_in, leak, out_dim, emb_dim=0, vocab=0, split_input=False):
        super().__init__()
        self.leak = leak
        self.n = N
        self.use_emb = emb_dim > 0 and vocab > 0
        self.use_gate = bool(wiring["use_gate"])
        self.kind = wiring["kind"]
        self.dmax = DMAX
        self.split_input = bool(split_input or wiring.get("split_input"))
        if self.use_emb:
            self.emb = nn.Embedding(vocab, emb_dim)
            n_in_eff = emb_dim
        else:
            self.emb = None
            n_in_eff = n_in

        mi = wiring["mask_intra"].astype(np.float32)
        mj = wiring["mask_inter"].astype(np.float32)
        self.register_buffer("mask_intra", torch.from_numpy(mi))
        self.register_buffer("mask_inter", torch.from_numpy(mj))
        delays = wiring["delays"].astype(np.int64)
        self.register_buffer("delays", torch.from_numpy(delays))
        reg = np.arange(N) // N_LOCAL
        self.register_buffer("reg_ids", torch.from_numpy(reg.astype(np.int64)))

        if self.use_gate:
            gates = precompute_gates(wiring["phases"])
            self.register_buffer("gates", torch.from_numpy(gates))
        else:
            self.register_buffer("gates", torch.ones(PERIOD, N_REG, N_REG))

        self.W_intra = nn.Parameter(0.08 * torch.randn(N, N))
        self.W_inter = nn.Parameter(0.08 * torch.randn(N, N))
        if self.kind == "rand":
            self.W_intra = nn.Parameter(torch.zeros(N, N))
            self.W_inter = nn.Parameter(0.08 * torch.randn(N, N))
        if self.kind == "fly":
            # Dale-constrained magnitudes: softplus(W_inter) * esign; no intra mask
            self.W_intra = nn.Parameter(torch.zeros(N, N))
            self.W_inter = nn.Parameter(torch.full((N, N), -1.5))  # softplus(-1.5)~0.20
            esm = wiring.get("esign_mat")
            if esm is None:
                esm = np.ones((N, N), dtype=np.float32)
            self.register_buffer("esign_mat", torch.from_numpy(np.asarray(esm, dtype=np.float32)))

        self.Win = nn.Parameter(0.6 * torch.randn(N, n_in_eff))
        if self.split_input and n_in_eff >= 2 and not self.use_emb:
            # left←ch0, right←ch1
            with torch.no_grad():
                self.Win[N_HEMI:, 0] = 0
                self.Win[:N_HEMI, 1] = 0
        elif self.split_input and self.use_emb:
            # CHAR callosum: right half of units get no direct drive
            with torch.no_grad():
                self.Win[N_HEMI:, :] = 0
        elif self.split_input and n_in_eff == 1:
            with torch.no_grad():
                self.Win[N_HEMI:, :] = 0

        self.b = nn.Parameter(torch.zeros(N))
        self.readout = nn.Linear(N, out_dim)

        # precompute per-delay inter masks for speed
        for d in range(1, DMAX + 1):
            md = (wiring["delays"] == d) & wiring["mask_inter"]
            self.register_buffer(f"mask_d{d}", torch.from_numpy(md.astype(np.float32)))

    def _recurrent(self, x, buf, t_step):
        """buf: (DMAX+1, B, N) with buf[0]=current x after roll write; buf[d]=x_{t-d}."""
        if self.kind == "rand":
            W = self.W_inter * self.mask_inter
            # use fixed delay INTER_DELAY_FIXED for rand
            d = INTER_DELAY_FIXED
            return buf[d] @ W.t()
        if self.kind == "fly":
            # All edges in mask_inter; unit delay; Dale signs via softplus * esign
            mag = F.softplus(self.W_inter) * self.mask_inter
            W = mag * self.esign_mat
            return buf[1] @ W.t()

        Wi = self.W_intra * self.mask_intra
        rec = x @ Wi.t()
        if self.use_gate:
            g = self.gates[t_step % PERIOD]
            rp = self.reg_ids
            G = g[rp[:, None], rp[None, :]]
        else:
            G = None

        for d in range(1, self.dmax + 1):
            md = getattr(self, f"mask_d{d}")
            if float(md.sum()) < 1:
                continue
            Wj = self.W_inter * md
            if G is not None:
                Wj = Wj * G
            rec = rec + buf[d] @ Wj.t()
        return rec

    def forward_sequence(self, u):
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
        for t in range(T):
            buf = torch.roll(buf, 1, dims=0)
            buf[0] = x
            rec = self._recurrent(x, buf, t)
            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            outs.append(self.readout(x))
        return torch.stack(outs, 0)


def count_params(m):
    n = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name == "W_intra":
            if getattr(m, "kind", None) == "fly":
                continue  # unused for FLY
            n += int(m.mask_intra.sum().item())
        elif name == "W_inter":
            n += int(m.mask_inter.sum().item())
        else:
            n += p.numel()
    return n


_CHAR_CACHE = {}
def load_alice():
    if "t" not in _CHAR_CACHE:
        text = (DATA / "alice.txt").read_text(errors="ignore")
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        _CHAR_CACHE.update(text=text, chars=chars, stoi=stoi, V=len(chars))
    return _CHAR_CACHE


def make_xor_batch(d, seq_len, batch, rng, split=False):
    """If split: 2-channel instantaneous XOR (callosum). Else delayed single-bit XOR."""
    T = seq_len
    if split:
        u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
        y = np.logical_xor(u[:, :, 0:1] > 0.5, u[:, :, 1:2] > 0.5).astype(np.float32)
        mask = np.ones((T, batch), dtype=np.float32)
    else:
        u = rng.integers(0, 2, size=(T, batch, 1)).astype(np.float32)
        y = np.zeros((T, batch, 1), dtype=np.float32)
        for t in range(d, T):
            y[t] = np.logical_xor(u[t] > 0.5, u[t - d] > 0.5).astype(np.float32)
        mask = np.zeros((T, batch), dtype=np.float32); mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_char_batch(seq_len, batch, start_limit, rng):
    C = load_alice()
    text = C["text"][:start_limit]
    stoi = C["stoi"]
    u = np.zeros((seq_len, batch), dtype=np.int64)
    y = np.zeros((seq_len, batch), dtype=np.int64)
    for b in range(batch):
        s = 0 if len(text) <= seq_len + 1 else int(rng.integers(0, len(text) - seq_len - 1))
        chunk = text[s:s + seq_len + 1]
        for t in range(seq_len):
            u[t, b] = stoi[chunk[t]]
            y[t, b] = stoi[chunk[t + 1]]
    return torch.from_numpy(u), torch.from_numpy(y)


def make_narma_batch(seq_len, batch, rng):
    T = seq_len
    u = rng.uniform(0, 0.5, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for b in range(batch):
        for i in range(NARMA_ORDER, T - 1):
            y[i + 1, b, 0] = (
                0.3 * y[i, b, 0]
                + 0.05 * y[i, b, 0] * y[i - 9:i + 1, b, 0].sum()
                + 1.5 * u[i - 9, b, 0] * u[i, b, 0]
                + 0.1
            )
    mask = np.zeros((T, batch), dtype=np.float32)
    mask[20:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def eval_metric(model, task, seed, hp, split=False, n_batches=4):
    rng = np.random.default_rng(seed)
    metrics = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng, split=split)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); target = y.squeeze(-1); m = mask
                acc = ((pred == target).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "CHAR":
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
                metrics.append(float(ce.item()))
            else:  # NARMA NRMSE
                u, y, mask = make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
                pred = model.forward_sequence(u).squeeze(-1)
                yt = y.squeeze(-1); m = mask
                mse = ((pred - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
                # std of target on masked steps
                yt_m = yt[m > 0]
                std = yt_m.std().clamp_min(1e-8)
                nrmse = torch.sqrt(mse) / std
                metrics.append(float(nrmse.item()))
    return float(np.mean(metrics))


def train_task(arm, task, seed, hp, epochs=None, record_curve=True):
    set_seed(seed)
    epochs = epochs if epochs is not None else hp["epochs"]
    wiring = make_wiring(arm, seed)
    split = bool(wiring.get("split_input"))
    if task == "XOR":
        n_in = 2 if split else 1
        out_dim, emb_dim, vocab = 1, 0, 0
    elif task == "CHAR":
        C = load_alice(); n_in, out_dim = hp["emb_dim"], C["V"]
        emb_dim, vocab = hp["emb_dim"], C["V"]
    elif task == "NARMA":
        n_in, out_dim, emb_dim, vocab = 1, 1, 0, 0
    else:
        raise KeyError(task)
    model = OscRNN(wiring, n_in, hp["leak"], out_dim, emb_dim=emb_dim, vocab=vocab, split_input=split)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    curve = []; rng = np.random.default_rng(seed + 17); best = None
    steps = int(hp.get("steps_per_epoch", 1))
    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng, split=split)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "CHAR":
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            else:
                u, y, mask = make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
                pred = model.forward_sequence(u).squeeze(-1)
                loss = ((pred - y.squeeze(-1)) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            last_loss = loss
        metric = eval_metric(model, task, seed + 1000 + ep, hp, split=split, n_batches=2)
        curve.append({"epoch": ep, "train_loss": float(last_loss.item()), "metric": metric})
        if task == "XOR":
            best = metric if best is None else max(best, metric)
        else:
            best = metric if best is None else min(best, metric)
    final = eval_metric(model, task, seed + 777, hp, split=split, n_batches=6)
    out = dict(
        arm=arm, task=task, seed=seed, metric=final, best_epoch_metric=best,
        n_params=count_params(model), nnz=int(wiring["nnz"]), n_hidden=N,
        kind=wiring["kind"], split_input=split,
        n_bridge=int(wiring.get("n_bridge", -1)),
        curve=curve if record_curve else [],
    )
    if wiring.get("kind") == "fly":
        out["dens_scale"] = float(wiring.get("dens_scale", float("nan")))
        out["frac_E"] = float(wiring.get("frac_E", float("nan")))
    return out


def tune():
    """Tune until NO_GATE and RAND_MATCH clearly learn XOR above chance on tune seeds."""
    grid = [
        dict(DEFAULT_HP, lr=1e-3, leak=0.4, epochs=30, steps_per_epoch=12),
        dict(DEFAULT_HP, lr=3e-3, leak=0.4, epochs=30, steps_per_epoch=12),
        dict(DEFAULT_HP, lr=3e-3, leak=0.3, epochs=30, steps_per_epoch=12),
        dict(DEFAULT_HP, lr=5e-3, leak=0.4, epochs=30, steps_per_epoch=12),
    ]
    scores = []
    for i, hp in enumerate(grid):
        vals = []
        floor_ok = True
        for arm in ("NO_GATE", "RAND_MATCH"):
            for s in TUNE_SEEDS:
                r_xor = train_task(arm, "XOR", s, hp, epochs=hp["epochs"], record_curve=False)
                r_ch = train_task(arm, "CHAR", s, hp, epochs=min(12, hp["epochs"]), record_curve=False)
                print(f"  tune{i} {arm} seed{s} xor={r_xor['metric']:.3f} char={r_ch['metric']:.3f}", flush=True)
                # Floor: NO_GATE must clearly learn (>=0.70); RAND_MATCH above chance (>=0.55).
                # RAND_MATCH is structurally harder on delayed XOR — do not require 0.70.
                thr = 0.70 if arm == "NO_GATE" else 0.55
                if r_xor["metric"] < thr:
                    floor_ok = False
                vals.append(r_xor["metric"] - 0.05 * r_ch["metric"])
        mean_s = float(np.mean(vals))
        # hard reject if baselines can't learn XOR
        if not floor_ok:
            mean_s -= 10.0
        scores.append((mean_s, i, hp, floor_ok))
        print("tune", i, "score", mean_s, "floor_ok", floor_ok, flush=True)
    scores.sort(reverse=True, key=lambda x: x[0])
    best = dict(scores[0][2])
    # restore full epoch budget for eval
    best["epochs"] = DEFAULT_HP["epochs"]
    best["steps_per_epoch"] = DEFAULT_HP["steps_per_epoch"]
    (OUT / "hparams.json").write_text(json.dumps(best, indent=2))
    meta = dict(chosen_grid=scores[0][1], floor_ok=scores[0][3], tune_scores=[
        dict(i=i, score=s, floor_ok=f) for s, i, _, f in scores
    ])
    (OUT / "tune_meta.json").write_text(json.dumps(meta, indent=2))
    print("froze hparams", best, flush=True)
    print("tune_meta", meta, flush=True)
    return best


def load_hp():
    p = OUT / "hparams.json"
    return json.loads(p.read_text()) if p.exists() else dict(DEFAULT_HP)


def _worker(arm, task, seed, hp):
    t0 = time.time()
    r = train_task(arm, task, seed, hp, record_curve=True)
    r["wall_s"] = time.time() - t0
    return r


def run_eval(arms=None, tasks=None, seeds=None, workers=2):
    hp = load_hp()
    arms = arms or ARMS
    tasks = tasks or PRIMARY_TASKS
    seeds = seeds or EVAL_SEEDS
    jobs = [(a, t, s) for a in arms for t in tasks for s in seeds]
    jf = OUT / "runs.jsonl"
    done = set()
    if jf.exists():
        for line in jf.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["arm"], r["task"], r["seed"]))
    todo = [j for j in jobs if j not in done]
    print(f"jobs {len(jobs)} todo {len(todo)} workers {workers} hp={hp}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, a, t, s, hp): (a, t, s) for a, t, s in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with open(jf, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['arm']} {r['task']} seed={r['seed']} metric={r['metric']:.4f} {r['wall_s']:.1f}s", flush=True)


def run_followup_fly(workers=2):
    """Secondary labeled FLY arm run (same seeds/epochs/tasks as main eval).

    Writes to out/fly_followup_runs.jsonl — does NOT touch main runs.jsonl.
    """
    hp = load_hp()
    arms = list(FLY_ARMS)
    tasks = list(PRIMARY_TASKS)
    seeds = list(EVAL_SEEDS)
    jobs = [(a, t, s) for a in arms for t in tasks for s in seeds]
    jf = OUT / "fly_followup_runs.jsonl"
    done = set()
    if jf.exists():
        for line in jf.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["arm"], r["task"], r["seed"]))
    todo = [j for j in jobs if j not in done]
    # sanity: param match
    w = make_wiring("FLY", EVAL_SEEDS[0])
    w_ng = make_wiring("NO_GATE", EVAL_SEEDS[0])
    meta = dict(
        label="SECONDARY_FLY_FOLLOWUP",
        compare_against=FOLLOWUP_COMPARE_ARMS,
        fly_nnz=int(w["nnz"]), modular_nnz=int(w_ng["nnz"]),
        dens_scale=float(w.get("dens_scale", float("nan"))),
        frac_E=float(w.get("frac_E", float("nan"))),
        n_hidden=N, seeds=seeds, tasks=tasks, hp=hp,
        note="Fly CX/EI_SBM wiring (bio_wired_net FRAC_E+EI_DENS), dens-scaled to match modular nnz; "
             "Dale softplus*esign; delay=1; no oscillation gate; same OscRNN harness.",
    )
    (OUT / "fly_followup_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[FLY FOLLOWUP] jobs {len(jobs)} todo {len(todo)} workers {workers}", flush=True)
    print(f"[FLY FOLLOWUP] nnz fly={w['nnz']} modular={w_ng['nnz']} dens_scale={w.get('dens_scale'):.4f} "
          f"frac_E={w.get('frac_E'):.3f}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, a, t, s, hp): (a, t, s) for a, t, s in todo}
        for fut in as_completed(futs):
            r = fut.result()
            r["run_label"] = "SECONDARY_FLY_FOLLOWUP"
            with open(jf, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[FLY] {r['arm']} {r['task']} seed={r['seed']} metric={r['metric']:.4f} "
                  f"{r['wall_s']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--followup-fly", action="store_true",
                    help="Secondary FLY arm only → out/fly_followup_runs.jsonl (after main RESULTS)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-floor", action="store_true",
                    help="Quick check NO_GATE/RAND_MATCH XOR above chance with DEFAULT_HP")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True); FIGS.mkdir(exist_ok=True)
    if args.smoke:
        hp = dict(DEFAULT_HP, epochs=3, steps_per_epoch=4, batch_xor=16, batch_char=16, batch_narma=16)
        for arm in list(ARMS) + list(FLY_ARMS):
            for task in PRIMARY_TASKS:
                r = train_task(arm, task, 90, hp, epochs=3, record_curve=False)
                print("smoke", r["arm"], r["task"], f"metric={r['metric']:.4f}",
                      f"nnz={r['nnz']}", f"npar={r['n_params']}", f"kind={r['kind']}", flush=True)
        nz = {arm: make_wiring(arm, 90)["nnz"] for arm in list(ARMS) + list(FLY_ARMS)}
        print("nnz by arm", nz, flush=True)
        return
    if args.check_floor:
        hp = dict(DEFAULT_HP)
        for arm in ("NO_GATE", "RAND_MATCH"):
            for s in TUNE_SEEDS:
                r = train_task(arm, "XOR", s, hp, record_curve=False)
                print(f"FLOOR {arm} seed{s} xor={r['metric']:.3f}", flush=True)
        return
    if args.tune:
        tune()
    if args.eval:
        run_eval(args.arms, args.tasks, args.seeds, workers=args.workers)
    if args.followup_fly:
        run_followup_fly(workers=args.workers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Realistic corpus-callosum prior vs controls (left-only readout, bridge-required XOR).
Prereg: PREREG_realistic_callosum.md — freeze before seeds 2100–2111.
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
BRIDGE_NPZ = ROOT / "connectomes" / "human" / "griffa70" / "griffa70_scale68_bridge.npz"
CALLOSUM_NPZ = ROOT / "connectomes" / "human" / "griffa70" / "griffa70_scale68_callosum.npz"

N_REG = 4
N_LOCAL = 40
N = N_REG * N_LOCAL  # 160
N_HEMI = N // 2      # 80
INTRA_DENS = 0.12
INTER_DELAY_FIXED = 2
DMAX = 5
GRAPH_SEED = 7
XOR_D = 5
NARMA_ORDER = 10
BRIDGE_BUDGET = 2 * N_HEMI  # 160 directed

EVAL_SEEDS = list(range(2100, 2112))
ARMS = [
    "REAL_CALLOSUM",
    "REAL_CALLOSUM_SHUFFLED",
    "INDIVIDUAL_CALLOSUM",
    "CALLOSUM",
    "RAND_BRIDGE",
    "NO_BRIDGE",
    "DENSE_BRIDGE",
]
PRIMARY_TASKS = ["XOR", "CHAR", "NARMA"]

DEFAULT_HP = dict(
    lr=5e-3, leak=0.4, epochs=45, steps_per_epoch=20,
    batch_xor=48, batch_char=40, batch_narma=32,
    seq_xor=36, seq_char=48, seq_narma=64,
    char_train_chars=30000, wd=0.0, emb_dim=32,
)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


_BRIDGE = None
_CALLOSUM = None


def load_bridge():
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = dict(np.load(BRIDGE_NPZ, allow_pickle=True))
    return _BRIDGE


def load_callosum_subj():
    global _CALLOSUM
    if _CALLOSUM is None:
        _CALLOSUM = dict(np.load(CALLOSUM_NPZ, allow_pickle=True))
    return _CALLOSUM


def parcel_unit_map(n_units, n_parcels, seed=GRAPH_SEED):
    """Assign each unit to a parcel index in [0, n_parcels). Roughly equal counts."""
    g = np.random.default_rng(seed + 11)
    base = np.arange(n_units) % n_parcels
    # shuffle within to avoid strict stripe
    g.shuffle(base)
    return base.astype(np.int32)


def _sample_directed_from_inter(
    sc_inter, fc_inter, homo_mask, n_units, n_parcels, budget, rng, presence=None,
    target_frac_homo=None,
):
    """Build directed bridge edges from parcel inter matrix → unit graph.

    Preserve real EDGE-COUNT homotopic fraction (not weight-proportional, which
    oversamples strong mirror links). Within homo/hetero classes, sample ∝ SC weight.
    """
    u_map = parcel_unit_map(n_units, n_parcels)
    cands_h, cands_e = [], []
    for i in range(n_parcels):
        for j in range(n_parcels):
            w = float(sc_inter[i, j])
            if w <= 0:
                continue
            if presence is not None and not bool(presence[i, j]):
                continue
            f = float(fc_inter[i, j]) if np.isfinite(fc_inter[i, j]) else 0.0
            item = (i, j, w, f, bool(homo_mask[i, j]))
            if item[4]:
                cands_h.append(item)
            else:
                cands_e.append(item)
    if not cands_h and not cands_e:
        return [], [], [], [], []

    # target fraction from data if not given
    if target_frac_homo is None:
        n_h = len(cands_h); n_e = len(cands_e)
        target_frac_homo = n_h / max(1, n_h + n_e)
    n_homo_target = int(round(budget * target_frac_homo))
    n_homo_target = max(0, min(budget, n_homo_target))
    n_hetero_target = budget - n_homo_target

    def _draw(cands, n_need, seen, edges):
        if not cands or n_need <= 0:
            return
        wts = np.array([c[2] for c in cands], dtype=np.float64)
        wts = wts / wts.sum()
        attempts = 0
        max_attempts = n_need * 80
        while len([e for e in edges if e[4] == cands[0][4]]) < n_need and attempts < max_attempts:
            # count current of this class
            attempts += 1
            cur = sum(1 for e in edges if e[4] == cands[0][4])
            if cur >= n_need:
                break
            k = int(rng.choice(len(cands), p=wts))
            pi, pj, w, f, homo = cands[k]
            left_units = np.where(u_map == pi)[0]
            right_units = np.where(u_map == pj)[0]
            if len(left_units) == 0 or len(right_units) == 0:
                continue
            lu = int(rng.choice(left_units))
            ru = int(rng.choice(right_units)) + n_units
            r, c = (lu, ru) if rng.random() < 0.5 else (ru, lu)
            key = r * (2 * n_units) + c
            if key in seen:
                continue
            seen.add(key)
            sign = 1.0 if f >= 0 else -1.0
            edges.append((r, c, w, sign, homo))
        # top-up if short
        cur = sum(1 for e in edges if e[4] == cands[0][4])
        if cur < n_need:
            order = np.argsort([-c[2] for c in cands])
            for idx in order:
                if sum(1 for e in edges if e[4] == cands[0][4]) >= n_need:
                    break
                pi, pj, w, f, homo = cands[idx]
                left_units = np.where(u_map == pi)[0]
                right_units = np.where(u_map == pj)[0]
                for lu in left_units:
                    for ru0 in right_units:
                        if sum(1 for e in edges if e[4] == cands[0][4]) >= n_need:
                            break
                        for r, c in ((int(lu), int(ru0) + n_units), (int(ru0) + n_units, int(lu))):
                            if sum(1 for e in edges if e[4] == cands[0][4]) >= n_need:
                                break
                            key = r * (2 * n_units) + c
                            if key in seen:
                                continue
                            seen.add(key)
                            sign = 1.0 if f >= 0 else -1.0
                            edges.append((r, c, w, sign, homo))

    edges = []
    seen = set()
    if cands_h:
        _draw(cands_h, n_homo_target, seen, edges)
    if cands_e:
        _draw(cands_e, n_hetero_target, seen, edges)
    # if still short (e.g. not enough homo unit pairs), fill from hetero then homo
    if len(edges) < budget:
        for pool in (cands_e, cands_h):
            if len(edges) >= budget:
                break
            if not pool:
                continue
            _draw(pool, budget - len(edges) + sum(1 for e in edges if e[4] == pool[0][4]), seen, edges)
            # simpler fill:
            wts = np.array([c[2] for c in pool], dtype=np.float64); wts /= wts.sum()
            attempts = 0
            while len(edges) < budget and attempts < budget * 100:
                attempts += 1
                k = int(rng.choice(len(pool), p=wts))
                pi, pj, w, f, homo = pool[k]
                left_units = np.where(u_map == pi)[0]
                right_units = np.where(u_map == pj)[0]
                if len(left_units) == 0 or len(right_units) == 0:
                    continue
                lu = int(rng.choice(left_units)); ru = int(rng.choice(right_units)) + n_units
                r, c = (lu, ru) if rng.random() < 0.5 else (ru, lu)
                key = r * (2 * n_units) + c
                if key in seen:
                    continue
                seen.add(key)
                edges.append((r, c, w, 1.0 if f >= 0 else -1.0, homo))

    edges = edges[:budget]
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    ws = [e[2] for e in edges]
    signs = [e[3] for e in edges]
    homos = [e[4] for e in edges]
    return rows, cols, ws, signs, homos


def assign_delays_real(rows, cols, ws, homos, n, rng):
    """Heterotopic / weaker → larger delay in 1..DMAX."""
    delays = np.ones((n, n), dtype=np.int32)
    if not rows:
        return delays
    # score: hetero gets +1, plus rank by inverse weight
    scores = []
    for w, h in zip(ws, homos):
        scores.append((0.0 if h else 1.0) + (1.0 / (w + 1e-9)))
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)  # small score → fast
    # assign sorted delays 1..DMAX proportional
    n_e = len(rows)
    # multiset: linear from 1 to DMAX
    raw = 1 + np.floor(np.linspace(0, DMAX - 1 + 1e-9, n_e)).astype(np.int32)
    raw = np.clip(raw, 1, DMAX)
    assigned = np.empty(n_e, dtype=np.int32)
    assigned[order] = raw
    for k, (r, c) in enumerate(zip(rows, cols)):
        delays[r, c] = int(assigned[k])
    return delays


def build_intra(seed=GRAPH_SEED):
    g = np.random.default_rng(seed + 909)
    mask_intra = np.zeros((N, N), dtype=bool)
    for sl in (slice(0, N_HEMI), slice(N_HEMI, N)):
        block = g.random((N_HEMI, N_HEMI)) < INTRA_DENS
        np.fill_diagonal(block, False)
        mask_intra[sl, sl] = block
    return mask_intra


def edges_to_masks(rows, cols, signs, delays, mask_intra):
    mask_inter = np.zeros((N, N), dtype=bool)
    esign = np.ones((N, N), dtype=np.float32)
    for r, c, s in zip(rows, cols, signs):
        mask_inter[r, c] = True
        esign[r, c] = float(s)
    return mask_intra, mask_inter, esign, delays, len(rows)


def build_real_bridge(arm, seed_eval):
    mask_intra = build_intra()
    br = load_bridge()
    n_parcels = int(br["n_lh"])
    assert n_parcels == int(br["n_rh"])
    sc = br["sc_inter_mean"].astype(np.float64)
    fc = br["fc_inter_mean"].astype(np.float64)
    homo = br["homo_mask"].astype(bool)
    presence = br["majority_mask"].astype(bool)
    rng = np.random.default_rng(GRAPH_SEED + 4242)

    if arm == "INDIVIDUAL_CALLOSUM":
        sub = load_callosum_subj()
        sid = int((seed_eval - EVAL_SEEDS[0]) % 70)
        sc_full = sub["sc_subj"][:, :, sid].astype(np.float64)
        fc_full = sub["fc_subj"][:, :, sid].astype(np.float64)
        nL = n_parcels
        sc = np.maximum(sc_full[:nL, nL:], sc_full[nL:, :nL].T)
        fc = 0.5 * (fc_full[:nL, nL:] + fc_full[nL:, :nL].T)
        # rebuild homo from names
        names = [str(x) for x in sub["names"]]
        homo = np.zeros((nL, nL), dtype=bool)
        for i in range(nL):
            for j in range(nL):
                if names[i] == names[nL + j]:
                    homo[i, j] = True
        presence = sc > 0
        rng = np.random.default_rng(GRAPH_SEED + 4242 + sid)
        subject_id = sid
    else:
        subject_id = -1

    # Group majority frac_homo≈0.157; individuals use their own presence-derived frac
    tfrac = 0.157 if arm != "INDIVIDUAL_CALLOSUM" else None
    rows, cols, ws, signs, homos = _sample_directed_from_inter(
        sc, fc, homo, N_HEMI, n_parcels, BRIDGE_BUDGET, rng, presence=presence,
        target_frac_homo=tfrac,
    )
    delays = assign_delays_real(rows, cols, ws, homos, N, rng)

    meta = dict(
        n_bridge=len(rows),
        frac_homo=float(np.mean(homos)) if homos else 0.0,
        frac_exc=float(np.mean([s > 0 for s in signs])) if signs else 1.0,
        subject_id=subject_id,
        mean_w=float(np.mean(ws)) if ws else 0.0,
        mean_delay=float(np.mean([delays[r, c] for r, c in zip(rows, cols)])) if rows else 0.0,
    )

    if arm == "REAL_CALLOSUM_SHUFFLED":
        # permute right endpoints among bridge edges; keep weights/signs/delays multiset
        rng_s = np.random.default_rng(seed_eval + 99991)
        # collect right-side nodes appearing as targets or sources
        new_rows, new_cols = [], []
        # For each edge, keep source hemi, remap cross target to a random opposite-hemi unit
        for r, c, w, s, h in zip(rows, cols, ws, signs, homos):
            if r < N_HEMI:
                # left -> right: shuffle right target
                new_c = int(rng_s.integers(N_HEMI, N))
                new_rows.append(r); new_cols.append(new_c)
            else:
                # right -> left: shuffle left target
                new_c = int(rng_s.integers(0, N_HEMI))
                new_rows.append(r); new_cols.append(new_c)
        # rebuild delays with same multiset assigned to new edges in same order
        old_d = [int(delays[r, c]) for r, c in zip(rows, cols)]
        delays = np.ones((N, N), dtype=np.int32)
        for (r, c), d in zip(zip(new_rows, new_cols), old_d):
            delays[r, c] = d
        rows, cols = new_rows, new_cols
        # homos become meaningless after shuffle — keep signs/ws aligned by index
        meta["frac_homo"] = float("nan")
        meta["shuffled"] = True

    return edges_to_masks(rows, cols, signs, delays, mask_intra) + (meta,)


def build_classic_bridge(arm, seed=GRAPH_SEED):
    mask_intra = build_intra(seed)
    g = np.random.default_rng(seed + 4242)
    rows, cols, signs = [], [], []
    delays = np.full((N, N), INTER_DELAY_FIXED, dtype=np.int32)

    def add(r, c):
        rows.append(r); cols.append(c); signs.append(1.0)

    if arm == "NO_BRIDGE":
        pass
    elif arm == "CALLOSUM":
        for i in range(N_HEMI):
            add(i, N_HEMI + i)
            add(N_HEMI + i, i)
    elif arm == "RAND_BRIDGE":
        seen = set()
        while len(rows) < BRIDGE_BUDGET:
            left = int(g.integers(0, N_HEMI))
            right = int(g.integers(N_HEMI, N))
            r, c = (left, right) if g.random() < 0.5 else (right, left)
            k = r * N + c
            if k in seen or r == c:
                continue
            seen.add(k); add(r, c)
    elif arm == "DENSE_BRIDGE":
        for i in range(N_HEMI):
            for j in range(N_HEMI):
                add(i, N_HEMI + j)
                add(N_HEMI + j, i)
    else:
        raise KeyError(arm)

    meta = dict(
        n_bridge=len(rows),
        frac_homo=1.0 if arm == "CALLOSUM" else (0.0 if arm != "NO_BRIDGE" else float("nan")),
        frac_exc=1.0,
        subject_id=-1,
        mean_w=1.0,
        mean_delay=float(INTER_DELAY_FIXED),
    )
    return edges_to_masks(rows, cols, signs, delays, mask_intra) + (meta,)


def make_wiring(arm, seed_eval):
    if arm in ("REAL_CALLOSUM", "REAL_CALLOSUM_SHUFFLED", "INDIVIDUAL_CALLOSUM"):
        mi, mj, esign, delays, n_bridge, meta = build_real_bridge(arm, seed_eval)
    else:
        mi, mj, esign, delays, n_bridge, meta = build_classic_bridge(arm)
    return dict(
        arm=arm, mask_intra=mi, mask_inter=mj, esign_mat=esign,
        delays=delays, nnz=int(mi.sum() + mj.sum()), n_bridge=n_bridge,
        kind="callosum", split_input=True, left_only_readout=True,
        use_gate=False, meta=meta,
    )


class OscRNN(nn.Module):
    def __init__(self, wiring, n_in, leak, out_dim, emb_dim=0, vocab=0):
        super().__init__()
        self.leak = leak
        self.n = N
        self.use_emb = emb_dim > 0 and vocab > 0
        self.dmax = DMAX
        self.left_only = bool(wiring.get("left_only_readout", True))
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
        self.register_buffer("delays", torch.from_numpy(wiring["delays"].astype(np.int64)))
        esm = wiring.get("esign_mat")
        if esm is None:
            esm = np.ones((N, N), dtype=np.float32)
        self.register_buffer("esign_mat", torch.from_numpy(np.asarray(esm, dtype=np.float32)))

        self.W_intra = nn.Parameter(0.08 * torch.randn(N, N))
        self.W_inter = nn.Parameter(0.08 * torch.randn(N, N))
        self.Win = nn.Parameter(0.6 * torch.randn(N, n_in_eff))
        with torch.no_grad():
            if n_in_eff >= 2 and not self.use_emb:
                self.Win[N_HEMI:, 0] = 0
                self.Win[:N_HEMI, 1] = 0
            else:
                # CHAR / NARMA: drive left only
                self.Win[N_HEMI:, :] = 0
        self.b = nn.Parameter(torch.zeros(N))
        read_in = N_HEMI if self.left_only else N
        self.readout = nn.Linear(read_in, out_dim)
        self.left_only_dim = read_in

        for d in range(1, DMAX + 1):
            md = (wiring["delays"] == d) & wiring["mask_inter"]
            self.register_buffer(f"mask_d{d}", torch.from_numpy(md.astype(np.float32)))

    def _recurrent(self, x, buf, t_step):
        Wi = self.W_intra * self.mask_intra
        rec = x @ Wi.t()
        for d in range(1, self.dmax + 1):
            md = getattr(self, f"mask_d{d}")
            if float(md.sum()) < 1:
                continue
            Wj = self.W_inter * md * self.esign_mat
            # for inhibitory signs, keep magnitude positive via abs on trainable then * sign
            # simpler: W_inter can go negative; multiply by sign so Dale-ish preference
            # Use softplus magnitude * sign for stability on signed bridges
            mag = F.softplus(self.W_inter) * md
            Wj = mag * self.esign_mat
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
            feat = x[:, :N_HEMI] if self.left_only else x
            outs.append(self.readout(feat))
        return torch.stack(outs, 0)


def count_params(m):
    n = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name == "W_intra":
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


def make_xor_batch(d, seq_len, batch, rng):
    """Delayed hemispheric XOR: left←ch0, right←ch1; y_t = ch0_{t-d} XOR ch1_{t-d}."""
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
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
    mask = np.zeros((T, batch), dtype=np.float32); mask[20:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def eval_metric(model, task, seed, hp, n_batches=4):
    rng = np.random.default_rng(seed)
    metrics = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); target = y.squeeze(-1); m = mask
                acc = ((pred == target).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "CHAR":
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
                metrics.append(float(ce.item()))
            else:
                u, y, mask = make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
                pred = model.forward_sequence(u).squeeze(-1)
                yt = y.squeeze(-1); m = mask
                mse = ((pred - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
                std = yt[m > 0].std().clamp_min(1e-8)
                metrics.append(float((torch.sqrt(mse) / std).item()))
    return float(np.mean(metrics))


def train_task(arm, task, seed, hp, epochs=None, record_curve=True):
    set_seed(seed)
    epochs = epochs if epochs is not None else hp["epochs"]
    wiring = make_wiring(arm, seed)
    if task == "XOR":
        n_in, out_dim, emb_dim, vocab = 2, 1, 0, 0
    elif task == "CHAR":
        C = load_alice(); n_in, out_dim = hp["emb_dim"], C["V"]
        emb_dim, vocab = hp["emb_dim"], C["V"]
    elif task == "NARMA":
        n_in, out_dim, emb_dim, vocab = 1, 1, 0, 0
    else:
        raise KeyError(task)
    model = OscRNN(wiring, n_in, hp["leak"], out_dim, emb_dim=emb_dim, vocab=vocab)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    curve = []; rng = np.random.default_rng(seed + 17); best = None
    steps = int(hp.get("steps_per_epoch", 1))
    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "XOR":
                u, y, mask = make_xor_batch(XOR_D, hp["seq_xor"], hp["batch_xor"], rng)
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
        metric = eval_metric(model, task, seed + 1000 + ep, hp, n_batches=2)
        curve.append({"epoch": ep, "train_loss": float(last_loss.item()), "metric": metric})
        if task == "XOR":
            best = metric if best is None else max(best, metric)
        else:
            best = metric if best is None else min(best, metric)
    final = eval_metric(model, task, seed + 777, hp, n_batches=6)
    out = dict(
        arm=arm, task=task, seed=seed, metric=final, best_epoch_metric=best,
        n_params=count_params(model), nnz=int(wiring["nnz"]), n_hidden=N,
        n_bridge=int(wiring["n_bridge"]), left_only_readout=True,
        curve=curve if record_curve else [],
        bridge_meta=wiring.get("meta", {}),
    )
    return out


def load_hp():
    p = OUT / "hparams.json"
    if p.exists():
        return json.loads(p.read_text())
    # copy from oscillation_gating if present
    src = HERE.parent / "oscillation_gating" / "out" / "hparams.json"
    if src.exists():
        hp = json.loads(src.read_text())
        p.write_text(json.dumps(hp, indent=2))
        return hp
    p.write_text(json.dumps(DEFAULT_HP, indent=2))
    return dict(DEFAULT_HP)


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
            print(f"{r['arm']} {r['task']} seed={r['seed']} metric={r['metric']:.4f} "
                  f"bridge={r['n_bridge']} {r['wall_s']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-bridge", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.check_bridge:
        for arm in ARMS:
            w = make_wiring(arm, EVAL_SEEDS[0])
            print(arm, "nnz", w["nnz"], "n_bridge", w["n_bridge"], "meta", w["meta"], flush=True)
        return
    if args.smoke:
        hp = dict(DEFAULT_HP, epochs=2, steps_per_epoch=2, batch_xor=16, batch_char=16, batch_narma=16)
        for arm in ARMS:
            for task in PRIMARY_TASKS:
                r = train_task(arm, task, 90, hp, epochs=2, record_curve=False)
                print("smoke", r["arm"], r["task"], f"metric={r['metric']:.4f}",
                      f"bridge={r['n_bridge']}", f"npar={r['n_params']}", flush=True)
        return
    if args.eval:
        load_hp()
        run_eval(args.arms, args.tasks, args.seeds, workers=args.workers)


if __name__ == "__main__":
    main()

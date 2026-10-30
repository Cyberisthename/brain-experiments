#!/usr/bin/env python3
"""Whole-brain human SC prior vs matched controls (left-only readout, hard XOR).
Prereg: PREREG_human_wholebrain.md — freeze before seeds 2200–2207.
Data stays under connectomes/ (gitignored); never upload.
"""
from __future__ import annotations
import os, json, time, argparse
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
WHOLE_NPZ = ROOT / "connectomes" / "human" / "griffa70" / "griffa70_scale68_whole.npz"

N_PARCEL = 68
N_LOCAL = 2
N = N_PARCEL * N_LOCAL  # 136
N_LH = 34
N_HEMI_UNITS = N_LH * N_LOCAL  # 68
DMAX = 5
GRAPH_SEED = 7
NARMA_ORDER = 10
XOR_HARD_D = 12
XOR2BIT_D = 8

EVAL_SEEDS = list(range(2200, 2208))
ARMS = [
    "HUMAN_SC",
    "HUMAN_SC_MYELIN",
    "DEGREE_PRESERVING_SHUFFLE",
    "WEIGHT_SHUFFLE",
    "ER_RANDOM",
    "MYELIN_ONLY",
    "INDIVIDUAL",
]
PRIMARY_TASKS = ["XOR_HARD", "XOR2BIT", "CHAR", "NARMA"]

DEFAULT_HP = dict(
    lr=5e-3, leak=0.4, epochs=45, steps_per_epoch=20,
    batch_xor=48, batch_char=40, batch_narma=32,
    seq_xor=48, seq_char=48, seq_narma=64,
    char_train_chars=30000, wd=0.0, emb_dim=32,
)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


_WHOLE = None


def load_whole():
    global _WHOLE
    if _WHOLE is None:
        _WHOLE = dict(np.load(WHOLE_NPZ, allow_pickle=True))
    return _WHOLE


def region_of(u: int) -> int:
    return u // N_LOCAL


def units_of(r: int):
    return range(r * N_LOCAL, (r + 1) * N_LOCAL)


def same_hemi(i: int, j: int) -> bool:
    return (i < N_LH) == (j < N_LH)


def delay_from_length(length: float, lengths_pos: np.ndarray) -> int:
    """Map fiber length (mm) → delay in 1..DMAX by quantile bins."""
    if not np.isfinite(length) or length <= 0:
        return 1
    qs = np.quantile(lengths_pos, np.linspace(0, 1, DMAX + 1)[1:-1])
    d = 1
    for q in qs:
        if length > q:
            d += 1
    return int(min(DMAX, max(1, d)))


def consensus_region_graph():
    z = load_whole()
    fl = z["fiber_length"].astype(np.float64)
    sc = z["sc_group"].astype(np.float64)
    fc = z["fc_group"].astype(np.float64)
    topo = (fl > 0) & (~np.eye(N_PARCEL, dtype=bool))
    # weights on consensus edges; fill tiny if missing
    W = np.where(topo, np.maximum(sc, 0.0), 0.0)
    if (W[topo] <= 0).any():
        med = float(np.median(W[topo & (W > 0)])) if (W[topo] > 0).any() else 1.0
        W = np.where(topo & (W <= 0), med, W)
    signs = np.where(np.isfinite(fc), np.where(fc >= 0, 1.0, -1.0), 1.0)
    # Dale fallback on missing: 80% +
    missing = ~np.isfinite(fc)
    if missing.any():
        rng = np.random.default_rng(GRAPH_SEED + 3)
        signs = signs.copy()
        signs[missing] = np.where(rng.random(missing.sum()) < 0.8, 1.0, -1.0)
    lengths = np.where(topo, fl, 0.0)
    return topo.astype(bool), W, signs.astype(np.float64), lengths


def subject_region_graph(sid: int):
    z = load_whole()
    sc = z["sc_subj"][:, :, sid].astype(np.float64)
    fc = z["fc_subj"][:, :, sid].astype(np.float64)
    fl = z["fiber_length"].astype(np.float64)
    topo = (sc > 0) & (~np.eye(N_PARCEL, dtype=bool))
    # match consensus density approximately by keeping strongest edges if denser
    cons_topo, _, _, _ = consensus_region_graph()
    target_nnz = int(cons_topo.sum())
    if topo.sum() > target_nnz * 1.15:
        flat = sc.copy(); flat[~topo] = -1
        thresh = np.partition(flat.ravel(), -target_nnz)[-target_nnz]
        topo = (sc >= thresh) & (~np.eye(N_PARCEL, dtype=bool))
    W = np.where(topo, sc, 0.0)
    signs = np.where(np.isfinite(fc), np.where(fc >= 0, 1.0, -1.0), 1.0)
    lengths = np.where(topo, np.where(fl > 0, fl, np.nanmedian(fl[fl > 0])), 0.0)
    return topo.astype(bool), W, signs.astype(np.float64), lengths


def maslov_sneppen(topo: np.ndarray, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Degree-preserving rewire on undirected-ish directed adjacency (symmetric preferred)."""
    A = topo.copy()
    # work with upper triangle undirected edges then symmetrize
    rows, cols = np.where(np.triu(A, 1))
    edges = list(zip(rows.tolist(), cols.tolist()))
    if len(edges) < 2:
        return A
    for _ in range(n_swaps):
        if len(edges) < 2:
            break
        i1, i2 = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i1]; c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        # try a-d, c-b
        if (not A[a, d]) and (not A[c, b]) and a != d and c != b:
            A[a, b] = A[b, a] = False
            A[c, d] = A[d, c] = False
            A[a, d] = A[d, a] = True
            A[c, b] = A[b, c] = True
            edges[i1] = (min(a, d), max(a, d))
            edges[i2] = (min(c, b), max(c, b))
    np.fill_diagonal(A, False)
    return A


def expand_to_units(topo, W, signs, lengths, use_myelin: bool, rng: np.random.Generator):
    """Expand parcel graph → unit masks. Returns mi, mj, esign, delays, meta."""
    mi = np.zeros((N, N), dtype=np.float32)
    mj = np.zeros((N, N), dtype=np.float32)
    esign = np.ones((N, N), dtype=np.float32)
    delays = np.ones((N, N), dtype=np.int64)
    lengths_pos = lengths[lengths > 0]
    if lengths_pos.size == 0:
        lengths_pos = np.array([30.0])

    # within-parcel instant
    for r in range(N_PARCEL):
        us = list(units_of(r))
        for u in us:
            for v in us:
                if u != v:
                    mi[u, v] = 1.0

    n_cross = 0
    n_callosal = 0
    delay_list = []
    for i in range(N_PARCEL):
        for j in range(N_PARCEL):
            if not topo[i, j] or i == j:
                continue
            w = float(W[i, j])
            s = float(signs[i, j])
            if use_myelin:
                d = delay_from_length(float(lengths[i, j]), lengths_pos)
            else:
                d = 1
            callosal = not same_hemi(i, j)
            for u in units_of(i):
                for v in units_of(j):
                    if u == v:
                        continue
                    esign[u, v] = s
                    delays[u, v] = d
                    # short same-hemi → intra; else inter with delay
                    if (not callosal) and d <= 1:
                        mi[u, v] = 1.0
                    else:
                        mj[u, v] = 1.0
                        delay_list.append(d)
                    n_cross += 1
                    if callosal:
                        n_callosal += 1

    meta = dict(
        n_region_edges=int(topo.sum()),
        n_unit_cross=int(n_cross),
        n_callosal_unit=int(n_callosal),
        n_intra=int(mi.sum()),
        n_inter=int(mj.sum()),
        mean_delay=float(np.mean(delay_list)) if delay_list else 1.0,
        frac_exc=float((esign[mj > 0] > 0).mean()) if mj.sum() > 0 else float("nan"),
        subject_id=-1,
        use_myelin=bool(use_myelin),
    )
    return mi, mj, esign, delays, meta


def build_myelin_only(target_nnz_region: int, n_callosal_region: int, rng: np.random.Generator):
    """Two-hemi modular ER + sparse random callosal bridge + delay diversity."""
    topo = np.zeros((N_PARCEL, N_PARCEL), dtype=bool)
    # within-hemi dens from target
    n_within_possible = 2 * (N_LH * (N_LH - 1))  # directed both ways each hemi
    # use undirected then symmetrize
    n_within_undirected_target = max(1, (target_nnz_region - n_callosal_region) // 2)
    # LH
    for hemi_start in (0, N_LH):
        nodes = list(range(hemi_start, hemi_start + N_LH))
        possible = [(a, b) for a in nodes for b in nodes if a < b]
        rng.shuffle(possible)
        k = n_within_undirected_target // 2
        for a, b in possible[:k]:
            topo[a, b] = topo[b, a] = True
    # callosal undirected
    callosal_possible = [(i, j) for i in range(N_LH) for j in range(N_LH, N_PARCEL)]
    rng.shuffle(callosal_possible)
    k_c = max(1, n_callosal_region // 2)
    for a, b in callosal_possible[:k_c]:
        topo[a, b] = topo[b, a] = True
    # fake lengths: within short, callosal longer + jitter
    lengths = np.zeros((N_PARCEL, N_PARCEL), dtype=np.float64)
    W = np.zeros_like(lengths)
    signs = np.ones_like(lengths)
    for i in range(N_PARCEL):
        for j in range(N_PARCEL):
            if not topo[i, j]:
                continue
            if same_hemi(i, j):
                lengths[i, j] = float(rng.uniform(10, 40))
            else:
                lengths[i, j] = float(rng.uniform(40, 120))
            W[i, j] = float(rng.uniform(0.1, 1.0))
            signs[i, j] = 1.0 if rng.random() < 0.8 else -1.0
    return topo, W, signs, lengths


def make_wiring(arm: str, seed_eval: int):
    rng = np.random.default_rng(GRAPH_SEED + 1000 * (hash(arm) % 10007) + seed_eval)
    cons_topo, cons_W, cons_S, cons_L = consensus_region_graph()
    target_nnz = int(cons_topo.sum())
    n_callosal = int(sum(
        1 for i in range(N_PARCEL) for j in range(N_PARCEL)
        if cons_topo[i, j] and not same_hemi(i, j)
    ))

    if arm == "HUMAN_SC":
        topo, W, S, L = cons_topo, cons_W, cons_S, cons_L
        use_myelin = False
        sid = -1
    elif arm == "HUMAN_SC_MYELIN":
        topo, W, S, L = cons_topo, cons_W, cons_S, cons_L
        use_myelin = True
        sid = -1
    elif arm == "DEGREE_PRESERVING_SHUFFLE":
        topo = maslov_sneppen(cons_topo, n_swaps=max(2000, 10 * target_nnz), rng=rng)
        # reuse weight/sign pools on new edges by random assignment from originals
        pool_w = cons_W[cons_topo]
        pool_s = cons_S[cons_topo]
        pool_l = cons_L[cons_topo]
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        idxs = np.where(topo)
        n = len(idxs[0])
        pick = rng.integers(0, len(pool_w), size=n)
        W[idxs] = pool_w[pick]; S[idxs] = pool_s[pick]; L[idxs] = pool_l[pick]
        use_myelin = True
        sid = -1
    elif arm == "WEIGHT_SHUFFLE":
        topo = cons_topo.copy()
        pool_w = cons_W[cons_topo].copy(); rng.shuffle(pool_w)
        pool_s = cons_S[cons_topo].copy(); rng.shuffle(pool_s)
        pool_l = cons_L[cons_topo].copy(); rng.shuffle(pool_l)
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        W[topo] = pool_w; S[topo] = pool_s; L[topo] = pool_l
        use_myelin = True
        sid = -1
    elif arm == "ER_RANDOM":
        dens = target_nnz / (N_PARCEL * (N_PARCEL - 1))
        topo = rng.random((N_PARCEL, N_PARCEL)) < dens
        np.fill_diagonal(topo, False)
        # symmetrize roughly to match undirected SC style
        topo = np.logical_or(topo, topo.T)
        # trim to target nnz
        if topo.sum() > target_nnz:
            rows, cols = np.where(np.triu(topo, 1))
            order = rng.permutation(len(rows))
            keep = order[: target_nnz // 2]
            new = np.zeros_like(topo)
            for k in keep:
                a, b = int(rows[k]), int(cols[k])
                new[a, b] = new[b, a] = True
            topo = new
        pool_w = cons_W[cons_topo]; pool_s = cons_S[cons_topo]; pool_l = cons_L[cons_topo]
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        idxs = np.where(topo)
        pick = rng.integers(0, len(pool_w), size=len(idxs[0]))
        W[idxs] = pool_w[pick]; S[idxs] = pool_s[pick]; L[idxs] = pool_l[pick]
        use_myelin = True
        sid = -1
    elif arm == "MYELIN_ONLY":
        topo, W, S, L = build_myelin_only(target_nnz, n_callosal, rng)
        use_myelin = True
        sid = -1
    elif arm == "INDIVIDUAL":
        sid = int(seed_eval % 70)
        topo, W, S, L = subject_region_graph(sid)
        use_myelin = True
    else:
        raise KeyError(arm)

    mi, mj, esign, delays, meta = expand_to_units(topo, W, S, L, use_myelin, rng)
    meta["subject_id"] = sid
    meta["arm"] = arm
    nnz = int(mi.sum() + mj.sum())
    return dict(
        arm=arm, mask_intra=mi, mask_inter=mj, esign_mat=esign,
        delays=delays, nnz=nnz, n_bridge=int(meta["n_callosal_unit"]),
        kind="human_wholebrain", split_input=True, left_only_readout=True,
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
                # hemispheric split: even channels → left, odd → right (for 2 or 4)
                self.Win[N_HEMI_UNITS:, 0::2] = 0
                self.Win[:N_HEMI_UNITS, 1::2] = 0
            else:
                self.Win[N_HEMI_UNITS:, :] = 0
        self.b = nn.Parameter(torch.zeros(N))
        read_in = N_HEMI_UNITS if self.left_only else N
        self.readout = nn.Linear(read_in, out_dim)

        for d in range(1, DMAX + 1):
            md = (wiring["delays"] == d) & (wiring["mask_inter"].astype(bool))
            self.register_buffer(f"mask_d{d}", torch.from_numpy(md.astype(np.float32)))

    def _recurrent(self, x, buf, t_step):
        Wi = self.W_intra * self.mask_intra
        rec = x @ Wi.t()
        for d in range(1, self.dmax + 1):
            md = getattr(self, f"mask_d{d}")
            if float(md.sum()) < 1:
                continue
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
            feat = x[:, :N_HEMI_UNITS] if self.left_only else x
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


def make_xor_hard_batch(d, seq_len, batch, rng):
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32); mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_xor2bit_batch(d, seq_len, batch, rng):
    """Two independent delayed XORs: L←(a0,a1), R←(b0,b1); y=(a0⊕b0, a1⊕b1) at lag d."""
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 4)).astype(np.float32)
    y = np.zeros((T, batch, 2), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
        y[t, :, 1] = np.logical_xor(u[t - d, :, 2] > 0.5, u[t - d, :, 3] > 0.5).astype(np.float32)
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
            if task == "XOR_HARD":
                u, y, mask = make_xor_hard_batch(XOR_HARD_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); target = y.squeeze(-1); m = mask
                acc = ((pred == target).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "XOR2BIT":
                u, y, mask = make_xor2bit_batch(XOR2BIT_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u)
                pred = (logits > 0).float(); m = mask.unsqueeze(-1).expand_as(pred)
                acc = ((pred == y).float() * m).sum() / m.sum().clamp_min(1.0)
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
    if task == "XOR_HARD":
        n_in, out_dim, emb_dim, vocab = 2, 1, 0, 0
    elif task == "XOR2BIT":
        n_in, out_dim, emb_dim, vocab = 4, 2, 0, 0
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
    higher_better = task in ("XOR_HARD", "XOR2BIT")
    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "XOR_HARD":
                u, y, mask = make_xor_hard_batch(XOR_HARD_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "XOR2BIT":
                u, y, mask = make_xor2bit_batch(XOR2BIT_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u)
                loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                loss = (loss * mask.unsqueeze(-1)).sum() / mask.unsqueeze(-1).sum().clamp_min(1.0)
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
        if higher_better:
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
                  f"nnz={r['nnz']} {r['wall_s']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-wiring", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.check_wiring:
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
                      f"nnz={r['nnz']}", f"npar={r['n_params']}", flush=True)
        return
    if args.eval:
        load_hp()
        run_eval(args.arms, args.tasks, args.seeds, workers=args.workers)


if __name__ == "__main__":
    main()

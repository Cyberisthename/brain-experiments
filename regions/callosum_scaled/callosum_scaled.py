#!/usr/bin/env python3
"""Scaled callosum / whole-SC prior vs matched controls (transfer tasks).
Prereg: PREREG_callosum_scaled.md — freeze commit before seeds 2300–2307.
Data stays under connectomes/ (gitignored); never upload.
NaN-safe NARMA: human_wholebrain/FIX_NOTES.md.
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
N_LH = 34
DMAX = 5
GRAPH_SEED = 7
NARMA_ORDER = 10
XOR_HARD_D = 12
SPLIT_BIND_D = 8
D_MATCH = 12

# Actual N = 68 * N_LOCAL
N_LABEL_TO_LOCAL = {512: 8, 1024: 15, 2048: 30}
DEFAULT_N_LABELS = [512, 1024]  # 2048 staged / deferred per prereg

EVAL_SEEDS = list(range(2300, 2308))
PRIMARY_ARMS = [
    "REAL_SC",
    "DEGREE_SHUFFLE",
    "WEIGHT_SHUFFLE",
    "SPARSE_BRIDGE",
    "NO_BRIDGE",
    "DENSE",
]
EXPLORATORY_ARMS = ["AGCC"]
ARMS = PRIMARY_ARMS + EXPLORATORY_ARMS
PRIMARY_TASKS = ["XOR_HARD", "CHAR", "NARMA", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"]

# Lausanne indices (0-based within 0..33, same offset +34 for RH)
VISUAL_LH = [20, 21, 22, 23]       # cuneus, pericalcarine, latOcc, lingual
MOTOR_LH = [9, 10]                # precentral, paracentral
VISUAL_RH = [i + N_LH for i in VISUAL_LH]
MOTOR_RH = [i + N_LH for i in MOTOR_LH]

# Caminiti 2013 Method3 velocities (m/s) → scale delay ∝ 1/v
CAMINITI_V = {
    "Prefrontal": 7.83,
    "Motor": 9.76,
    "Parietal": 7.67,
    "Temporal": 8.5,   # interpolate (not in Method3 table; mid)
    "Visual": 9.81,
    "Cingulate": 8.0,
    "Insula": 8.0,
}
V_REF = 8.5  # reference velocity for delay scaling

DEFAULT_HP = dict(
    lr=5e-3, leak=0.4, epochs=30, steps_per_epoch=12,
    batch_xor=32, batch_char=32, batch_narma=24, batch_xfer=32,
    seq_xor=40, seq_char=40, seq_narma=56, seq_xfer=40,
    char_train_chars=30000, wd=0.0, emb_dim=32, eval_every=3,
)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


_WHOLE = None
_SECTOR = None


def load_whole():
    global _WHOLE
    if _WHOLE is None:
        _WHOLE = dict(np.load(WHOLE_NPZ, allow_pickle=True))
    return _WHOLE


def sector_of_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("orbitofrontal", "frontalpole", "parstriangularis",
                            "parsorbitalis", "parsopercularis", "rostralmiddlefrontal",
                            "superiorfrontal", "caudalmiddlefrontal")):
        return "Prefrontal"
    if any(k in n for k in ("precentral", "paracentral")):
        return "Motor"
    if any(k in n for k in ("postcentral", "supramarginal", "superiorparietal",
                            "inferiorparietal", "precuneus")):
        return "Parietal"
    if any(k in n for k in ("cuneus", "pericalcarine", "lateraloccipital", "lingual")):
        return "Visual"
    if "cingulate" in n or "isthmus" in n:
        return "Cingulate"
    if "insula" in n:
        return "Insula"
    if any(k in n for k in ("temporal", "fusiform", "parahippocampal", "entorhinal",
                            "bankssts")):
        return "Temporal"
    return "Prefrontal"


def parcel_sectors():
    global _SECTOR
    if _SECTOR is None:
        z = load_whole()
        names = [str(x) for x in z["names"]]
        _SECTOR = [sector_of_name(n) for n in names]
    return _SECTOR


def same_hemi(i: int, j: int) -> bool:
    return (i < N_LH) == (j < N_LH)


def units_of(r: int, n_local: int):
    return range(r * n_local, (r + 1) * n_local)


def delay_from_length(length: float, lengths_pos: np.ndarray, speed_scale: float = 1.0) -> int:
    """Map fiber length → delay in 1..DMAX by quantile bins, then scale by 1/speed."""
    if not np.isfinite(length) or length <= 0:
        return 1
    qs = np.quantile(lengths_pos, np.linspace(0, 1, DMAX + 1)[1:-1])
    d = 1
    for q in qs:
        if length > q:
            d += 1
    # faster axons → shorter delay; speed_scale = v / V_REF
    if speed_scale > 0 and speed_scale != 1.0:
        d = int(round(d / speed_scale))
    return int(min(DMAX, max(1, d)))


def edge_speed_scale(i: int, j: int) -> float:
    """Caminiti sector velocity geometric mean / V_REF; callosal FA bump ~1.15."""
    sec = parcel_sectors()
    vi = CAMINITI_V.get(sec[i], V_REF)
    vj = CAMINITI_V.get(sec[j], V_REF)
    v = (vi * vj) ** 0.5
    scale = v / V_REF
    if not same_hemi(i, j):
        # FA callosal ~1.30× other WM → delay *= 1/1.15 (mild; see delay_prior_suggestions)
        scale *= 1.15
    return float(scale)


def consensus_region_graph():
    z = load_whole()
    fl = z["fiber_length"].astype(np.float64)
    sc = z["sc_group"].astype(np.float64)
    fc = z["fc_group"].astype(np.float64)
    topo = (fl > 0) & (~np.eye(N_PARCEL, dtype=bool))
    W = np.where(topo, np.maximum(sc, 0.0), 0.0)
    if (W[topo] <= 0).any():
        med = float(np.median(W[topo & (W > 0)])) if (W[topo] > 0).any() else 1.0
        W = np.where(topo & (W <= 0), med, W)
    signs = np.where(np.isfinite(fc), np.where(fc >= 0, 1.0, -1.0), 1.0)
    missing = ~np.isfinite(fc)
    if missing.any():
        rng = np.random.default_rng(GRAPH_SEED + 3)
        signs = signs.copy()
        signs[missing] = np.where(rng.random(missing.sum()) < 0.8, 1.0, -1.0)
    lengths = np.where(topo, fl, 0.0)
    return topo.astype(bool), W, signs.astype(np.float64), lengths


def maslov_sneppen(topo: np.ndarray, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    A = topo.copy()
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
        if (not A[a, d]) and (not A[c, b]) and a != d and c != b:
            A[a, b] = A[b, a] = False
            A[c, d] = A[d, c] = False
            A[a, d] = A[d, a] = True
            A[c, b] = A[b, c] = True
            edges[i1] = (min(a, d), max(a, d))
            edges[i2] = (min(c, b), max(c, b))
    np.fill_diagonal(A, False)
    return A


def expand_to_units(topo, W, signs, lengths, n_local: int, use_delays: bool,
                    rng: np.random.Generator, force_callosal_delay: int | None = None,
                    unit_edges_per_region: int | None = None):
    """Expand parcel graph → units.

    Within-parcel: all-to-all instant.
    Cross-parcel: **sparse** bio2art-style expansion — each region edge becomes
    `unit_edges_per_region` (default = n_local) random directed unit edges, not a
    full n_local×n_local block. Keeps N large while nnz ~ O(E_region * n_local).
    """
    N = N_PARCEL * n_local
    mi = np.zeros((N, N), dtype=np.float32)
    mj = np.zeros((N, N), dtype=np.float32)
    esign = np.ones((N, N), dtype=np.float32)
    delays = np.ones((N, N), dtype=np.int64)
    lengths_pos = lengths[lengths > 0]
    if lengths_pos.size == 0:
        lengths_pos = np.array([30.0])
    k_edges = int(unit_edges_per_region) if unit_edges_per_region else int(n_local)

    for r in range(N_PARCEL):
        us = list(units_of(r, n_local))
        for u in us:
            for v in us:
                if u != v:
                    mi[u, v] = 1.0

    n_cross = 0
    n_callosal = 0
    delay_list = []
    for i in range(N_PARCEL):
        src_u = list(units_of(i, n_local))
        for j in range(N_PARCEL):
            if not topo[i, j] or i == j:
                continue
            s = float(signs[i, j])
            callosal = not same_hemi(i, j)
            if use_delays:
                if force_callosal_delay is not None and callosal:
                    d = int(force_callosal_delay)
                else:
                    d = delay_from_length(float(lengths[i, j]), lengths_pos,
                                          speed_scale=edge_speed_scale(i, j))
            else:
                d = 1
            dst_u = list(units_of(j, n_local))
            # sample k directed unit edges (with replacement then unique)
            n_need = min(k_edges, len(src_u) * len(dst_u))
            seen = set()
            tries = 0
            while len(seen) < n_need and tries < n_need * 40:
                tries += 1
                u = int(rng.choice(src_u)); v = int(rng.choice(dst_u))
                if u == v or (u, v) in seen:
                    continue
                seen.add((u, v))
            for u, v in seen:
                esign[u, v] = s
                delays[u, v] = d
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
        use_delays=bool(use_delays),
        n_local=int(n_local),
        n_hidden=int(N),
    )
    return mi, mj, esign, delays, meta


def build_sparse_bridge(target_nnz_region: int, n_callosal_region: int, rng: np.random.Generator):
    topo = np.zeros((N_PARCEL, N_PARCEL), dtype=bool)
    n_within_undirected_target = max(1, (target_nnz_region - n_callosal_region) // 2)
    for hemi_start in (0, N_LH):
        nodes = list(range(hemi_start, hemi_start + N_LH))
        possible = [(a, b) for a in nodes for b in nodes if a < b]
        rng.shuffle(possible)
        k = n_within_undirected_target // 2
        for a, b in possible[:k]:
            topo[a, b] = topo[b, a] = True
    callosal_possible = [(i, j) for i in range(N_LH) for j in range(N_LH, N_PARCEL)]
    rng.shuffle(callosal_possible)
    k_c = max(1, n_callosal_region // 2)
    for a, b in callosal_possible[:k_c]:
        topo[a, b] = topo[b, a] = True
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


def make_wiring(arm: str, seed_eval: int, n_local: int):
    rng = np.random.default_rng(GRAPH_SEED + 1000 * (abs(hash(arm)) % 10007) + seed_eval + n_local)
    cons_topo, cons_W, cons_S, cons_L = consensus_region_graph()
    target_nnz = int(cons_topo.sum())
    n_callosal = int(sum(
        1 for i in range(N_PARCEL) for j in range(N_PARCEL)
        if cons_topo[i, j] and not same_hemi(i, j)
    ))
    force_cd = None

    if arm == "REAL_SC":
        topo, W, S, L = cons_topo, cons_W, cons_S, cons_L
    elif arm == "DEGREE_SHUFFLE":
        # Rewire WITHIN intra-LH, intra-RH, and callosal classes separately so
        # bridge edge count / hemi modularity stay matched (research note 2:
        # "degree-preserving shuffle of inter + intra"). Full-graph MS would
        # flood the bridge and confound the map contrast.
        topo = np.zeros_like(cons_topo)
        for mask_fn in (
            lambda i, j: same_hemi(i, j) and i < N_LH and j < N_LH,
            lambda i, j: same_hemi(i, j) and i >= N_LH and j >= N_LH,
            lambda i, j: not same_hemi(i, j),
        ):
            sub = np.zeros_like(cons_topo)
            for i in range(N_PARCEL):
                for j in range(N_PARCEL):
                    if cons_topo[i, j] and mask_fn(i, j):
                        sub[i, j] = True
            n_sub = int(sub.sum())
            if n_sub >= 4:
                sub = maslov_sneppen(sub, n_swaps=max(500, 10 * n_sub), rng=rng)
            topo |= sub
        # reassign pools WITHIN class so delay multiset per class matches REAL
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        for mask_fn in (
            lambda i, j: same_hemi(i, j) and i < N_LH and j < N_LH,
            lambda i, j: same_hemi(i, j) and i >= N_LH and j >= N_LH,
            lambda i, j: not same_hemi(i, j),
        ):
            src_ij = [(i, j) for i in range(N_PARCEL) for j in range(N_PARCEL)
                      if cons_topo[i, j] and mask_fn(i, j)]
            dst_ij = [(i, j) for i in range(N_PARCEL) for j in range(N_PARCEL)
                      if topo[i, j] and mask_fn(i, j)]
            if not src_ij or not dst_ij:
                continue
            pool_w = np.array([cons_W[i, j] for i, j in src_ij])
            pool_s = np.array([cons_S[i, j] for i, j in src_ij])
            pool_l = np.array([cons_L[i, j] for i, j in src_ij])
            pick = rng.integers(0, len(pool_w), size=len(dst_ij))
            for k, (i, j) in enumerate(dst_ij):
                W[i, j] = pool_w[pick[k]]; S[i, j] = pool_s[pick[k]]; L[i, j] = pool_l[pick[k]]
    elif arm == "WEIGHT_SHUFFLE":
        topo = cons_topo.copy()
        pool_w = cons_W[cons_topo].copy(); rng.shuffle(pool_w)
        pool_s = cons_S[cons_topo].copy(); rng.shuffle(pool_s)
        pool_l = cons_L[cons_topo].copy(); rng.shuffle(pool_l)
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        W[topo] = pool_w; S[topo] = pool_s; L[topo] = pool_l
    elif arm == "SPARSE_BRIDGE":
        topo, W, S, L = build_sparse_bridge(target_nnz, n_callosal, rng)
    elif arm == "NO_BRIDGE":
        topo = cons_topo.copy()
        for i in range(N_PARCEL):
            for j in range(N_PARCEL):
                if not same_hemi(i, j):
                    topo[i, j] = False
        W = np.where(topo, cons_W, 0.0)
        S = np.where(topo, cons_S, 1.0)
        L = np.where(topo, cons_L, 0.0)
    elif arm == "DENSE":
        topo = cons_topo.copy()
        for i in range(N_LH):
            for j in range(N_LH, N_PARCEL):
                topo[i, j] = topo[j, i] = True
        # keep within-hemi from consensus; callosal all-to-all
        for i in range(N_PARCEL):
            for j in range(N_PARCEL):
                if same_hemi(i, j) and not cons_topo[i, j]:
                    topo[i, j] = False
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        # fill from pools
        pool_w = cons_W[cons_topo]; pool_s = cons_S[cons_topo]; pool_l = cons_L[cons_topo]
        idxs = np.where(topo)
        pick = rng.integers(0, len(pool_w), size=len(idxs[0]))
        W[idxs] = pool_w[pick]; S[idxs] = pool_s[pick]; L[idxs] = pool_l[pick]
        # longer callosal lengths
        for i in range(N_PARCEL):
            for j in range(N_PARCEL):
                if topo[i, j] and not same_hemi(i, j):
                    L[i, j] = float(rng.uniform(80, 140))
        force_cd = 3
    elif arm == "AGCC":
        # no callosum + boost intra (~1.5× within edges)
        topo = cons_topo.copy()
        for i in range(N_PARCEL):
            for j in range(N_PARCEL):
                if not same_hemi(i, j):
                    topo[i, j] = False
        for hemi_start in (0, N_LH):
            nodes = list(range(hemi_start, hemi_start + N_LH))
            possible = [(a, b) for a in nodes for b in nodes if a < b and not topo[a, b]]
            rng.shuffle(possible)
            n_add = max(1, int(0.5 * sum(1 for a, b in
                        [(x, y) for x in nodes for y in nodes if x < y and topo[x, y]])))
            for a, b in possible[:n_add]:
                topo[a, b] = topo[b, a] = True
        W = np.zeros_like(cons_W); S = np.ones_like(cons_S); L = np.zeros_like(cons_L)
        pool_w = cons_W[cons_topo]; pool_s = cons_S[cons_topo]; pool_l = cons_L[cons_topo]
        idxs = np.where(topo)
        pick = rng.integers(0, len(pool_w), size=len(idxs[0]))
        W[idxs] = pool_w[pick]; S[idxs] = pool_s[pick]; L[idxs] = pool_l[pick]
    else:
        raise KeyError(arm)

    mi, mj, esign, delays, meta = expand_to_units(
        topo, W, S, L, n_local, use_delays=True, rng=rng, force_callosal_delay=force_cd
    )
    meta["arm"] = arm
    N = N_PARCEL * n_local
    nnz = int(mi.sum() + mj.sum())

    # unit index lists for specialized readouts / inputs
    vis_rh_u = [u for r in VISUAL_RH for u in units_of(r, n_local)]
    mot_lh_u = [u for r in MOTOR_LH for u in units_of(r, n_local)]
    return dict(
        arm=arm, mask_intra=mi, mask_inter=mj, esign_mat=esign,
        delays=delays, nnz=nnz, n_bridge=int(meta["n_callosal_unit"]),
        n_hidden=N, n_local=n_local, n_hemi_units=N_LH * n_local,
        vis_rh_units=np.asarray(vis_rh_u, dtype=np.int64),
        mot_lh_units=np.asarray(mot_lh_u, dtype=np.int64),
        kind="callosum_scaled", left_only_readout=True,
        meta=meta,
    )


class OscRNN(nn.Module):
    """Sparse-edge OscRNN: softplus only on existing inter edges (speed at large N)."""

    def __init__(self, wiring, n_in, leak, out_dim, emb_dim=0, vocab=0,
                 readout_mode="left_hemi", input_mode="hemi_split"):
        super().__init__()
        self.leak = leak
        self.n = int(wiring["n_hidden"])
        self.n_hemi = int(wiring["n_hemi_units"])
        self.use_emb = emb_dim > 0 and vocab > 0
        self.dmax = DMAX
        self.readout_mode = readout_mode
        self.input_mode = input_mode
        if self.use_emb:
            self.emb = nn.Embedding(vocab, emb_dim)
            n_in_eff = emb_dim
        else:
            self.emb = None
            n_in_eff = n_in

        mi = wiring["mask_intra"].astype(bool)
        mj = wiring["mask_inter"].astype(bool)
        esign = np.asarray(wiring.get("esign_mat", np.ones((self.n, self.n))), dtype=np.float32)
        delays = wiring["delays"].astype(np.int64)

        # Intra edges (instant)
        ii, jj = np.where(mi)
        self.register_buffer("intra_src", torch.from_numpy(ii.astype(np.int64)))
        self.register_buffer("intra_dst", torch.from_numpy(jj.astype(np.int64)))
        self.w_intra = nn.Parameter(0.08 * torch.randn(len(ii)))

        # Inter edges (delayed), packed
        ii, jj = np.where(mj)
        self.register_buffer("inter_src", torch.from_numpy(ii.astype(np.int64)))
        self.register_buffer("inter_dst", torch.from_numpy(jj.astype(np.int64)))
        self.register_buffer("inter_delay", torch.from_numpy(delays[ii, jj]))
        self.register_buffer("inter_sign", torch.from_numpy(esign[ii, jj]))
        self.w_inter = nn.Parameter(0.08 * torch.randn(len(ii)))

        # Keep dense masks only for param counting / meta compatibility
        self.register_buffer("mask_intra", torch.from_numpy(mi.astype(np.float32)))
        self.register_buffer("mask_inter", torch.from_numpy(mj.astype(np.float32)))
        self.register_buffer("vis_rh", torch.from_numpy(wiring["vis_rh_units"]))
        self.register_buffer("mot_lh", torch.from_numpy(wiring["mot_lh_units"]))

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
                idx = wiring["vis_rh_units"]
                self.Win[idx, :] = 0.8 * torch.randn(len(idx), n_in_eff)
            elif input_mode == "delayed_match":
                idx = wiring["vis_rh_units"]
                self.Win[idx, 0] = 0.8 * torch.randn(len(idx))
                self.Win[: self.n_hemi, 1] = 0.6 * torch.randn(self.n_hemi)
            else:
                self.Win[: self.n_hemi, :] = 0.6 * torch.randn(self.n_hemi, n_in_eff)

        self.b = nn.Parameter(torch.zeros(self.n))
        if readout_mode == "lh_motor":
            read_in = int(len(wiring["mot_lh_units"]))
        elif readout_mode == "left_hemi":
            read_in = self.n_hemi
        else:
            read_in = self.n
        self.readout = nn.Linear(read_in, out_dim)

        # Precompute delay selectors as bool buffers
        for d in range(1, DMAX + 1):
            self.register_buffer(f"sel_d{d}", self.inter_delay == d)

    def _feat(self, x):
        if self.readout_mode == "lh_motor":
            return x[:, self.mot_lh]
        if self.readout_mode == "left_hemi":
            return x[:, : self.n_hemi]
        return x

    def _scatter_edges(self, src_states, src_idx, dst_idx, weights):
        """src_states: (B, N); return (B, N) messages aggregated onto destinations."""
        if src_idx.numel() == 0:
            return torch.zeros_like(src_states)
        msgs = src_states[:, src_idx] * weights.unsqueeze(0)  # B, E
        out = torch.zeros_like(src_states)
        out.index_add_(1, dst_idx, msgs)
        return out

    def _recurrent(self, x, buf):
        rec = self._scatter_edges(x, self.intra_src, self.intra_dst, self.w_intra)
        w_mag = F.softplus(self.w_inter) * self.inter_sign
        for d in range(1, self.dmax + 1):
            sel = getattr(self, f"sel_d{d}")
            if not bool(sel.any()):
                continue
            rec = rec + self._scatter_edges(
                buf[d], self.inter_src[sel], self.inter_dst[sel], w_mag[sel]
            )
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
            rec = self._recurrent(x, buf)
            ut = self.emb(u[t]) if self.use_emb else u[t]
            drive = ut @ self.Win.t() + self.b
            x = (1 - self.leak) * x + self.leak * torch.tanh(rec + drive)
            outs.append(self.readout(self._feat(x)))
        return torch.stack(outs, 0)



def count_params(m):
    n = 0
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
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


def make_transfer_batch(d, seq_len, batch, rng):
    """RH-only bit → LH-motor readout must recover delayed identity (Poffenberger-like)."""
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = u[t - d, :, 0]
    mask = np.zeros((T, batch), dtype=np.float32); mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_split_bind_batch(d, seq_len, batch, rng):
    """L bit + R bit → XOR binding; left readout needs the bridge."""
    T = seq_len
    u = rng.integers(0, 2, size=(T, batch, 2)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for t in range(d, T):
        y[t, :, 0] = np.logical_xor(u[t - d, :, 0] > 0.5, u[t - d, :, 1] > 0.5).astype(np.float32)
    mask = np.zeros((T, batch), dtype=np.float32); mask[d:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def make_delayed_match_batch(d_match, seq_len, batch, rng):
    """Cue on ch0 (RH), probe on ch1 (LH) after lag; target = match."""
    T = seq_len
    u = np.zeros((T, batch, 2), dtype=np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    mask = np.zeros((T, batch), dtype=np.float32)
    # place cue early, probe at cue+d_match
    for b in range(batch):
        t_cue = int(rng.integers(2, max(3, T - d_match - 2)))
        t_probe = t_cue + d_match
        cue = float(rng.integers(0, 2))
        probe = float(rng.integers(0, 2))
        u[t_cue, b, 0] = cue
        u[t_probe, b, 1] = probe
        y[t_probe, b, 0] = 1.0 if cue == probe else 0.0
        mask[t_probe, b] = 1.0
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


def make_narma_batch(seq_len, batch, rng, max_abs=50.0, max_tries=50):
    """NARMA-10 with rejection of unstable targets (FIX_NOTES.md)."""
    T = seq_len
    u = rng.uniform(0, 0.5, size=(T, batch, 1)).astype(np.float32)
    y = np.zeros((T, batch, 1), dtype=np.float32)
    for b in range(batch):
        for attempt in range(max_tries):
            if attempt > 0:
                u[:, b, 0] = rng.uniform(0, 0.5, size=(T,)).astype(np.float32)
            y[:, b, 0] = 0.0
            ok = True
            for i in range(NARMA_ORDER, T - 1):
                val = (
                    0.3 * y[i, b, 0]
                    + 0.05 * y[i, b, 0] * y[i - 9:i + 1, b, 0].sum()
                    + 1.5 * u[i - 9, b, 0] * u[i, b, 0]
                    + 0.1
                )
                if (not np.isfinite(val)) or abs(val) > max_abs:
                    ok = False
                    break
                y[i + 1, b, 0] = val
            if ok:
                break
        else:
            y[:, b, 0] = np.nan_to_num(y[:, b, 0], nan=0.0, posinf=max_abs, neginf=-max_abs)
            np.clip(y[:, b, 0], -max_abs, max_abs, out=y[:, b, 0])
    mask = np.zeros((T, batch), dtype=np.float32); mask[20:] = 1.0
    return torch.from_numpy(u), torch.from_numpy(y), torch.from_numpy(mask)


def task_io(task, hp):
    """Return n_in, out_dim, emb, vocab, readout_mode, input_mode, transfer_d."""
    xfer_d = max(8, DMAX + 2)
    if task == "XOR_HARD":
        return 2, 1, 0, 0, "left_hemi", "hemi_split", XOR_HARD_D
    if task == "CHAR":
        C = load_alice()
        return hp["emb_dim"], C["V"], hp["emb_dim"], C["V"], "left_hemi", "left_drive", 0
    if task == "NARMA":
        return 1, 1, 0, 0, "left_hemi", "left_drive", 0
    if task == "TRANSFER":
        return 1, 1, 0, 0, "lh_motor", "transfer_rh_visual", xfer_d
    if task == "SPLIT_BIND":
        return 2, 1, 0, 0, "left_hemi", "hemi_split", SPLIT_BIND_D
    if task == "DELAYED_MATCH":
        return 2, 1, 0, 0, "left_hemi", "delayed_match", D_MATCH
    raise KeyError(task)


HIGHER_BETTER = {"XOR_HARD", "TRANSFER", "SPLIT_BIND", "DELAYED_MATCH"}


def eval_metric(model, task, seed, hp, n_batches=4):
    rng = np.random.default_rng(seed)
    metrics = []
    _, _, _, _, _, _, d_task = task_io(task, hp)
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            if task == "XOR_HARD":
                u, y, mask = make_xor_hard_batch(XOR_HARD_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); target = y.squeeze(-1); m = mask
                acc = ((pred == target).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "CHAR":
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
                metrics.append(float(ce.item()))
            elif task == "NARMA":
                u, y, mask = make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
                pred = model.forward_sequence(u).squeeze(-1)
                yt = y.squeeze(-1); m = mask
                mse = ((pred - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
                std = yt[m > 0].std().clamp_min(1e-8)
                metrics.append(float((torch.sqrt(mse) / std).item()))
            elif task == "TRANSFER":
                u, y, mask = make_transfer_batch(d_task, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); m = mask
                acc = ((pred == y.squeeze(-1)).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "SPLIT_BIND":
                u, y, mask = make_split_bind_batch(SPLIT_BIND_D, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); m = mask
                acc = ((pred == y.squeeze(-1)).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            elif task == "DELAYED_MATCH":
                u, y, mask = make_delayed_match_batch(D_MATCH, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                pred = (logits > 0).float(); m = mask
                acc = ((pred == y.squeeze(-1)).float() * m).sum() / m.sum().clamp_min(1.0)
                metrics.append(float(acc.item()))
            else:
                raise KeyError(task)
    return float(np.mean(metrics))


def train_task(arm, task, seed, hp, n_label, epochs=None, record_curve=True):
    set_seed(seed + 17 * n_label)
    epochs = epochs if epochs is not None else hp["epochs"]
    n_local = N_LABEL_TO_LOCAL[n_label]
    wiring = make_wiring(arm, seed, n_local)
    n_in, out_dim, emb_dim, vocab, readout_mode, input_mode, d_task = task_io(task, hp)
    model = OscRNN(
        wiring, n_in, hp["leak"], out_dim, emb_dim=emb_dim, vocab=vocab,
        readout_mode=readout_mode, input_mode=input_mode,
    )
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    curve = []; rng = np.random.default_rng(seed + 17); best = None
    steps = int(hp.get("steps_per_epoch", 1))
    higher_better = task in HIGHER_BETTER
    for ep in range(epochs):
        model.train()
        last_loss = None
        for _step in range(steps):
            if task == "XOR_HARD":
                u, y, mask = make_xor_hard_batch(XOR_HARD_D, hp["seq_xor"], hp["batch_xor"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "CHAR":
                u, y = make_char_batch(hp["seq_char"], hp["batch_char"], hp["char_train_chars"], rng)
                pred = model.forward_sequence(u)
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), y.reshape(-1))
            elif task == "NARMA":
                u, y, mask = make_narma_batch(hp["seq_narma"], hp["batch_narma"], rng)
                pred = model.forward_sequence(u).squeeze(-1)
                loss = ((pred - y.squeeze(-1)) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "TRANSFER":
                u, y, mask = make_transfer_batch(d_task, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "SPLIT_BIND":
                u, y, mask = make_split_bind_batch(SPLIT_BIND_D, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            elif task == "DELAYED_MATCH":
                u, y, mask = make_delayed_match_batch(D_MATCH, hp["seq_xfer"], hp["batch_xfer"], rng)
                logits = model.forward_sequence(u).squeeze(-1)
                # only supervised at probe steps
                loss = F.binary_cross_entropy_with_logits(logits, y.squeeze(-1), reduction="none")
                loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                raise KeyError(task)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            last_loss = loss
        ev = int(hp.get("eval_every", 1))
        if (ep % ev == 0) or (ep == epochs - 1):
            metric = eval_metric(model, task, seed + 1000 + ep, hp, n_batches=2)
        else:
            metric = float("nan")
        curve.append({"epoch": ep, "train_loss": float(last_loss.item()), "metric": metric})
        if np.isfinite(metric):
            if higher_better:
                best = metric if best is None else max(best, metric)
            else:
                best = metric if best is None else min(best, metric)
    final = eval_metric(model, task, seed + 777, hp, n_batches=4)
    out = dict(
        arm=arm, task=task, seed=seed, n_label=n_label, n_hidden=int(wiring["n_hidden"]),
        n_local=int(wiring["n_local"]), metric=final, best_epoch_metric=best,
        n_params=count_params(model), nnz=int(wiring["nnz"]),
        n_bridge=int(wiring["n_bridge"]), left_only_readout=True,
        readout_mode=readout_mode, input_mode=input_mode,
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


def _worker(arm, task, seed, hp, n_label):
    t0 = time.time()
    r = train_task(arm, task, seed, hp, n_label, record_curve=True)
    r["wall_s"] = time.time() - t0
    return r


def _done_key(r):
    return (r["arm"], r["task"], r["seed"], int(r.get("n_label", -1)))


def run_eval(arms=None, tasks=None, seeds=None, n_labels=None, workers=2):
    hp = load_hp()
    arms = arms or ARMS
    tasks = tasks or PRIMARY_TASKS
    seeds = seeds or EVAL_SEEDS
    n_labels = n_labels or DEFAULT_N_LABELS
    jobs = [(a, t, s, n) for n in n_labels for a in arms for t in tasks for s in seeds]
    jf = OUT / "runs.jsonl"
    done = set()
    if jf.exists():
        for line in jf.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add(_done_key(r))
    todo = [j for j in jobs if j not in done]
    print(f"jobs {len(jobs)} todo {len(todo)} workers {workers} n_labels={n_labels} hp={hp}",
          flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, a, t, s, hp, n): (a, t, s, n) for a, t, s, n in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with open(jf, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"N={r['n_label']}({r['n_hidden']}) {r['arm']} {r['task']} seed={r['seed']} "
                f"metric={r['metric']:.4f} nnz={r['nnz']} {r['wall_s']:.1f}s",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n-labels", nargs="*", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-one", action="store_true",
                    help="Full-hp single job timing probe (seed 91, not counted)")
    ap.add_argument("--check-wiring", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.check_wiring:
        for n_label in (args.n_labels or [512]):
            n_local = N_LABEL_TO_LOCAL[n_label]
            print(f"=== N_label={n_label} N_local={n_local} N={68*n_local} ===", flush=True)
            for arm in ARMS:
                w = make_wiring(arm, EVAL_SEEDS[0], n_local)
                print(arm, "nnz", w["nnz"], "n_bridge", w["n_bridge"],
                      "mean_delay", w["meta"]["mean_delay"], flush=True)
        return
    if args.smoke:
        hp = dict(DEFAULT_HP, epochs=2, steps_per_epoch=2,
                  batch_xor=16, batch_char=16, batch_narma=16, batch_xfer=16)
        n_labels = args.n_labels or [512]
        arms = args.arms or ARMS
        tasks = args.tasks or PRIMARY_TASKS
        for n_label in n_labels:
            for arm in arms:
                for task in tasks:
                    r = train_task(arm, task, 91, hp, n_label, epochs=2, record_curve=False)
                    print("smoke", f"N={n_label}", r["arm"], r["task"],
                          f"metric={r['metric']:.4f}", f"nnz={r['nnz']}",
                          f"npar={r['n_params']}", flush=True)
        return
    if args.time_one:
        hp = load_hp()
        n_label = (args.n_labels or [512])[0]
        arm = (args.arms or ["REAL_SC"])[0]
        task = (args.tasks or ["TRANSFER"])[0]
        t0 = time.time()
        r = train_task(arm, task, 91, hp, n_label, record_curve=False)
        dt = time.time() - t0
        print(json.dumps(dict(
            n_label=n_label, n_hidden=r["n_hidden"], arm=arm, task=task,
            metric=r["metric"], nnz=r["nnz"], wall_s=dt,
        )), flush=True)
        return
    if args.eval:
        load_hp()
        run_eval(args.arms, args.tasks, args.seeds, args.n_labels, workers=args.workers)


if __name__ == "__main__":
    main()

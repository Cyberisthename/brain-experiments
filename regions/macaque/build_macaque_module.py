"""Expand Markov FLNe 29×29 → ~2000-neuron macaque_module.npz.

Prereg: PREREG_macaque.md. Sources: SOURCES.md.
Frozen template seed=11. Measured SLN not available → hierarchy_rank proxy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

H = Path(__file__).resolve().parent
DATA = H / "data"
OUT = DATA / "macaque_module.npz"
META = DATA / "macaque_module_meta.json"
FLN_PATH = DATA / "markov_fln29.npz"

N_PER = 69          # 29*69 = 2001
P_INTRA = 0.12
FRAC_E = 0.80
LAYER_FRACS = [("L2/3", 0.35), ("L4", 0.25), ("L5", 0.25), ("L6", 0.15)]
DMAX = 8
TEMPLATE_SEED = 11

# Coarse hierarchy rank (low = early sensory / posterior). Proxy for SLN FF/FB.
HIER = {
    "V1": 0, "V2": 1, "V4": 2, "MT": 3, "TEO": 4, "TEPD": 5, "DP": 6,
    "7A": 7, "7M": 8, "7B": 9, "STPC": 10, "STPI": 10, "STPR": 10, "PBR": 11,
    "2": 12, "5": 13, "F1": 14, "F5": 15, "F2": 16, "PROM": 17, "F7": 18,
    "8L": 19, "8M": 20, "8B": 21, "46D": 22, "9/46D": 23, "9/46V": 24,
    "10": 25, "24C": 26,
}


def _assign_layers(n, g):
    labs = []
    counts = []
    rem = n
    for i, (L, f) in enumerate(LAYER_FRACS):
        c = int(round(f * n)) if i < len(LAYER_FRACS) - 1 else rem
        c = max(1, min(c, rem - (len(LAYER_FRACS) - 1 - i)))
        labs.extend([L] * c)
        counts.append(c)
        rem -= c
    labs = np.array(labs[:n], dtype=object)
    if len(labs) < n:
        labs = np.concatenate([labs, np.array(["L6"] * (n - len(labs)), dtype=object)])
    g.shuffle(labs)
    return labs


def _row_sum_shuffle(FLN, g):
    """Permute nonzero destinations within each row, preserving row sums and nnz/row."""
    A = FLN.copy()
    n = A.shape[0]
    for i in range(n):
        cols = np.where(np.arange(n) != i)[0]
        vals = A[i, cols].copy()
        g.shuffle(vals)
        A[i, cols] = vals
        A[i, i] = 0.0
    return A


def expand(FLN, DIST, areas, seed=TEMPLATE_SEED, tag="real"):
    g = np.random.default_rng(seed)
    n_areas = FLN.shape[0]
    n_per = N_PER
    N = n_areas * n_per
    area_of = np.repeat(np.arange(n_areas), n_per)
    # signs + layers per neuron
    sign = np.ones(N, dtype=np.int8)
    layer = np.empty(N, dtype=object)
    cell_type = np.empty(N, dtype=object)
    for a in range(n_areas):
        idx = np.where(area_of == a)[0]
        n_i = len(idx)
        n_e = int(round(FRAC_E * n_i))
        s = np.ones(n_i, dtype=np.int8)
        s[n_e:] = -1
        g.shuffle(s)
        sign[idx] = s
        layer[idx] = _assign_layers(n_i, g)
        cell_type[idx] = np.where(s > 0, "E", "I")
    rows, cols, data, delays = [], [], [], []

    # within-area
    for a in range(n_areas):
        idx = np.where(area_of == a)[0]
        n_i = len(idx)
        # ER-like without self
        n_possible = n_i * (n_i - 1)
        n_edges = int(round(P_INTRA * n_possible))
        # sample pairs
        r = g.integers(0, n_i, size=n_edges * 2)
        c = g.integers(0, n_i, size=n_edges * 2)
        m = r != c
        r, c = r[m][:n_edges], c[m][:n_edges]
        pre = idx[r]; post = idx[c]
        mag = g.exponential(1.0, size=len(pre))
        w = mag * sign[pre].astype(float)
        rows.append(pre); cols.append(post); data.append(w)
        delays.append(np.ones(len(pre), dtype=np.int32))  # local delay=1

    # inter-area from FLN
    fln_max = float(FLN.max()) if FLN.max() > 0 else 1.0
    d_obs = DIST[DIST > 0]
    d_max = float(d_obs.max()) if len(d_obs) else 1.0
    ranks = np.array([HIER.get(str(areas[i]), 15) for i in range(n_areas)])

    for src in range(n_areas):
        for tgt in range(n_areas):
            if src == tgt:
                continue
            fln = float(FLN[src, tgt])
            if fln <= 0:
                continue
            p = float(np.clip(0.08 * (fln / fln_max) ** 0.5, 0.002, 0.08))
            src_idx = np.where(area_of == src)[0]
            tgt_idx = np.where(area_of == tgt)[0]
            # FF vs FB laminar preference
            ff = ranks[src] < ranks[tgt]
            if ff:
                # prefer L2/3 → L4
                pre_pool = src_idx[layer[src_idx] == "L2/3"]
                post_pool = tgt_idx[layer[tgt_idx] == "L4"]
            else:
                pre_pool = src_idx[layer[src_idx] == "L5"]
                post_pool = tgt_idx[layer[tgt_idx] == "L2/3"]
            if len(pre_pool) == 0:
                pre_pool = src_idx
            if len(post_pool) == 0:
                post_pool = tgt_idx
            # also allow some non-preferred (30%) for realism
            n_try = int(round(p * len(src_idx) * len(tgt_idx)))
            n_try = max(1, min(n_try, 800))
            n_pref = int(round(0.7 * n_try))
            n_other = n_try - n_pref
            pre = np.concatenate([
                g.choice(pre_pool, n_pref, replace=True),
                g.choice(src_idx, n_other, replace=True) if n_other else np.array([], dtype=int),
            ])
            post = np.concatenate([
                g.choice(post_pool, n_pref, replace=True),
                g.choice(tgt_idx, n_other, replace=True) if n_other else np.array([], dtype=int),
            ])
            mag = fln * g.exponential(1.0, size=len(pre))
            w = mag * sign[pre].astype(float)
            dist = float(DIST[src, tgt])
            if dist > 0 and d_max > 0:
                dly = int(1 + round((dist / d_max) * (DMAX - 1)))
                dly = int(np.clip(dly, 1, DMAX))
            else:
                dly = int(g.integers(1, DMAX + 1))
            rows.append(pre); cols.append(post); data.append(w)
            delays.append(np.full(len(pre), dly, dtype=np.int32))

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    data = np.concatenate(data); delays = np.concatenate(delays)
    W = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    W.sum_duplicates()
    # rebuild delay aligned to coo after sum_duplicates — store as dict of (i,j)->delay then map
    # simpler: keep parallel coo before sum; take first delay per unique edge
    delay_map = {}
    for r, c, d in zip(rows, cols, delays):
        key = (int(r), int(c))
        if key not in delay_map:
            delay_map[key] = int(d)
    Wc = W.tocoo()
    edge_delay = np.array([delay_map.get((int(r), int(c)), 1) for r, c in zip(Wc.row, Wc.col)], dtype=np.int32)
    meta = dict(
        N=int(N), nnz=int(W.nnz), n_areas=int(n_areas), n_per=int(n_per),
        frac_E=float((sign > 0).mean()), dens=float(W.nnz) / max(1, N * (N - 1)),
        fln_dens=float((FLN > 0).mean()), dmax=DMAX,
        sln_proxy="hierarchy_rank", template_seed=int(seed), tag=tag,
        areas=[str(a) for a in areas],
        layer_counts={L: int((layer == L).sum()) for L, _ in LAYER_FRACS},
        plain_words=(
            f"Markov FLNe among {n_areas} injected macaque cortical areas expanded to "
            f"N={N} ({n_per}/area). Within-area sparse Dale recurrent (p={P_INTRA}, "
            f"frac_E={FRAC_E}). Inter-area p∝√FLNe with hierarchy-proxy FF/FB laminar "
            f"targeting (measured SLN not in download). Delays from Dist_mm → 1..{DMAX}."
        ),
    )
    return W, sign, layer.astype(str), cell_type.astype(str), area_of, edge_delay, Wc.row, Wc.col, meta


def save_module(W, sign, layer, cell_type, area_of, edge_delay, row, col, meta, path=OUT):
    # store CSR + coo delay alignment
    np.savez_compressed(
        path,
        W_data=W.data, W_indices=W.indices, W_indptr=W.indptr,
        W_shape=np.array(W.shape),
        root_id=np.arange(W.shape[0], dtype=np.int64),
        sign=sign.astype(np.int8),
        layer=np.asarray(layer, dtype=object),
        cell_type=np.asarray(cell_type, dtype=object),
        area_of=area_of.astype(np.int32),
        edge_row=row.astype(np.int32),
        edge_col=col.astype(np.int32),
        edge_delay=edge_delay.astype(np.int32),
        dmax=np.array([DMAX]),
    )
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    z = np.load(FLN_PATH, allow_pickle=True)
    FLN = z["FLN"].astype(np.float64)
    DIST = z["DIST_mm"].astype(np.float64)
    areas = z["areas"]
    np.fill_diagonal(FLN, 0.0)
    W, sign, layer, ct, area_of, ed, row, col, meta = expand(FLN, DIST, areas, TEMPLATE_SEED, "real")
    save_module(W, sign, layer, ct, area_of, ed, row, col, meta, OUT)
    # also stash a shuffled-FLN template seed base for controls (built per-eval in runner)
    (DATA / "build_ok").write_text("ok\n")


if __name__ == "__main__":
    main()

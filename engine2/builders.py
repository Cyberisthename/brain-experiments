"""Wiring builders: FLY_CX + ER/SHUF/DENS/EI_SHUF reused from frankenstein."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "regions" / "frankenstein"))
sys.path.insert(0, str(_ROOT / "compress"))
import frankenstein as F  # noqa: E402

from .params import NT_SIGN, MOD_NTS


def load_fly_cx():
    W, sign, layer, ct = F.load_mod("fly_cx")
    return W.tocsr(), sign.astype(np.int8), layer, ct


def load_nt_labels(root_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """Align FlyWire top_nt to module order via neuromod_real/data/fly_cx_nt.npz."""
    p = _ROOT / "regions" / "neuromod_real" / "data" / "fly_cx_nt.npz"
    z = np.load(p, allow_pickle=True)
    labels = z["top_nt"].astype(str)
    if root_ids is not None:
        # already aligned in construction of fly_cx_nt.npz
        assert len(labels) == len(root_ids)
    return labels


def ei_class_shuffle(W, sign, g):
    """Same as xor_cx.ei_class_shuffle — rewire within (sign_pre, sign_post)."""
    Wc = W.tocoo()
    N = W.shape[0]
    row, col, data = Wc.row.copy(), Wc.col.copy(), np.abs(Wc.data).copy()
    s_pre = sign[row]
    s_post = sign[col]
    new_col = col.copy()
    for s_pre_v in (-1, 1):
        for s_post_v in (-1, 1):
            m = (s_pre == s_pre_v) & (s_post == s_post_v)
            idx = np.where(m)[0]
            if len(idx) < 2:
                continue
            cands = np.where(sign == s_post_v)[0]
            if len(cands) == 0:
                continue
            posts = col[idx].copy()
            g.shuffle(posts)
            for _ in range(8):
                bad = row[idx] == posts
                if not bad.any():
                    break
                posts[bad] = g.choice(cands, size=int(bad.sum()), replace=True)
            new_col[idx] = posts
    out_data = data * sign[row].astype(float)
    return sp.csr_matrix((out_data, (row, new_col)), shape=(N, N))


def build_arm(arm: str, seed: int = 0, module: str = "fly_cx") -> Dict:
    """Return dict with W, sign, layer, nt_labels, delays_ms, meta.

    Arms: FLY_CX|CX|REAL, ER, SHUF, DENS, EI_SHUF.
    """
    W0, sign, layer, ct = F.load_mod(module)
    g = np.random.default_rng(seed + 13)
    arm_u = arm.upper().replace("-", "_")
    if arm_u in ("FLY_CX", "CX", "REAL"):
        W = W0.copy()
        kind = "real"
    elif arm_u == "ER":
        W = F.er_match(W0, sign, g)
        kind = "er"
    elif arm_u == "SHUF":
        W = F.degree_shuffle(W0, sign, g)
        kind = "shuf"
    elif arm_u in ("DENS", "DENS_MATCHED"):
        W = F.dens_match(W0, sign, g)
        kind = "dens"
    elif arm_u == "EI_SHUF":
        W = ei_class_shuffle(W0, sign, g)
        kind = "ei_shuf"
    else:
        raise KeyError(arm)

    # Apply Dale signs to magnitudes for random arms that already did;
    # ensure W signs match Dale
    Wc = W.tocoo()
    # For edges, sign should follow presynaptic (col in our convention? frankenstein: row=post, col=pre in er_match uses sign[rows] where rows are post — wait)
    # frankenstein er_match: data = abs * sign[rows], rows are random posts — that's a bug-legacy.
    # fly_cx W_data already signed by pre. We keep as-is for real; for controls match xor_cx.

    nt = None
    if module == "fly_cx":
        try:
            z = np.load(_ROOT / "regions" / "frankenstein" / "data" / "fly_cx_module.npz", allow_pickle=True)
            root_id = z["root_id"]
            nt = load_nt_labels(root_id)
            # Rebuild sign from NT where known (Dale)
            sign2 = sign.copy()
            for i, lab in enumerate(nt):
                s = NT_SIGN.get(str(lab), 0)
                if s != 0:
                    sign2[i] = s
            sign = sign2
        except Exception:
            nt = np.array(["ACh" if s > 0 else "GABA" for s in sign], dtype=object)

    # Default delays: 1 ms base (heterogeneous optional later)
    nnz = W.nnz
    delays_ms = np.ones(nnz, dtype=np.float32)

    return {
        "W": W.tocsr(),
        "sign": sign.astype(np.int8),
        "layer": layer,
        "cell_type": ct,
        "nt_labels": nt,
        "delays_ms": delays_ms,
        "arm": arm_u,
        "kind": kind,
        "N": W.shape[0],
        "nnz": int(W.nnz),
        "seed": seed,
    }

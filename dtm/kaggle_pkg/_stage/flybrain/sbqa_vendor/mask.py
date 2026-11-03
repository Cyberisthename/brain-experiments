"""Biological sparsity mask from Griffa scale-68 group SC (CC BY 4.0)."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_NPZ = HERE / "data" / "griffa70_scale68_group_minimal.npz"


def _npz_path() -> Path:
    env = os.environ.get("FLYBRAIN_GRIFFA_WHOLE_NPZ") or os.environ.get("SBQA_GRIFFA_NPZ")
    if env:
        return Path(env)
    return DEFAULT_NPZ


def load_sc(npz: Path | None = None) -> np.ndarray:
    p = Path(npz) if npz else _npz_path()
    z = np.load(p, allow_pickle=True)
    sc = np.asarray(z["sc_group"], dtype=np.float64)
    sc = 0.5 * (sc + sc.T)
    np.fill_diagonal(sc, 0.0)
    return sc


def expand_mask(sc: np.ndarray, n_local: int, thresh: float = 0.0) -> np.ndarray:
    """Bio2art-style expansion: N = 68 * n_local.

    Intra-parcel: all-to-all (no self). Inter-parcel: full bipartite block iff SC>thresh.
    Returns bool mask (N, N).
    """
    R = sc.shape[0]
    assert R == 68, R
    N = R * n_local
    M = np.zeros((N, N), dtype=bool)
    # intra
    for r in range(R):
        sl = slice(r * n_local, (r + 1) * n_local)
        M[sl, sl] = True
    np.fill_diagonal(M, False)
    # inter
    for i in range(R):
        for j in range(R):
            if i == j:
                continue
            if sc[i, j] > thresh:
                si = slice(i * n_local, (i + 1) * n_local)
                sj = slice(j * n_local, (j + 1) * n_local)
                M[si, sj] = True
    return M


def edge_list(mask: np.ndarray) -> np.ndarray:
    """(E, 2) int32 rows = (post, pre) for True entries."""
    ii, jj = np.where(mask)
    return np.stack([ii, jj], axis=1).astype(np.int32)


def make_mask(n_local: int = 4, thresh: float = 0.0, npz: Path | None = None):
    sc = load_sc(npz)
    M = expand_mask(sc, n_local, thresh=thresh)
    edges = edge_list(M)
    return {
        "mask": M,
        "edges": edges,
        "N": M.shape[0],
        "E_mask": int(edges.shape[0]),
        "n_local": n_local,
        "sc": sc,
    }

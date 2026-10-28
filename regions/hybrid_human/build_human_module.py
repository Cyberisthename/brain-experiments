"""Build human cortex adjacency subset for fly–human hybrid graft.
Mirrors regions/hybrid/build_mouse_module.py schema → data/human_module.npz.

Primary path (disk-safe): HuggingFace NathanRoll/h01-cortex-snn edges.npz
derived from ALL 166 public Avro shards of
  gs://h01-release/data/20210729/c3/synapses/exported/
plus official somas.csv for verification.

Does NOT require CAVE / Google login. Does NOT download the ~33GB Avro dump.
No fake edges. See data/SOURCE.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp

H = Path(__file__).resolve().parent
DATA = H / "data"
OUT = DATA / "human_module.npz"
EDGES = DATA / "hf_edges.npz"
META = DATA / "hf_metadata.npz"
SOMAS = DATA / "somas.csv"
SRC = DATA / "SOURCE.json"

QUOTAS = {"Layer 2": 550, "Layer 3": 450, "Layer 4": 500, "Layer 5": 350, "Layer 6": 250}
CTYPE_KEEP = {"PYRAMIDAL", "INTERNEURON", "SPINY_STELLATE", "SPINY_ATYPICAL"}

def map_layer(L: str) -> str:
    return {
        "Layer 1": "L1", "Layer 2": "L2/3", "Layer 3": "L2/3",
        "Layer 4": "L4", "Layer 5": "L5", "Layer 6": "L6",
        "White matter": "WM", "unclassified": "unk", "unknown": "unk",
    }.get(L, "unk")

def main():
    if not EDGES.exists() or not META.exists():
        print("Missing hf_edges.npz / hf_metadata.npz — download from HuggingFace NathanRoll/h01-cortex-snn", file=sys.stderr)
        sys.exit(2)
    meta = np.load(META, allow_pickle=True)
    edges = np.load(EDGES)["edges"]
    seg = meta["segment_ids"].astype(np.int64)
    layers = meta["layers"].astype(str)
    ctypes = meta["cell_types"].astype(str)
    is_e = meta["is_excitatory"]; is_i = meta["is_inhibitory"]
    N = len(seg)
    agg = {}
    for a, b in edges[:, :2]:
        a, b = int(a), int(b)
        if a == b: continue
        agg[(a, b)] = agg.get((a, b), 0) + 1
    deg = np.zeros(N, dtype=np.int64)
    for (a, b), c in agg.items():
        deg[a] += c; deg[b] += c

    selected = []
    for L, q in QUOTAS.items():
        cand = [i for i in range(N) if layers[i] == L and ctypes[i] in CTYPE_KEEP]
        cand.sort(key=lambda i: -deg[i])
        selected.extend(cand[:q])
    selected = np.array(selected, dtype=np.int64)

    def within_deg(idxs):
        s = set(map(int, idxs)); d = {int(i): 0 for i in idxs}
        for (a, b), c in agg.items():
            if a in s and b in s:
                d[a] += c; d[b] += c
        return d

    for _ in range(4):
        d = within_deg(selected)
        alive = np.array([i for i in selected if d[int(i)] > 0], dtype=np.int64)
        have = Counter(layers[alive].tolist())
        selected_set = set(map(int, alive))
        extra = []
        for L, q in QUOTAS.items():
            need = q - have.get(L, 0)
            if need <= 0: continue
            cand = [i for i in range(N) if layers[i] == L and ctypes[i] in CTYPE_KEEP and i not in selected_set]
            cand.sort(key=lambda i: -deg[i])
            for i in cand[:need]:
                extra.append(i); selected_set.add(i)
        selected = np.concatenate([alive, np.asarray(extra, dtype=np.int64)]) if extra else alive
    d = within_deg(selected)
    selected = np.array([i for i in selected if d[int(i)] > 0], dtype=np.int64)

    def sign_of(i):
        if is_i[i]: return -1
        if is_e[i]: return 1
        return 0

    idx_map = {int(o): n for n, o in enumerate(selected)}
    rows, cols, data = [], [], []
    for (a, b), c in agg.items():
        if a not in idx_map or b not in idx_map: continue
        sg = sign_of(a)
        if sg == 0: continue
        rows.append(idx_map[a]); cols.append(idx_map[b]); data.append(float(c) * sg)
    Nh = len(selected)
    W = sp.coo_matrix((data, (rows, cols)), shape=(Nh, Nh)).tocsr()
    W.sum_duplicates()
    signs = np.array([sign_of(i) for i in selected], dtype=np.int8)
    layer_out = np.array([map_layer(layers[i]) for i in selected])
    ct_out = ctypes[selected]
    root_id = seg[selected]
    np.savez_compressed(
        OUT,
        W_data=W.data, W_indices=W.indices, W_indptr=W.indptr, W_shape=np.array(W.shape),
        root_id=root_id, sign=signs, layer=layer_out, cell_type=ct_out, hf_index=selected,
    )
    print(json.dumps({
        "out": str(OUT), "n": Nh, "nnz": int(W.nnz),
        "E": int((signs == 1).sum()), "I": int((signs == -1).sum()),
        "layers": dict(Counter(layer_out.tolist())),
    }, indent=2))

if __name__ == "__main__":
    main()

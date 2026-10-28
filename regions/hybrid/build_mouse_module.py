"""Build proofread-subset mouse cortex adjacency from Zenodo MICrONS mm3 connectome
or from GCS static tables. Writes regions/hybrid/data/mouse_module.npz.
See PREREGISTRATION_hybrid.md. No fake edges.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

H = Path(__file__).resolve().parent
DATA = H / "data"
DATA.mkdir(parents=True, exist_ok=True)
H5 = DATA / "microns_mm3_connectome_v1412.h5"
CT = DATA / "proofread_celltypes_v1412.parquet"
IDS = DATA / "proofread_root_ids_v1412.txt"
OUT = DATA / "mouse_module.npz"
SRC = DATA / "SOURCE.json"


def _load_ids():
    ids = [int(x) for x in IDS.read_text().splitlines() if x.strip()]
    return np.asarray(ids, dtype=np.int64)


def _ei_layer():
    ct = pd.read_parquet(CT)
    return ct.set_index("pt_root_id")


def from_zenodo_h5(path: Path):
    """Extract condensed adjacency among proofread roots from ConnectomeUtilities HDF5."""
    import h5py
    ids = _load_ids()
    id_set = set(ids.tolist())
    ct = _ei_layer()
    with h5py.File(path, "r") as f:
        # discover layout
        keys = list(f.keys())
        info = {"h5_keys": keys}
        # Common layouts: 'full'/'condensed' groups, or top-level edges
        def find(name_parts):
            for k in keys:
                if all(p in k.lower() for p in name_parts):
                    return k
            return None

        # Try ConnectomeUtilities style
        gname = None
        for cand in ("condensed", "full", "connectome", "default"):
            if cand in f:
                gname = cand
                break
        if gname is None and "edges" in keys:
            gname = None  # top-level

        def read_nodes(g):
            # vertex ids
            for nk in ("vertex_properties/node_ids", "nodes/node_ids", "node_ids",
                       "vertices/id", "vertex_ids", "gids", "root_id"):
                if nk in g:
                    return np.asarray(g[nk][:], dtype=np.int64)
            # nested
            for nk in g.keys():
                if "node" in nk.lower() or "vert" in nk.lower() or "gid" in nk.lower():
                    obj = g[nk]
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.shape[0] > 100:
                        return np.asarray(obj[:], dtype=np.int64)
                    if isinstance(obj, h5py.Group):
                        for sk in obj.keys():
                            if any(t in sk.lower() for t in ("id", "gid", "root")):
                                return np.asarray(obj[sk][:], dtype=np.int64)
            return None

        def read_edges(g):
            # source/target or pre/post
            src = tgt = w = None
            for sk, tk in [("source", "target"), ("pre", "post"), ("pre_gid", "post_gid"),
                           ("row", "col"), ("i", "j"), ("src", "dst")]:
                # search datasets
                def find_ds(name):
                    for path in (name, f"edges/{name}", f"edge_properties/{name}",
                                 f"connectivity/{name}"):
                        if path in g:
                            return g[path]
                    # recursive shallow
                    for k in g.keys():
                        if name == k.lower() or name in k.lower():
                            if isinstance(g[k], h5py.Dataset):
                                return g[k]
                        if isinstance(g[k], h5py.Group):
                            for sk2 in g[k].keys():
                                if name in sk2.lower() and isinstance(g[k][sk2], h5py.Dataset):
                                    return g[k][sk2]
                    return None
                sds, tds = find_ds(sk), find_ds(tk)
                if sds is not None and tds is not None:
                    src = np.asarray(sds[:], dtype=np.int64)
                    tgt = np.asarray(tds[:], dtype=np.int64)
                    break
            for wk in ("count", "size", "weight", "n_synapses", "sum_size"):
                for path in (wk, f"edges/{wk}", f"edge_properties/{wk}"):
                    if path in g:
                        w = np.asarray(g[path][:], dtype=np.float64)
                        break
                if w is not None:
                    break
            return src, tgt, w

        root = f[gname] if gname else f
        nodes = read_nodes(root)
        if nodes is None and gname:
            # try sibling
            for alt in keys:
                nodes = read_nodes(f[alt])
                if nodes is not None:
                    break
        src, tgt, w = read_edges(root)
        if src is None:
            # dump structure for debugging
            def walk(g, prefix="", depth=0):
                out = []
                if depth > 3:
                    return out
                for k in list(g.keys())[:40]:
                    obj = g[k]
                    p = f"{prefix}/{k}"
                    if isinstance(obj, h5py.Dataset):
                        out.append(f"{p} shape={obj.shape} dtype={obj.dtype}")
                    else:
                        out.append(f"{p}/")
                        out.extend(walk(obj, p, depth + 1))
                return out
            info["structure"] = walk(f)
            (DATA / "h5_structure.txt").write_text("\n".join(info["structure"]))
            raise RuntimeError(f"Could not parse edge layout; structure written to h5_structure.txt keys={keys}")

        info["n_nodes_file"] = int(len(nodes)) if nodes is not None else None
        info["n_edges_file"] = int(len(src))
        # Map root ids
        if nodes is not None and len(nodes) == len(np.unique(np.concatenate([src, tgt]))) or True:
            # edges may already be root ids OR indices into nodes
            # Heuristic: if max(src) < len(nodes)*2 and nodes look like large root ids, treat as indices
            if nodes is not None and src.max() < len(nodes) and nodes.max() > 10**12:
                src_ids = nodes[src]
                tgt_ids = nodes[tgt]
            else:
                src_ids, tgt_ids = src, tgt
        else:
            src_ids, tgt_ids = src, tgt

        # Filter to proofread subset
        mask = np.isin(src_ids, ids) & np.isin(tgt_ids, ids) & (src_ids != tgt_ids)
        src_ids, tgt_ids = src_ids[mask], tgt_ids[mask]
        ww = w[mask] if w is not None else np.ones(mask.sum(), dtype=np.float64)
        info["n_edges_proofread"] = int(mask.sum())

        # Keep only neurons that appear + have cell type
        keep = [i for i in ids if i in ct.index]
        keep = np.asarray(keep, dtype=np.int64)
        idx = {int(r): i for i, r in enumerate(keep)}
        rows, cols, data = [], [], []
        signs = ct.loc[keep, "sign"].to_numpy()
        layers = ct.loc[keep, "layer"].astype(str).to_numpy()
        cell_types = ct.loc[keep, "cell_type"].astype(str).to_numpy()
        for s, t, c in zip(src_ids, tgt_ids, ww):
            if int(s) not in idx or int(t) not in idx:
                continue
            i, j = idx[int(s)], idx[int(t)]
            # weight = count * sign(pre)
            rows.append(i); cols.append(j); data.append(float(c) * float(signs[i]))
        N = len(keep)
        W = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        # collapse duplicates
        W.sum_duplicates()
        info["n_neurons"] = N
        info["n_nnz"] = int(W.nnz)
        info["density"] = float(W.nnz) / (N * N)
        info["n_E"] = int((signs == 1).sum())
        info["n_I"] = int((signs == -1).sum())
        np.savez_compressed(
            OUT,
            W_data=W.data, W_indices=W.indices, W_indptr=W.indptr, W_shape=np.array(W.shape),
            root_id=keep, sign=signs, layer=layers, cell_type=cell_types,
        )
        meta = json.loads(SRC.read_text()) if SRC.exists() else {}
        meta.update({
            "module_file": str(OUT),
            "synapse_source": f"zenodo:{path.name}",
            "zenodo_record": "https://zenodo.org/records/16744240",
            "materialization": 1412,
            "n_neurons_module": N,
            "n_synapses_or_edges": int(W.nnz),
            "h5_info": info,
            "download_date": "2026-10-27",
            "cave_token_required": False,
            "note": "E/I signs from aibs_metamodel_mtypes_v661_v2 classification_system; edge weight = synapse_count * pre_sign",
        })
        SRC.write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2)[:2000])
        return OUT


def main():
    if not CT.exists() or not IDS.exists():
        print("Missing cell-type / id files; run metadata extract first", file=sys.stderr)
        sys.exit(2)
    if H5.exists() and H5.stat().st_size > 500_000_000:
        return from_zenodo_h5(H5)
    print(f"Zenodo H5 not ready yet (size={H5.stat().st_size if H5.exists() else 0}); waiting required", file=sys.stderr)
    sys.exit(3)


if __name__ == "__main__":
    main()

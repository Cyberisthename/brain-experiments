"""Build mega_franken modules: subsampled finch, layered worm, symlinks for base+octo."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp

H = Path(__file__).resolve().parent
DATA = H / "data"
DATA.mkdir(exist_ok=True)

_ROOT = Path(__file__).resolve().parents[2]
FINCH_SRC = _ROOT / "regions/finch/data/areax_module.npz"
WORM_SRC = _ROOT / "regions/celegans/data/COOK_H.npz"
FRANK = _ROOT / "regions/frankenstein/data"
MYELIN = _ROOT / "regions/hybrid_human/data/human_myelin.npz"

# Command / sensory sets (C. elegans)
SENSORY = {
    "ASEL","ASER","AWCL","AWCR","ASGL","ASGR","AFDL","AFDR","ASHL","ASHR",
    "ADLL","ADLR","ASKL","ASKR","ALML","ALMR","PLML","PLMR","FLPL","FLPR",
    "OLQDL","OLQDR","OLQVL","OLQVR","IL1DL","IL1DR","IL1L","IL1R","IL1VL","IL1VR",
    "IL2DL","IL2DR","IL2L","IL2R","IL2VL","IL2VR","CEPDL","CEPDR","CEPVL","CEPVR",
}
COMMAND = {
    "AVAL","AVAR","AVBL","AVBR","AVDL","AVDR","AVEL","AVER","AVJL","AVJR",
    "AVKL","AVKR","PVCL","PVCR","PVNL","PVNR","PVDL","PVDR","PVPL","PVPR",
    "DVA","DVB","DVC","RIBL","RIBR","RID","RIML","RIMR","RMED","RMEV","RMEL","RMER",
}


def symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    print(f"link {dst.name} -> {src}")


def subsample_finch(target_n=2000, seed=42):
    z = np.load(FINCH_SRC, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    ct = z["cell_type"].astype(str)
    layer = z["layer"].astype(str)
    sign = z["sign"].astype(np.int8)
    nid = z["neuron_id"] if "neuron_id" in z.files else np.arange(W.shape[0])
    N0 = W.shape[0]
    g = np.random.default_rng(seed)
    # Stratified: keep all rare types fully if small; subsample MSN heavily
    types, counts = np.unique(ct, return_counts=True)
    # Aim ~2000 with type proportions, but always keep ALL HVC/LMAN/DA/GPe/GPi/STN/INT*
    keep_all = {"HVC", "LMAN", "DA", "GPe", "GPi", "STN", "INT1", "INT2", "INT3", "INT4", "TAN", "LTS"}
    chosen = []
    for t, c in zip(types, counts):
        idx = np.where(ct == t)[0]
        if t in keep_all or c <= 80:
            chosen.append(idx)
        else:
            # proportional remainder
            pass
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    remaining_slots = max(0, target_n - len(chosen))
    # Fill from MSN (and any other large types not keep_all)
    pool = []
    for t, c in zip(types, counts):
        if t in keep_all or c <= 80:
            continue
        idx = np.where(ct == t)[0]
        pool.append(idx)
    if pool and remaining_slots > 0:
        pool = np.concatenate(pool)
        take = min(remaining_slots, len(pool))
        chosen = np.concatenate([chosen, g.choice(pool, size=take, replace=False)])
    chosen = np.unique(chosen)
    if len(chosen) > target_n:
        # trim MSN preferentially
        msn = chosen[ct[chosen] == "MSN"]
        other = chosen[ct[chosen] != "MSN"]
        need = target_n - len(other)
        if need < 0:
            chosen = g.choice(chosen, size=target_n, replace=False)
        else:
            chosen = np.concatenate([other, g.choice(msn, size=need, replace=False)])
    chosen = np.sort(chosen.astype(int))
    # induce subgraph
    Wsub = W[chosen][:, chosen].tocsr()
    meta = dict(
        source="j0251 areax_module.npz",
        parent_N=int(N0),
        N=int(len(chosen)),
        nnz=int(Wsub.nnz),
        seed=seed,
        target_n=target_n,
        type_counts={t: int((ct[chosen] == t).sum()) for t in np.unique(ct[chosen])},
        note="Stratified subsample; keep-all projection/rare types; MSN fill to ~2000",
    )
    out = DATA / "finch_module.npz"
    np.savez_compressed(
        out,
        W_data=Wsub.data.astype(np.float64),
        W_indices=Wsub.indices.astype(np.int32),
        W_indptr=Wsub.indptr.astype(np.int32),
        W_shape=np.array(Wsub.shape, dtype=np.int64),
        sign=sign[chosen],
        layer=layer[chosen],
        cell_type=ct[chosen],
        root_id=nid[chosen],
        parent_idx=chosen.astype(np.int32),
        meta_json=json.dumps(meta),
    )
    (DATA / "finch_module_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"finch subsample N={meta['N']} nnz={meta['nnz']} types={meta['type_counts']}")
    return meta


def build_worm():
    z = np.load(WORM_SRC, allow_pickle=True)
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    names = np.asarray(z["names"]).astype(str)
    sign = z["sign"].astype(np.int8)
    N = W.shape[0]
    layer = np.array(["L2/3"] * N, dtype="<U4")
    cell_type = np.array(["inter"] * N, dtype="<U8")
    for i, n in enumerate(names):
        base = n  # already short names
        if base in SENSORY or base.rstrip("LR") in {s.rstrip("LR") for s in SENSORY}:
            # exact match preferred
            pass
        if base in SENSORY:
            layer[i] = "L4"
            cell_type[i] = "sensory"
        elif base in COMMAND:
            layer[i] = "L5"
            cell_type[i] = "command"
        else:
            # heuristic: *L/*R sensory prefixes
            stem = base[:-1] if base[-1] in "LR" and len(base) > 2 else base
            sens_stems = {s[:-1] if s[-1] in "LR" else s for s in SENSORY}
            cmd_stems = {s[:-1] if s[-1] in "LR" else s for s in COMMAND}
            if base in SENSORY or stem in sens_stems:
                layer[i] = "L4"
                cell_type[i] = "sensory"
            elif base in COMMAND or stem in cmd_stems:
                layer[i] = "L5"
                cell_type[i] = "command"
    meta = dict(
        source="Cook2019 COOK_H.npz",
        N=int(N),
        nnz=int(W.nnz),
        n_L4=int((layer == "L4").sum()),
        n_L5=int((layer == "L5").sum()),
        n_L23=int((layer == "L2/3").sum()),
        sensory_names=sorted(names[layer == "L4"].tolist()),
        command_names=sorted(names[layer == "L5"].tolist()),
    )
    out = DATA / "worm_module.npz"
    np.savez_compressed(
        out,
        W_data=W.data.astype(np.float64),
        W_indices=W.indices.astype(np.int32),
        W_indptr=W.indptr.astype(np.int32),
        W_shape=np.array(W.shape, dtype=np.int64),
        sign=sign,
        layer=layer,
        cell_type=cell_type,
        root_id=names,
        names=names,
        meta_json=json.dumps({k: v for k, v in meta.items() if k not in ("sensory_names", "command_names")}),
    )
    (DATA / "worm_module_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"worm N={meta['N']} nnz={meta['nnz']} L4={meta['n_L4']} L5={meta['n_L5']} L23={meta['n_L23']}")
    return meta


def main():
    for name in ("mouse_module.npz", "human_module.npz", "fly_cx_module.npz",
                 "octo_vl_synthetic.npz", "octo_vl_random.npz"):
        symlink(FRANK / name, DATA / name)
    # also expose octo as octo_module.npz
    symlink(FRANK / "octo_vl_synthetic.npz", DATA / "octo_module.npz")
    symlink(MYELIN, DATA / "human_myelin.npz")
    fm = subsample_finch(2000, seed=42)
    wm = build_worm()
    manifest = dict(finch=fm, worm=wm, base=["mouse", "human", "fly_cx"],
                    octo="SYNTHETIC octo_vl_synthetic.npz")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done", DATA)


if __name__ == "__main__":
    main()

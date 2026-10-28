import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigs
D = "/workspace/flybrain/data"
a = pd.read_csv(f"{D}/ann.tsv", sep="\t", low_memory=False, usecols=["root_id", "super_class", "cell_class", "cell_type"])
ct = a.cell_type.astype(str); g = np.random.default_rng(0)
grp = {"CX": a.cell_class == "CX", "MBON": a.cell_class == "MBON", "PAM": ct.str.startswith("PAM"), "PPL": ct.str.startswith("PPL"),
       "OA": ct.str.startswith("OA-"), "DN": a.super_class == "descending", "ALPN": a.cell_class == "ALPN", "TuBu": a.cell_class == "TuBu",
       "ORN": a.cell_class == "olfactory"}
vpn = a.root_id[a.super_class == "visual_projection"].values; vs = set(g.choice(vpn, 1000, replace=False))
grp["VPN"] = a.root_id.isin(vs)
ids = {k: np.unique(a.root_id[m].values) for k, m in grp.items()}
order = []; lab = {}
for k, v in ids.items():
    v = [x for x in v if x not in lab]; lab.update({x: k for x in v}); order += v
order = np.array(order); idx = pd.Series(np.arange(len(order)), index=order)
d = pd.read_parquet(f"{D}/Drosophila_brain_model/Connectivity_783.parquet", columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"])
d = d[d.Presynaptic_ID.isin(idx.index) & d.Postsynaptic_ID.isin(idx.index)]
W = sp.csr_matrix((d["Excitatory x Connectivity"].values.astype(float), (idx[d.Postsynaptic_ID].values, idx[d.Presynaptic_ID].values)), shape=(len(order),) * 2)
r = abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-6)[0]); W = W / r
labs = np.array([lab[x] for x in order])
sp.save_npz("out/circuit.npz", W.tocsr()); np.savez("out/circuit_meta.npz", root_id=order, group=labs)
print({k: int((labs == k).sum()) for k in ids}, "N", len(order), "edges", W.nnz)

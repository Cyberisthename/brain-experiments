"""Plastic hippocampus (CA3 covariance+BCM Hebbian on CA3->CA3 and DG->CA3; CA1 contrastive delta rule;
CA1->CX targets at E-PG/P-EG, FB tangential, or random CX). See PREREGISTRATION_hippo_plastic.md. Resumable."""
import os, sys, json, time
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
sys.path.insert(0, "/workspace/flybrain/regions/hippo"); sys.path.insert(0, "/workspace/flybrain")
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
import hippo as Hh, flybrain_reservoir as F
from hippo import EC, DG, CA3, CA1, NM, NC, nEC, nDG, nCA3, nCA1, CXW, proj, kwta
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out"); JF = os.path.join(OUT, "jobs.jsonl")
ann = pd.read_csv("/workspace/flybrain/data/ann.tsv", sep="\t", low_memory=False, usecols=["root_id", "cell_class", "cell_type"])
CXID = np.array(sorted(set(ann.root_id[ann.cell_class == "CX"]))); assert len(CXID) == NC
ct = ann.drop_duplicates("root_id").set_index("root_id").cell_type.astype(str).reindex(CXID).values
TGT = {"EPG": np.where(np.isin(ct, ["EPG", "PEG", "EPGt"]))[0], "FBT": np.where(pd.Series(ct).str.match(r"^FB\d").values)[0], "RCX": np.arange(NC)}
# cond: (wiring, plastic, targets)
CONDS = {"CX": ("CX", 0, None), "HP_EPG": ("HIPPO", 1, "EPG"), "HP_FBT": ("HIPPO", 1, "FBT"), "HP_RCX": ("HIPPO", 1, "RCX"),
         "HS_EPG": ("HIPPO", 0, "EPG"), "RANDP_EPG": ("RAND", 1, "EPG"), "SHUFP_EPG": ("SHUF", 1, "EPG")}
LEAKS, RHOS, TUNE, TEST = Hh.LEAKS, Hh.RHOS, Hh.TUNE, Hh.TEST
ETA, ETA1, EPOCHS, WCLIP = 15.0, 15.0, 5, 3.0

def build(cond, s):
    wire, pl, tg = CONDS[cond]; g = np.random.default_rng(5000 + s)
    if wire == "CX": return dict(W=CXW, dg=np.arange(0), ro=np.arange(NC), pl=0)
    parts = [proj(g, EC, DG, 20), proj(g, DG, CA3, 10, 4.0), proj(g, EC, CA3, 20), proj(g, CA3, CA3, 40), proj(g, CA3, CA1, 40), proj(g, EC, CA1, 20), proj(g, CA1, EC, 10)]
    pre, post, w = [np.concatenate(x) for x in zip(*parts)]; k = pre != post; pre, post, w = pre[k], post[k], w[k]
    dg, outn, ca3, ecin = DG, CA1, CA3, EC
    if wire == "SHUF": pre, post, w, _ = F.deg(NM, pre, post, w, g)
    if wire == "RAND":
        E = len(pre); key = np.unique(g.integers(0, NM, (int(E * 1.05), 2)) @ np.array([NM, 1]))
        key = g.permutation(key[(key // NM) != (key % NM)])[:E]; pre, post = key // NM, key % NM; w = g.permutation(w)[:len(pre)]
        perm = g.permutation(NM); dg, outn, ca3, ecin = [np.sort(perm[a:b]) for a, b in [(0, nDG), (nDG, nDG + nCA1), (nDG + nCA1, nDG + nCA1 + nCA3), (nDG + nCA1 + nCA3, NM)]]
    M = sp.csr_matrix((w, (post, pre)), shape=(NM, NM)); M = M / Hh.eigs(M, k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-6)[0].__abs__()
    a_pre = np.concatenate([g.choice(NC, 50, replace=False) for _ in ecin]); a_post = np.repeat(ecin, 50)
    tp = TGT[tg]; b_pre = np.repeat(outn, 30); b_post = np.concatenate([g.choice(tp, 30, replace=len(tp) < 30) for _ in outn])
    sc = 1 / np.sqrt(50)
    A = sp.csr_matrix((sc * g.lognormal(0, .5, len(a_pre)), (a_post, a_pre)), shape=(NM, NC))
    B = sp.csr_matrix((sc * g.choice([-1, 1], len(b_pre)) * g.lognormal(0, .5, len(b_pre)), (b_post, b_pre)), shape=(NC, NM))
    return dict(CXW=CXW, A=A, B=B, M=M.tocsr(), dg=NC + dg, ro=np.concatenate([np.arange(NC), NC + outn]), ca3=ca3, ca1=outn, ec=ecin, mdg=dg, pl=pl)
def assemble(G): return sp.bmat([[G["CXW"], G["B"]], [G["A"], G["M"]]]).tocsr() if "M" in G else G["W"]

def masked_block(M, rows, cols):
    sub = M[rows][:, cols].tocoo(); return sub.row, sub.col, sub.data.copy()
def learn(G, leak, rho, Pin, rng):
    """Storage phase: Pin (P,NC) clean pattern drives. Online, batch-averaged updates for EPOCHS x 10 steps.
    CA3 covariance rule with BCM sliding threshold on CA3->CA3 and DG->CA3 (existing synapses only), row L1-renormalised and clipped.
    CA1: contrastive (GeneRec-style) delta on CA3->CA1: plus phase = CA3+EC drive, minus phase = CA3-only drive."""
    M = G["M"].tolil(); M = M.tocsr(); ca3, ca1, mdg, ec = G["ca3"], G["ca1"], G["mdg"], G["ec"]
    blocks = {"rec": (ca3, ca3), "mf": (ca3, mdg), "c31": (ca1, ca3)}
    BL = {k: masked_block(M, r, c) for k, (r, c) in blocks.items()}; norm0 = {}
    for k, (r, c, d) in BL.items(): norm0[k] = np.bincount(r, np.abs(d), minlength=len(blocks[k][0])) + 1e-9
    theta = np.full(len(ca3), 0.05); ec_ca1 = M[ca1][:, ec]
    for ep in range(EPOCHS):
        W = assemble(G) * rho; S = np.zeros((len(Pin), W.shape[0])); WT = W.T.tocsr()
        for t in range(10):
            S = (1 - leak) * S + leak * kwta(np.tanh(S @ WT + Pin), G["dg"]); x = S[:, NC:]
            post = x[:, ca3]; theta = 0.9 * theta + 0.1 * (post ** 2).mean(0)  # BCM sliding threshold
            phi = post * (post - theta)                                             # BCM post term
            for k in ("rec", "mf"):
                r, c, d = BL[k]; pre = x[:, blocks[k][1]]; prec = pre - pre.mean(0)   # covariance (pre centred)
                d += ETA * np.einsum("bi,bi->i", phi[:, r], prec[:, c]) / len(Pin)
            # CA1 contrastive: plus = tanh(CA3 + EC input), minus = tanh(CA3-only input)
            r, c, d = BL["c31"]; c3 = x[:, ca3]
            h3 = np.zeros((len(Pin), len(ca1))); np.add.at(h3.T, r, (d[:, None] * c3[:, c].T))
            plus = np.tanh(rho * (h3 + x[:, ec] @ ec_ca1.T)); minus = np.tanh(rho * h3)
            d += ETA1 * np.einsum("bi,bi->i", (plus - minus)[:, r], c3[:, c]) / len(Pin)
        for k, (r, c, d) in BL.items():  # stabilise: clip, then renormalise each row to its original L1 norm
            np.clip(d, -WCLIP, WCLIP, out=d); n = np.bincount(r, np.abs(d), minlength=len(blocks[k][0])) + 1e-9; d *= (norm0[k] / n)[r]
        M = M.tolil()
        for k, (r, c, d) in BL.items(): M[blocks[k][0][r], blocks[k][1][c]] = d
        G["M"] = M.tocsr()
    return G

def pc(G, leak, rho, rng, val):
    P, d = 20, 20; ntr, nte = (800, 200) if val else (800, 400)
    pats = rng.choice([-1.0, 1.0], (P, d)); lab = rng.integers(0, P, ntr + nte)
    cue = pats[lab] * (rng.random((ntr + nte, d)) > 0.5) + 0.5 * rng.standard_normal((ntr + nte, d))
    Win, b = Hh.inp(NC + (NM if "M" in G else 0), rng, d)
    G = dict(G)
    if G["pl"]: G = learn(G, leak, rho, pats @ Win.T + b, rng)
    W = assemble(G) * rho; on = cue @ Win.T + b; off = np.zeros_like(on) + b
    S = Hh.batch(W, G["dg"], [on] * 5 + [off] * 5, leak, 10)[:, G["ro"]]
    B = F.ridge(S[:ntr], pats[lab[:ntr]], 1e-4); out = F.pred(S[ntr:], B)
    return float(np.mean(np.argmax(out @ pats.T, 1) == lab[ntr:]))
# robustness: PC after deleting 10% of module+CX synapses at random (worst over 3 draws), test only
def job(a):
    c, s, leak, rho, val = a; t = time.time(); G = build(c, s); g = np.random.default_rng(s + 3 if val else 10 * s + 3)
    v = pc(G, leak, rho, g, val); rob = None
    if not val:
        rs = []
        for r in range(3):
            G2 = build(c, s); gg = np.random.default_rng(777 + 10 * s + r)
            for key in ("M", "CXW", "W"):
                if key in G2: X = G2[key].tocoo(); k = gg.random(X.nnz) > 0.1; G2[key] = sp.csr_matrix((X.data[k], (X.row[k], X.col[k])), shape=X.shape)
            rs.append(pc(G2, leak, rho, np.random.default_rng(10 * s + 3), False))
        rob = min(rs)
    return dict(cond=c, seed=s, leak=leak, rho=rho, val=val, value=v, worst_lesion=rob, sec=time.time() - t)
def done():
    return {(r["cond"], r["seed"], r["leak"], r["rho"], r["val"]): r for r in map(json.loads, open(JF))} if os.path.exists(JF) else {}
def runall(P, J):
    have = done(); J = [j for j in J if tuple(j) not in have]
    with open(JF, "a") as f:
        for r in P.imap_unordered(job, J, chunksize=1): f.write(json.dumps(r) + "\n"); f.flush()
if __name__ == "__main__":
    if len(sys.argv) > 1:
        for c in CONDS: t = time.time(); print(c, round(pc(build(c, 0), .5, .95, np.random.default_rng(0), True), 3), round(time.time() - t, 1), flush=True)
        sys.exit()
    P = Pool(int(os.environ.get("WORKERS", 3)))
    runall(P, [(c, s, l, r, True) for c in CONDS for s in TUNE for l in LEAKS for r in RHOS])
    sw = pd.DataFrame(list(done().values())); sw = sw[sw.val]; m = sw.groupby(["cond", "leak", "rho"]).value.mean().reset_index()
    best = {c: tuple(map(float, mm.loc[mm.value.idxmax(), ["leak", "rho"]])) for c, mm in m.groupby("cond")}
    json.dump(best, open(os.path.join(OUT, "best.json"), "w"), indent=1)
    runall(P, [(c, s, *best[c], False) for c in CONDS for s in TEST]); print("all done", flush=True)

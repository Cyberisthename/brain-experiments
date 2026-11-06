#!/usr/bin/env python3
"""NEURON-4: autonomous relationship discovery. See PREREG.md. No QPU, no external models."""
import sys, os, json, re, hashlib, random, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
assert hashlib.sha256((ROOT/"n3lib.py").read_bytes()).hexdigest().startswith("2a392b927bdb"), "n3lib changed"
import n3lib as N
OUT = ROOT/"results"; OUT.mkdir(exist_ok=True)
SEEDS = list(range(int(os.environ.get("N4_NSEEDS", "20"))))
GOLD_Q = {"gold","kind","cluster","gold_fact","src","hop_src","true_entity","qid"}
GOLD_F = {"uncertain","prereg"}
SYSTEMS = ["exact","rag","A","D_auto","E_auto","E_oracle"]

# ---------- file-read audit ----------
READS = []; PHASE = ["eval"]
def hook(ev, args):
    if ev == "open" and PHASE[0] == "system" and isinstance(args[0], (str, bytes, os.PathLike)):
        READS.append(str(args[0]))
sys.addaudithook(hook)

# ---------- corpora (evaluator side) ----------
C2_ROOTS = ["brain_qa","bookish-umbrella","chatbot","chatbot_upload","chatbot-site-src","chatbot_newest","pr164","jarvis_model","Jarvis-2v","v","aee"]
def c2_files():
    ps = set()
    for r in C2_ROOTS:
        for p in Path("/workspace", r).rglob("*.md"):
            s = str(p)
            if any(x in s for x in ("/node_modules/","/.venv/","/.git/","__MACOSX")) or not p.is_file(): continue
            ps.add(p)
    return sorted(ps)
def load(corpus):
    if corpus == 2:
        orig = N.list_source_files; N.list_source_files = c2_files
        try: r = N.build_corpus()
        finally: N.list_source_files = orig
        # drop anything also in corpus 1 by content
        return r
    return N.build_corpus()

# ---------- system-side views ----------
HDR = re.compile(r"^\[meta date=(\S+) project=(.+?) file=(.+?)\]\n", re.S)
def mem_view(m):
    return dict(id=m["id"], text=m["text"], project=HDR.match(m["text"]).group(2), path=m["id"].split("#")[0])
def q_view(q):
    v = {k: q.get(k) for k in ("text","intent","project","entity","attr","sattr","sval")}
    assert not (set(v) & GOLD_Q); return v
def auto_extract(views):
    facts = []
    for m in views:
        mt = HDR.match(m["text"]); date, proj, fname = mt.group(1), mt.group(2), mt.group(3)
        body = m["text"][mt.end():]
        for f in N.extract_facts_from_text(body, "/x/" + fname, date, proj):
            for k in GOLD_F: f.pop(k)
            f["path"] = m["path"]; f["line"] = (m["id"], f["line"]); f["memory_id"] = m["id"]; f["fname"] = fname
            facts.append(f)
    for i, f in enumerate(facts): f["id"] = i
    return facts

# ---------- uncertainty classifier (numpy logreg, hashed n-grams) ----------
D = 2**12
def feats(f):
    toks = []
    raw = f["raw"].lower()
    for n in (1,2,3):
        toks += ["c%d:%s" % (n, raw[i:i+n]) for i in range(len(raw)-n+1) if not raw[i:i+n].isdigit()]
    toks += ["w:"+w for w in re.findall(r"[a-z]+", raw)]
    toks += ["h:"+w for w in re.findall(r"[a-z]+", f["heading"])]
    toks += ["f:"+w for w in re.findall(r"[a-z]+", f["fname"].lower())]
    idx = sorted({int(hashlib.md5(t.encode()).hexdigest()[:8], 16) % D for t in toks})
    return idx
import scipy.sparse as sp
def to_X(F):
    r, c = [], []
    for i, f in enumerate(F):
        ix = feats(f); r += [i]*len(ix); c += ix
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(F), D))
def fit(X, y, l2=1e-3, it=300, lr=0.5):
    w = np.zeros(X.shape[1], np.float32); b = 0.0
    pw = (len(y) - y.sum()) / max(1, y.sum())
    sw = np.where(y == 1, pw, 1.0).astype(np.float32); sw /= sw.mean()
    for _ in range(it):
        p = 1/(1+np.exp(-(X@w+b))); g = (p - y) * sw
        w -= lr * (X.T@g/len(y) + l2*w); b -= lr * g.mean()
    return w, b
def f1(y, p):
    tp = ((y==1)&(p==1)).sum(); fp = ((y==0)&(p==1)).sum(); fn = ((y==1)&(p==0)).sum()
    return 2*tp/max(1, 2*tp+fp+fn)
def train_clf(F, y, seed):
    X = to_X(F); rng = np.random.default_rng(seed); fold = rng.integers(0, 3, len(y)); P = np.zeros(len(y))
    for k in range(3):
        w, b = fit(X[fold != k], y[fold != k]); P[fold == k] = 1/(1+np.exp(-(X[fold == k]@w+b)))
    thr = max(np.arange(0.05, 0.96, 0.05), key=lambda t: f1(y, (P >= t).astype(int)))
    w, b = fit(X, y)
    return w, b, float(thr)

def gold_key(f, mid): return (mid, f["entity"], f["attr"], f["value"])

# ---------- systems ----------
class Named:
    def __init__(self, s, name, extra=0, fmap=None): self.s = s; self.name = name; self.native = s.native + extra; self.fmap = fmap
def build(views, afacts, pred_unc, gold_mem, gold_facts, seed, cap, clf_bytes):
    def mp(F):
        d = defaultdict(list)
        for f in F: d[f["memory_id"]].append(f)
        return d
    am = mp(afacts)
    S = [Named(N.Exact(views, cap), "exact", fmap=am), Named(N.RAG(views, cap), "rag", fmap=am)]
    plain = [dict(f, uncertain=False, prereg=False) for f in afacts]   # A never reads flags; constant False, not gold
    S.append(Named(N.Graph("A", views, plain, N.build_edges(plain), "A", seed, cap), "A", fmap=mp(plain)))
    pf = [dict(f, uncertain=bool(pred_unc[f["id"]]), prereg=False) for f in afacts]
    E2 = N.build_edges(pf); pm = mp(pf)
    S.append(Named(N.Graph("D", views, pf, E2, "D", seed, cap), "D_auto", clf_bytes, pm))
    S.append(Named(N.Graph("E", views, pf, E2, "E", seed, cap), "E_auto", clf_bytes, pm))
    S.append(Named(N.Graph("Eo", gold_mem, gold_facts, N.build_edges(gold_facts), "E", seed, cap), "E_oracle", fmap=mp(gold_facts)))
    return S
def ask(sysn, q, auto_map, gold_map):
    N.FACTS_BY_MEM.clear(); N.FACTS_BY_MEM.update(sysn.fmap)
    if sysn.name != "E_oracle":
        assert all(not (GOLD_F & set(f)) or f.get("prereg") is False for fl in sysn.fmap.values() for f in fl[:1])
    if sysn.name == "E_oracle": return sysn.s.answer(q)
    PHASE[0] = "system"
    try: return sysn.s.answer(q_view(q))
    finally: PHASE[0] = "eval"

def run_seed(seed, C1, C2):
    t0 = time.time(); files, mems, facts = C1
    gmap = defaultdict(list)
    for f in facts: gmap[f["memory_id"]].append(f)
    projs = sorted({m["project"] for m in mems})
    test_p = {p for p in projs if N.h01("proj", seed, p) < 0.5}
    if not test_p or test_p == set(projs): test_p = set(projs[: len(projs)//2])
    held = {m["path"] for m in mems if m["project"] in test_p}
    dev, test, _ = N.make_questions(facts, mems, seed, held)
    blob = "\n".join(m["text"] for m in mems)
    assert not [q for q in test if q["text"] in blob]
    PHASE[0] = "system"
    views = [mem_view(m) for m in mems]; af = auto_extract(views)
    PHASE[0] = "eval"
    amap = defaultdict(list)
    for f in af: amap[f["memory_id"]].append(f)
    # train labels: gold uncertainty on TRAIN projects only
    glab = {gold_key(f, f["memory_id"]): f["uncertain"] for f in facts}
    tr = [f for f in af if f["project"] not in test_p and gold_key(f, f["memory_id"]) in glab]
    y = np.array([int(glab[gold_key(f, f["memory_id"])]) for f in tr])
    print(f"  seed {seed} corpus/extract {time.time()-t0:.0f}s", flush=True)
    w, b, thr = train_clf(tr, y, seed)
    print(f"  seed {seed} clf {time.time()-t0:.0f}s", flush=True)
    clf_bytes = N.zb(w.astype(np.float16).tobytes())
    PHASE[0] = "system"
    Xa = to_X(af); pred = ((1/(1+np.exp(-(Xa@w+b)))) >= thr).astype(int)
    PHASE[0] = "eval"
    te = [f for f in af if f["project"] in test_p and gold_key(f, f["memory_id"]) in glab]
    yt = np.array([int(glab[gold_key(f, f["memory_id"])]) for f in te]); pt = np.array([pred[f["id"]] for f in te])
    clf = dict(seed=seed, thr=thr, n_train=len(y), train_pos=int(y.sum()), test_f1=float(f1(yt, pt)),
               test_prec=float(((yt==1)&(pt==1)).sum()/max(1,pt.sum())), test_rec=float(((yt==1)&(pt==1)).sum()/max(1,yt.sum())),
               n_test_facts=len(te), auto_facts=len(af), gold_facts=len(facts), clf_bytes=clf_bytes)
    # equal budget
    S = build(views, af, pred, mems, facts, seed, None, clf_bytes); reads = []
    for s in S:
        for q in dev: ask(s, q, amap, gmap)
        reads.append(s.s.read / max(1, s.s.nq))
    cap = max(reads); print(f"  seed {seed} dev {time.time()-t0:.0f}s", flush=True)
    S = build(views, af, pred, mems, facts, seed, cap, clf_bytes)
    rows = run_qs(S, test, mems, amap, gmap, seed, 1)
    # corpus 2 (cross-source, all unseen)
    print(f"  seed {seed} test1 {time.time()-t0:.0f}s", flush=True); rows2 = []
    if C2:
        f2, m2, fa2 = C2
        g2 = defaultdict(list)
        for f in fa2: g2[f["memory_id"]].append(f)
        _, t2, _ = N.make_questions(fa2, m2, seed + 500, {m["path"] for m in m2})
        PHASE[0] = "system"
        v2 = [mem_view(m) for m in m2]; a2 = auto_extract(v2); p2 = ((1/(1+np.exp(-(to_X(a2)@w+b)))) >= thr).astype(int)
        PHASE[0] = "eval"
        am2 = defaultdict(list)
        for f in a2: am2[f["memory_id"]].append(f)
        S2 = build(v2, a2, p2, m2, fa2, seed, cap, clf_bytes)
        rows2 = run_qs(S2, t2, m2, am2, g2, seed, 2)
    storage = {s.name: s.native for s in S}
    print(f"seed {seed} n1={len(test)} n2={len(rows2)} f1={clf['test_f1']:.3f} t={time.time()-t0:.0f}s", flush=True)
    return rows, rows2, clf, storage, cap

def run_qs(S, test, mems, amap, gmap, seed, corpus):
    out = []
    for q in test:
        rec = dict(seed=seed, corpus=corpus, qid=q["qid"], kind=q["kind"], cluster=q["cluster"], question=q["text"], gold=q["gold"])
        for s in S:
            s.s.read = 0
            ans, conf = ask(s, q, amap, gmap)
            c, a, wr, u = N.score(q["gold"], ans)
            rec[s.name] = dict(answer=ans if ans is not None else "ABSTAIN", confidence=round(float(conf), 4),
                               correct=c, abstain=a, wrong=wr, bytes_read=s.s.read)
        out.append(rec)
    return out

_C = None
def _worker(seed):
    READS.clear(); out = run_seed(seed, *_C); return out + (list(READS),)
def main():
    t0=time.time(); C1 = load(1); print('c1',time.time()-t0,flush=True)
    c1hash = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in {m["path"] for m in C1[1]}}
    f2, m2, fa2 = load(2)
    m2 = [m for m in m2 if hashlib.sha256(Path(m["path"]).read_bytes()).hexdigest() not in c1hash]
    keep = {m["id"] for m in m2}; fa2 = [f for f in fa2 if f["memory_id"] in keep]
    for i, f in enumerate(fa2): f["id"] = i
    meta = dict(c1_files=len({m["path"] for m in C1[1]}), c1_mem=len(C1[1]), c1_facts=len(C1[2]),
                c2_files=len({m["path"] for m in m2}), c2_mem=len(m2), c2_facts=len(fa2),
                c2_projects=sorted({m["project"] for m in m2}))
    print(json.dumps(meta), flush=True); (OUT/"corpus_meta.json").write_text(json.dumps(meta, indent=1))
    C2 = (None, m2, fa2) if fa2 else None
    allr, allr2, clfs, stor = [], [], [], []
    (OUT/"seeds").mkdir(exist_ok=True)
    if os.environ.get("N4_ONE"):
        sd = int(os.environ["N4_ONE"]); res = run_seed(sd, C1, C2)
        (OUT/"seeds"/f"seed{sd}.json").write_text(json.dumps(list(res) + [list(READS)])); return
    if os.environ.get("N4_MERGE"):
        res = [tuple(json.loads((OUT/"seeds"/f"seed{sd}.json").read_text())) for sd in SEEDS]
    global _C
    _C = (C1, C2)
    import multiprocessing as mp
    if not os.environ.get("N4_MERGE"):
        with mp.get_context("fork").Pool(int(os.environ.get("N4_PROCS", "7"))) as pool:
            res = pool.map(_worker, SEEDS)
    for seed, (r, r2, c, st, cap, rd) in zip(SEEDS, res):
        allr += r; allr2 += r2; clfs.append(c); stor.append(dict(seed=seed, cap=cap, **st)); READS.extend(rd)
    bad = [p for p in READS if "/results" in p or "neuron3" in p or "per_question" in p]
    (OUT/"system_file_reads.json").write_text(json.dumps(dict(n_reads=len(READS), reads=sorted(set(READS)), forbidden=bad), indent=1))
    assert not bad, bad
    with open(OUT/"per_question.jsonl", "w") as fh:
        for r in allr + allr2: fh.write(json.dumps(r) + "\n")
    (OUT/"classifier.json").write_text(json.dumps(clfs, indent=1)); (OUT/"storage.json").write_text(json.dumps(stor, indent=1))
    (OUT/"run_meta.json").write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seeds=SEEDS), indent=1))
    print("done")
if __name__ == "__main__": main()

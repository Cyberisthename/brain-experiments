#!/usr/bin/env python3
"""QMS-1: classical-first search over NEURON-4 E-auto boundary-policy switches. No QPU. See PREREG.md"""
import sys, os, json, time, itertools, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/workspace/jarvis/neuron/neuron4")
import run_neuron4 as R4
N = R4.N
OUT = Path(__file__).resolve().parent / "results"; (OUT/"seeds").mkdir(parents=True, exist_ok=True)
NV, B = 10, 64
DEFAULT = (1,1,1,1,0,0,0,0,0,0)
CONFIGS = list(itertools.product((0,1), repeat=NV))
OFF = {(0,0):0.0,(1,0):-0.15,(0,1):0.15,(1,1):0.3}

def capture(sysobj, q):
    rec = []
    orig = sysobj.read_facts
    sysobj.read_facts = lambda qq, c: (rec.append((dict(qq), list(c))), (None, 0.0))[1]
    try: sysobj.answer(R4.q_view(q))
    finally: sysobj.read_facts = orig
    return rec[0] if rec else None

def read(x, qq, cands, prob, thr, fb):
    t = thr + OFF[(x[4], x[5])]
    unc = lambda f: prob[f["id"]] >= t or (x[8] and f["fname"].startswith("PREREG"))
    c = [f for f in cands if f["entity"] == qq["entity"] and f["attr"] == qq["attr"]]
    if not c and x[9] and fb: c = [f for f in fb[1] if f["entity"] == qq["entity"] and f["attr"] == qq["attr"]]
    if x[3]: c = [f for f in c if f["project"] == qq["project"]]
    if not c: return None
    cert = [f for f in c if not unc(f)]
    if x[0]:
        cert = [f for f in cert if not any(g["date"] > f["date"] and g["value"] != f["value"] for g in cert)] or cert
    if x[6] and qq["intent"] == "latest" and cert:
        md = max(f["date"] for f in cert); cert = [f for f in cert if f["date"] == md]
    pool = cert
    if not cert:
        if x[2]: return None
        pool = c
    vals = defaultdict(int)
    for f in pool: vals[f["value"]] += 1
    if len(vals) > 1 and x[1]: return None
    v, k = max(vals.items(), key=lambda kv: (kv[1], kv[0]))
    if x[7] and k < 2: return None
    return v

def evaluate(x, Q):
    res = []
    for q, rec, fb, prob, thr in Q:
        a = read(x, rec[0], rec[1], prob, thr, fb) if rec else None
        res.append(N.score(q["gold"], a)[:3] + (a,))
    return res
def obj(x, Q):
    r = evaluate(x, Q); return np.mean([c for c, a, w, _ in r]) - 2*np.mean([w for c, a, w, _ in r])

def prep(seed, extra_mem=None):
    files, mems, facts = N.build_corpus()
    projs = sorted({m["project"] for m in mems}); test_p = {p for p in projs if N.h01("proj", seed, p) < 0.5}
    if not test_p or test_p == set(projs): test_p = set(projs[: len(projs)//2])
    held = {m["path"] for m in mems if m["project"] in test_p}; train_paths = {m["path"] for m in mems} - held
    _, test, _ = N.make_questions(facts, mems, seed, held)
    _, optq, _ = N.make_questions(facts, mems, seed + 7919, train_paths)
    assert not ({q["cluster"] for q in optq} & {q["cluster"] for q in test})
    views = [R4.mem_view(m) for m in mems]
    if extra_mem: views += [R4.mem_view(m) for m in extra_mem]
    af = R4.auto_extract(views)
    glab = {R4.gold_key(f, f["memory_id"]): f["uncertain"] for f in facts}
    tr = [f for f in af if f["project"] not in test_p and R4.gold_key(f, f["memory_id"]) in glab]
    y = np.array([int(glab[R4.gold_key(f, f["memory_id"])]) for f in tr]); w, b, thr = R4.train_clf(tr, y, seed)
    prob = 1/(1+np.exp(-(R4.to_X(af)@w+b))); pred = (prob >= thr).astype(int)
    S = R4.build(views, af, pred, mems, facts, seed, None, 0)
    dev, _, _ = N.make_questions(facts, mems, seed, held)
    reads = []
    for s in S:
        for q in dev: R4.ask(s, q, None, None)
        reads.append(s.s.read/max(1, s.s.nq))
    S = {s.name: s for s in R4.build(views, af, pred, mems, facts, seed, max(reads), 0)}
    def cache(qs):
        out = []
        for q in qs:
            N.FACTS_BY_MEM.clear(); N.FACTS_BY_MEM.update(S["E_auto"].fmap); rec = capture(S["E_auto"].s, q)
            N.FACTS_BY_MEM.clear(); N.FACTS_BY_MEM.update(S["exact"].fmap); fb = capture(S["exact"].s, q)
            out.append((q, rec, fb, prob, thr))
        return out
    return cache(optq), cache(test), facts, dict(thr=thr, test_projects=sorted(test_p), n_opt=len(optq), n_test=len(test))

def s_exact(f):
    t = time.time(); v = {x: f(x) for x in CONFIGS}; return max(CONFIGS, key=lambda x: (v[x], x == DEFAULT)), len(v), time.time()-t, v
def s_random(f, rng):
    t = time.time(); xs = [tuple(int(v) for v in rng.integers(0, 2, NV)) for _ in range(B)]; return max(xs, key=f), B, time.time()-t
def s_greedy(f):
    t = time.time(); x = DEFAULT; fx = f(x); n = 1
    while n < B:
        best = None
        for i in range(NV):
            if n >= B: break
            y = list(x); y[i] ^= 1; y = tuple(y); fy = f(y); n += 1
            if fy > fx + 1e-12 and (best is None or fy > best[1]): best = (y, fy)
        if not best: break
        x, fx = best
    return x, n, time.time()-t
def s_sa(f, rng):
    t = time.time(); x = DEFAULT; fx = f(x); bx, bf = x, fx
    for k in range(B-1):
        T = 0.3*(1-k/B)+1e-3; i = int(rng.integers(NV)); y = list(x); y[i] ^= 1; y = tuple(y); fy = f(y)
        if fy >= fx or rng.random() < np.exp((fy-fx)/T): x, fx = y, fy
        if fx > bf: bx, bf = x, fx
    return bx, B, time.time()-t
def feats(x):
    x = np.array(x, float); return np.concatenate([[1], x, [x[i]*x[j] for i in range(NV) for j in range(i+1, NV)]])
def fit_surrogate(f, rng):
    xs = [tuple(int(v) for v in rng.integers(0, 2, NV)) for _ in range(B)]; ys = np.array([f(x) for x in xs])
    X = np.array([feats(x) for x in xs]); lam = 1e-2
    w = np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@ys)
    pred = np.array([feats(x)@w for x in CONFIGS]); return w, pred
def qaoa(pred, p, rng):
    C = -(pred - pred.mean()) / (pred.std() + 1e-9); n = NV; dim = 2**n
    # CONFIGS[i]: x0 is the most significant bit of i
    bits = [((np.arange(dim) >> (n-1-q)) & 1) == 0 for q in range(n)]
    def state(th):
        psi = np.full(dim, 1/np.sqrt(dim), complex)
        for l in range(p):
            psi = psi*np.exp(-1j*th[l]*C)
            c, s = np.cos(th[p+l]), -1j*np.sin(th[p+l])
            for q in range(n):
                lo = bits[q]; a0 = psi[lo].copy(); a1 = psi[~lo].copy()
                psi[lo] = c*a0 + s*a1; psi[~lo] = s*a0 + c*a1
        return psi
    E = lambda th: float(np.real(np.sum(np.abs(state(th))**2 * C)))
    best = None
    for g0 in np.linspace(0.1, 1.5, 4):
        for b0 in np.linspace(0.1, 1.2, 4):
            r = minimize(E, [g0]*p + [b0]*p, method="Nelder-Mead", options=dict(maxiter=200))
            if best is None or r.fun < best.fun: best = r
    pr = np.abs(state(best.x))**2; samp = rng.choice(dim, size=B, p=pr/pr.sum())
    bi = int(max(samp, key=lambda i: pred[i]))
    return CONFIGS[bi], dict(exp_cost=float(best.fun), p_surrogate_opt=float(pr[int(np.argmax(pred))]), uniform_p=1/dim, sampled_unique=int(len(set(samp.tolist()))))

def metrics(rows, Q):
    qs = [q for q, *_ in Q]; n = len(rows)
    ans = [(r, q) for r, q in zip(rows, qs) if r[3] is not None]; g = [(r, q) for r, q in zip(rows, qs) if q["gold"] != "ABSTAIN"]
    bk = [(r, q) for r, q in zip(rows, qs) if q["kind"] in ("conflict", "uncertain", "latest", "distractor")]
    mh = [(r, q) for r, q in zip(rows, qs) if q["kind"] == "multi_hop_3"]
    return dict(correct=float(np.mean([r[0] for r in rows])), coverage=len(ans)/n, ans_cov=sum(r[3] is not None for r, q in g)/max(1, len(g)),
                sel_acc=sum(r[0] for r, q in ans)/max(1, len(ans)), conf_wrong=float(np.mean([r[2] for r in rows])),
                boundary_acc=float(np.mean([r[0] for r, q in bk])) if bk else None, multi_hop=float(np.mean([r[0] for r, q in mh])) if mh else None)

def run_seed(seed):
    t0 = time.time()
    OPT, TEST, facts, info = prep(seed)
    f = lambda x: obj(x, OPT)
    ex, nex, tex, allv = s_exact(f)
    picks = {"default_E_auto": (DEFAULT, 0, 0.0), "exact": (ex, nex, tex)}
    picks["random"] = s_random(f, np.random.default_rng(seed+1)); picks["greedy"] = s_greedy(f); picks["sa"] = s_sa(f, np.random.default_rng(seed+2))
    t = time.time(); w, pred = fit_surrogate(f, np.random.default_rng(seed+3)); tfit = time.time()-t
    tv = np.array([allv[x] for x in CONFIGS]); r2 = 1 - np.sum((pred-tv)**2)/max(1e-12, np.sum((tv-tv.mean())**2))
    t = time.time(); cs = CONFIGS[int(np.argmax(pred))]; picks["surrogate_classical"] = (cs, B, tfit + time.time()-t)
    qinfo = {}
    for p in (1, 2):
        t = time.time(); xq, qi = qaoa(pred, p, np.random.default_rng(seed+10+p)); qinfo[p] = qi
        picks[f"surrogate_qaoa_p{p}"] = (xq, B, tfit + time.time()-t)
    out = {}
    for nm, (x, ne, wt) in picks.items():
        x = tuple(int(v) for v in x); rows = evaluate(x, TEST)
        out[nm] = dict(config=list(x), n_true_evals=ne, wall_s=wt, opt_obj=float(allv[x]), opt_gap=float(allv[ex]-allv[x]),
                       test_obj=float(np.mean([r[0] for r in rows]) - 2*np.mean([r[2] for r in rows])), test=metrics(rows, TEST),
                       per_q=[dict(qid=q["qid"], kind=q["kind"], cluster=q["cluster"], gold=q["gold"], answer=r[3], correct=r[0], wrong=r[2])
                              for r, (q, *_) in zip(rows, TEST)])
    m2, _ = N.synth_interference(1000, seed, facts)
    m2 = [dict(m, text=m["text"].replace("]\n", " file=synthetic.md]\n", 1)) for m in m2]
    _, TEST2, _, _ = prep(seed, extra_mem=m2)
    for nm in out:
        x = tuple(out[nm]["config"]); a = evaluate(x, TEST); b2 = evaluate(x, TEST2)
        out[nm]["acc_after"] = float(np.mean([r[0] for r in b2]))
        out[nm]["F"] = float(max(0, out[nm]["test"]["correct"] - out[nm]["acc_after"]) + np.mean([r1[0] and r2_[2] for r1, r2_ in zip(a, b2)]))
    res = dict(seed=seed, info=info, surrogate_R2_all1024=float(r2), qaoa=qinfo, solvers=out, runtime_s=time.time()-t0)
    (OUT/"seeds"/f"seed{seed}.json").write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"seed {seed} nopt={info['n_opt']} ntest={info['n_test']} R2={r2:.2f} exact={ex} t={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for sd in map(int, sys.argv[1:]): run_seed(sd)

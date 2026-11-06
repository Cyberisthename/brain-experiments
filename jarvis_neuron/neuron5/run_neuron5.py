#!/usr/bin/env python3
"""NEURON-5 physics-inspired (energy-minimisation) memory engine. See PREREG.md. No QPU."""
import sys, os, json, time, math, random, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/workspace/jarvis/neuron/neuron4")
import run_neuron4 as R4
N = R4.N
OUT = Path(__file__).resolve().parent / "results"; (OUT / "seeds").mkdir(parents=True, exist_ok=True)
LAM, J = 0.25, 10.0
BEST_N4 = os.environ.get("N5_BEST", "E_auto")

class PEnergy(N.Graph):
    def __init__(self, views, facts, punc, seed, cap, b=0.5, tau=0.25):
        super().__init__("P", views, facts, N.build_edges(facts), "A", seed, cap)
        self.punc = punc; self.b = b; self.tau = tau; self.sa_checks = []
    def energy_solve(self, h):
        vals = list(h); n = len(vals); best = []
        for bits in itertools.product((0, 1), repeat=n):
            e = -sum((h[v] - self.b) * s for v, s in zip(vals, bits)) + J * sum(bits[i]*bits[j] for i in range(n) for j in range(i+1, n))
            best.append((e, bits))
        best.sort(key=lambda x: x[0])
        return vals, best
    def sa(self, h, rng, steps=400):
        vals = list(h); s = [0]*len(vals)
        E = lambda s: -sum((h[v]-self.b)*x for v, x in zip(vals, s)) + J*sum(s[i]*s[j] for i in range(len(s)) for j in range(i+1, len(s)))
        cur = E(s); bestE, bestS = cur, list(s)
        for t in range(steps):
            T = 2.0 * (1 - t/steps) + 1e-3; i = rng.randrange(len(s)); s[i] ^= 1; ne = E(s)
            if ne <= cur or rng.random() < math.exp(-(ne-cur)/T): cur = ne
            else: s[i] ^= 1
            if cur < bestE: bestE, bestS = cur, list(s)
        return bestS
    def read_facts(self, q, cands):
        c = [f for f in cands if f["entity"] == q["entity"] and f["attr"] == q["attr"] and f["project"] == q["project"]]
        if not c: return None, 0.0
        h = defaultdict(float)
        for f in c:
            newer = sum(1 for g in c if g["date"] > f["date"] and g["value"] != f["value"])
            h[f["value"]] += (1 - self.punc[f["id"]]) * (LAM ** newer)
        h = dict(list(sorted(h.items(), key=lambda x: -x[1]))[:12])
        vals, ranked = self.energy_solve(h)
        e0, s0 = ranked[0]; gap = ranked[1][0] - e0 if len(ranked) > 1 else 99
        if random.random() < 0.05:
            sa = self.sa(h, random.Random(len(self.sa_checks)))
            self.sa_checks.append(tuple(sa) == tuple(s0))
        if sum(s0) != 1 or gap < self.tau: return None, gap
        return vals[s0.index(1)], gap

def setup(seed, mems, facts, extra=None):
    """returns systems dict built system-side from raw text only."""
    projs = sorted({m["project"] for m in mems if not m["project"].startswith("synthetic")})
    test_p = {p for p in projs if N.h01("proj", seed, p) < 0.5}
    if not test_p or test_p == set(projs): test_p = set(projs[: len(projs)//2])
    return test_p

def systems(views, af, w, b, thr, cap, seed, pb=0.5, ptau=0.25):
    Xa = R4.to_X(af); prob = 1/(1+np.exp(-(Xa@w+b))); pred = (prob >= thr).astype(int)
    S = R4.build(views, af, pred, [], [], seed, cap, 0) if False else None
    mp = lambda F: R4.defaultdict(list) if False else _mp(F)
    am = _mp(af); plain = [dict(f, uncertain=False, prereg=False) for f in af]
    pf = [dict(f, uncertain=bool(pred[f["id"]]), prereg=False) for f in af]; E2 = N.build_edges(pf); pm = _mp(pf)
    clfb = N.zb(w.astype(np.float16).tobytes())
    out = {"exact": R4.Named(N.Exact(views, cap), "exact", fmap=am), "rag": R4.Named(N.RAG(views, cap), "rag", fmap=am),
           "A": R4.Named(N.Graph("A", views, plain, N.build_edges(plain), "A", seed, cap), "A", fmap=_mp(plain))}
    mode = "E" if BEST_N4 == "E_auto" else "D"
    out[BEST_N4] = R4.Named(N.Graph(mode, views, pf, E2, mode, seed, cap), BEST_N4, clfb, pm)
    P = PEnergy(views, plain, {f["id"]: float(prob[f["id"]]) for f in af}, seed, cap, pb, ptau)
    out["P_energy"] = R4.Named(P, "P_energy", clfb, _mp(plain))
    return out
def _mp(F):
    d = defaultdict(list)
    for f in F: d[f["memory_id"]].append(f)
    return d

def run_seed(seed):
    t0 = time.time(); files, mems, facts = N.build_corpus()
    test_p = setup(seed, mems, facts); held = {m["path"] for m in mems if m["project"] in test_p}
    dev, test, _ = N.make_questions(facts, mems, seed, held)
    assert not [q for q in test if q["text"] in "\n".join(m["text"] for m in mems)]
    views = [R4.mem_view(m) for m in mems]; af = R4.auto_extract(views)
    glab = {R4.gold_key(f, f["memory_id"]): f["uncertain"] for f in facts}
    tr = [f for f in af if f["project"] not in test_p and R4.gold_key(f, f["memory_id"]) in glab]
    y = np.array([int(glab[R4.gold_key(f, f["memory_id"])]) for f in tr]); w, b, thr = R4.train_clf(tr, y, seed)
    S = systems(views, af, w, b, thr, None, seed); reads = []
    for s in S.values():
        for q in dev: R4.ask(s, q, None, None)
        reads.append(s.s.read / max(1, s.s.nq))
    cap = max(reads)
    # tune P's b, tau on TRAIN dev questions only
    S = systems(views, af, w, b, thr, cap, seed); P = S["P_energy"].s; best = None
    for pb, pt in itertools.product((0.25, 0.5, 0.75), (0.1, 0.25, 0.5, 0.75)):
        P.b, P.tau = pb, pt; u = 0
        for q in dev:
            c, a, wr, _ = N.score(q["gold"], R4.ask(S["P_energy"], q, None, None)[0]); u += c - 2*wr
        if best is None or u > best[0]: best = (u, pb, pt)
    P.b, P.tau = best[1], best[2]
    def evaluate(S, qs):
        rows = []
        for q in qs:
            rec = dict(seed=seed, qid=q["qid"], kind=q["kind"], cluster=q["cluster"], gold=q["gold"])
            for nm, s in S.items():
                t1 = time.perf_counter(); ans, conf = R4.ask(s, q, None, None); ms = (time.perf_counter()-t1)*1000
                c, a, wr, _ = N.score(q["gold"], ans)
                rec[nm] = dict(answer=ans if ans is not None else "ABSTAIN", confidence=round(float(conf), 4), correct=c, abstain=a, wrong=wr, ms=ms)
            rows.append(rec)
        return rows
    rows = evaluate(S, test)
    storage = {nm: s.native for nm, s in S.items()}
    # forgetting: +1000 synthetic memories appended (append-only), systems rebuilt from text
    m2, _ = N.synth_interference(1000, seed, facts)
    m2 = [dict(m, text=m["text"].replace("]\n", " file=synthetic.md]\n", 1)) for m in m2]
    views2 = views + [R4.mem_view(m) for m in m2]; af2 = R4.auto_extract(views2)
    S2 = systems(views2, af2, w, b, thr, cap, seed, P.b, P.tau)
    rows2 = evaluate(S2, test)
    sa_ok = P.sa_checks
    print(f"seed {seed} n={len(test)} b={P.b} tau={P.tau} sa_agree={sum(sa_ok)}/{len(sa_ok)} t={time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, rows=rows, rows_after=rows2, storage=storage, cap=cap, tuned=[P.b, P.tau], sa=[sum(sa_ok), len(sa_ok)], test_projects=sorted(test_p))

if __name__ == "__main__":
    sd = int(sys.argv[1]); r = run_seed(sd)
    (OUT / "seeds" / f"seed{sd}.json").write_text(json.dumps(r, default=float))

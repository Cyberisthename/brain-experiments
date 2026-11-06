import json, sys, glob, itertools, numpy as np
from collections import defaultdict
sys.path.insert(0, "/workspace/jarvis/neuron/neuron4"); import n3lib as N
R = [json.load(open(f)) for f in sorted(glob.glob("results/seeds/seed*.json"))]
S = ["exact", "rag", "A", "E_auto", "P_energy"]; BEST = "E_auto"; rng = np.random.default_rng(0)
rows = [r for x in R for r in x["rows"]]; after = {(r["seed"], r["qid"]): r for x in R for r in x["rows_after"]}
def m(rs, s):
    n = len(rs); a = [r for r in rs if r[s]["answer"] != "ABSTAIN"]; g = [r for r in rs if r["gold"] != "ABSTAIN"]
    acc_b = np.mean([r[s]["correct"] for r in rs]); acc_a = np.mean([after[(r["seed"], r["qid"])][s]["correct"] for r in rs])
    corr = np.mean([r[s]["correct"] and after[(r["seed"], r["qid"])][s]["wrong"] for r in rs])
    return dict(correct=acc_b, coverage=len(a)/n, ans_cov=sum(r[s]["answer"] != "ABSTAIN" for r in g)/len(g),
        sel_acc=sum(r[s]["correct"] for r in a)/max(1, len(a)), conf_wrong=np.mean([r[s]["wrong"] for r in rs]),
        multi_hop=np.mean([r[s]["correct"] for r in rs if r["kind"] == "multi_hop_3"]),
        lat_ms=float(np.median([r[s]["ms"] for r in rs])), acc_after=acc_a, F=max(0, acc_b-acc_a)+corr,
        storage=float(np.mean([x["storage"][s] for x in R])))
out = {s: m(rows, s) for s in S}
def score(w2, w3, w4):
    per = defaultdict(list)
    for x in R:
        rs = x["rows"]; M = {s: m(rs, s) for s in S}; Lm = max(v["lat_ms"] for v in M.values()); Sm = max(v["storage"] for v in M.values())
        for s in S: per[s].append(M[s]["correct"] - w2*M[s]["lat_ms"]/Lm - w3*M[s]["storage"]/Sm - w4*M[s]["F"])
    return {s: float(np.mean(v)) for s, v in per.items()}
S0 = score(1, 1, 1); sweep = [score(*w) for w in itertools.product((0.5, 1, 2), repeat=3)]
first = sum(max(d, key=d.get) == "P_energy" for d in sweep) / len(sweep)
def bc(f):
    d = defaultdict(list)
    for r in rows: d[(r["seed"], r["cluster"])].append(f(r))
    return d
T = []
o, p = N.cluster_signflip(bc(lambda r: r["P_energy"]["wrong"] - r[BEST]["wrong"]), 10000, rng); T.append(["H5a conf_wrong P<best", o, p, o < 0])
G = [r for r in rows if r["gold"] != "ABSTAIN"]
def ni(f, margin, rs):
    d = defaultdict(list)
    for r in rs: d[(r["seed"], r["cluster"])].append(f(r))
    ks = list(d); su = np.array([sum(d[k]) for k in ks]); ln = np.array([len(d[k]) for k in ks]); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(ks), len(ks)); bs.append(su[i].sum()/ln[i].sum())
    return su.sum()/ln.sum(), (np.sum(np.array(bs) <= margin)+1)/2001
o, p = ni(lambda r: int(r["P_energy"]["answer"] != "ABSTAIN") - int(r[BEST]["answer"] != "ABSTAIN"), -0.10, G); T.append(["H5b ans-coverage NI 10pp", o, p, True])
o, p = N.cluster_signflip(bc(lambda r: np.log(r["P_energy"]["ms"]) - np.log(r[BEST]["ms"])), 10000, rng); T.append(["H5c latency P<best (log ms)", o, p, o < 0])
o, p = ni(lambda r: r["P_energy"]["correct"] - r[BEST]["correct"], -0.02, rows); T.append(["H5d correct NI 2pp", o, p, True])
adj = N.holm([t[2] for t in T]); ok = [a < 0.05 and t[3] for t, a in zip(T, adj)]
prim = (ok[0] and ok[1]) or (ok[2] and ok[3] and out["P_energy"]["storage"] <= out[BEST]["storage"])
res = dict(metrics=out, S_equal=S0, sweep_share_P_first=first, tests=[dict(name=t[0], diff=float(t[1]), p=float(t[2]), p_holm=float(a), pass_=bool(k)) for t, a, k in zip(T, adj, ok)],
           primary=bool(prim), n_q=len(rows), seeds=len(R), tuned=[x["tuned"] for x in R], sa=[sum(x["sa"][0] for x in R), sum(x["sa"][1] for x in R)])
json.dump(res, open("results/analysis.json", "w"), indent=1, default=float)
L = [f"# NEURON-5 RESULTS  (n={len(rows)} q, {len(R)} seeds)", "| system | correct | coverage | ans-cov | sel-acc | conf-wrong | multi-hop | median ms | storage B | acc after +1000 | F | S(equal) |", "|"+"---|"*12]
for s in S:
    v = out[s]; L.append(f"| {s} | {v['correct']:.3f} | {v['coverage']:.3f} | {v['ans_cov']:.3f} | {v['sel_acc']:.3f} | {v['conf_wrong']:.3f} | {v['multi_hop']:.3f} | {v['lat_ms']:.2f} | {v['storage']:.0f} | {v['acc_after']:.3f} | {v['F']:.3f} | {S0[s]:+.3f} |")
L += [f"- {t['name']}: diff {t['diff']:+.4f} p {t['p']:.4f} Holm {t['p_holm']:.4f} {'PASS' if t['pass_'] else 'FAIL'}" for t in res["tests"]]
L += [f"**Primary supported: {prim}**", f"P_energy ranks first in {first:.0%} of 27 weight combos", f"SA/exact solver agreement: {res['sa']}"]
open("RESULTS.md", "w").write("\n".join(L)); print("\n".join(L))

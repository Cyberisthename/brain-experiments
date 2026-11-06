import json, glob, sys, numpy as np
from collections import defaultdict
sys.path.insert(0, "/workspace/jarvis/neuron/neuron4"); import n3lib as N
R = [json.load(open(f)) for f in sorted(glob.glob("results/seeds/seed*.json"))]
SV = list(R[0]["solvers"]); rng = np.random.default_rng(0)
L = [f"# QMS-1 RESULTS ({len(R)} seeds, {sum(r['info']['n_test'] for r in R)} held-out test q)", "| solver | true evals | wall s | opt-set gap to exact | =exact cfg | =default cfg | test correct | coverage | ans-cov | sel-acc | conf-wrong | boundary acc | multi-hop | F | test O |", "|"+"---|"*15]
agg = {}
for s in SV:
    P = [q for r in R for q in r["solvers"][s]["per_q"]]; X = [r["solvers"][s] for r in R]
    tm = lambda k: np.mean([x["test"][k] * r["info"]["n_test"] for x, r in zip(X, R) if x["test"][k] is not None]) * len(R) / sum(r["info"]["n_test"] for r in R)
    c = np.mean([q["correct"] for q in P]); w = np.mean([q["wrong"] for q in P]); ans = [q for q in P if q["answer"] is not None]
    g = [q for q in P if q["gold"] != "ABSTAIN"]; bk = [q for q in P if q["kind"] in ("conflict", "uncertain", "latest", "distractor")]; mh = [q for q in P if q["kind"] == "multi_hop_3"]
    agg[s] = dict(correct=c, conf_wrong=w, coverage=len(ans)/len(P), ans_cov=sum(q["answer"] is not None for q in g)/len(g), sel=np.mean([q["correct"] for q in ans]),
                  bacc=np.mean([q["correct"] for q in bk]), mh=np.mean([q["correct"] for q in mh]), F=np.mean([x["F"] for x in X]))
    a = agg[s]
    L.append(f"| {s} | {np.mean([x['n_true_evals'] for x in X]):.0f} | {np.mean([x['wall_s'] for x in X]):.2f} | {np.mean([x['opt_gap'] for x in X]):.3f} | {sum(x['config']==r['solvers']['exact']['config'] for x, r in zip(X, R))}/20 | {sum(x['config']==[1,1,1,1,0,0,0,0,0,0] for x in X)}/20 | {a['correct']:.3f} | {a['coverage']:.3f} | {a['ans_cov']:.3f} | {a['sel']:.3f} | {a['conf_wrong']:.3f} | {a['bacc']:.3f} | {a['mh']:.3f} | {a['F']:.3f} | {a['correct']-2*a['conf_wrong']:+.3f} |")
def pairs(s1, s2, key):
    d = defaultdict(list)
    for r in R:
        for q1, q2 in zip(r["solvers"][s1]["per_q"], r["solvers"][s2]["per_q"]): d[(r["seed"], q1["cluster"])].append(q1[key]-q2[key])
    return d
o1, p1 = N.cluster_signflip(pairs("exact", "default_E_auto", "wrong"), 10000, rng)
d = pairs("exact", "default_E_auto", "correct"); ks = list(d); su = np.array([sum(d[k]) for k in ks]); ln = np.array([len(d[k]) for k in ks])
bs = [su[i].sum()/ln[i].sum() for i in (rng.integers(0, len(ks), len(ks)) for _ in range(2000))]; p2 = (np.sum(np.array(bs) <= -0.02)+1)/2001
h = N.holm([p1, p2])
L += [f"\nT1 exact-optimum vs default conf-wrong: diff {o1:+.4f}, p {p1:.4f}, Holm {h[0]:.4f} -> {'PASS' if h[0]<.05 and o1<0 else 'FAIL'}",
      f"T2 correct non-inferior 2pp: diff {su.sum()/ln.sum():+.4f}, p {p2:.4f}, Holm {h[1]:.4f} -> {'PASS' if h[1]<.05 else 'FAIL'}"]
qd = sum(r["solvers"]["surrogate_qaoa_p1"]["config"] != r["solvers"]["surrogate_classical"]["config"] for r in R)
qd2 = sum(r["solvers"]["surrogate_qaoa_p2"]["config"] != r["solvers"]["surrogate_classical"]["config"] for r in R)
dd = max(abs(agg[k]["correct"]-agg["surrogate_classical"]["correct"]) for k in ("surrogate_qaoa_p1", "surrogate_qaoa_p2"))
L += [f"\nSurrogate R2 vs true objective (all 1024): median {np.median([r['surrogate_R2_all1024'] for r in R]):.2f}, min {min(r['surrogate_R2_all1024'] for r in R):.2f}",
      f"QAOA vs classical surrogate argmax config differ: p1 {qd}/20, p2 {qd2}/20; max test-correct diff {dd:.3f}",
      f"QAOA prob on surrogate optimum: p1 mean {np.mean([r['qaoa']['1']['p_surrogate_opt'] for r in R]):.4f}, p2 {np.mean([r['qaoa']['2']['p_surrogate_opt'] for r in R]):.4f} (uniform {1/1024:.4f})",
      f"Hardware rule (>=25% seeds differ OR >=2pp): {'MET' if (max(qd,qd2)>=5 or dd>=0.02) else 'NOT MET'}"]
open("RESULTS.md", "w").write("\n".join(L)); print("\n".join(L))

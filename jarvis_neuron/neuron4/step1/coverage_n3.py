import json,collections,numpy as np
P=[json.loads(l) for l in open("/workspace/jarvis/neuron/neuron3/results/per_question.jsonl")]
S=["0_exact_zstd","0_rag","A_normal","B_random","C_wronglabel","D_weighted","E_boundary"]
def m(rows,s):
    n=len(rows); ans=[r for r in rows if r[s]["answer"]!="ABSTAIN"]
    ab=[r for r in rows if r[s]["answer"]=="ABSTAIN"]
    corr_ans=sum(r[s]["correct"] for r in ans)
    absg=[r for r in rows if r["gold"]=="ABSTAIN"]
    return dict(n=n,coverage=len(ans)/n,sel_acc=corr_ans/max(1,len(ans)),overall_correct=np.mean([r[s]["correct"] for r in rows]),
      answered_correct=corr_ans/n,abstain=len(ab)/n,conf_wrong=np.mean([r[s]["wrong"] for r in rows]),
      approp_abst=(np.mean([r[s]["abstain"] for r in absg]) if absg else None))
def aurc(rows,s):
    # rank answered by confidence desc; abstentions appended last (conf=-1); risk = wrong among covered
    x=sorted(rows,key=lambda r:-(r[s]["confidence"] if r[s]["answer"]!="ABSTAIN" else -1))
    err=np.cumsum([1-r[s]["correct"] if r[s]["answer"]!="ABSTAIN" else 1-r[s]["correct"] for r in x])/np.arange(1,len(x)+1)
    ans=[r for r in x if r[s]["answer"]!="ABSTAIN"]
    e2=np.cumsum([1-r[s]["correct"] for r in ans])/np.arange(1,len(ans)+1) if ans else np.array([0])
    pts={c:float(e2[max(0,int(c*len(rows))-1)]) if c*len(rows)<=len(ans) else None for c in (0.1,0.25,0.5,0.75)}
    return float(np.mean(e2)),pts,len(set(r[s]["confidence"] for r in ans))
out={"overall":{},"by_kind":collections.defaultdict(dict),"aurc":{}}
for s in S:
    out["overall"][s]=m(P,s); out["aurc"][s]=aurc(P,s)
    for k in sorted({r["kind"] for r in P}): out["by_kind"][k][s]=m([r for r in P if r["kind"]==k],s)
json.dump(out,open("n3_coverage.json","w"),indent=1,default=float)
f=lambda v:"  -  " if v is None else f"{v:.3f}"
print("ALL n=",len(P)); print("sys            cov   selacc overall abst  cwrong apprAbs AURC(answered) risk@25/50% #confLevels")
for s in S:
    x=out["overall"][s];a=out["aurc"][s];print(f"{s:14} {f(x['coverage'])} {f(x['sel_acc'])} {f(x['overall_correct'])} {f(x['abstain'])} {f(x['conf_wrong'])} {f(x['approp_abst'])} {a[0]:.3f} {f(a[1][0.25])}/{f(a[1][0.5])} {a[2]}")
for k,d in out["by_kind"].items():
    print("\n",k,"n=",d[S[0]]["n"])
    for s in S:x=d[s];print(f"{s:14} cov {f(x['coverage'])} selacc {f(x['sel_acc'])} correct {f(x['overall_correct'])} cwrong {f(x['conf_wrong'])} apprAbs {f(x['approp_abst'])}")

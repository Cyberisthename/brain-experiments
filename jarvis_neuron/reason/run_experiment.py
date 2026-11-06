#!/usr/bin/env python3
"""Frozen CPU-only experiment runner and preregistered scorer. No production imports."""
import os,sys
sys.dont_write_bytecode=True
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[var]='1'
try:os.nice(max(0,19-os.nice(0)))
except OSError:pass
import json,hashlib,time,itertools,math,random,unittest,io
from pathlib import Path
from decimal import Decimal,InvalidOperation
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
from system import Reasoner,ARMS,SETTINGS,norm,degree,edges_for
ROOT=Path(__file__).resolve().parent
SLICE=['direct','hop','conflict','uncertain','similar_unrelated','conflict_uncertain','all']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def tree(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def grade(gold,ans):
    if ans is None:return dict(correct=int(gold=='ABSTAIN'),answered=0,wrong=0)
    if gold=='ABSTAIN':return dict(correct=0,answered=1,wrong=1)
    try:ok=Decimal(str(ans))==Decimal(str(gold))
    except InvalidOperation:ok=norm(ans)==norm(gold)
    return dict(correct=int(ok),answered=1,wrong=int(not ok))
def subset(rows,sl):return rows if sl=='all' else [r for r in rows if r['kind'] in ('conflict','uncertain')] if sl=='conflict_uncertain' else [r for r in rows if r['kind']==sl]
def metrics(rs):
    n=len(rs);a=sum(r['answered'] for r in rs);w=sum(r['wrong'] for r in rs);c=sum(r['correct'] for r in rs)
    na=sum(r['answerable'] for r in rs);aa=sum(r['answered'] for r in rs if r['answerable'])
    return dict(n=n,correct=c,answered=a,abstentions=n-a,wrong=w,answerable=na,answered_answerable=aa,
        accuracy=c/n if n else None,coverage=a/n if n else None,answerable_coverage=aa/na if na else None,
        confident_wrong=w/a if a else None,selective_accuracy=(a-w)/a if a else None)
def means(metrics_list):
    out={}
    for k in metrics_list[0]:
        xs=[m[k] for m in metrics_list if m[k] is not None];out[k]=sum(xs)/len(xs) if xs else None
    out['defined_confident_wrong_seeds']=sum(m['confident_wrong'] is not None for m in metrics_list)
    return out

def contrast(rows,other):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['kind']=='hop' and r['arm'] in ('E',other):groups[(r['qid'],r['cluster'])][r['arm']].append(r['correct'])
    d=defaultdict(float);n=0
    for (qid,cl),rs in groups.items():
        d[cl]+=sum(rs['E'])/len(rs['E'])-sum(rs[other])/len(rs[other]);n+=1
    vals=list(d.values());total=sum(vals);effect=total/n if n else 0.
    if len(vals)<=20:
        null=[sum(a*b for a,b in zip(v,vals)) for v in itertools.product((-1,1),repeat=len(vals))]
        p=sum(v>=total-1e-12 for v in null)/len(null);reverse=sum(v<=total+1e-12 for v in null)/len(null);method='exact sign flips'
    else:
        rng=random.Random(20261106);ge=le=0
        for _ in range(100000):
            v=sum(rng.choice((-1,1))*a for a in vals);ge+=v>=total-1e-12;le+=v<=total+1e-12
        p=(ge+1)/100001;reverse=(le+1)/100001;method='100000 Monte Carlo sign flips, plus-one'
    return dict(comparator=other,effect_accuracy=effect,p_one_sided=p,p_reverse=reverse,n_clusters=len(vals),
                nonzero_clusters=sum(abs(x)>1e-12 for x in vals),n_questions=n,cluster_sums=dict(d),method=method)
def holm(ps):
    adj=[0.]*len(ps);last=0.
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        last=max(last,min(1.,(len(ps)-rank)*ps[i]));adj[i]=last
    return adj

def main():
    start=time.monotonic();manifest=json.loads((ROOT/'frozen_manifest.json').read_text())
    for name,h in manifest['sha256'].items():assert sha(ROOT/name)==h,f'FROZEN INPUT CHANGED: {name}'
    prereg=json.loads((ROOT/'prereg_lock.json').read_text())
    assert sha(ROOT/'PREREG.md')==prereg['sha256']['PREREG.md']
    out=ROOT/'runs'/datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_EST');out.mkdir(parents=True)
    baseline=json.loads((ROOT/'neuron4_hashes_before.json').read_text());n4=Path('/workspace/jarvis/neuron/neuron4')
    assert tree(n4)==baseline,'neuron4 changed before experiment'
    default_paths=['chat.py','answer_modes.py','reason_sort.py','shadow_eauto.py']
    chat=Path('/workspace/jarvis/chat_rag');defaults={name:sha(chat/name) for name in default_paths};dump(out/'default_hashes_before.json',defaults)
    cards=json.loads((ROOT/'cards.json').read_text());facts=json.loads((ROOT/'facts.json').read_text());Q=json.loads((ROOT/'questions.json').read_text())
    # Mechanical tests use fixtures only. Outcomes below are never fed into the systems.
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.discover(str(ROOT),pattern='test_system.py'))
    (out/'unit_tests.log').write_text(stream.getvalue());assert tests.wasSuccessful()
    audit=[];rows=[];baseedges=edges_for(facts)
    for arm in ARMS:
        for seed in SETTINGS['seeds']:
            s=Reasoner(cards,facts,arm,seed)
            audit.append(dict(arm=arm,seed=seed,n_edges=len(s.edges),degree_preserved=(degree(s.edges)==degree(baseedges)) if arm!='bag' else None,
                label_counts=dict(Counter(e[2] for e in s.edges)),self_loops=sum(a==b for a,b,_ in s.edges),
                changed_endpoint_pairs=sum((a,b)!=(c,d) for (a,b,_),(c,d,_) in zip(s.edges,baseedges))))
            for q in Q:
                pred=s.answer(q['query']);g=grade(q['gold'],pred['answer'])
                rows.append(dict(qid=q['qid'],kind=q['kind'],cluster=q['cluster'],answerable=q['answerable'],
                     arm=arm,seed=seed,question=q['query']['text'],gold=q['gold'],**g,**pred))
            print(f'{arm} seed={seed} complete',flush=True)
    # Invariants: deterministic arms repeated only for paired seed alignment.
    for arm in ('bag','plain','E'):
        byq=defaultdict(set)
        for r in rows:
            if r['arm']==arm:byq[r['qid']].add((r['answer'],r['correct'],r['answered']))
        assert all(len(x)==1 for x in byq.values())
    with (out/'per_question.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    dump(out/'graph_audit.json',audit)
    perseed={arm:{sl:[metrics(subset([r for r in rows if r['arm']==arm and r['seed']==seed],sl)) for seed in SETTINGS['seeds']] for sl in SLICE} for arm in ARMS}
    avg={arm:{sl:means(ms) for sl,ms in slices.items()} for arm,slices in perseed.items()}
    tests_h={h:contrast(rows,c) for h,c in [('H1','bag'),('H2','degree_shuffle'),('H3','label_shuffle')]}
    for h,p in zip(('H1','H2'),holm([tests_h[h]['p_one_sided'] for h in ('H1','H2')])):tests_h[h]['p_holm']=p
    eh=avg['E']['hop'];ed=avg['E']['direct'];bd=avg['bag']['direct']
    guard={
        'hop_answerable_coverage_at_least_80pct':eh['answerable_coverage']>=.8,
        'direct_accuracy_within_5pp_of_bag':ed['accuracy']>=bd['accuracy']-.05-1e-12,
        'direct_answerable_coverage_within_5pp_of_bag':ed['answerable_coverage']>=bd['answerable_coverage']-.05-1e-12,
        'direct_confident_wrong_not_increased':ed['confident_wrong'] is not None and bd['confident_wrong'] is not None and ed['confident_wrong']<=bd['confident_wrong']+1e-12}
    for c in ('bag','degree_shuffle'):
        ch=avg[c]['hop']
        guard[f'hop_coverage_within_5pp_of_{c}']=eh['answerable_coverage']>=ch['answerable_coverage']-.05-1e-12
        guard[f'hop_confident_wrong_not_above_{c}']=eh['confident_wrong'] is not None and ch['confident_wrong'] is not None and eh['confident_wrong']<=ch['confident_wrong']+1e-12
    for h,d in tests_h.items():
        p=d.get('p_holm',d['p_one_sided'])
        ok=d['effect_accuracy']>0 and p<=.05
        # Falsification requires a reverse test plus a reverse effect in every shuffle seed.
        reverse_each=[]
        for seed in SETTINGS['seeds']:
            reverse_each.append(perseed['E']['hop'][seed]['accuracy']<perseed[d['comparator']]['hop'][seed]['accuracy'])
        rev=d['effect_accuracy']<0 and d['p_reverse']<=.025 and all(reverse_each)
        if h in ('H1','H2'):ok=ok and all(guard.values())
        d['verdict']='supported' if ok else 'falsified' if rev else 'not supported'
        d['secondary_exploratory']=h=='H3'
    overall=all(tests_h[h]['verdict']=='supported' for h in ('H1','H2')) and all(guard.values())
    after=tree(n4);dump(out/'neuron4_hashes_after.json',after);dump(ROOT/'neuron4_hashes_after.json',after)
    default_after={name:sha(chat/name) for name in default_paths};dump(out/'default_hashes_after.json',default_after)
    safety=dict(neuron4_unchanged=after==baseline,neuron4_files=len(after),default_files_unchanged=defaults==default_after,
                no_training=True,no_qpu=True,no_upload=True,no_git_push=True,no_institutional_books=True)
    assert all(v for k,v in safety.items() if isinstance(v,bool))
    finished=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S EST')
    summary=dict(status='completed',finished_et=finished,runtime_seconds=time.monotonic()-start,run_directory=str(out),
       seeds=SETTINGS['seeds'],n_questions=len(Q),dataset=json.loads((ROOT/'dataset_audit.json').read_text()),
       means_by_slice=avg,per_seed=perseed,hypotheses=tests_h,guardrails=guard,overall_supported=overall,
       safety=safety,mechanical_tests=dict(run=tests.testsRun,success=tests.wasSuccessful()),
       prereg_sha256=sha(ROOT/'PREREG.md'),frozen_manifest_sha256=sha(ROOT/'frozen_manifest.json'))
    dump(out/'summary.json',summary);dump(ROOT/'summary.json',summary)
    def fmt(x):return 'undefined' if x is None else f'{x:.3f}'
    lines=['# Association reasoning experiment',f'Finished {finished} (America/New_York).',
      '',f"**Overall: {'SUPPORTED' if overall else 'NOT SUPPORTED'} under the locked win rule.**",'',
      '## Hypotheses', '| Hypothesis | Comparator | Hop accuracy difference | Raw one-sided p | Holm p | Verdict |',
      '|---|---|---:|---:|---:|---|']
    for h,d in tests_h.items():lines.append(f"| {h}{' (secondary/exploratory)' if h=='H3' else ''} | {d['comparator']} | {d['effect_accuracy']:+.3f} | {d['p_one_sided']:.6f} | {fmt(d.get('p_holm'))} | {d['verdict']} |")
    lines+=['','Primary paired sign flips use seven research-topic source clusters, not 248 independent hop trials. Questions are averaged over seeds before source-cluster aggregation. H1/H2 use Holm correction; H3 is unadjusted exploratory. No nonsignificant null is called falsification.',
      '', '## Slice means across eight seeds', '| Slice | Arm | Questions/seed | Accuracy | Coverage | Answerable coverage | Wrong / answered | Selective accuracy |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for sl in SLICE:
        for arm in ARMS:
            m=avg[arm][sl];lines.append(f"| {sl} | {arm} | {m['n']:.0f} | {fmt(m['accuracy'])} | {fmt(m['coverage'])} | {fmt(m['answerable_coverage'])} | {fmt(m['confident_wrong'])} | {fmt(m['selective_accuracy'])} |")
    lines+=['','Counts, abstentions, denominator-defined seed counts, per-seed ratios, source clusters, and predictions are in summary.json and runs/*/per_question.jsonl. Undefined error with zero answers is not reported as zero.',
      '', '## Methods and audit',
      '- Raw-text numerical clause-prefix extractor plus literal author/capital patterns; no model training and no existing answer/question/fact_statement fields as system input. Questions supply literal slot/project cues, not a learned natural-language parser.',
      '- All five arms see the same extracted facts and lexical top-12 source cards. Graphs spread for three hops only within that candidate set. The bag baseline may extract a second span from an already retrieved seed card: it is deliberately not forced to abstain on every hop. Graph readout has no unrestricted global entity-table join.',
      '- Degree control uniformly permutes all undirected fact-edge stubs. Every seed preserves the exact node-labelled degree sequence, edge count, and label multiset; multiedges and self-loops are retained. Label control preserves endpoints and label counts. See graph_audit.json.',
      '- E boundaries block unrelated_similar links, downweight uncertain_source links, reject raw-text weak support and unresolved multiple values. Plain graph/bag use ordinary weighted extraction. No dated supersession examples were available in the selected source cards.',
      '- Same-card associations are explicit source structure; cross-topic lexical similarity is a lure, not evidence. Shared-topic notes are clustered together conservatively. Frozen n3lib/run_neuron4 were inspected read-only but not imported; a local prose adapter avoids their unrestricted two-retrieval hop shortcut.',
      '', '## Guardrails', '```json',json.dumps(guard,indent=2),'```',
      '', '## Limitations',
      '- Small, automatically templated benchmark. Gold numeric/string answers are literal source spans; conflict/uncertainty labels are text rules, not independent human judgments. The extraction and question construction share a representation, so this tests that representation rather than discovering general understanding.',
      '- All primary hop items use two fields co-located in eight existing cards, seven source-topic clusters. This is a limited association test, not a difficult cross-document deduction test. Plain source-card extraction can already solve it; do not claim graph-specific reasoning from high absolute accuracy alone.',
      '- Existing jarvis_notes were previously used for sorter development. Fresh templates/slots do not make those sources independent held-out data. The historical 30 chat questions and conservatively matched author topics were excluded from primary evaluation; their facts remain retrieval distractors.',
      '- One conflict probe and one weak probe were already synthetic examples in jarvis_notes. They are explicitly local probes, not real scientific claims, and two items cannot validate robust abstention. No new fake scientific claims were invented.',
      '- Similar-unrelated slice contains seven superficially similar metric-slot questions, not independently authored adversarial lures. Literal query-slot matching makes entity disambiguation easier than free chat.',
      '- Source cards summarize earlier experiments; the underlying scientific claims were not independently re-audited here. No Institutional Books corpus was loaded. WordNet raw cards remain candidate distractors but are not used as definition gold questions.',
      '- Eight seeds randomize control graphs; deterministic bag/plain/E repeats are not independent replications. Very few source clusters constrain power and generalization. Nominal confidence is uncalibrated; every wrong non-abstaining answer counts as confident-wrong.',
      '- A motif abstraction only: no claim about biological brains, consciousness, or training the 319M model to understand. No production promotion.',
      '', '## Safety', '```json',json.dumps(safety,indent=2),'```',
      'CAD hybrid default and opt-in --reason/--shadow paths were untouched. All 69 neuron4 files match the original before-run SHA256 manifest. No fine-tune, QPU, Kaggle upload, git push, or external service call.',
      '', '## Preregistration / deviations',
      '- Original PREREG.md hash remains '+sha(ROOT/'PREREG.md')+'. Corpus, question set, arm settings, system, tests and scorer were SHA256-frozen before any system outcomes. No outcome-informed changes.',
      '- Local execution resumed after the initial unrun status; that status was archived under status_before_run/. This is an execution availability change, not a hypothesis/metric/seed/win-rule amendment.',
      '', '## Rerun', '```bash','cd /workspace/jarvis/neuron/reason',
      'nice -n 19 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_experiment.py','```',
      'Uses frozen snapshots; verifies hashes and creates a new run directory. Do not rerun prepare.py/make_questions.py to regenerate the locked data.',
      '', 'Files: PREREG.md, IMPLEMENTATION.md, frozen_manifest.json, source_manifest.json, dataset_audit.json, cards.json, facts.json, questions.json, settings.json, system.py, prepare.py, make_questions.py, test_system.py, run_experiment.py, summary.json, neuron4_hashes_before/after.json, runs/*/.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'hypotheses':tests_h,'overall_supported':overall,'safety':safety,'runtime_seconds':summary['runtime_seconds']},indent=2))
if __name__=='__main__':main()

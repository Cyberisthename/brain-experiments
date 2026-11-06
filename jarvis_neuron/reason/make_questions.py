"""Evaluator preparation: source-backed template questions, NOT independent human labels."""
import json,re,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from prepare import dump,sha,ROOT
from system import norm,topic,SETTINGS,edges_for,extract

def main():
    assert not (ROOT/'frozen_manifest.json').exists()
    cards=json.loads((ROOT/'cards.json').read_text());facts=extract(cards)
    dump('facts.json',facts)
    provenance=json.loads((ROOT/'source_manifest.json').read_text())['cards']
    full=Path('/workspace/jarvis/chat_rag/results/full_eval.json')
    dev=json.loads(full.read_text())['items']
    devtexts='\n'.join(x.get('question','') for x in dev).casefold()
    bykey=defaultdict(list);bycard=defaultdict(list)
    for f in facts:bykey[(f['project'],f['slot'])].append(f);bycard[f['card']].append(f)
    Q=[];excluded=[]
    def support(fs):
        return [dict(card=f['card'],**provenance[f['card']],span=f['span'],raw=f['raw']) for f in fs]
    def add(kind,project,slot,gold,fs,seed=None):
        text=f"In the {project} notes, what value is recorded for '{slot}'?"
        public=dict(text=text,project=project,slot=slot)
        if seed:
            text=f"In {project}, find the note containing '{seed['slot']}' = {seed['value']}. What value is attached to '{slot}' in that same note?"
            public.update(text=text,seed_slot=seed['slot'],seed_value=seed['value'])
        Q.append(dict(qid=f'q{len(Q):03d}',kind=kind,cluster=project,query=public,gold=gold,
                      support=support(fs),answerable=gold!='ABSTAIN'))
    usable={}
    for (proj,slot),fs in sorted(bykey.items()):
        # Do not evaluate fictional novel cards, unsourced book/year numeric artefacts,
        # or historical 30Q slots. Uploaded author facts are retained only if new.
        if proj=='uploaded':
            if not slot.startswith('author of '):continue
            work=slot.removeprefix('author of ')
            # Conservative prefix overlap catches paraphrased/short historical titles.
            if any(norm(work) in norm(x.get('question','')) or norm(x.get('question','')).find(norm(work).split(';')[0])>=0 for x in dev):
                excluded.append(dict(slot=slot,reason='historical 30Q topic'));continue
            if any(provenance[f['card']]['source'] in ('novel','upload') for f in fs):continue
        else:
            if any(t in slot for t in ('bonferroni','mean contrast','no hardware case.','at n=')) or slot=='all':
                excluded.append(dict(slot=slot,reason='identifier/statistical ancillary/count field'));continue
        certain=[f for f in fs if not f['weak']];values={norm(f['value']) for f in certain}
        if not certain: add('uncertain',proj,slot,'ABSTAIN',fs);continue
        if len(values)>1:add('conflict',proj,slot,'ABSTAIN',fs);continue
        gold=certain[0]['value'];usable[(proj,slot)]=gold
        add('direct',proj,slot,gold,fs)
    # Disjoint lure slice: literal methods with overlapping metric labels but different values.
    # These are existing notes, not fabricated scientific claims.
    lures={'training-free chat_rag full eval hybrid accuracy','training-free chat_rag generation accuracy',
           'training-free chat_rag cad accuracy','e-auto coverage','answerable coverage',
           'e-auto confident-wrong','exact search confident-wrong'}
    for q in Q:
        if q['kind']=='direct' and q['query']['slot'] in lures:q['kind']='similar_unrelated'
    # A hop must have two distinct explicitly co-located fields, no hidden true entity.
    # Up to five ordered pairs/card, deterministic hash order fixed before outcomes.
    hop_cards=[]
    for cid,fs in sorted(bycard.items()):
        if not cid.startswith('jarvis_notes'):continue
        fs=[f for f in fs if not f['weak'] and (f['project'],f['slot']) in usable]
        pairs=[]
        for a in fs:
            matching={f['card'] for f in facts if f['project']==a['project'] and f['slot']==a['slot'] and norm(f['value'])==norm(a['value'])}
            if matching!={cid}:continue
            for b in fs:
                if a['slot']==b['slot']:continue
                pairs.append((hashlib.sha256((cid+a['slot']+'|'+b['slot']).encode()).hexdigest(),a,b))
        for _,a,b in sorted(pairs,key=lambda t:t[0])[:5]:
            add('hop',b['project'],b['slot'],b['value'],[a,b],seed=a)
        if pairs:hop_cards.append(cid)
    # Raw source-span checks guard against invented answers; source spans never enter Reasoner.
    cmap={c['id']:c['text'] for c in cards}
    for q in Q:
        assert all(cmap[s['card']][s['span'][0]:s['span'][1]]==s['raw'] for s in q['support'])
        if q['answerable']:assert any(norm(q['gold']) in norm(s['raw']).lstrip('~≈') for s in q['support'])
        assert q['query']['text'].casefold() not in devtexts
    dump('questions.json',Q)
    dump('dataset_audit.json',dict(counts=dict(Counter(q['kind'] for q in Q)),n_facts=len(facts),
         n_cards=len(cards),edge_types=dict(Counter(r for _,_,r in edges_for(facts))),
         hop_cards=hop_cards,source_clusters=sorted({q['cluster'] for q in Q if q['kind']=='hop'}),
         exclusions=excluded,old_30q_manifest_sha256=sha(full),
         gold_policy='Literal source spans/templates; uncertainty/conflict labels from text rules, not independent human judgments. No existing answer/question/fact_statement fields used as system inputs.',
         cluster_policy='Raw-text research topic groups; all cards about the same experiment count as one cluster. Similar-unrelated edges are not shared evidence and do not merge clusters.'))
    print(json.dumps(json.loads((ROOT/'dataset_audit.json').read_text()),indent=2))
if __name__=='__main__':main()

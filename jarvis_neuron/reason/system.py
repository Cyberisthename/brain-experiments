"""Training-free note association motif. No evaluator metadata is accepted here."""
import re, math, random
from collections import defaultdict, Counter

NUM = re.compile(r'(?<![\w.\-])(?:~|≈)?[+-]?\d+(?:\.\d+)?(?![\w-]|\.\d)')
UNC = re.compile(r'~|≈|\b(?:approx\w*|unverified|prelim\w*|pending|tbd|estimate\w*)\b', re.I)
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
STOP = {'on','n','at','vs','versus','was','is','in','of','and','for','the','to','from','mean contrast','holm p','p','diff'}
ARMS = ['bag', 'plain', 'E', 'degree_shuffle', 'label_shuffle']
SETTINGS = {'candidate_cards':12,'hops':3,'decay':0.7,'seeds':list(range(8)),
            'edge_shuffle':'uniform permutation of undirected endpoint stubs; multigraph/self-loops retained',
            'label_shuffle':'uniform permutation of all typed labels; fixed endpoints',
            'normalization':'casefold, whitespace, strip punctuation; numeric Decimal-equivalence',
            'bag_hops':'local seed-card span extraction only, not global entity-table join'}

def norm(s):
    return re.sub(r'\s+',' ', str(s).casefold()).strip(' \t\n.,;:!?\"')

def tokens(s): return re.findall(r'[a-z0-9_]+',str(s).lower())

def topic(text):
    m=re.search(r'NEURON-\d+|QMS-\d+|chat_rag|hypothalamus|callosum|engine_v3|tutor|shadow probe',text,re.I)
    return norm(m.group()) if m else 'uploaded'

def extract(cards):
    """Raw text only; never uses pre-existing question/answer/fact_statement fields.

    Numeric spans receive literal clause-prefix slot keys (not semantic parsing).
    Source offsets retained for auditing, never used as relevance features.
    """
    out=[]
    for c in cards:
        text=c['text']; proj=topic(text); seen=set()
        def add(slot,value,a,b,unc=False):
            slot=norm(re.sub(r'\b(?:was|is|were|equals)\s*$','',slot,flags=re.I))
            if len(slot)<3 or slot in STOP: return
            key=(slot,norm(value))
            if key in seen:return
            seen.add(key)
            out.append(dict(id=len(out),project=proj,slot=slot,value=str(value),card=c['id'],
                            raw=text[a:b],span=[a,b],weak=bool(unc)))
        # Literal number fields in prose. Ignore module identifiers and numeric indices.
        for sm in re.finditer(r'.+?(?:[.!?](?=\s+[A-Z])|$)',text,re.S):
            sentence=sm.group();base=sm.start(); prev=0
            for nm in NUM.finditer(sentence):
                prefix=sentence[prev:nm.start()]
                prefix=re.split(r'[;,:]|\b(?:versus|vs)\b',prefix,flags=re.I)[-1].strip()
                # Last unmatched brackets are separators, not relation names.
                prefix=prefix.strip(' ()[]')
                start=base+nm.start();end=base+nm.end()
                add(prefix,nm.group().lstrip('~≈'),start,end,UNC.search(sentence))
                prev=nm.end()
        # Conventional uploaded author/card facts, no language-model or external calls.
        for sm in SENT.split(text):
            st=sm.strip(); m=re.fullmatch(r'(.+?) wrote (.+?)\.',st)
            if m:
                author,work=m.groups();i=text.find(author)
                add('author of '+work,author,i,i+len(author))
            m=re.fullmatch(r'The capital of (.+?) is (.+?)\.',st)
            if m:
                place,val=m.groups();i=text.find(val)
                add('capital of '+place,val,i,i+len(val))
    return out

def edges_for(facts):
    """Typed raw-structure associations, independent of questions or labels."""
    bycard=defaultdict(list); byslot=defaultdict(list); bylex=defaultdict(list)
    for f in facts:
        bycard[f['card']].append(f);byslot[(f['project'],f['slot'])].append(f)
        bylex[f['slot']].append(f)
    es=[]
    # A card is an explicit source grouping. Use a clique so order is irrelevant.
    for fs in bycard.values():
        for i,a in enumerate(fs):
            for b in fs[i+1:]:
                es.append((a['id'],b['id'],'uncertain_source' if a['weak'] or b['weak'] else 'same_card'))
    for fs in byslot.values():
        for i,a in enumerate(fs):
            for b in fs[i+1:]:
                if a['card']==b['card']:continue
                r='supported' if norm(a['value'])==norm(b['value']) else 'contradicts'
                if a['weak'] or b['weak']:r='uncertain_source'
                es.append((a['id'],b['id'],r))
    # Lexically similar slots across distinct raw-text topics are lures, not evidence.
    generic={'the','was','is','on','for','of','at','and','in','to','beat','scored','rate'}
    for i,a in enumerate(facts):
        ta=set(tokens(a['slot']))-generic
        for b in facts[i+1:]:
            if a['project']==b['project']:continue
            tb=set(tokens(b['slot']))-generic
            if len(ta & tb)>=2 and len(ta & tb)/max(1,len(ta|tb))>=0.25:
                es.append((a['id'],b['id'],'unrelated_similar'))
    return es

def degree(es):
    c=Counter()
    for a,b,_ in es:c[a]+=1;c[b]+=1
    return c

def shuffled(es,arm,seed):
    rng=random.Random(seed*31+7)
    if arm=='degree_shuffle':
        stubs=[n for a,b,_ in es for n in (a,b)];rng.shuffle(stubs)
        out=[(stubs[2*i],stubs[2*i+1],r) for i,(_,_,r) in enumerate(es)]
        assert len(out)==len(es) and degree(out)==degree(es)
        assert Counter(r for _,_,r in out)==Counter(r for _,_,r in es)
        return out
    if arm=='label_shuffle':
        labs=[r for _,_,r in es];rng.shuffle(labs)
        out=[(a,b,r) for (a,b,_),r in zip(es,labs)]
        assert [(a,b) for a,b,_ in out]==[(a,b) for a,b,_ in es]
        assert Counter(labs)==Counter(r for _,_,r in es)
        return out
    return list(es)

class Reasoner:
    def __init__(self,cards,facts,arm,seed=0):
        self.cards={c['id']:c for c in cards};self.facts=facts;self.F={f['id']:f for f in facts};self.arm=arm
        self.edges=shuffled(edges_for(facts),arm,seed) if arm!='bag' else []
        self.adj=defaultdict(list)
        for a,b,r in self.edges:self.adj[a].append((b,r));self.adj[b].append((a,r))
        self.bycard=defaultdict(list)
        for f in facts:self.bycard[f['card']].append(f)
        self.card_tokens={i:Counter(tokens(c['text'])) for i,c in self.cards.items()}
        self.df=Counter(t for co in self.card_tokens.values() for t in co)
    def retrieve(self,q):
        ts=tokens(q['text']);scores={}
        for cid,co in self.card_tokens.items():
            scores[cid]=sum((1+math.log((1+len(self.cards))/(1+self.df[t])))*min(co[t],1) for t in ts if t in co)
        cs=sorted(scores,key=lambda i:(-scores[i],i))[:SETTINGS['candidate_cards']]
        return cs,[f for c in cs for f in self.bycard[c]]
    def spread(self,starts,allowed):
        act={i:1. for i in starts};front=list(act);paths={i:[] for i in starts}
        for _ in range(SETTINGS['hops']):
            nxt=[]
            for a in front:
                for b,r in self.adj[a]:
                    if b not in allowed:continue
                    if self.arm!='plain' and r=='unrelated_similar':continue
                    weight=0.3 if self.arm!='plain' and r=='uncertain_source' else 1.
                    v=act[a]*SETTINGS['decay']*weight
                    if v>act.get(b,0)+1e-9:
                        act[b]=v;paths[b]=paths[a]+[[a,b,r]];nxt.append(b)
            front=nxt
        return act,paths
    def read(self,fs,q,weights):
        if not fs:return None,0.,'no supported candidate'
        boundary=self.arm in ('E','degree_shuffle','label_shuffle')
        if boundary:
            fs=[f for f in fs if not f['weak']]
            if not fs:return None,.2,'only weak support'
            # Like E, factual conflict check does not require a correct edge label.
            vals={norm(f['value']) for f in fs}
            if len(vals)>1:return None,.3,'unresolved conflicting values'
        sc=Counter()
        for f in fs:sc[f['value']]+=weights.get(f['id'],1.)
        val,w=sorted(sc.items(),key=lambda kv:(-kv[1],norm(kv[0])))[0]
        conf=w/sum(sc.values())
        return val,conf,'supported answer'
    def answer(self,q):
        # Public query has text + parsed literal cues, NEVER kind/gold/support/source-card.
        assert set(q)<= {'text','project','slot','seed_slot','seed_value'}
        cs,cands=self.retrieve(q);allowed={f['id'] for f in cands}
        if q.get('seed_slot') is not None:
            seeds=[f for f in cands if f['project']==q['project'] and f['slot']==q['seed_slot'] and norm(f['value'])==norm(q['seed_value'])]
            if self.arm=='bag':
                # Strong local-card extract baseline: co-located facts are visible without graph spreading.
                mids={f['card'] for f in seeds};weights={f['id']:1. for f in cands if f['card'] in mids};paths={}
            else:
                seeds=[f for f in seeds if self.arm=='plain' or not f['weak']]
                weights,paths=self.spread([f['id'] for f in seeds],allowed)
            targets=[f for f in cands if f['id'] in weights and f['slot']==q['slot'] and f['project']==q['project']]
        else:
            targets=[f for f in cands if f['slot']==q['slot'] and f['project']==q['project']]
            weights={f['id']:1. for f in targets};paths={}
        ans,conf,why=self.read(targets,q,weights)
        supporting=[f['id'] for f in targets if ans is not None and norm(f['value'])==norm(ans)]
        return dict(answer=ans,confidence=conf,decision=why,candidate_cards=cs,
                    target_ids=[f['id'] for f in targets],trace=[p for i in supporting for p in paths.get(i,[])])

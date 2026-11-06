"""Prepare/freeze data before outcomes. Evaluator-only gold creation from raw spans."""
import json,hashlib,re
from pathlib import Path
from collections import defaultdict,Counter
from system import extract,norm,topic,SETTINGS,edges_for
ROOT=Path(__file__).resolve().parent
KNOW=Path('/workspace/jarvis/chat_rag/knowledge')

def dump(name,data): (ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    assert not (ROOT/'frozen_manifest.json').exists(),'Already frozen; do not overwrite'
    cards=[];provenance={};metadata={};source_hashes={}
    for filename in ['uploaded_facts.jsonl','jarvis_notes.jsonl']:
        p=KNOW/filename;source_hashes[str(p)]=sha(p)
        for line,s in enumerate(p.read_text().splitlines(),1):
            row=json.loads(s);cid=f'{filename}:{line}'
            cards.append(dict(id=cid,text=row['text']))
            provenance[cid]=dict(path=str(p),line=line,card_id=row.get('id'),source=row.get('source'))
            # ONLY evaluator: never written into raw cards or extracted facts.
            metadata[cid]=row
    facts=extract(cards);dump('cards.json',cards);dump('facts.json',facts)
    dump('source_manifest.json',dict(sha256=source_hashes,cards=provenance))
    # Show raw-source extraction inventory for source audit, NOT system outcomes.
    for f in facts:
        if f['project']!='uploaded':print(f['id'],f['card'],f['project'],f['slot'],repr(f['value']),'weak='+str(f['weak']))
    dump('settings.json',SETTINGS)
    return cards,facts,provenance,metadata
if __name__=='__main__':prepare()

"""Keep near-duplicate topic families on the same side of train/validation.

Names alone miss synonyms. For each scope, retrieve 50 nearby papers and connect
families when scopes share at least half those works. Split connected components,
not individual scopes or examples. Definitions steer retrieval. This is an
additional leakage guard, not a proof that every semantic neighbor is detected.
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import faiss
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from srl_search.field import Ctx
ROOT=Path(os.environ.get('FIELDJUDGE_V2','/mnt/fast/scholarsreadinglist/fieldjudge/v2'))

def run():
    src=ROOT/'topics.json'; digest=hashlib.sha256(src.read_bytes()).hexdigest()
    dest=ROOT/'split.json'
    if dest.exists():
        if json.loads(dest.read_text())['topics_sha256']!=digest:raise ValueError('topics changed')
        return
    families=json.loads(src.read_text())['families']
    ctx=Ctx(); eng=ctx.engine; faiss.omp_set_num_threads(16)
    parent={f['id']:f['id'] for f in families}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[b]=a
    postings=defaultdict(list); scopes=[]; edges=[]
    for f in families:
        for s in f['scopes']:
            q=eng.encoder.encode(s['topic']+': '+s['definition'])[0]
            _,idx=eng.index.search(q.reshape(1,-1),50,params=faiss.SearchParametersIVF(nprobe=128))
            ws={int(w) for w in eng.work_ids[idx[0][idx[0]>=0]]}
            hits=defaultdict(int)
            for w in ws:
                for old in postings[w]:hits[old]+=1
            for old,count in hits.items():
                other=scopes[old]
                if other['family']!=f['id'] and count>=.5*min(len(ws),other['n']):
                    union(other['family'],f['id'])
                    edges.append([other['scope'],s['id'],count])
            n=len(scopes)
            scopes.append({'scope':s['id'],'family':f['id'],'n':len(ws)})
            for w in ws:postings[w].append(n)
    components=defaultdict(list)
    for f in families:components[root(f['id'])].append(f['id'])
    groups=list(components.values());random.Random(5026).shuffle(groups)
    # Prefer small intact components; a giant generic cluster must not absorb the whole validation set.
    validation=set()
    for group in groups:
        if len(validation)>=50:break
        if len(group)<=60-len(validation):validation.update(group)
    if not 40<=len(validation)<=60:
        raise ValueError('no meaningful component-disjoint validation partition; inspect topic breadth')
    output={'topics_sha256':digest,'components':groups,'overlap_edges':edges,
            'family_split':{f['id']:('validation' if f['id'] in validation else 'train') for f in families},
            'validation_families':len(validation),'release':ctx.release['id']}
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2));tmp.replace(dest)
    print('SPLIT',len(groups),'components;',len(validation),'validation families',flush=True)

if __name__=='__main__':run()

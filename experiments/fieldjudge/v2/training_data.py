"""Freeze splits and validation now; add late TRAIN labels in a second snapshot.

No provider calls. Guard by topic-family retrieval overlap, work IDs and normalized
titles against validation and all test sets. The whole queue determines exclusions,
not just whichever labels finished first. Validation examples/labels never change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from stream_queue import ROOT, FAST, digest, write_json
from partition import load_answers
from export_fast import resolve

HERE=Path(__file__).resolve().parent
TRAIN=ROOT/'training'
GOOD={'provisional_agreement','provisional_single_teacher'}

def title_key(text):
    return hashlib.sha256(re.sub(r'[^\w]','',text.splitlines()[0].casefold()).encode()).hexdigest()

def catalog():
    batches={p.stem:json.loads(p.read_text()) for p in sorted((FAST/'queue').glob('*.json'))}
    rows=[r for b in batches.values() for r in b['pairs']]
    if len(rows)!=100000 or len({r['id'] for r in rows})!=len(rows):raise ValueError('queue not complete/unique')
    return batches,rows,digest({k:digest(v) for k,v in batches.items()})

def freeze_split(rows,queue_hash):
    path=TRAIN/'split.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data['queue_hash']!=queue_hash:raise ValueError('queue changed under frozen split')
        return data
    sys.path.insert(0,str(HERE.parents[2]))
    from srl_search.field import Ctx
    import numpy as np
    import faiss
    ctx=Ctx();eng=ctx.engine;faiss.omp_set_num_threads(16)
    scopes={}
    for r in rows:
        scopes[(r['family_id'],r['topic'],r.get('definition') or '')]=False
    recipe=json.loads((HERE/'recipe.json').read_text())
    reserved=[]
    for h in recipe['new_holdouts']+recipe['development_holdouts']:
        family='test:'+digest(h)[:20];reserved.append(family)
        scopes[(family,h['topic'],h['definition'])]=True
    parent={f:f for f,_,_ in scopes}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[b]=a
    postings=defaultdict(list);previous=[];edges=[]
    cache=TRAIN/'retrieval-overlap';cache.mkdir(parents=True,exist_ok=True)
    for i,(family,topic,definition) in enumerate(sorted(scopes)):
        key=digest([topic,definition,ctx.release['id']]);p=cache/f'{key}.json'
        if p.exists():ws=set(json.loads(p.read_text()))
        else:
            q=eng.encoder.encode(topic+(': '+definition if definition else ''))[0]
            _,ix=eng.index.search(q.reshape(1,-1),50,params=faiss.SearchParametersIVF(nprobe=128))
            ws={int(w) for w in eng.work_ids[ix[0][ix[0]>=0]]};write_json(p,sorted(ws))
        counts=Counter(old for w in ws for old in postings[w])
        for old,n in counts.items():
            other=previous[old]
            if other['family']!=family and n>=.5*min(len(ws),other['n']):
                union(family,other['family']);edges.append([family,other['family'],n])
        previous.append({'family':family,'n':len(ws)})
        for w in ws:postings[w].append(i)
        if (i+1)%200==0:print('SPLIT CHECK',i+1,'/',len(scopes),flush=True)
    test_roots={root(f) for f in reserved}
    excluded={f for f in parent if root(f) in test_roots}
    components=defaultdict(list)
    for f in sorted(parent):
        if f not in excluded:components[root(f)].append(f)
    groups=list(components.values());random.Random(8526).shuffle(groups)
    vfamily=set()
    for group in groups:
        if len(vfamily)>=50:break
        if len(group)<=60-len(vfamily):vfamily.update(group)
    if len(vfamily)<40:raise ValueError('not enough independent validation families')
    data={'queue_hash':queue_hash,'release':ctx.release['id'],'validation_families':sorted(vfamily),
          'excluded_families':sorted(excluded),'components':list(components.values()),'overlap_edges':edges,
          'note':'Retrieval-overlap guard, not proof that every semantic relationship was detected.'}
    write_json(path,data);return data

def labels(batches,rows):
    votes=load_answers(batches)
    assignment=json.loads((FAST/'assignment.json').read_text())
    assigned={m:{k for ids in assignment['batches'][m].values() for k in ids} for m in votes}
    reference={(r['topic_id'],r['id']):r['y'] for r in map(json.loads,open(ROOT.parent/'labels.jsonl'))}
    result={};counts=Counter()
    for r in rows:
        k=r['id'];answers={m:({'label':a[k]} if k in a else None) for m,a in votes.items()}
        old=reference.get((int(r['family_id'][7:]),r['work_id'])) if r['family_id'].startswith('legacy-') else None
        pending=any(k in assigned[m] and k not in votes[m] for m in votes)
        reason,y,sources=resolve(answers,old=old,single_mode=True,assigned_pending=pending)
        counts[reason]+=1
        if reason in GOOD:result[k]=(y,sources)
    return result,dict(counts)

def forbidden(rows,split):
    vfamily=set(split['validation_families']);excluded=set(split['excluded_families'])
    vrows=[r for r in rows if r['family_id'] in vfamily]
    vid={r['work_id'] for r in vrows};vtitle={r['text_group'] for r in vrows}
    testid=set();testtitle=set()
    for r in map(json.loads,open(ROOT.parent/'tests.jsonl')):
        testid.add(r['id']);testtitle.add(title_key(r['text']))
    for folder in ('benchmark','benchmark-titles'):
        for p in (ROOT/folder).glob('*.json'):
            for r in json.loads(p.read_text())['pairs']:
                testid.add(r['work_id']);testtitle.add(r['text_group'])
    # Exclude all papers in test-overlapping families as well, not just sampled test records.
    for r in rows:
        if r['family_id'] in excluded:
            testid.add(r['work_id']);testtitle.add(r['text_group'])
    return vfamily,excluded,vid,vtitle,testid,testtitle

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(path)

def build(final=False):
    TRAIN.mkdir(exist_ok=True)
    batches,rows,qhash=catalog();split=freeze_split(rows,qhash)
    if final:
        plan=json.loads((FAST/'assignment.json').read_text());votes=load_answers(batches)
        for m in votes:
            required={k for ids in plan['batches'][m].values() for k in ids}
            if not required<=set(votes[m]):raise ValueError(f'{m} not finished yet')
    accepted,counts=labels(batches,rows)
    vf,ex,vid,vt,tid,tt=forbidden(rows,split)
    train=[];val=[];rejected=Counter()
    for r in rows:
        if r['family_id'] in ex or r['work_id'] in tid or r['text_group'] in tt:
            rejected['test_overlap']+=1;continue
        if r['id'] not in accepted:continue
        y,teachers=accepted[r['id']]
        row={k:r[k] for k in ('id','work_id','text_group','topic','definition','text','family_id')}
        row.update(y=y,teachers=teachers)
        if r['family_id'] in vf:val.append(row)
        elif r['work_id'] in vid or r['text_group'] in vt:rejected['validation_paper_overlap']+=1
        else:train.append(row)
    valpath=TRAIN/'validation.jsonl'
    if not valpath.exists():
        # Freeze a bounded, family-balanced validation set, not a moving target.
        selected=[]
        for f in sorted(vf):
            candidates=sorted([r for r in val if r['family_id']==f],key=lambda r:digest(['val',r['id']]))
            selected+=candidates[:80]
        if len(selected)<1000 or len({r['y'] for r in selected})!=2:raise ValueError('validation snapshot too small')
        write_rows(valpath,selected)
    val=json.loads('[]') if not valpath.exists() else [json.loads(l) for l in open(valpath)]
    for r in val:
        if r['id'] not in accepted or accepted[r['id']][0]!=r['y']:raise ValueError('frozen validation label changed')
    if len(train)<10000 or len({r['y'] for r in train})!=2:raise ValueError('insufficient training data')
    name='final' if final else 'initial';path=TRAIN/f'{name}.jsonl'
    train.sort(key=lambda r:r['id'])
    if path.exists():
        old=[json.loads(l) for l in open(path)]
        if old!=train and not final:
            print('Initial snapshot already frozen; leaving it untouched',flush=True);return
    if final and (TRAIN/'initial.jsonl').exists():
        by={r['id']:r for r in train}
        for r in map(json.loads,open(TRAIN/'initial.jsonl')):
            if by.get(r['id'])!=r:raise ValueError('initial train row disappeared/changed; inspect late labels')
    assert not {r['work_id'] for r in train}&{r['work_id'] for r in val}
    assert not {r['text_group'] for r in train}&{r['text_group'] for r in val}
    write_rows(path,train)
    report={'snapshot':name,'train':len(train),'validation':len(val),'train_positive':sum(r['y'] for r in train),
            'train_families':len({r['family_id'] for r in train}),'label_status':counts,'excluded':dict(rejected),
            'queue_hash':qhash,'train_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'validation_sha256':hashlib.sha256(valpath.read_bytes()).hexdigest(),'mechanical_checks_passed':True,
            'reference':'Silver teacher labels, not reviewed ground truth; experimental training only.'}
    write_json(TRAIN/f'{name}-manifest.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');build(ap.parse_args().final)

"""Freeze a disjoint 50/50 assignment of the remaining unanswered pairs.

Run only after stopping both workers; acquire their locks too. Completed labels
(including uncertainty) are preserved. A prior failed/refused request is assigned
to the other teacher if possible; if both refused, retain for review. Allocate
whole batches where possible, balancing within families and globally, splitting
only when necessary to hit the pair-count target. No provider calls.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from stream_queue import FAST, digest, write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))  # the fieldjudge folder: teachers.py
import teachers  # noqa: E402

MODELS=teachers.ROLES
A,B=MODELS
VALID={'on_topic','off_topic','uncertain'}

def make_plan(batches, answers):
    rows={}
    for batch_id,batch in batches.items():
        for row in batch['pairs']:
            if row['id'] in rows:raise ValueError('duplicate queue pair')
            rows[row['id']]=(batch_id,row)
    for model in MODELS:
        if set(answers[model])-set(rows):raise ValueError('answer names unknown pair')
        if any(v not in VALID and v is not None for v in answers[model].values()):raise ValueError('invalid saved label')
    covered={k for k in rows if any(answers[m].get(k) in VALID for m in MODELS)}
    allocations={m:defaultdict(list) for m in MODELS}
    counts=Counter(); per_family=defaultdict(Counter); review=[]; flexible=defaultdict(list)
    def assign(pair_id,model):
        batch_id,row=rows[pair_id]
        allocations[model][batch_id].append(pair_id)
        counts[model]+=1
        per_family[row.get('family_id','')][model]+=1
    for k,(batch_id,row) in rows.items():
        if k in covered:continue
        eligible=[m for m in MODELS if k not in answers[m]]  # don't repeat a refused/unreadable request
        if not eligible:review.append(k)
        elif len(eligible)==1:assign(k,eligible[0])
        else:flexible[batch_id].append(k)
    total=len(rows)-len(covered)-len(review)
    a_target=max(counts[A],min(total//2,total-counts[B]))
    targets={A:a_target,B:total-a_target}
    order=sorted(flexible)
    random.Random(15726).shuffle(order)
    for batch_id in order:
        remaining=list(flexible[batch_id])
        while remaining:
            family=rows[remaining[0]][1].get('family_id','')
            possible=[m for m in MODELS if counts[m]<targets[m]]
            if not possible:raise ValueError('assignment capacity exhausted')
            model=min(possible,key=lambda m:(per_family[family][m],counts[m],digest([batch_id,m])))
            n=min(len(remaining),targets[model]-counts[model])
            for k in remaining[:n]:assign(k,model)
            remaining=remaining[n:]
    sets={m:{k for ks in allocations[m].values() for k in ks} for m in MODELS}
    if sets[A]&sets[B] or (sets[A]|sets[B])&covered:raise ValueError('overlap in partition')
    if sets[A]|sets[B]|covered|set(review)!=set(rows):raise ValueError('incomplete partition')
    return {'version':1,'mode':'single-teacher-remainder','created':time.time(),
            'queue_pairs':len(rows),'already_answered_pairs':len(covered),'review_only_ids':sorted(review),
            'remaining_pairs':total,'assigned_pairs':{m:counts[m] for m in MODELS},'targets':targets,
            'batches':{m:dict(allocations[m]) for m in MODELS},
            'source_batch_hashes':{b:digest(batch) for b,batch in batches.items()},
            'answer_snapshot_sha256':digest(answers)}

def load_answers(batches, root=FAST):
    out={m:{} for m in MODELS}
    for model in MODELS:
        for path in sorted((root/'answers'/model).glob('*.json')):
            if path.name=='status.json':continue
            result=json.loads(path.read_text())
            if path.stem not in batches or result['batch_sha256']!=digest(batches[path.stem]):
                raise ValueError('answer/input checksum mismatch')
            expected={r['id'] for r in batches[path.stem]['pairs']}
            requested=set(result.get('requested_ids',expected))
            actual=[v['id'] for v in result['answers']]
            if not requested <= expected or set(actual)!=requested or len(actual)!=len(set(actual)):
                raise ValueError('incomplete or duplicate answer records')
            for v in result['answers']:
                if v['id'] not in expected or v['id'] in out[model]:raise ValueError('bad answer identity')
                out[model][v['id']]=v['label']
    return out

def create(root=FAST):
    target=root/'assignment.json'
    if target.exists():
        p=json.loads(target.read_text())
        print('Existing immutable assignment:',p['assigned_pairs']);return p
    producer=root/'producer-done.json'
    if not producer.exists():raise ValueError('finish collection before freezing the remaining assignments')
    with ExitStack() as stack:
        for m in MODELS:
            folder=root/'answers'/m;folder.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context(open(folder/'worker.lock','a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        batches={p.stem:json.loads(p.read_text()) for p in sorted((root/'queue').glob('*.json'))}
        plan=make_plan(batches,load_answers(batches,root))
        write_json(target,plan)  # immutable: a resume reads this same ownership plan
    print(json.dumps({k:v for k,v in plan.items() if k not in ('batches','source_batch_hashes','review_only_ids')},indent=2))
    return plan

if __name__=='__main__':
    create()

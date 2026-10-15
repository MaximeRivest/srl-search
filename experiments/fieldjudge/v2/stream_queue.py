"""Standard-library-only queue producer shared by CPU collection and teacher workers."""
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
ROOT=Path(os.environ.get('FIELDJUDGE_V2','/mnt/fast/scholarsreadinglist/fieldjudge/v2'))
FAST=ROOT/'fast'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    os.replace(tmp,path)

def enqueue(rows,source):
    folder=FAST/'queue';folder.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in rows:
        r={**r,'id':digest([r['topic'],r.get('definition'),r['work_id'],r['text']])[:24]}
        groups[(r['topic'],r.get('definition'))].append(r)
    count=0
    for key,group in groups.items():
        random.Random(digest([source,key])).shuffle(group)
        for i in range(0,len(group),25):
            pairs=group[i:i+25]
            payload={'source':source,'pairs':pairs}
            batch_id=digest(payload)[:24]
            path=folder/f'{batch_id}.json'
            if path.exists() and json.loads(path.read_text())!=payload:raise ValueError('queue collision')
            if not path.exists():write_json(path,payload)
            count+=len(pairs)
    return count

def inventory():
    ids=set();count=0
    for path in (FAST/'queue').glob('*.json'):
        for row in json.loads(path.read_text())['pairs']:
            if row['id'] in ids:raise ValueError('duplicate pair in queue')
            ids.add(row['id']);count+=1
    return ids,count

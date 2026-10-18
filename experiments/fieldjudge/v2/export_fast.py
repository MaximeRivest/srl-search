"""Join streaming labels; reuse old reference-model answers (round 1) as free checks where available.

This exports PROVISIONAL supervision, not training approval. Missing answers,
refusals, uncertainty, disagreements, and cross-split duplicates remain explicit.
Whole-topic-family overlap still needs auditing before a training release.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from stream_queue import ROOT,FAST,digest,write_json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))  # the fieldjudge folder: teachers.py
import teachers  # noqa: E402

def resolve(answers, old=None, overlap=False, single_mode=False, assigned_pending=False):
    """Single-teacher labels are usable provisionally but never reported as consensus."""
    available={m:a['label'] for m,a in answers.items() if a is not None and a.get('label') is not None}
    labels=list(available.values())
    reason=('validation_overlap' if overlap else
            'pending' if assigned_pending or (not single_mode and any(a is None for a in answers.values())) else
            ('unanswered' if any(a is not None for a in answers.values()) else 'pending') if not labels else
            'uncertain' if 'uncertain' in labels else
            'disagreement' if len(set(labels))!=1 else
            'reference_disagrees' if old is not None and (old>=.5)!=(labels[0]=='on_topic') else
            'provisional_agreement' if len(available)>1 else 'provisional_single_teacher')
    y=int(labels[0]=='on_topic') if reason in ('provisional_agreement','provisional_single_teacher') else None
    return reason,y,list(available)

def run(partial=False):
    if not partial and not (FAST/'producer-done.json').exists():raise ValueError('collection still running')
    rows={}
    for p in sorted((FAST/'queue').glob('*.json')):
        for r in json.loads(p.read_text())['pairs']:
            if r['id'] in rows:raise ValueError('duplicate input pair')
            rows[r['id']]=r
    assignment_path=FAST/'assignment.json'
    assignment=json.loads(assignment_path.read_text()) if assignment_path.exists() else None
    allocated={m:{k for ids in assignment['batches'][m].values() for k in ids} if assignment else set() for m in teachers.ROLES}
    votes={}
    for model in teachers.ROLES:
        by={}
        for path in (FAST/'answers'/model).glob('*.json'):
            if path.name=='status.json':continue
            result=json.loads(path.read_text())
            batch=json.loads((FAST/'queue'/path.name).read_text())
            if result['batch_sha256']!=digest(batch):raise ValueError('answer/input checksum mismatch')
            for v in result['answers']:
                if v['id'] in by:raise ValueError('duplicate teacher answer')
                by[v['id']]=v
        if not partial:
            if assignment:
                if not allocated[model] <= set(by):raise ValueError(f'{model} assigned work incomplete')
            elif set(by)!=set(rows):raise ValueError(f'{model} incomplete')
        votes[model]=by
    if not partial and set().union(*votes.values()) != set(rows):
        raise ValueError('some pairs have no completed attempt')
    reference={(r['topic_id'],r['id']):r['y'] for r in map(json.loads,open(ROOT.parent/'labels.jsonl'))}
    val=[r for r in rows.values() if r['split']=='validation']
    vwork={r['work_id'] for r in val};vtext={r['text_group'] for r in val}
    norm=lambda s:' '.join(re.sub(r'[^\w\s]',' ',s.casefold()).split())
    vtopic={norm(r['topic']) for r in val}
    counts=Counter(); output=[]
    for k,r in rows.items():
        answers={m:v.get(k) for m,v in votes.items()}
        old=None
        if r['family_id'].startswith('legacy-'):
            old=reference.get((int(r['family_id'][7:]),r['work_id']))
        overlap=r['split']!='validation' and (r['work_id'] in vwork or r['text_group'] in vtext or norm(r['topic']) in vtopic)
        waiting=any(k in allocated[m] and k not in votes[m] for m in votes)
        reason,y,sources=resolve(answers,old,overlap,assignment is not None,waiting)
        counts[reason]+=1
        row={**r,'answers':answers,'reference_v1':old,'status':reason,'label_sources':sources,'training_approved':False}
        if y is not None:row['y']=y
        output.append(row)
    dest=FAST/'exports';dest.mkdir(exist_ok=True)
    path=dest/'joined.jsonl';tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output));tmp.replace(path)
    summary={'counts':dict(counts),'pairs':len(rows),'partial':partial,'training_approved':False,
             'assignment_mode':assignment['mode'] if assignment else 'dual',
             'note':'Whole-family leakage and remaining disagreements require audit before training.'}
    write_json(dest/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');run(ap.parse_args().partial)

"""Join streaming labels; reuse old Kimi answers as free checks where available.

This exports PROVISIONAL supervision, not training approval. Missing answers,
refusals, uncertainty, disagreements, and cross-split duplicates remain explicit.
Whole-topic-family overlap still needs auditing before a training release.
"""
import argparse
import json
import re
from collections import Counter
from stream_queue import ROOT,FAST,digest,write_json

def run(partial=False):
    if not partial and not (FAST/'producer-done.json').exists():raise ValueError('collection still running')
    rows={}
    for p in sorted((FAST/'queue').glob('*.json')):
        for r in json.loads(p.read_text())['pairs']:
            if r['id'] in rows:raise ValueError('duplicate input pair')
            rows[r['id']]=r
    votes={}
    for model in ('sol','opus'):
        by={}
        for path in (FAST/'answers'/model).glob('*.json'):
            if path.name=='status.json':continue
            result=json.loads(path.read_text())
            batch=json.loads((FAST/'queue'/path.name).read_text())
            if result['batch_sha256']!=digest(batch):raise ValueError('answer/input checksum mismatch')
            for v in result['answers']:
                if v['id'] in by:raise ValueError('duplicate teacher answer')
                by[v['id']]=v
        if not partial and set(by)!=set(rows):raise ValueError(f'{model} incomplete')
        votes[model]=by
    kimi={(r['topic_id'],r['id']):r['y'] for r in map(json.loads,open(ROOT.parent/'labels.jsonl'))}
    val=[r for r in rows.values() if r['split']=='validation']
    vwork={r['work_id'] for r in val};vtext={r['text_group'] for r in val}
    norm=lambda s:' '.join(re.sub(r'[^\w\s]',' ',s.casefold()).split())
    vtopic={norm(r['topic']) for r in val}
    counts=Counter(); output=[]
    for k,r in rows.items():
        answers={m:v.get(k) for m,v in votes.items()}
        ys=[a.get('label') if a else None for a in answers.values()]
        old=None
        if r['family_id'].startswith('legacy-'):
            old=kimi.get((int(r['family_id'][7:]),r['work_id']))
        overlap=r['split']!='validation' and (r['work_id'] in vwork or r['text_group'] in vtext or norm(r['topic']) in vtopic)
        reason=('validation_overlap' if overlap else 'pending' if any(a is None for a in answers.values()) else
                'unanswered' if None in ys else 'uncertain' if 'uncertain' in ys else
                'disagreement' if len(set(ys))!=1 else
                'kimi_disagrees' if old is not None and (old>=.5)!=(ys[0]=='on_topic') else 'provisional_agreement')
        counts[reason]+=1
        row={**r,'answers':answers,'kimi_v1':old,'status':reason,'training_approved':False}
        if reason=='provisional_agreement':row['y']=int(ys[0]=='on_topic')
        output.append(row)
    dest=FAST/'exports';dest.mkdir(exist_ok=True)
    path=dest/'joined.jsonl';tmp=path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output));tmp.replace(path)
    summary={'counts':dict(counts),'pairs':len(rows),'partial':partial,'training_approved':False,
             'note':'Whole-family leakage and remaining disagreements require audit before training.'}
    write_json(dest/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');run(ap.parse_args().partial)

"""Matched old/new 0.8B evaluation on eight reserved topics plus old diagnostics.
Exclude papers/titles seen in v1 training from the new-topic comparison for BOTH
models. Kimi labels are a silver reference; no threshold is tuned on these tests.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from srl_search import judges as J
from stream_queue import ROOT,write_json,digest
from training_data import title_key
from mine import gate,VPY
ANSWER=' Answer with one word: "yes" (on topic) or "no" (not on topic).'
OUT=ROOT/'evaluation-08';OUT.mkdir(exist_ok=True)

def data():
    old=ROOT.parent
    labels=[json.loads(l) for l in open(old/'labels.jsonl')]
    excluded={113,117,137}
    val=set(random.Random(0).sample(sorted({r['topic_id'] for r in labels if r['topic_id'] not in excluded}),40))
    keys={(r['topic_id'],r['id']) for r in labels if r['topic_id'] not in excluded|val}
    train_ids={w for t,w in keys}
    train_titles={title_key(r['text']) for r in map(json.loads,open(old/'samples.jsonl')) if (r['topic_id'],r['id']) in keys}
    ref=defaultdict(set)
    for f in (ROOT/'benchmark-kimi').glob('wave-*.json'):
        for r in json.loads(f.read_text())['answers']:
            if r['y'] is not None:ref[r['id']].add(r['y'])
    rows=[];removed=defaultdict(int)
    for folder in ('benchmark','benchmark-titles'):
        for f in sorted((ROOT/folder).glob('*.json')):
            for r in json.loads(f.read_text())['pairs']:
                if r['work_id'] in train_ids or r['text_group'] in train_titles:
                    removed['seen_by_v1']+=1;continue
                if len(ref[r['id']])!=1:
                    removed['no_unique_reference']+=1;continue
                rows.append({**r,'kind':folder,'y':next(iter(ref[r['id']])),'suite':'new_topics'})
    for r in map(json.loads,open(old/'tests.jsonl')):
        if r['kimi'] is None:continue
        rows.append({'id':r['id'],'topic':r['topic'],'definition':r['definition'],'text':r['text'],'kind':r['set'],
                     'suite':'development','y':r['kimi'],'weight':r.get('weight',1.),'stratum':r.get('stratum',r['set'])})
    return rows,dict(removed)

def metrics(rows,p):
    y=np.array([r['y'] for r in rows]);w=np.array([r['weight'] for r in rows]);yes=p>=.5
    tp=float((w*y*yes).sum());precision=tp/max(1e-12,float((w*yes).sum()));recall=tp/max(1e-12,float((w*y).sum()))
    return {'n':len(y),'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-12,precision+recall),
            'agreement':float(np.average(yes==y,weights=w)), 'sample_agreement':float((yes==y).mean()),
            'auc':float(roc_auc_score(y,p,sample_weight=w)) if len(set(y))==2 else None,
            'average_precision':float(average_precision_score(y,p,sample_weight=w)) if y.sum()>0 else None}

def run():
    rows,removed=data();fingerprint=digest(rows)
    write_json(OUT/'case-manifest.json',{'n':len(rows),'excluded':removed,'case_hash':fingerprint})
    scores={};results={}
    models={'v1':ROOT.parent/'m08/merged','v2':ROOT/'models/0.8B/merged'}
    for name,model in models.items():
        file=OUT/f'{name}-scores.json';meta=OUT/f'{name}-manifest.json'
        contract={'cases':fingerprint,'model':str(model),'model_mtime':(model/'model.safetensors').stat().st_mtime}
        if file.exists() and meta.exists() and json.loads(meta.read_text())==contract:
            p=np.array(json.loads(file.read_text()))
        else:
            gate()
            src=OUT/f'{name}-input.jsonl'
            src.write_text(''.join(json.dumps({'text':r['text'],'system':J.instruction(r['topic'],r['definition'])+ANSWER})+'\n' for r in rows))
            J.run([VPY,str(Path(J.__file__).with_name('vllm_decider.py')),str(model),str(src),str(file),J.GPU],OUT/f'{name}-log.txt')
            p=np.array(json.loads(file.read_text()))
            if len(p)!=len(rows) or not np.isfinite(p).all():raise ValueError('invalid prediction output')
            write_json(meta,contract)
        scores[name]=p
        grouped={}
        for suite,topic,kind in sorted({(r['suite'],r['topic'],r['kind']) for r in rows}):
            idx=[i for i,r in enumerate(rows) if (r['suite'],r['topic'],r['kind'])==(suite,topic,kind)]
            grouped[f'{suite} / {topic} / {kind}']=metrics([rows[i] for i in idx],p[idx])
        results[name]=grouped
        print('SCORED',name,flush=True)
    topics=sorted({r['topic'] for r in rows if r['suite']=='new_topics' and r['kind']=='benchmark'})
    macro={name:{metric:float(np.mean([results[name][f'new_topics / {t} / benchmark'][metric] for t in topics]))
                       for metric in ('precision','recall','f1','auc','sample_agreement','average_precision')} for name in models}
    # Paired, within-stratum bootstrap; conditional on these topics and one training seed.
    strata=defaultdict(list)
    for i,r in enumerate(rows):
        if r['suite']=='new_topics' and r['kind']=='benchmark':strata[(r['topic'],r['stratum'])].append(i)
    rng=np.random.default_rng(22026);deltas=[]
    for _ in range(400):
        by=defaultdict(list)
        for (topic,s),idx in strata.items():by[topic].extend(rng.choice(idx,len(idx),replace=True).tolist())
        vals={name:np.mean([metrics([rows[i] for i in idx],scores[name][idx])['f1'] for idx in by.values()]) for name in models}
        deltas.append(vals['v2']-vals['v1'])
    report={'reference':'Kimi silver labels, not human gold. Fixed 0.5 threshold.', 'excluded':removed,'macro':macro,
            'paired_macro_f1_difference':macro['v2']['f1']-macro['v1']['f1'],
            'paired_f1_ci95':np.percentile(deltas,[2.5,97.5]).tolist(), 'per_topic':results,
            'caveats':['Single training seed each; intervals are conditional on these topics.',
                       'Old diagnostic topics have guided development; not untouched tests.',
                       'Original sampling weights retained after removing v1 training overlaps; compare these matched samples, not an exact census.',
                       'Standalone judges, not yet a full map/reading-list replay.']}
    write_json(OUT/'comparison.json',report)
    print(json.dumps({'macro':macro,'delta':report['paired_macro_f1_difference'],'ci95':report['paired_f1_ci95']}),flush=True)

if __name__=='__main__':run()

"""Wait for both models, then score the frozen eight-topic benchmark on GPU 1.
No provider calls. Kimi is an independent silver reference, not human truth.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from srl_search import judges as J
from mine import gate,VPY
from stream_queue import ROOT,write_json
ANSWER=' Answer with one word: "yes" (on topic) or "no" (not on topic).'

def run():
    while not all((ROOT/'models'/m/'DONE.json').exists() for m in ('0.8B','2B')):
        time.sleep(60)
    labels={}
    for f in (ROOT/'benchmark-kimi').glob('wave-*.json'):
        for r in json.loads(f.read_text())['answers']:
            if r['y'] is not None:labels.setdefault(r['id'],set()).add(r['y'])
    rows=[]
    for folder in ('benchmark','benchmark-titles'):
        for f in sorted((ROOT/folder).glob('*.json')):
            d=json.loads(f.read_text())
            for r in d['pairs']:
                ys=labels.get(r['id'],set())
                if len(ys)==1:rows.append({**r,'test_set':folder,'reference':next(iter(ys))})
    results={}
    for size in ('0.8B','2B'):
        model=ROOT/'models'/size/'merged';work=ROOT/'evaluation'/size;work.mkdir(parents=True,exist_ok=True)
        scores=[]
        for i in range(0,len(rows),1000):
            chunk=rows[i:i+1000];src=work/f'{i}.jsonl';dst=work/f'{i}-scores.json'
            if not dst.exists():
                gate()
                src.write_text(''.join(json.dumps({'text':r['text'],'system':J.instruction(r['topic'],r['definition'])+ANSWER})+'\n' for r in chunk))
                J.run([VPY,str(Path(J.__file__).with_name('vllm_decider.py')),str(model),str(src),str(dst),J.GPU],work/'log.txt')
            p=json.loads(dst.read_text())
            if len(p)!=len(chunk):raise ValueError('prediction length mismatch')
            scores+=p
        p=np.array(scores);out={}
        for topic,kind in sorted({(r['topic'],r['test_set']) for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['topic']==topic and r['test_set']==kind]
            y=np.array([rows[i]['reference'] for i in ix]);w=np.array([rows[i]['weight'] for i in ix]);yes=p[ix]>=.5
            tp=float((w*y*yes).sum());pp=tp/max(1e-9,float((w*yes).sum()));rr=tp/max(1e-9,float((w*y).sum()))
            out[f'{topic} / {kind}']={'n':len(ix),'precision':pp,'recall':rr,'f1':2*pp*rr/max(1e-9,pp+rr),
                 'auc':float(roc_auc_score(y,p[ix],sample_weight=w)) if len(set(y))==2 else None}
        results[size]=out
        write_json(ROOT/'evaluation/results.json',{'reference':'Kimi silver labels; retrieval and title-only populations reported separately','models':results})
        print('EVALUATED',size,flush=True)

if __name__=='__main__':run()

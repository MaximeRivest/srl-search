"""Stream already-screened new families into labeling; no GPU mining or global wait.

Uses the existing CPU sampler and test-paper exclusion. The 50 extra examples
are random retrieval candidates, explicitly NOT model-mined. Raw labels are
candidate data: validation/seed cross-over is removed by export_fast.py before
anything may train on it. The target includes the existing-paper enrichment.
"""
import json
import random
import time
from pathlib import Path
import collect
from stream_queue import ROOT,FAST,digest,enqueue,inventory,write_json

TARGET=100000

def run():
    manifest=json.loads((ROOT/'topics-available.json').read_text())
    families=manifest['families']
    seen,count=inventory()
    c=collect.Collector(family_dir=FAST/'families');c.reserve()
    for family in sorted(families,key=lambda f:(f['split']!='validation',f['id'])):
        if count>=TARGET:break
        c.family(family)
        data=json.loads((c.family_dir/f"{family['id']}.json").read_text())
        pool=list(data['mining_pool']);random.Random(family['id']).shuffle(pool)
        extra=[{**r,'source':'retrieval_extra_unmined'} for r in pool[:50]]
        rows=data['coverage']+data['cross_scope']+extra
        fresh=[]
        for r in rows:
            pair_id=digest([r['topic'],r.get('definition'),r['work_id'],r['text']])[:24]
            if pair_id not in seen and count+len(fresh)<TARGET:
                seen.add(pair_id);fresh.append({**r,'id':pair_id})
        if fresh:
            enqueue(fresh,'screened-family-'+family['id'])
            count+=len(fresh)
        write_json(FAST/'producer-status.json',{'state':'collecting','pairs_queued':count,'target':TARGET,
                    'last_family':family['id'],'updated':time.time(),'gpu_mining':False})
        print('STREAM QUEUED',count,'/',TARGET,flush=True)
    write_json(FAST/'producer-done.json',{'pairs_queued':count,'target':TARGET,'target_met':count>=TARGET,
               'available_families':len(families),'deferred_subjects':manifest.get('deferred',[]),
               'gpu_mining':False,'training_approved':False})
    print('COLLECTION DONE',count,'pairs',flush=True)

if __name__=='__main__':run()

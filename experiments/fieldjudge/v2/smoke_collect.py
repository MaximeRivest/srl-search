"""CPU integration test: one family in an isolated scratch directory, no model calls."""
import hashlib
import json
import tempfile
from pathlib import Path
import collect

def run():
    proposals=sorted((collect.ROOT/'topic-generation').glob('*.json'))
    raw=next(json.loads(p.read_text()) for p in proposals
             if isinstance(json.loads(p.read_text()),dict) and 'args' in json.loads(p.read_text()) and 'families' in json.loads(p.read_text()))
    scopes=raw['families'][0]['scopes']
    for i,s in enumerate(scopes):s['id']=f'smoke-scope-{i}'
    family={'id':'a'*20,'field':raw['args']['field'],'split':'validation','scopes':scopes}
    real=collect.ROOT
    with tempfile.TemporaryDirectory(prefix='v2-smoke-',dir=real.parent) as temp:
        collect.ROOT=Path(temp)
        c=collect.Collector()
        c.family(family)
        path=collect.ROOT/'families'/f"{family['id']}.json"
        data=json.loads(path.read_text())
        assert len(data['coverage'])==100 and len(data['cross_scope'])==50 and len(data['mining_pool'])==100
        all_rows=data['coverage']+data['cross_scope']+data['mining_pool']
        assert len({r['id'] for r in all_rows})==250
        assert all(r['split']=='validation' for r in all_rows)
        stamp=hashlib.sha256(path.read_bytes()).hexdigest()
        c.family(family)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==stamp
        other={**data['coverage'][0],'split':'train'}
        assert not c.own(other)
        print('SMOKE PASSED: counts, unique pairs, resume, cross-split rejection',flush=True)

if __name__=='__main__':run()

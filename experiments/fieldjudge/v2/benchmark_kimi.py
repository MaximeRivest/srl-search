"""Independent silver labels for the frozen benchmark only, never training data.

Budget: $20 for this benchmark run, checkpointed between chunks. Reserve an
intentionally conservative UTF-8 byte token bound for each wave before sending;
if the remaining budget is insufficient, stop instead of guessing labels.
Kimi labels are a diagnostic reference, not reviewed ground truth.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from srl_search import judges as J
ROOT=Path(os.environ.get('FIELDJUDGE_V2','/mnt/fast/scholarsreadinglist/fieldjudge/v2'))
BUDGET=20.0

def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));tmp.replace(path)

def run():
    outdir=ROOT/'benchmark-kimi';outdir.mkdir(exist_ok=True)
    spent=sum(json.loads(p.read_text())['usd'] for p in outdir.glob('wave-*.json'))
    for path in sorted((ROOT/'benchmark').glob('*.json'))+sorted((ROOT/'benchmark-titles').glob('*.json')):
        data=json.loads(path.read_text());scope=data['scope'];pairs=data['pairs']
        for start in range(0,len(pairs),64):
            chunk=pairs[start:start+64]
            key=f"{path.parent.name}-{path.stem}-{start:04d}"
            dest=outdir/f"wave-{key}.json"
            if dest.exists():continue
            system=J.instruction(scope['topic'],scope['definition'])
            # UTF-8 bytes upper-bound tokenizer size conservatively, including chat boilerplate.
            reserve=sum((len((system+r['text']).encode())+512)*3.0/1e6 + 8*15/1e6 for r in chunk)
            if spent+reserve*6>BUDGET:  # six provider attempts is the teacher client's maximum
                print('PAUSED before budget risk',spent,reserve,flush=True);return
            teacher=J.Teacher(BUDGET-spent)
            y=teacher.label([r['text'] for r in chunk],scope['topic'],scope['definition'])
            spent+=teacher.usd
            write(dest,{'usd':teacher.usd,'source_sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
                        'answers':[{'id':r['id'],'y':None if v!=v else float(v)} for r,v in zip(chunk,y)],
                        'reference':'Kimi silver labels, not human-reviewed'})
            print('BENCHMARK KIMI',key,'total $',round(spent,3),flush=True)
    print('BENCHMARK KIMI DONE $',round(spent,3),flush=True)

if __name__=='__main__':run()

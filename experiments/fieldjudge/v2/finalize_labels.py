"""Freeze final train snapshot when both assigned labelers finish; leave validation fixed."""
import json
import sys
import time
from pathlib import Path
from stream_queue import FAST
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))  # the fieldjudge folder: teachers.py
import teachers  # noqa: E402
from training_data import build

if __name__=='__main__':
    while True:
        ready=True;status={}
        for m in teachers.ROLES:
            p=FAST/'answers'/m/'status.json'
            s=json.loads(p.read_text()) if p.exists() else {}
            status[m]=s.get('assigned_remaining')
            ready &= s.get('state')=='done' and s.get('assigned_remaining')==0
        if ready:break
        print('Awaiting final labels:',status,flush=True);time.sleep(30)
    build(final=True)

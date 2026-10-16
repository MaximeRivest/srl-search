"""Private live metrics dashboard. Exposes counts/curves only, never papers or credentials."""
import argparse
import json
import math
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from stream_queue import ROOT,FAST
HERE=Path(__file__).resolve().parent
LOCK=threading.Lock(); ANSWERS={'sol':{},'opus':{}}; SEEN={'sol':set(),'opus':set()}

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def unit(name):
    r=subprocess.run(['systemctl','--user','show',name,'-p','ActiveState','-p','SubState','-p','ExecMainStatus'],capture_output=True,text=True,timeout=5)
    return dict(l.split('=',1) for l in r.stdout.splitlines() if '=' in l)

def logs(path):
    out=[]
    try:
        for l in path.read_text().splitlines():
            try:out.append(json.loads(l))
            except ValueError:pass
    except OSError:pass
    return out

def snapshot():
    with LOCK:
        for m in ANSWERS:
            for path in (FAST/'answers'/m).glob('*.json'):
                if path.name=='status.json' or path.stem in SEEN[m]:continue
                value=read(path)
                if value is None:continue
                for r in value.get('answers',[]):ANSWERS[m][r['id']]=r.get('label')
                SEEN[m].add(path.stem)
        ids=set(ANSWERS['sol'])|set(ANSWERS['opus'])
        labeled=sum(any(ANSWERS[m].get(k) is not None for m in ANSWERS) for k in ids)
        binary=sum(any(ANSWERS[m].get(k) in ('on_topic','off_topic') for m in ANSWERS) for k in ids)
        initial=read(ROOT/'training/initial-manifest.json',{})
        final=read(ROOT/'training/final-manifest.json',{})
        service=unit('fj-v2-models.service')
        models=[]
        for size in ('0.8B','2B'):
            path=ROOT/'models'/size
            events=logs(path/'log.jsonl')
            loss=[e for e in events if e.get('event')=='train']
            val=[e for e in events if e.get('event')=='validation']
            done=read(path/'DONE.json')
            last=events[-1] if events else {}
            last_train=loss[-1] if loss else {}
            rate=[e['papers_s'] for e in loss[-10:] if e.get('papers_s',0)>0]
            speed=sum(rate)/len(rate) if rate else None
            total=initial.get('train',0)+final.get('train',initial.get('train',0))
            epoch=last_train.get('epoch',1)
            seen=(initial.get('train',0) if epoch==2 else 0)+last_train.get('seen',0)
            state=('done' if done else 'queued' if not events else 'paused' if last.get('event') in ('yielding_to_live_map','waiting_for_late_labels') else
                   'stopped' if service.get('ActiveState')=='failed' else 'training')
            if done:seen=total
            models.append({'size':size,'state':state,'event':last.get('event'),'step':last.get('step',0),
                           'epoch':epoch,'seen':seen,'total':total,'rate':speed,
                           'eta_s':max(0,(total-seen)/speed*1.15) if speed and not done else 0 if done else None,
                           'loss':[{'x':e['step'],'y':e['loss']} for e in loss],
                           'validation':[{'x':e['step'],'auc':e['auc'],'agreement':e['agreement']} for e in val],
                           'updated':last.get('time')})
        return {'updated':time.time(),'labeling':{'target':100000,'attempted':len(ids),'answered':labeled,'binary':binary,
                'teachers':{m:read(FAST/'answers'/m/'status.json',{}) for m in ANSWERS}},
                'initial':initial,'final':final,'models':models,'training_service':service,
                'next':'Late accepted training labels enter pass two; validation stays fixed. 2B follows 0.8B. No automatic deployment.'}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path.split('?')[0] in ('/status.json','/status'):
            try:body=json.dumps(snapshot(),allow_nan=False).encode();code=200
            except Exception as e:body=json.dumps({'error':type(e).__name__}).encode();code=503
            kind='application/json'
        elif self.path.split('?')[0] in ('/','/index.html'):
            body=(HERE/'dashboard/index.html').read_bytes();kind='text/html; charset=utf-8';code=200
        else:body=b'Not found';kind='text/plain';code=404
        self.send_response(code);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8943);a=ap.parse_args()
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

"""Two-pass experimental fine-tune: frozen initial data, then frozen final data.

Validation is immutable. Late labels enter only at the epoch boundary, with the
optimizer preserved. Checkpoints include row order, cursor, RNG and dataset hashes.
Exit 3 yields GPU to live maps; exit 4 waits for final data without holding GPU.
The exported model is the FINAL state, so it has seen every usable late example.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from srl_search import judges as J
from srl_search.field import FieldStore, ROOT as FIELDS

DATA=Path(os.environ.get('FIELDJUDGE_V2','/mnt/fast/scholarsreadinglist/fieldjudge/v2'))/'training'
ANSWER=' Answer with one word: "yes" (on topic) or "no" (not on topic).'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(l) for l in open(path)]

def wanted():
    p=FieldStore(FIELDS).plan()
    return J.RESERVED.exists() or bool(p['running'] and p['running']['st'].get('kind')=='full') or any(
        r['st'].get('kind')=='full' and r['start']-p['now']<300 for r in p['waiting'][:1])

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    log=open(out/'log.jsonl','a')
    def say(**v):
        v['time']=time.time();s=json.dumps(v);print(s,flush=True);log.write(s+'\n');log.flush()
    val_path=DATA/'validation.jsonl';initial=DATA/'initial.jsonl'
    contract={'base':a.base,'rank':32,'lr':1e-4,'batch':32,'seed':0,'max_length':768,
              'validation_sha256':sha(val_path),'initial_sha256':sha(initial)}
    cp=out/'checkpoint.pt'
    state={'epoch':0,'cursor':0,'order':[],'step':0,'data_hashes':{},'best_auc':-1}
    saved=torch.load(cp,map_location='cpu',weights_only=False) if cp.exists() else None
    if saved:
        if saved['contract']!=contract:raise ValueError('checkpoint contract differs')
        state=saved['state']
    if state['epoch']>=2 and (out/'DONE.json').exists():return
    if state['epoch']==1 and not (DATA/'final-manifest.json').exists():raise SystemExit(4)
    if wanted():raise SystemExit(3)
    torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(a.base);tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    yes,no=tok.encode('yes',add_special_tokens=False)[0],tok.encode('no',add_special_tokens=False)[0]
    head=torch.nn.Parameter(model.get_output_embeddings().weight[[yes,no]].detach().clone().float())
    free=torch.cuda.mem_get_info()[0]/2**30;small='0.8B' in a.base
    micro=4 if small and free>=9 else 2
    checkpointing=not small and free<9
    model.config.use_cache=False
    if getattr(model.config,'text_config',None):model.config.text_config.use_cache=False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules='all-linear',task_type='CAUSAL_LM'))
    params=[p for p in model.parameters() if p.requires_grad]+[head]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    if saved:
        set_peft_model_state_dict(model,saved['lora']);head.data.copy_(saved['head'].to('cuda'))
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    def checkpoint():
        tmp=cp.with_suffix('.tmp')
        torch.save({'contract':contract,'state':state,'lora':get_peft_model_state_dict(model),'head':head.detach(),
                    'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},tmp)
        os.replace(tmp,cp)
    def tokens(row):
        system=J.instruction(row['topic'],row.get('definition'))+ANSWER
        def render(text):
            prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
            return tok(prompt,add_special_tokens=False)['input_ids']
        overhead=len(render(''));budget=768-overhead-16
        if budget<32:raise ValueError('topic definition consumes the context')
        body=tok(row['text'],add_special_tokens=False)['input_ids'][:budget]
        ids=render(tok.decode(body))
        while len(ids)>768:
            body=body[:max(0,len(body)-(len(ids)-768)-8)];ids=render(tok.decode(body))
        return ids
    def margin(xs):
        enc=tok.pad({'input_ids':xs},return_tensors='pt',pad_to_multiple_of=128).to('cuda')
        h=model.get_base_model().model(**enc,use_cache=False).last_hidden_state[:,-1,:].float()
        logits=h@head.T
        return logits[:,0]-logits[:,1]
    val=read(val_path);Xv=[tokens(r) for r in val];yv=np.array([r['y'] for r in val])
    def validate():
        model.eval();pred=np.zeros(len(val));order=sorted(range(len(val)),key=lambda i:len(Xv[i]))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(order),8):
                b=order[start:start+8];pred[b]=torch.sigmoid(margin([Xv[i] for i in b])).float().cpu().numpy()
        auc=float(roc_auc_score(yv,pred));acc=float(((pred>=.5)==yv).mean())
        say(event='validation',step=state['step'],epoch=state['epoch'],auc=auc,agreement=acc,n=len(val))
        if auc>state['best_auc']:
            state['best_auc']=auc
            tmp=out/'best.tmp'
            torch.save({'lora':get_peft_model_state_dict(model),'head':head.detach(),'step':state['step'],'auc':auc},tmp)
            os.replace(tmp,out/'best.pt')
        model.train()
    say(event='start_or_resume',base=a.base,epoch=state['epoch'],step=state['step'],validation=len(val),
        micro_batch=micro,gradient_checkpointing=checkpointing,free_gib=round(free,2))
    for epoch in range(state['epoch'],2):
        source=DATA/('initial.jsonl' if epoch==0 else 'final.jsonl')
        if epoch==1 and not (DATA/'final-manifest.json').exists():
            state.update(epoch=1,cursor=0,order=[]);checkpoint();say(event='waiting_for_late_labels');raise SystemExit(4)
        train=read(source);source_hash=sha(source)
        manifest=json.loads(source.with_name(source.stem+'-manifest.json').read_text())
        if not manifest['mechanical_checks_passed'] or manifest['train_sha256']!=source_hash or manifest['validation_sha256']!=sha(val_path):
            raise ValueError('data manifest validation failed')
        old=state['data_hashes'].get(str(epoch))
        if old and old!=source_hash:raise ValueError('epoch data changed underneath a checkpoint')
        state['data_hashes'][str(epoch)]=source_hash
        if not state['order']:
            state['order']=list(range(len(train)));random.Random(7100+epoch).shuffle(state['order']);state['cursor']=0
        if len(state['order'])!=len(train):raise ValueError('checkpoint data length mismatch')
        X=[tokens(r) for r in train]
        say(event='epoch',epoch=epoch+1,rows=len(train),data_sha256=source_hash)
        model.train();tick=time.time();seen=0
        while state['cursor']<len(train):
            if state['step']%20==0 and wanted():
                checkpoint();say(event='yielding_to_live_map');raise SystemExit(3)
            b=state['order'][state['cursor']:state['cursor']+32]
            progress=(epoch+state['cursor']/len(train))/2
            lr=1e-4*min(1,(state['step']+1)/50)*.5*(1+math.cos(math.pi*progress))
            for group in opt.param_groups:group['lr']=lr
            loss_sum=0.
            for start in range(0,len(b),micro):
                sub=b[start:start+micro];y=torch.tensor([train[i]['y'] for i in sub],device='cuda',dtype=torch.float32)
                with torch.autocast('cuda',dtype=torch.bfloat16):m=margin([X[i] for i in sub])
                loss=F.binary_cross_entropy_with_logits(m.float(),y,reduction='sum')/len(b)
                loss.backward();loss_sum+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step();opt.zero_grad(set_to_none=True)
            state['step']+=1;state['cursor']+=len(b);seen+=len(b)
            if state['step']%20==0:
                say(event='train',epoch=epoch+1,step=state['step'],seen=state['cursor'],total=len(train),
                    loss=loss_sum,papers_s=round(seen/(time.time()-tick),1));seen=0;tick=time.time()
            if state['step']%100==0:checkpoint()
            if state['step']%400==0:validate();checkpoint();tick=time.time();seen=0
        validate()
        state.update(epoch=epoch+1,cursor=0,order=[]);checkpoint()
        del X,train
    # Export the final state, not an earlier best checkpoint that may not have seen the late rows.
    del opt
    torch.cuda.empty_cache()
    merged=model.merge_and_unload()
    W=merged.get_output_embeddings().weight.detach().clone()
    W[[yes,no]]=head.detach().to(W.dtype)
    merged.lm_head=torch.nn.Linear(W.shape[1],W.shape[0],bias=False,dtype=W.dtype,device=W.device)
    merged.lm_head.weight=torch.nn.Parameter(W);merged.config.tie_word_embeddings=False
    if getattr(merged.config,'text_config',None):merged.config.text_config.tie_word_embeddings=False
    merged.save_pretrained(out/'merged',safe_serialization=True);tok.save_pretrained(out/'merged')
    (out/'merged/decider.json').write_text(json.dumps({'system':None,'yes':yes,'no':no,'base':a.base,'answer':ANSWER}))
    (out/'DONE.json').write_text(json.dumps({'contract':contract,'data_hashes':state['data_hashes'],'steps':state['step'],
                                           'best_validation_auc':state['best_auc'],'exported':'final_state_including_late_labels'}))
    say(event='done',steps=state['step'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--out',required=True)
    run(ap.parse_args())

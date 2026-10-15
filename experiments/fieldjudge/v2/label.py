"""Dual-teacher labeling in validated batches, through FunctAI's disk reply cache.

Run one process per teacher (flock prevents duplicate workers). Before each wave
of two requests: fail-closed quota check. An API/parser failure stops the process;
resume later without redoing completed waves. Misquoted evidence is quarantined,
not accepted into training. Missing/duplicate/unknown IDs invalidate the batch.
No fallback to a billed API, and no automatic launch without a passed pilot.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
from common import ROOT, PROMPT_VERSION, Case, classify, settings, validate_response, presented, make_cases, digest, write_json
from allowance import check
from control import provider_failure


def run(model, split, batch_size, limit_pairs=None):
    report = ROOT / "pilot-defined" / "acceptance.json"
    if batch_size != 25 or not report.exists() or not json.loads(report.read_text())["passed"]:
        raise ValueError("dual-teacher/order-check pilot not passed; refuse full run")
    source = ROOT / f"pairs-{split}.jsonl"
    rows = [json.loads(l) for l in open(source)]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty/duplicate inputs")
    if any(r["split"] != split for r in rows):
        raise ValueError("mixed splits")
    folder = ROOT / "judgments" / split / model
    folder.mkdir(parents=True, exist_ok=True)
    lock = open(folder / "worker.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = {"input_sha256": digest(rows), "prompt": PROMPT_VERSION, "protocol": "dual-teacher-order-check-v1", "batch_size": batch_size, "model": model}
    manifest = folder / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("labeling contract changed under existing results")
    write_json(manifest, contract)
    fn = classify.using(**settings(model))
    batches = [make_cases(rows[i:i+batch_size]) for i in range(0, len(rows), batch_size)]
    for start in range(0,len(batches),2):
        if limit_pairs is not None and start*batch_size >= limit_pairs:
            print('COHORT READY: stopped at',limit_pairs,'pairs for audit; not the full dataset',flush=True)
            return
        group = batches[start:start+2]
        path = folder / f"wave-{start:06d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["input_sha256"] != digest(group):
                raise ValueError("wave input changed")
            continue
        check(model)
        import random
        shuffled = [c for b in group for c in b]
        random.Random(9326 + start).shuffle(shuffled)
        again = [shuffled[i:i+batch_size] for i in range(0,len(shuffled),batch_size)]
        requests = group + again
        raw_path = folder / f"raw-{start:06d}.json"
        prior = json.loads(raw_path.read_text()) if raw_path.exists() else None
        if prior and any('Refusal' in str(r.get('error')) for r in prior):
            if len(prior) != len(requests):
                raise ValueError('saved raw wave length mismatch')
            result = list(prior)
            # Preserve refusals, but retry transient errors in OTHER requests of the same wave.
            retry_indices = [i for i,r in enumerate(prior) if r.get('error') and 'Refusal' not in str(r['error'])]
            if retry_indices:
                retries = fn.map([{'cases': [Case(**c) for c in presented(requests[i])]} for i in retry_indices], threads=2).to_dicts()
                for i,row in zip(retry_indices,retries):
                    result[i] = row
                write_json(raw_path,result)
        else:
            result = fn.map([{"cases": [Case(**c) for c in presented(b)]} for b in requests], threads=2).to_dicts()
            write_json(raw_path, result)
        passes = [[], []]
        for i,(batch,row) in enumerate(zip(requests,result)):
            if row.get("error"):
                if 'Refusal' in str(row['error']):
                    passes[int(i >= len(group))] += [dict(id=c['id'],label='insufficient_evidence',relation='ambiguous',
                        evidence='',reason='No classification: provider refused.',evidence_verified=False,provider_refused=True) for c in batch]
                    continue
                provider_failure(row["error"], f"{model} labeling wave {start}")
            passes[int(i >= len(group))] += validate_response(batch,row["pred_result"])
        repeat = {v['id']:v for v in passes[1]}
        verdicts = [{**v, 'order_stable':v['label']==repeat[v['id']]['label'], 'repeat':repeat[v['id']]} for v in passes[0]]
        write_json(path,{"input_sha256":digest(group),"answers":verdicts})
        print(model,split,min((start+len(group))*batch_size,len(rows)),"/",len(rows),flush=True)
    write_json(folder / "DONE.json",contract)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("model",choices=["sol","opus"])
    ap.add_argument("split",choices=["train","validation"])
    ap.add_argument("--batch-size",type=int,choices=[5,10,25],default=25)
    ap.add_argument('--limit-pairs',type=int,default=None)
    args=ap.parse_args()
    if args.limit_pairs is not None and args.limit_pairs % (2*args.batch_size):
        ap.error('limit must be a whole two-request wave')
    run(args.model,args.split,args.batch_size,args.limit_pairs)

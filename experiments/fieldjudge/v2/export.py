"""Join independent judgments without pretending that a two-vote panel is truth.

An agreeing, non-abstaining pair with verified evidence is provisional training
supervision. Disagreements, abstentions and invalid evidence go to review.jsonl;
raw votes/reasons are retained. No majority with two votes, and no synthetic
probability 0.5 passed off as calibrated uncertainty.
"""
import json
from collections import Counter
from common import ROOT, write_json

def run():
    summary={}
    outdir=ROOT/"export"
    outdir.mkdir(exist_ok=True)
    for split in ("train","validation"):
        rows=[json.loads(l) for l in open(ROOT/f"pairs-{split}.jsonl")]
        expected={r['id'] for r in rows}
        judgments={}
        for model in ("sol","opus"):
            folder=ROOT/"judgments"/split/model
            if not (folder/"DONE.json").exists():
                raise ValueError(f"{split}/{model} incomplete")
            answers={}
            for path in sorted(folder.glob("wave-*.json")):
                for v in json.loads(path.read_text())["answers"]:
                    if v['id'] in answers:
                        raise ValueError("duplicate judgment")
                    answers[v['id']]=v
            if set(answers)!=expected:
                raise ValueError("judgment/input mismatch")
            judgments[model]=answers
        counts=Counter()
        records,review=[],[]
        for row in rows:
            votes={m:j[row['id']] for m,j in judgments.items()}
            labels={v['label'] for v in votes.values()}
            reason=("order_instability" if not all(v.get('order_stable') for v in votes.values()) else
                    "abstention" if "insufficient_evidence" in labels else "disagreement" if len(labels)!=1
                    else "unverified_evidence" if not all(v.get('evidence_verified') and v.get('repeat',{}).get('evidence_verified') for v in votes.values()) else "agreement")
            record={**row,"teachers":votes,"supervision_status":reason}
            counts[reason]+=1
            if reason=="agreement":
                record['y']=int(next(iter(labels))=="on_topic")
                record['label_source']='sol_opus_agreement_unreviewed'
                records.append(record)
            else:
                review.append(record)
        for suffix,data in (("provisional",records),("review",review)):
            path=outdir/f"{split}-{suffix}.jsonl"
            tmp=path.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
            tmp.replace(path)
        summary[split]=dict(counts)
    write_json(outdir/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    run()

"""Test a stricter dual-teacher + order-check protocol after raw batching failed.

Predeclared gate: at least 60/100 paired cases yield two agreeing, stable,
non-abstaining, evidence-backed labels; at least 95% of those also agree with
BOTH singleton answers. This is a consistency check, not an accuracy claim.
Every rejected case remains in the review pool. The held-out benchmark must not
be filtered to agreement cases when assessing model quality.
"""
import json
from common import ROOT, teachers, write_json, validate_response
from pilot import select, PILOT

def phase(model, name):
    out={}
    for path in sorted((PILOT/model).glob(f'{name}-*.json')):
        if path.stem.endswith(('-raw','-failure')):
            continue
        content=json.loads(path.read_text())
        for v in content['answers']:
            out[v['id']]=v
    return out

def run():
    cases=select()[:100]
    judgments={m:{s:phase(m,s) for s in ('batch25','single','reorder25')} for m in teachers.ROLES}
    if any(c['id'] not in judgments[m][s] for c in cases for m in judgments for s in judgments[m]):
        raise ValueError('incomplete paired pilot')
    accepted,consistent,review=[],[],[]
    for c in cases:
        four=[judgments[m][s][c['id']] for m in judgments for s in ('batch25','reorder25')]
        labels={v['label'] for v in four}
        # Revalidate earlier checkpoints as evidence verification evolved during the pilot.
        verified=[]
        for v in four:
            from common import validate
            verified.append(validate([c],[{k:x for k,x in v.items() if k!='evidence_verified'}],strict_evidence=False)[0])
        ok=len(labels)==1 and 'insufficient_evidence' not in labels and all(v['evidence_verified'] for v in verified)
        if not ok:
            review.append(c['id']);continue
        accepted.append(c['id'])
        label=next(iter(labels))
        if all(judgments[m]['single'][c['id']]['label']==label for m in judgments):
            consistent.append(c['id'])
    report={'protocol':'dual-teacher-order-check-v1','n':len(cases),'accepted':len(accepted),
            'consistent_with_both_singletons':len(consistent),'agreement_on_accepted':len(consistent)/max(1,len(accepted)),
            'passed':len(accepted)>=60 and len(consistent)/max(1,len(accepted))>=.95,
            'review_ids':review,'inconsistent_ids':sorted(set(accepted)-set(consistent)),
            'note':'Small development pilot. Measures stability, not correctness; batch-order checks double label-generation calls.'}
    write_json(ROOT/'pilot-defined'/'acceptance.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()

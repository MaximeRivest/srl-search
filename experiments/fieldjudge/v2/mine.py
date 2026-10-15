"""Select difficult pairs using the OLD 0.8B/2B models, without seeing teacher answers.

50,000 candidate pairs -> 25,000 selected: half near the 2B decision boundary,
half where the two models disagree most. Coverage and cross-scope examples stay
in the mix. This is uncertainty/disagreement mining, not proof of a wrong label.
Each GPU chunk waits for live-map work and free memory. Does not stop services.
"""
from __future__ import annotations
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search import field as F, judges as J

ROOT = Path(os.environ.get("FIELDJUDGE_V2", "/mnt/fast/scholarsreadinglist/fieldjudge/v2"))
VPY = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python"
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'

def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False))
    os.replace(temp, path)

def gate():
    while True:
        plan = F.FieldStore().plan()
        running = plan["running"]
        imminent = any(j["st"].get("kind") == "full" and j["start"] - plan["now"] < 300 for j in plan["waiting"][:1])
        if not (running and running["st"].get("kind") == "full") and not imminent and not J.RESERVED.exists() and J.gpu_free_mb() >= 10000:
            return
        print("Waiting for live maps/GPU memory", flush=True)
        time.sleep(30)

def run():
    manifest=json.loads((ROOT/'topics.json').read_text())
    families = [json.loads(p.read_text()) for p in sorted((ROOT / "families").glob("*.json"))]
    expected={f['id'] for f in manifest['families']}
    if not expected or {f['family_id'] for f in families} != expected:
        raise ValueError("family shards do not match the screened topic manifest")
    expected_pairs = 200 * len(expected)
    rows = [r for f in families for r in f["mining_pool"]]
    work = ROOT / "mining"
    work.mkdir(exist_ok=True)
    predictions = {}
    for name in ("m08", "m2b"):
        pred = {}
        for start in range(0, len(rows), 2000):
            chunk = rows[start:start+2000]
            destination = work / f"{name}-{start:06d}.json"
            if destination.exists():
                stored = json.loads(destination.read_text())
                if stored["ids"] != [r["id"] for r in chunk]:
                    raise ValueError("mining inputs changed")
            else:
                gate()
                source = work / f"{name}-{start:06d}.jsonl"
                source.write_text("".join(json.dumps({"text": r["text"], "system": J.instruction(r["topic"],r["definition"]) + ANSWER}) + "\n" for r in chunk))
                scored = work / f"{name}-{start:06d}-scores.json"
                J.run([VPY, str(Path(J.__file__).with_name("vllm_decider.py")), str(ROOT.parent/name/"merged"),
                       str(source), str(scored), J.GPU], work/"log.txt")
                scores = json.loads(scored.read_text())
                if len(scores) != len(chunk) or not all(np.isfinite(v) and 0 <= v <= 1 for v in scores):
                    raise ValueError("invalid mining scores")
                stored = {"ids": [r["id"] for r in chunk], "scores": scores}
                write(destination, stored)
            pred.update(zip(stored["ids"], stored["scores"]))
            print("MINING", name, start+len(chunk), "/", len(rows), flush=True)
        predictions[name] = pred
    all_rows = []
    for f in families:
        candidates = f["mining_pool"]
        boundary = sorted(candidates, key=lambda r: (abs(predictions['m2b'][r['id']]-.5), r['id']))[:25]
        used = {r["id"] for r in boundary}
        disagree = sorted((r for r in candidates if r["id"] not in used),
                          key=lambda r: (-abs(predictions['m08'][r['id']]-predictions['m2b'][r['id']]), r['id']))[:25]
        selected = [{**r, "source": source, "mining_scores": {n:p[r['id']] for n,p in predictions.items()}}
                    for group, source in ((boundary,"boundary"),(disagree,"model_disagreement")) for r in group]
        all_rows += f["coverage"] + f["cross_scope"] + selected
    if len(all_rows) != expected_pairs or len({r["id"] for r in all_rows}) != expected_pairs:
        raise ValueError("target count/unique pair mismatch")
    # Only export after cross-split identity checks, independent of collection's own guard.
    owners = {}
    for r in all_rows:
        for identity in (f"work:{r['work_id']}", f"text:{r['text_group']}"):
            if identity in owners and owners[identity] != r["split"]:
                raise ValueError("train/validation paper leakage")
            owners[identity] = r["split"]
    random.Random(4526).shuffle(all_rows)
    for split in ("train", "validation"):
        subset = [r for r in all_rows if r["split"] == split]
        path = ROOT / f"pairs-{split}.jsonl"
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in subset))
        os.replace(temp,path)
    write(ROOT/'dataset-status.json', {'pairs':len(all_rows),'target_pairs':100000,'deferred':manifest.get('deferred',[]),
                                      'complete':bool(manifest.get('complete',True)),'training_approved':False})
    print("EXPORTED",len(all_rows),"pairs of target 100000; deferred subjects:",len(manifest.get('deferred',[])),flush=True)

if __name__ == "__main__":
    run()

"""Batching acceptance test. Existing TRAINING papers only, never the test benchmark.

200 cases in eight batches of 25; 100 of the same cases as singletons; the same
100 in shuffled batches. Same prompt/schema and reasoning in all conditions.
This measures batching consistency, not correctness. No full labeling launch if
IDs/evidence fail or batching noticeably changes labels. Calls use FunctAI's
persistent reply cache; checkpoint files are exported results, not another cache.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from common import ROOT, OLD, Case, classify, settings, validate_response, presented, write_json, digest
from allowance import check

PILOT = ROOT / "pilot-defined"

def select():
    p = PILOT / "cases.json"
    if p.exists():
        return json.loads(p.read_text())
    topics = {t["id"]: t for t in json.loads((OLD / "topics.json").read_text())["topics"]}
    labels = {(r["topic_id"], r["id"]): r["y"] for r in map(json.loads, open(OLD / "labels.jsonl"))}
    pools = {}
    for r in map(json.loads, open(OLD / "samples.jsonl")):
        definition = topics[r['topic_id']].get('definition')
        if r["topic_id"] in {113, 117, 137} or not definition or str(definition).lower() in ('null', 'none'):
            continue
        pools.setdefault(r["topic_id"], []).append(r)
    # Eight fields, not eight variants of a single topic.
    rng = random.Random(2108)
    ts = sorted(pools)
    rng.shuffle(ts)
    fields, chosen = set(), []
    for t in ts:
        if topics[t]["field"] not in fields:
            chosen.append(t)
            fields.add(topics[t]["field"])
        if len(chosen) == 8:
            break
    cases = []
    for t in chosen:
        candidates = pools[t]
        rng.shuffle(candidates)
        # New production scopes all have definitions; include ~20% title-only cases.
        strata = {}
        for r in candidates:
            s = (r["kind"] == "cited_title_only", labels[(t, r["id"])])
            strata.setdefault(s, []).append(r)
        picked = []
        while len(picked) < 25:
            for s in sorted(strata):
                title_count = sum(r['kind'] == 'cited_title_only' for r in picked)
                if strata[s] and len(picked) < 25 and (not s[0] or title_count < 5):
                    picked.append(strata[s].pop())
        for r in picked:
            cases.append({"id": digest([t, r["id"], r["text"]])[:24], "topic": topics[t]["topic"],
                          "definition": topics[t]["definition"], "text": r["text"]})
    rng.shuffle(cases)
    write_json(p, cases)
    return cases

def calls(model, phase, batches):
    fn = classify.using(**{**settings(model), "replicate": 1 if phase == "single-repeat" else 0})
    output = []
    for start in range(0, len(batches), 2):
        path = PILOT / model / f"{phase}-{start:04d}.json"
        group = batches[start:start + 2]
        fingerprint = digest(group)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["inputs_sha256"] != fingerprint:
                raise ValueError("pilot input changed; use a new pilot version")
            output.extend(saved["answers"])
            continue
        raw_path = PILOT / model / f"{phase}-{start:04d}-raw.json"
        prior = json.loads(raw_path.read_text()) if raw_path.exists() else None
        if prior and any('Refusal' in str(r.get('error')) for r in prior):
            run = prior  # do not repeat refused requests
        else:
            check(model)
            run = fn.map([{"cases": [Case(**c) for c in presented(batch)]} for batch in group], threads=2).to_dicts()
            write_json(raw_path, run)
        answers = []
        for batch, row in zip(group, run):
            if row.get("error"):
                write_json(PILOT / model / f"{phase}-{start:04d}-failure.json", row)
                if 'Refusal' in str(row['error']):
                    answers.extend(dict(id=c['id'],label='insufficient_evidence',relation='ambiguous',evidence='',
                                        reason='No classification: provider refused.',evidence_verified=False,provider_refused=True) for c in batch)
                    continue
                raise RuntimeError(f"{model}/{phase}: provider/parser error; inspect saved failure")
            answers.extend(validate_response(batch, row["pred_result"]))
        write_json(path, {"inputs_sha256": fingerprint, "answers": answers})
        output.extend(answers)
        print(model, phase, start + len(group), "/", len(batches), flush=True)
    return {a["id"]: a for a in output}

def run(model, smoke=False, batch_size=25, repeat=False):
    cases = select()
    if smoke:
        calls(model, "smoke", [cases[:2]])
        return
    subset = cases[:100]
    if repeat:
        first = calls(model, "single", [[c] for c in subset])
        again = calls(model, "single-repeat", [[c] for c in subset])
        write_json(PILOT / f"{model}-repeat.json", {"n":len(subset), "same":sum(first[c['id']]['label'] == again[c['id']]['label'] for c in subset)})
        return
    batch = calls(model, f"batch{batch_size}", [cases[i:i+batch_size] for i in range(0, len(cases), batch_size)])
    single = calls(model, "single", [[c] for c in subset])
    shuffled = list(subset)
    random.Random(17).shuffle(shuffled)
    reorder = calls(model, f"reorder{batch_size}", [shuffled[i:i+batch_size] for i in range(0, len(shuffled), batch_size)])
    same = sum(batch[c["id"]]["label"] == single[c["id"]]["label"] for c in subset)
    stable = sum(batch[c["id"]]["label"] == reorder[c["id"]]["label"] for c in subset)
    disagreements = [{"id": c["id"], "topic": c["topic"], "text": c["text"], "batch": batch[c["id"]],
                      "single": single[c["id"]], "reorder": reorder[c["id"]]} for c in subset
                     if len({batch[c["id"]]["label"], single[c["id"]]["label"], reorder[c["id"]]["label"]}) > 1]
    result = {"model": model, "batch_size": batch_size, "paired_n": len(subset), "same_batch_single": same,
              "unverified_evidence": sum(not v.get("evidence_verified", bool(v["evidence"])) for v in batch.values() if v['label'] != 'insufficient_evidence'),
              "same_reordered": stable, "labels": dict(Counter(a["label"] for a in batch.values())),
              "passed": same >= 95 and stable >= 95, "disagreements": disagreements,
              "note": "95/100 is an operational pilot gate, not a population accuracy estimate."}
    write_json(PILOT / f"{model}-batch{batch_size}-report.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "disagreements"}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["sol", "opus"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, choices=[5, 10, 25], default=25)
    ap.add_argument("--repeat-single", action="store_true")
    args = ap.parse_args()
    run(args.model, args.smoke, args.batch_size, args.repeat_single)

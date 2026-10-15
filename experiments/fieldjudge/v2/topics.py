"""Create and screen topic families through subscription connections only.

Generation is separate from labeling. Both teachers independently flag overlap
with reserved evaluation topics. Rejected families and every raw generation are
kept. Once topics.json exists it is immutable; a changed recipe needs a new run.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from pydantic import BaseModel, Field
from functai import ai, _ai
from common import ROOT, OLD, settings, write_json, digest
from allowance import check
from control import provider_failure

RECIPE = json.loads(Path(__file__).with_name("recipe.json").read_text())

class Scope(BaseModel):
    role: str = Field(description="anchor, sibling, broader, narrower, or alternative; one of each per family")
    topic: str
    definition: str = Field(description="One sentence specifying scope and any ambiguous sense; do not assert paper labels.")

class ProposedScope(Scope):
    family: int = Field(description="Family number, 1 through 24; five scopes share each number.")

@ai
def propose(field: str, reserved: list[dict], already_used: list[str], variant: int) -> list[ProposedScope]:
    """Propose 24 diverse research-topic families within the scientific field.
    Each contains five distinct scopes: a mid-sized anchor topic, a neighboring
    sibling, a moderately broader research topic (not an entire discipline), a
    narrower topic, and an alternative application or word sense. Use natural
    search queries, acronyms where appropriate, with precise one-sentence scope
    definitions. Include classics-heavy and recent work, methods and applications,
    theoretical and empirical work, and varied organisms or materials. Avoid
    duplicate anchor topics in already_used and near-duplicates within this output.
    Do not propose reserved topics, synonyms, their subtopics, or a broad topic
    whose central literature substantially includes a reserved topic. Avoid
    assigning five aliases to the same topic: the boundaries must genuinely differ.
    Return 120 scope records: 24 family numbers, each with all five roles once. The variant number
    asks for a different selection, not a different classification rule.
    """
    return _ai

class Screening(BaseModel):
    id: str
    overlap: bool
    reserved_topic: str | None
    reason: str

@ai
def screen(families: list[dict], reserved: list[dict]) -> list[Screening]:
    """Check each family's five scopes for leakage into reserved evaluation topics.
    Flag overlap if a scope is equivalent to, a close subtopic of, or a broad topic
    substantially encompassing a reserved topic. Ordinary adjacent research is not
    itself leakage. Read definitions, not just keywords (electrical and neural
    transformers differ). Return exactly one item per input id in input order.
    """
    return _ai

def normalize(s):
    return " ".join(s.casefold().replace("-", " ").split())

def generate(defer_blocked_fields=False, available_only=False):
    target = ROOT / ('topics-available.json' if available_only else 'topics.json')
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["recipe_sha256"] != digest(RECIPE):
            raise ValueError("recipe changed under frozen topics")
        if saved.get('deferred') and not defer_blocked_fields:
            raise RuntimeError('frozen manifest has deferred subjects; explicit --defer-blocked-fields required')
        print("Topics already frozen:", len(saved["families"]), 'deferred subjects:', len(saved.get('deferred', [])))
        return
    reserved = RECIPE["new_holdouts"] + RECIPE["development_holdouts"]
    fields = sorted({t["field"] for t in json.loads((OLD / "topics.json").read_text())["topics"]})
    accepted, seen, rejects = [], set(), []
    deferred = []
    quota = {field: 500 // len(fields) + (i < 500 % len(fields)) for i, field in enumerate(fields)}
    for field in fields:
        selected = []
        for variant in range(4):
            if len(selected) >= quota[field]:
                break
            already = [f["scopes"][0]["topic"] for f in selected]
            args = dict(field=field, reserved=reserved, already_used=already, variant=variant)
            key = digest(args)[:20]
            path = ROOT / "topic-generation" / f"{key}.json"
            if path.exists():
                proposals = json.loads(path.read_text())["families"]
            else:
                if available_only:
                    break  # export completed screening only; do not wait or make any model calls
                check("sol")
                result = propose.using(**settings("sol")).map([args], threads=1).to_dicts()[0]
                if result.get("error"):
                    write_json(path.with_suffix(".failure.json"), result)
                    provider_failure(result["error"], "topic generation")
                grouped = {}
                for value in result["pred_result"]:
                    scope = ProposedScope.model_validate(value).model_dump()
                    family_number = scope.pop("family")
                    grouped.setdefault(family_number, []).append(scope)
                proposals = [{"scopes": scopes} for scopes in grouped.values()]
                write_json(path, {"args": args, "families": proposals})
            candidates = []
            for family in proposals:
                scopes = family["scopes"]
                if Counter(s["role"] for s in scopes) != Counter(["anchor", "sibling", "broader", "narrower", "alternative"]):
                    rejects.append({"family": family, "reason": "invalid roles"})
                    continue
                family_id = digest([field, scopes])[:20]
                texts = [normalize(s["topic"]) for s in scopes]
                if len(set(texts)) != 5 or any(t in seen for t in texts):
                    rejects.append({"family": family, "reason": "duplicate scope"})
                    continue
                joined = normalize(" ".join(s["topic"] + " " + s["definition"] for s in scopes))
                if any(normalize(term) in joined for r in reserved for term in r["exclude"]):
                    rejects.append({"family": family, "reason": "reserved keyword"})
                    continue
                candidates.append({"id": family_id, "field": field, "scopes": scopes})
            if not candidates:
                continue
            flagged = set()
            for model in ("sol", "opus"):
                screen_path = ROOT / 'topic-generation' / f'{key}-{model}-screen.json'
                if screen_path.exists():
                    verdicts = json.loads(screen_path.read_text())
                    if [v['id'] for v in verdicts] != [c['id'] for c in candidates]:
                        raise ValueError('screen checkpoint/input mismatch')
                    flagged.update(v['id'] for v in verdicts if v['overlap'])
                    continue
                failure = ROOT / "topic-generation" / f"{key}-{model}-failure.json"
                if failure.exists() and 'Refusal' in str(json.loads(failure.read_text()).get('error')):
                    flagged.update(c['id'] for c in candidates)
                    break  # no repeat of a refused screening request; retain and reject this proposal batch
                if available_only:
                    flagged.update(c['id'] for c in candidates)
                    break
                check(model)
                result = screen.using(**settings(model)).map([dict(families=candidates, reserved=reserved)], threads=1).to_dicts()[0]
                if result.get("error"):
                    write_json(failure, result)
                    if 'Refusal' in str(result['error']):
                        flagged.update(c['id'] for c in candidates)
                        break
                    provider_failure(result["error"], "topic screening")
                verdicts = [Screening.model_validate(v).model_dump() for v in result["pred_result"]]
                if [v["id"] for v in verdicts] != [c["id"] for c in candidates]:
                    raise ValueError("screen IDs mismatch")
                write_json(ROOT / "topic-generation" / f"{key}-{model}-screen.json", verdicts)
                flagged.update(v["id"] for v in verdicts if v["overlap"])
            for family in candidates:
                texts = [normalize(s["topic"]) for s in family["scopes"]]
                if family["id"] in flagged or any(t in seen for t in texts):
                    rejects.append({"family": family, "reason": "overlap/duplicate"})
                    continue
                if len(selected) >= quota[field]:
                    continue
                seen.update(texts)
                for scope in family["scopes"]:
                    scope["id"] = digest([family["id"], scope["role"]])[:20]
                selected.append(family)
            print(field, len(selected), "/", quota[field], flush=True)
        if len(selected) < quota[field]:
            shortfall = {'field': field, 'selected': selected, 'quota': quota[field], 'missing': quota[field]-len(selected)}
            write_json(ROOT / 'topic-generation' / f'shortfall-{digest(field)[:12]}.json', shortfall)
            if not defer_blocked_fields:
                raise RuntimeError("Insufficient distinct families: inspect shortfall, do not silently lower coverage")
            deferred.append({'field': field, 'accepted': len(selected), 'quota': quota[field], 'missing': quota[field]-len(selected)})
            print('DEFERRED FOR REVIEW',field,'missing',quota[field]-len(selected),'families; other subjects continue',flush=True)
        accepted.extend(selected)
    # Group siblings/parents/alternative meanings before splitting; never random-split paper pairs.
    random.Random(3026).shuffle(accepted)
    for i, family in enumerate(accepted):
        family["split"] = "validation" if i < 50 else "train"
    write_json(target, {"recipe_sha256": digest(RECIPE), "families": accepted, "rejected": rejects,
                        'deferred': deferred, 'target_families': 500, 'complete': not deferred, 'available_only': available_only})
    print("FROZEN", len(accepted), "families", flush=True)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--defer-blocked-fields',action='store_true',help='Continue other subjects while preserving explicit coverage shortfalls; no training approval.')
    ap.add_argument('--available-only',action='store_true',help='Export already-screened families, without model calls.')
    args=ap.parse_args()
    generate(args.defer_blocked_fields or args.available_only, args.available_only)

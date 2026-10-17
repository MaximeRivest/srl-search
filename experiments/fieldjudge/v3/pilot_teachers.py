"""Can the teachers do the task? Sol and Opus on the 200 expert-classified articles (jevarticle, Science-Metrix 176
subfields), 20 papers a request, the round-3 teacher prompt. Scored like jevarticle/scripts/score.py.
The benchmark stays out of student training; this only validates the teachers and their shares (calibration).

    ~/Projects/scholarsreadinglist/sciencemadereadable/.venv/bin/python pilot_teachers.py
"""
import csv, collections, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import teacher as T

JA = Path.home() / "Projects/jevarticle"
tax = list(csv.DictReader(open(JA / "data/taxonomy.csv")))
arts = list(csv.DictReader(open(JA / "data/articles_resolved.csv")))
codes = T.codes_for(len(tax))
options = [{"code": c, "name": t["subfield"], "about": f"{t['field']} ({t['domain']})"} for c, t in zip(codes, tax)]
name_of = {o["code"]: o["name"] for o in options}
field_of = {t["subfield"]: t["field"] for t in tax}
SCHEME = "Science-Metrix classification of science (176 subfields within 22 fields and 6 domains)"
QUESTION = "Which subfield best describes the research content of this paper?"


def text(a):
    s = f"Title: {a['title']}\nVenue: {a['journal']}\nYear: {a['year']}"
    return s + (f"\nAbstract: {a['abstract'][:1500]}" if a["abstract"] else "\n(no abstract)")


def label(model):
    out = T.ROOT / f"pilot-{model}.json"
    if out.exists():
        return json.loads(out.read_text())
    fn = T.classify_batch.using(**T.settings(model))
    batches = [arts[i:i + 20] for i in range(0, len(arts), 20)]
    reqs = [{"scheme": SCHEME, "question": QUESTION, "options": options,
             "papers": [{"id": f"p{j + 1:02d}", "text": text(a)} for j, a in enumerate(b)]} for b in batches]
    t = time.time()
    rows = fn.map(reqs, threads=4).to_dicts()
    res, errors = {}, 0
    for b, req, row in zip(batches, reqs, rows):
        try:
            got = T.validate(req["papers"], set(codes), row["pred_result"])
        except Exception as e:                       # noqa: BLE001
            errors += 1
            print(model, "invalid batch:", str(e)[:200], str(row.get("error"))[:200], flush=True)
            continue
        for a, g in zip(b, got):
            res[a["scopus_id"]] = {name_of[c]: p for c, p in g["dist"].items()}
    r = {"seconds": round(time.time() - t), "errors": errors, "answers": res}
    out.write_text(json.dumps(r))
    return r


votes = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(JA / "data/gold_standard_long.csv")):
    votes[r["scopus_id"]][r["evaluator"]].append(r["subfield"])


def modes(xs):
    c = collections.Counter(xs); m = max(c.values())
    return {s for s, n in c.items() if n == m}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.0f}% [{100 * (c - h):.0f}-{100 * (c + h):.0f}]"


with ThreadPoolExecutor(2) as ex:
    results = dict(zip(("sol", "opus"), ex.map(label, ("sol", "opus"))))
for model, r in results.items():
    A = r["answers"]
    by = collections.defaultdict(list); fhit = []; share_on_expert = []; conf_hit = []
    for a in arts:
        d = A.get(a["scopus_id"])
        if d is None:
            continue
        firsts = [v[0] for v in votes[a["scopus_id"]].values()]
        top = max(d, key=d.get)
        by[a["subset"]].append(top in modes(firsts))
        fhit.append(field_of[top] in modes([field_of[x] for x in firsts]))
        share_on_expert.append(sum(d.get(x, 0) for x in firsts) / len(firsts))      # expected share of votes matched
        conf_hit.append((d[top], top in modes(firsts)))
    conf_hit.sort()
    third = len(conf_hit) // 3
    print(f"{model}: disciplinary {wilson(sum(by['disciplinary']), len(by['disciplinary']))}, multidisciplinary "
          f"{wilson(sum(by['multidisciplinary']), len(by['multidisciplinary']))}, field {100 * sum(fhit) / len(fhit):.0f}%, "
          f"mean share given to the experts' picks {sum(share_on_expert) / len(share_on_expert):.2f}, "
          f"hit low/high-confidence third {100 * sum(h for _, h in conf_hit[:third]) / third:.0f}%/"
          f"{100 * sum(h for _, h in conf_hit[-third:]) / third:.0f}%, mean top share "
          f"{sum(c for c, _ in conf_hit) / len(conf_hit):.2f}, {r['seconds']} s, invalid batches {r['errors']}")

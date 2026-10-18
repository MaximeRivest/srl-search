"""Teacher work units: one option list (a scheme level, a set of siblings, or a subset) + 20 papers.

Shapes:  level     every option of one level (<= MAX_OPTIONS)
         siblings  the children of one parent (the step down a hierarchy), papers from inside the parent
         subset    2-40 options of one level, papers near some of them (best among the offered)
"None of these" is added to NONE_SHARE of units, which then also carry 3-5 papers from elsewhere in science.
Papers: our search API (title, venue, year, abstract), never the record's own OpenAlex topic/field (that would leak one
scheme's answers). 15% shown as title only. Excluded: every test paper (round-2 benchmarks, frag/LLM tests,
jevarticle). A paper goes to one teacher only (teachers A and B see different papers); used at most 3 times.
Split: train units, and eval units (held-out schemes, plus held-out papers on training schemes; both teachers label
eval units, which also measures how far the two teachers agree).

    ../../../.venv/bin/python build_queue.py      -> v3/queue/{train,eval}/*.json
"""
import glob, hashlib, json, random, re, sys, time, urllib.parse, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the fieldjudge folder: teachers.py
import teachers  # noqa: E402
A, B = teachers.ROLES                         # teacher roles (models: the private settings)

ROOT = Path("/mnt/fast/scholarsreadinglist/fieldjudge/v3")
SCHEMES = ROOT / "schemes"
Q = ROOT / "queue"
API = "http://127.0.0.1:8811"                 # the dev copy: never load the public API
MAX_OPTIONS, PAPERS, NONE_SHARE, TITLE_ONLY = 300, 20, 0.35, 0.15
NONE = "None of these (outside the options listed)"
rng = random.Random(17)

BUDGET = {"openalex": 350, "acm-ccs": 200, "mesh": 200, "oecd-ford": 100, "arxiv": 150, "msc2020": 150, "anzsrc-for": 200}
TEACHER = {"openalex": 0, "acm-ccs": 0, "mesh": 0, "oecd-ford": 0, "arxiv": 1, "msc2020": 1,   # teacher A (0) or B (1)
           "anzsrc-for": 1}
MAPS_UNITS, INVENTED_UNITS = 300, 850
EVAL = {"jel": 30, "heldout-invented": 40, "heldout-papers": 60}
QUESTIONS = ["Which {level} best describes the research content of this paper?",
             "Which {level} does this paper belong to?",
             "In this classification, which {level} fits this paper best?",
             "Pick the {level} that best matches what this paper studies."]


def titlekey(t):
    """Round 2's key (fast.py seed / text_group): the first line of "Title: ...", letters and digits only."""
    first = ((t or "").splitlines() or [""])[0]
    if not first.startswith("Title: "):
        first = "Title: " + first
    return hashlib.sha256(re.sub(r"[^\w]", "", first.casefold()).encode()).hexdigest()


def reserved():
    D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
    ids, titles = set(), set()
    for r in map(json.loads, open(D / "tests.jsonl")):
        ids.add(int(r["id"])); titles.add(titlekey(r["text"]))
    for folder in ("benchmark", "benchmark-titles"):
        for p in (D / "v2" / folder).glob("*.json"):
            for r in json.loads(p.read_text())["pairs"]:
                ids.add(int(r["work_id"])); titles.add(r["text_group"])
    import csv
    for a in csv.DictReader(open(Path.home() / "Projects/jevarticle/data/articles_resolved.csv")):
        titles.add(titlekey(a["title"]))
    return ids, titles


RES_IDS, RES_TITLES = reserved()
cache = {}


def search(q, k=40):
    if q in cache:
        return cache[q]
    u = f"{API}/search?" + urllib.parse.urlencode({"q": q[:300], "k": k, "abstracts": "true"})
    for attempt in range(8):
        try:
            rows = json.load(urllib.request.urlopen(u, timeout=120))["results"]
            break
        except Exception as e:                              # noqa: BLE001
            print("search failed, retrying", q[:60], e, file=sys.stderr); time.sleep(30)
    else:
        raise RuntimeError("search API down")
    out = []
    for r in rows:
        w = int(r["id"][1:])
        ab = r.get("abstract") or ""
        if ab.strip().casefold().startswith("abstract not provided"):
            ab = ""
        if w in RES_IDS or titlekey(r.get("title")) in RES_TITLES or (r.get("junk") or 0) > 0.5 or len(r.get("title") or "") < 15:
            continue
        if r.get("type") in ("erratum", "paratext", "retraction", "editorial", "letter", "peer-review"):
            continue
        out.append({"work_id": w, "title": r["title"], "venue": r.get("venue"), "year": r.get("year"), "abstract": ab,
                    "title_key": titlekey(r["title"])})
    cache[q] = out
    return out


def text(p, title_only):
    s = f"Title: {re.sub(r'<[^>]+>', '', p['title']).strip()}"
    v = (p.get("venue") or "").strip()
    if v and v != "None" and not re.match(r"(https?:|www\.|doi:)", v, re.I) and len(v) <= 120:
        s += f"\nVenue: {v}"
    if p.get("year"):
        s += f"\nYear: {p['year']}"
    ab = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", p.get("abstract") or "")).strip()
    return s + (f"\nAbstract: {ab[:1500]}" if ab and not title_only else "\n(no abstract)")


owner, uses, seen_title = {}, Counter(), {}
eval_papers = set()


def take(pool, n, teacher, split):
    """n papers from pool (shuffled) that this teacher may use."""
    rng.shuffle(pool)
    got, keys = [], set()
    for p in pool:
        w = p["work_id"]
        if p["title_key"] in keys:
            continue
        if split == "train" and (owner.get(w, teacher) != teacher or uses[w] >= 3 or w in eval_papers):
            continue
        if split == "eval" and (owner.get(w) is not None or w in eval_papers and uses[w] >= 2):
            continue
        if seen_title.get(p["title_key"], w) != w:             # another record with the same title
            continue
        got.append(p); keys.add(p["title_key"])
        if len(got) == n:
            break
    for p in got:
        w = p["work_id"]; uses[w] += 1; seen_title[p["title_key"]] = w
        if split == "train":
            owner[w] = teacher
        else:
            eval_papers.add(w)
    return got


def load(sid):
    return json.loads((SCHEMES / f"{sid}.json").read_text())


def unit(s, teacher, split, uid):
    nodes = s["nodes"]
    by_level = defaultdict(list)
    for n in nodes:
        by_level[n["level"]].append(n)
    kids = defaultdict(list)
    for n in nodes:
        if n["parent"]:
            kids[n["parent"]].append(n)
    name_of = {n["id"]: n["name"] for n in nodes}
    shapes = []
    for lv, ns in by_level.items():
        if 2 <= len(ns) <= MAX_OPTIONS:
            shapes += [("level", lv)] * 2
        if len(ns) >= 4:
            shapes.append(("subset", lv))
    if any(len(v) >= 2 for v in kids.values()):
        shapes += [("siblings", None)] * 3
    kind, lv = rng.choice(shapes)
    if kind == "level":
        opts = list(by_level[lv]); anchors = rng.sample(opts, min(8, len(opts)))
    elif kind == "subset":
        k = min(len(by_level[lv]), rng.choice([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40]))
        opts = rng.sample(by_level[lv], k); anchors = rng.sample(opts, min(6, k))
    else:
        parent = rng.choice([p for p, v in kids.items() if len(v) >= 2])
        opts = list(kids[parent]); lv = opts[0]["level"]; anchors = rng.sample(opts, min(6, len(opts)))
    level_name = s["levels"][min(lv, len(s["levels"]) - 1)]
    with_none = rng.random() < NONE_SHARE
    area = s.get("area") or s["name"]
    queries = []
    for a in anchors:
        ctx = name_of.get(a["parent"]) if a.get("parent") else None
        queries.append(" ".join(x for x in (a["name"], ctx if kind != "siblings" or rng.random() < 0.5 else None,
                                             area if s["id"].startswith(("inv-", "map-")) else None) if x))
    with ThreadPoolExecutor(4) as ex:
        pool = [p for got in ex.map(search, queries) for p in got]
    n_out = rng.randint(3, 5) if with_none else 0
    papers = take(pool, PAPERS - n_out, teacher, split)
    if n_out:
        far = rng.sample([n for n in load("openalex")["nodes"] if n["level"] == 3], 3)
        papers += take([p for f in far for p in search(f["name"], 20)], n_out, teacher, split)
    if len(papers) < 8:
        return None
    rng.shuffle(papers)
    options = [{"node": o["id"], "name": o["name"],
                "about": (o.get("about") or "")[:160] or None,
                "parent": name_of.get(o["parent"]) if o.get("parent") and kind != "siblings" else None} for o in opts]
    if rng.random() < 0.5:
        rng.shuffle(options)                                    # teacher order: half natural, half shuffled
    if with_none:
        options.append({"node": "NONE", "name": NONE, "about": None, "parent": None})
    describe = len(options) <= 80 and rng.random() < 0.6
    q = rng.choice(QUESTIONS).format(level=level_name)
    sname = s["name"] + (f" ({s['purpose']})" if s.get("purpose") else "")
    if kind == "siblings":
        sname += f" — within: {name_of[parent]}"
    return {"id": uid, "scheme": s["id"], "split": split, "teacher": teacher, "kind": kind, "level": level_name,
            "scheme_text": sname, "question": q, "describe": describe, "options": options,
            "papers": [{"work_id": p["work_id"], "title_key": p["title_key"],
                        "text": text(p, rng.random() < TITLE_ONLY)} for p in papers]}


def main():
    (Q / "train").mkdir(parents=True, exist_ok=True); (Q / "eval").mkdir(parents=True, exist_ok=True)
    maps = sorted(Path(f).stem for f in glob.glob(str(SCHEMES / "map-*.json")))
    inv = sorted(Path(f).stem for f in glob.glob(str(SCHEMES / "inv-*.json")))
    inv_train = [s for s in inv if load(s)["split"] == "train"]
    inv_held = [s for s in inv if load(s)["split"] == "heldout"]
    for f in sorted(Q.glob("*/*.json")):                       # resume: who owns which paper, what is reserved
        u = json.loads(f.read_text())
        for p in u["papers"]:
            w = p["work_id"]; uses[w] += 1; seen_title[p["title_key"]] = w
            if u["split"] == "train":
                owner[w] = u["teacher"]
            else:
                eval_papers.add(w)
    plan = []
    for sid, n in BUDGET.items():
        plan += [(sid, teachers.ROLES[TEACHER[sid]], "train")] * n
    for i in range(MAPS_UNITS):
        plan.append((maps[i % len(maps)], A if (i // len(maps) + maps.index(maps[i % len(maps)])) % 2 else B, "train"))
    for i in range(INVENTED_UNITS):
        s = inv_train[i % len(inv_train)]
        plan.append((s, A if load(s)["source"] == f"invented by {A}" else B, "train"))   # a scheme's inventor labels it (invent.py)
    # eval first, so its papers are reserved before training takes any
    ev = [("jel", "both", "eval")] * EVAL["jel"] + [(inv_held[i % len(inv_held)], "both", "eval") for i in range(EVAL["heldout-invented"])]
    real = list(BUDGET) + maps[:10]
    ev += [(real[i % len(real)], "both", "eval") for i in range(EVAL["heldout-papers"])]
    rng.shuffle(plan)
    made = Counter()
    for split, items in (("eval", ev), ("train", plan)):
        for j, (sid, teacher, sp) in enumerate(items):
            uid = f"{split}-{j:05d}"
            f = Q / split / f"{uid}.json"
            if f.exists():
                continue
            u = None
            for _ in range(4):
                u = unit(load(sid), teacher, sp, uid)
                if u:
                    break
            if not u:
                made["failed"] += 1; continue
            f.write_text(json.dumps(u))
            made[(split, teacher)] += 1
            if j % 200 == 0:
                print(split, j, dict(made), flush=True)
    print("DONE", dict(made), "papers", len(owner), "eval papers", len(eval_papers), flush=True)
    (Q / "DONE").write_text(json.dumps(dict((str(k), v) for k, v in made.items())))


if __name__ == "__main__":
    main()

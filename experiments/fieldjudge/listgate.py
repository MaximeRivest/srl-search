"""Step 9: can the judges make the reading lists better? On the round-2 free maps (fields-r4: frag and LLM post-training,
made with the round-2 0.8B and 2B), each group's lists are rebuilt DEPTH deep instead of 8 (same release, same groups),
every candidate is judged by both round-2 judges, and each list keeps the first 8 candidates that pass a rule:

  base          today's lists (the first 8)
  in_map        only papers that are themselves in the mapped field (free: no model)
  j08 / j2B / both   only papers the judge says are on topic (P >= TAU); "both" = mean log-odds of the two judges.
                Judged on the paper itself, so a classic outside the map can still be kept (and off-topic
                foundations such as BERT dropped)
  both_strict   "both" with TAU_STRICT
  in_map_both   in the field AND "both"
  members       core and rising made of the group's own papers ranked by how often the whole literature cites them
                (rising: per year, last 5 years; start here: its reviews and surveys of the last 8 years), for groups where few papers have reference lists (< SPARSE of them),
                which starves the lists counted from the group's references; other groups and lists as in_map_both.
                All lists then judged as in_map_both

Measured: Kimi K3 on every paper any rule shows (silver reference, cached; ~$0.0015 a paper), per list; how many of the
classics (frag: studies 19 meta-analyses used; LLM: 44 landmarks) appear anywhere in the lists; how cited the core
lists stay (median "cited by N papers of the group"); lists left short of 8 (the pool ran out).

    SRL_RELEASE=<the runs' release> SRL_FIELDS_GPU=1 ../../.venv/bin/python listgate.py   (Kimi key in the environment)
"""
import inspect, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J, topics as T
from srl_search.engine import passes
from srl_search.scope import NON_PUBLICATION_TYPES
from srl_search.topic_service import finish, VERSION

R4 = Path("/mnt/fast/scholarsreadinglist/search/fields-r4")
WORK = R4 / "listgate"; WORK.mkdir(exist_ok=True)
D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
MODELS = {"08": D / "v2/models/0.8B/merged", "2B": D / "v2/models/2B/merged"}
VPY = os.environ.get("SRL_VLLM_PYTHON", "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/python")
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
DEPTH, SHOW, TAU, TAU_STRICT = 40, 8, 0.5, 0.8
TOPICS = {"frag": ("habitat fragmentation", None),
          "llm": ("LLM post-training", "training large language models after pretraining: supervised fine-tuning and instruction "
                  "tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement "
                  "learning for reasoning, and alignment training")}
ARMS = ("new08-only", "new2B-only")
LISTS = ("start_here", "core", "distinctive", "rising", "bridges", "newest")
RULES = ("base", "in_map", "j08", "j2B", "both", "both_strict", "in_map_both", "members")
SPARSE = 0.3
SURVEY = re.compile(r"\b(survey|review|overview|tutorial|primer)\b", re.I)

# Deeper lists without changing which papers the first 8 are: "only here" must still skip today's 8 core papers
# (not the 40), so the experiment patches that one line of a copy of lenses_for (production code untouched).
src = inspect.getsource(T.lenses_for).replace("core_set = set(core)", f"core_set = set(core[:{SHOW}])")
assert f"core[:{SHOW}]" in src
exec(compile(src, T.__file__, "exec"), T.__dict__)

ctx = F.Ctx()
cit = ctx.cit
bad = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]


def publications(ws):                                   # as readinglists.py
    det = F._details(ctx, ws)
    ok = []
    for w in ws:
        d = det.get(int(w))
        if d is not None:
            ok.append(passes(d, {}))
        else:
            nd = int(cit.nodes([int(w)])[0])
            ok.append(nd >= 0 and int(cit.kind[nd]) not in bad and len(cit.title(nd)) >= 10)
    return np.array(ok, bool)


def deep_map(t, arm):
    """build_maps' main map (field.py), lists DEPTH deep, no AI group names."""
    f = WORK / f"{t}-{arm}-deep{DEPTH}.json"
    if f.exists():
        return json.loads(f.read_text())
    Z = np.load(R4 / f"{t}-{arm}" / "work" / "scores.npz")
    ids, final, cut = Z["ids"], Z["final"], float(Z["cut"])
    yes = np.flatnonzero(final >= cut)
    ws = ids[yes][publications(ids[yes])]
    p = dict(zip(ids.tolist(), final.tolist()))
    ws = np.array(sorted(int(w) for w in ws), np.int64)
    det = F._details(ctx, ws)
    V, has = F._vectors(ctx, ws)
    details = []
    for w in ws:
        d = det.get(int(w))
        if d is None:
            nd = int(cit.nodes([int(w)])[0])
            d = {"title": cit.title(nd) if nd >= 0 else None, "year": int(cit.year[nd]) if nd >= 0 else None,
                 "type": cit.type(nd) if nd >= 0 else None, "citations": int(cit.cites[nd]) if nd >= 0 else None}
        details.append(d)
    seed = T.Seed(ws, V, has, details, np.array([round(p[int(w)], 3) for w in ws], np.float32))
    seed = seed.subset(T.collapse_versions(seed, cit))
    T.PARALLEL_MIN = 1500
    bg = ctx.keywords.phrase_counts if ctx.keywords is not None else None
    raw = T.build(seed, cit, TOPICS[t][0], T.Options(refs_top=30, workers=F.LEIDEN_WORKERS, lens_size=DEPTH), background=bg)
    out = finish(ctx, cit, raw, seed, ctx.release["id"])
    f.write_text(json.dumps(out))
    return out


def text_of(ws):
    det = F._details(ctx, ws)
    out = []
    for w in ws:
        d = det.get(int(w))
        if d:
            out.append(J.paper_text(d.get("title"), d.get("abstract")))
        else:
            nd = int(cit.nodes([int(w)])[0])
            out.append(J.paper_text(cit.title(nd) if nd >= 0 else "", None))
    return out


def judge(t, ws, name):
    """P(on topic) from one judge for these papers, cached per topic."""
    f = WORK / f"{t}-{name}-scores.json"
    have = {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}
    todo = [w for w in ws if w not in have]
    if todo:
        q, defin = TOPICS[t]
        src, dst = WORK / f"{t}-{name}-in.jsonl", WORK / f"{t}-{name}-out.json"
        system = J.instruction(q, defin) + ANSWER
        src.write_text("".join(json.dumps({"text": x, "system": system}) + "\n" for x in text_of(todo)))
        J.wait_for_gpu(10000, print)
        J.run([VPY, str(Path(J.__file__).with_name("vllm_decider.py")), str(MODELS[name]), str(src), str(dst), J.GPU],
              WORK / "vllm-log.txt")
        have.update(zip(todo, json.loads(dst.read_text())))
        f.write_text(json.dumps({str(k): v for k, v in have.items()}))
    return have


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
maps = {(t, a): deep_map(t, a) for t in TOPICS for a in ARMS}

# the first 8 are today's lists (same groups, same order): a check that the deeper build changed nothing else
for (t, a), m in maps.items():
    today = json.loads((R4 / f"{t}-{a}" / "maps" / VERSION / "main.json").read_text())
    same = sum(g["name"] == h["name"] and all([p["id"] for p in (g["lenses"].get(n) or [])[:SHOW]] ==
                                              [p["id"] for p in (h["lenses"].get(n) or [])[:SHOW]] for n in LISTS)
               for g, h in zip(m["groups"], today["groups"]))
    print(t, a, f"groups whose first {SHOW} match today's lists: {same} of {len(today['groups'])}", flush=True)

scores = {}
for t in TOPICS:
    ws = sorted({int(p["id"][1:]) for a in ARMS for g in maps[(t, a)]["groups"] for n in LISTS
                 for p in (g["lenses"].get(n) or [])})
    s08, s2b = judge(t, ws, "08"), judge(t, ws, "2B")
    scores[t] = {w: {"08": s08[w], "2B": s2b[w],
                     "both": float(1 / (1 + np.exp(-(lg(s08[w]) + lg(s2b[w])) / 2)))} for w in ws}
    print(t, len(ws), "candidates judged by both models", flush=True)


def keep(rule, t, p):
    s = scores[t][int(p["id"][1:])]
    return {"base": True, "in_map": p.get("in_map", False), "j08": s["08"] >= TAU, "j2B": s["2B"] >= TAU,
            "both": s["both"] >= TAU, "both_strict": s["both"] >= TAU_STRICT,
            "in_map_both": p.get("in_map", False) and s["both"] >= TAU,
            "members": p.get("in_map", False) and s["both"] >= TAU}[rule]


def member_pools(m):
    """Per sparse group: its papers by citations from the whole literature (core) and by citations a year (rising)."""
    ids = np.array([int(x[1:]) for x in m["papers"]["id"]]); grp = np.array(m["papers"]["group"])
    nd = cit.nodes(ids); ok = nd >= 0
    cites = np.where(ok, cit.cites[np.maximum(nd, 0)], 0).astype(float)
    year = np.where(ok, cit.year[np.maximum(nd, 0)], 0)
    pools = {}
    for g in m["groups"]:
        if g.get("with_references", 0) >= SPARSE * g["size"]:
            continue
        ix = np.flatnonzero(grp == g["id"])
        core = ix[np.argsort(-cites[ix], kind="stable")][:DEPTH]
        rec = ix[year[ix] >= T.YEAR_NOW - 5]
        rate = cites[rec] / np.maximum(T.YEAR_NOW - year[rec] + 1, 1)
        rising = rec[np.argsort(-rate, kind="stable")][:DEPTH]
        mk = lambda i, why: {"id": f"W{ids[i]}", "in_map": True, "cited_here": None, "citations": int(cites[i]), "why": why}
        det = F._details(ctx, ids[ix])
        rev = [i for i in ix[np.argsort(-cites[ix], kind="stable")] if year[i] >= T.YEAR_NOW - 8 and (
            (det.get(int(ids[i])) or {}).get("type") == "review" or SURVEY.search((det.get(int(ids[i])) or {}).get("title") or ""))][:DEPTH]
        pools[g["id"]] = {"core": [mk(i, "most cited") for i in core], "rising": [mk(i, "rising") for i in rising],
                          "start_here": [mk(i, "review") for i in rev]}
    return pools


pools = {k: member_pools(m) for k, m in maps.items()}
for t in TOPICS:
    extra = sorted({int(p["id"][1:]) for a in ARMS for gp in pools[(t, a)].values() for v in gp.values() for p in v}
                   - set(scores[t]))
    if extra:
        s08, s2b = judge(t, extra, "08"), judge(t, extra, "2B")
        for w in extra:
            scores[t][w] = {"08": s08[w], "2B": s2b[w], "both": float(1 / (1 + np.exp(-(lg(s08[w]) + lg(s2b[w])) / 2)))}
    print(t, "sparse groups:", {a: sorted(pools[(t, a)]) for a in ARMS}, len(extra), "more candidates judged", flush=True)

shown = {}                                               # (topic, arm, rule) -> [(group, list, paper)]
for (t, a), m in maps.items():
    for rule in RULES:
        rows = []
        for g in m["groups"]:
            for n in LISTS:
                items = g["lenses"].get(n) or []
                if rule == "members" and n in ("core", "rising", "start_here") and g["id"] in pools[(t, a)]:
                    items = pools[(t, a)][g["id"]][n]
                pick = items[:SHOW] if rule == "base" else [p for p in items if keep(rule, t, p)][:SHOW]
                rows += [(g["id"], n, p) for p in pick]
        shown[(t, a, rule)] = rows

# Kimi on every paper any rule shows (answers from earlier runs kept)
prev = {}
for f in [*Path(__file__).parent.glob("readinglists*.json"), WORK / "kimi.json"]:
    if f.exists():
        for t, d in json.loads(f.read_text()).get("labels", json.loads(f.read_text())).items():
            prev.setdefault(t, {}).update({int(k): v for k, v in d.items()})
labels = {}
usd = 0.0
for t, (q, defin) in TOPICS.items():
    old = prev.get(t, {})
    ws = sorted({int(p["id"][1:]) for (tt, a, r), rows in shown.items() if tt == t for _, _, p in rows} - set(old))
    y = J.Teacher(8.0).label(text_of(ws), q, defin) if ws else np.zeros(0)
    labels[t] = {**old, **dict(zip(ws, y.tolist()))}
    print(t, len(ws), "new papers judged by Kimi", flush=True)
(WORK / "kimi.json").write_text(json.dumps({t: {str(k): v for k, v in d.items()} for t, d in labels.items()}))

classics = {}
for r in map(json.loads, open(D / "tests.jsonl")):
    if r["set"] in ("landmarks", "meta_refs"):
        classics.setdefault(r["test"], set()).add(int(r["id"]))

out = {}
for (t, a, rule), rows in shown.items():
    lab = labels[t]
    ok = lambda p: lab.get(int(p["id"][1:]))
    ys = [ok(p) for _, _, p in rows if ok(p) == ok(p) and ok(p) is not None]
    per = {}
    for n in LISTS:
        v = [ok(p) for _, nn, p in rows if nn == n and ok(p) is not None and ok(p) == ok(p)]
        per[n] = round(float(np.mean(v)), 3) if v else None
    groups = maps[(t, a)]["groups"]
    short = sum(1 for g in groups for n in LISTS
                if len([1 for gid, nn, _ in rows if gid == g["id"] and nn == n]) < min(SHOW, len(g["lenses"].get(n) or [])))
    papers = {int(p["id"][1:]) for _, _, p in rows}
    core_cited = [p["cited_here"] for _, n, p in rows if n == "core" and p.get("cited_here") is not None]
    out[f"{t}/{a}/{rule}"] = {"shown": len(rows), "distinct": len(papers), "on_topic": round(float(np.mean(ys)), 3),
                             "by_list": per, "classics_in_lists": len(papers & classics[t]),
                             "core_median_cited_here": float(np.median(core_cited)) if core_cited else None,
                             "lists_left_short": short}
    print(f"{t:4} {a:10} {rule:11}", json.dumps(out[f"{t}/{a}/{rule}"]), flush=True)
(Path(__file__).parent / "listgate.json").write_text(json.dumps(out, indent=1))
print("LISTGATE DONE", flush=True)

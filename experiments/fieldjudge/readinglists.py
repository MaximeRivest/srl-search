"""Step 8: are the reading lists different? Each replay arm's selection (replay.py: work/scores.npz) is mapped into
groups with reading lists (srl_search.field.build_maps, main map only), against the runs' release. Then:
  - what readers see first: the top 10 of each paper list (start_here, core, distinctive, rising, bridges, newest)
    of every group, and the group names
  - overlap between arms (the same papers?)
  - Kimi K3 judges every distinct reading-list paper once, for its topic: the share on topic, per arm and per list
    ~$3

    ../../.venv/bin/python readinglists.py      (SRL_RELEASE=<the runs' release>; keys in the environment)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J
from srl_search.engine import passes
from srl_search.scope import NON_PUBLICATION_TYPES
from srl_search.topic_service import VERSION

R3 = Path("/mnt/fast/scholarsreadinglist/search/fields-r3")
TOP = 10
TOPICS = {"frag": ("habitat fragmentation", None),
          "llm": ("LLM post-training", "training large language models after pretraining: supervised fine-tuning and instruction "
                  "tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement "
                  "learning for reasoning, and alignment training")}
ARMS = ("per-topic", "general-2B", "general-0.8B")
ctx = F.Ctx()
cit = ctx.cit
bad = [i for i, tn in enumerate(cit.types) if tn in NON_PUBLICATION_TYPES]
F.SPLIT_MIN = 10 ** 9                                   # main map only: no subgroups


def publications(ws):
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


maps = {}
for t, (q, defin) in TOPICS.items():
    for arm in ARMS:
        job = F.Job(R3, f"{t}-{arm}")
        main = job.dir / "maps" / VERSION / "main.json"
        if not main.exists():
            Z = np.load(job.dir / "work" / "scores.npz")
            ids, final, cut, route = Z["ids"], Z["final"], float(Z["cut"]), Z["route"]
            yes = np.flatnonzero(final >= cut)
            ok = publications(ids[yes])
            via = {0: "meaning", 1: "phrase", 2: "cited"}
            sel = {"topic": q, "definition": defin, "made": F._now(), "release": ctx.release["id"], "report": {"arm": arm},
                   "papers": {str(int(ids[i])): {"via": via[int(route[i])], "p": round(float(final[i]), 3)} for i, o in zip(yes, ok) if o}}
            F.build_maps(ctx, job, sel, VERSION)
        maps[(t, arm)] = json.loads(main.read_text())
        print(t, arm, "groups", len(maps[(t, arm)]["groups"]), flush=True)

LISTS = ("start_here", "core", "distinctive", "rising", "bridges", "newest")
seen = {}                                                # (topic, arm) -> {(list, paper id)}
for (t, arm), m in maps.items():
    s = set()
    for g in m["groups"]:
        for name in LISTS:
            for p in (g["lenses"].get(name) or [])[:TOP]:
                s.add((name, p["id"]))
    seen[(t, arm)] = s

# Kimi on every distinct reading-list paper, once per topic
labels = {}
for t, (q, defin) in TOPICS.items():
    ws = sorted({int(pid[1:]) for arm in ARMS for _, pid in seen[(t, arm)]})
    det = F._details(ctx, ws)
    texts = []
    for w in ws:
        d = det.get(w)
        if d:
            texts.append(J.paper_text(d.get("title"), d.get("abstract")))
        else:
            nd = int(cit.nodes([w])[0])
            texts.append(J.paper_text(cit.title(nd) if nd >= 0 else "", None))
    k = J.Teacher(6.0)
    y = k.label(texts, q, defin)
    labels[t] = {w: v for w, v in zip(ws, y.tolist())}
    print(t, len(ws), "reading-list papers judged, $", round(k.usd, 2), "failed", k.failed, flush=True)

out = {}
for t in TOPICS:
    base = {pid for _, pid in seen[(t, "per-topic")]}
    for arm in ARMS:
        papers = {pid for _, pid in seen[(t, arm)]}
        ys = [labels[t][int(pid[1:])] for pid in papers if labels[t].get(int(pid[1:])) == labels[t].get(int(pid[1:]))]
        per_list = {}
        for name in LISTS:
            v = [labels[t][int(pid[1:])] for n, pid in seen[(t, arm)] if n == name and labels[t].get(int(pid[1:])) == labels[t].get(int(pid[1:]))]
            per_list[name] = round(float(np.mean(v)), 3) if v else None
        off = sorted(({pid for pid in papers if labels[t].get(int(pid[1:])) == 0.0}))
        out[f"{t}/{arm}"] = {"groups": [g["name"] for g in maps[(t, arm)]["groups"]], "papers_in_lists": len(papers),
                             "on_topic": round(float(np.mean(ys)), 3), "on_topic_by_list": per_list,
                             "same_as_per_topic": round(len(papers & base) / max(1, len(papers | base)), 3),
                             "off_topic_examples": off[:0]}
        print(t, arm, json.dumps({k: v for k, v in out[f"{t}/{arm}"].items() if k != "groups"}), flush=True)
        print("   groups:", " | ".join(out[f"{t}/{arm}"]["groups"]), flush=True)
(Path(__file__).parent / "readinglists.json").write_text(json.dumps({"summary": out, "labels": {t: {str(k): v for k, v in d.items()} for t, d in labels.items()},
                                                                     "lists": {f"{t}/{a}": sorted(map(list, s)) for (t, a), s in seen.items()}}, indent=1))
print("READING LISTS DONE", flush=True)

import os, sys, time, json, collections
sys.path.insert(0, ".")
for l in open(os.path.expanduser("~/Projects/lm15-dev/.env")):
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip('"').strip("'"))
import numpy as np
from pathlib import Path
from srl_search import field as Fm, judges as J
ctx = Fm.Ctx()
TOPIC = "LLM post-training"
DEF = "training large language models after pretraining: supervised fine-tuning and instruction tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement learning for reasoning, and alignment training"
surveys = [4417305769, 4412886765, 4416545265, 4386081793, 4416296080, 4390215524, 4394839102, 4415070954, 4417080162, 4406549374]
nd = ctx.cit.nodes(surveys)
refs_by = {}
for w, n in zip(surveys, nd):
    r = [ctx.cit.work(int(x)) for x in ctx.cit.refs(int(n))] if n >= 0 else []
    refs_by[w] = r; print(w, "refs", len(r))
ref_ws = sorted(set(x for r in refs_by.values() for x in r) - set(surveys))
print("distinct references", len(ref_ws))
det = Fm._details(ctx, ref_ws)
def pair(w):
    d = det.get(int(w))
    if d: return d.get("title"), d.get("abstract")
    n = int(ctx.cit.nodes([int(w)])[0]); return (ctx.cit.title(n) if n >= 0 else ""), None
rp = [pair(w) for w in ref_ws]
tK = J.Teacher(2.0); t = time.time(); ry = tK.label([J.paper_text(*p) for p in rp], TOPIC, DEF)
print("Kimi", round(time.time() - t), "s $", round(tK.usd, 3), "failed", tK.failed, "on topic", int(np.nansum(ry >= .5)))
REFS = {"surveys": surveys, "ids": ref_ws, "y": [None if y != y else float(y) for y in ry], "abstract": [bool(p[1]) for p in rp]}
Path("/home/maxime/Projects/scholarsreadinglist/search-dev/experiments/topicmap/teacher/llm_posttraining_survey_refs.json").write_text(json.dumps(REFS))

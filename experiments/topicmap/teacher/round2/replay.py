import os, sys, time, json, math, collections, importlib
sys.path.insert(0, ".")
for l in open(os.path.expanduser("~/Projects/lm15-dev/.env")):
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip('"').strip("'"))
import numpy as np
from pathlib import Path
from srl_search import field as Fm, judges as J
ctx = Fm.Ctx()
D4 = "/mnt/fast/scholarsreadinglist/search/fields-test4/a699318c9a209d289e4c3702"
f4 = json.load(open(D4 + "/field.json"))
t4 = {int(w): v for w, v in f4["labels"]["teacher"].items()}; d4 = {int(w): v for w, v in f4["labels"]["decider"].items()}
def text_map(ids):
    det = Fm._details(ctx, list(ids)); out = {}
    for w, v in ids.items():
        d = det.get(w)
        if d: tx = J.paper_text(d.get("title"), d.get("abstract"))
        else:
            nd = int(ctx.cit.nodes([w])[0]); tx = J.paper_text(ctx.cit.title(nd) if nd >= 0 else "", None)
        out[tx] = v
    return out
tt, dt = text_map(t4), text_map(d4)
class FakeTeacher:
    def __init__(self, b): self.usd, self.failed, self.slowest = 0.0, 0, {}
    def label(self, texts, q, d):
        print("teacher texts not cached:", sum(t not in tt for t in texts)); return np.array([tt.get(t, np.nan) for t in texts])
saved = (J.Teacher, J.train_decider, J.decider_scores)
J.Teacher = FakeTeacher; J.train_decider = lambda *a, **k: None
def fake_dec(md, texts, say):
    print("decider texts not cached:", sum(t not in dt for t in texts)); return np.array([dt.get(t, 0.0) for t in texts])
J.decider_scores = fake_dec
R = Path("/mnt/fast/scholarsreadinglist/search/fields-r2"); jr = Fm.Job(R, "replay4"); jr.dir.mkdir(parents=True, exist_ok=True)
try:
    t = time.time(); sel = Fm.select_field(ctx, jr, "habitat fragmentation", None, None); print("replay", round(time.time() - t), "s")
finally:
    J.Teacher, J.train_decider, J.decider_scores = saved
inter = len(set(sel["papers"]) & set(f4["papers"])); print("papers", len(sel["papers"]), "v4 run", len(f4["papers"]), "overlap", inter)
Z = np.load(jr.dir / "work" / "scores.npz"); univ, final = Z["ids"], Z["final"]; cut4 = sel["report"]["judging"]["cut"]

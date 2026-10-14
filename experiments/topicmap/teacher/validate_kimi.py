"""Recall and precision of whole-field selections, with Kimi K3 as the yardstick (no Jev):
recall = share of the meta-analyses' references Kimi calls on topic that the field holds (refs_labelled.jsonl);
precision = share of 400 random field papers Kimi calls on topic.

    search/.venv/bin/python validate_kimi.py NAME=FIELD_DIR [NAME=FIELD_DIR ...]      (Kimi keys in the environment)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from srl_search import judges as J
from srl_search.field import Ctx, _details
T = Path.home() / "Projects/tiny-classifiers/data/tasks/frag"
refs = [json.loads(l) for l in open(T / "refs_labelled.jsonl")]
meta = json.load(open(T / "meta.json"))
pos = [r["id"] for r in refs if r["label"] == "on topic"]
ctx = Ctx()
out = {}
for arg in sys.argv[1:]:
    name, d = arg.split("=", 1)
    field = json.load(open(Path(d) / "field.json"))["papers"]
    found = np.array([r[1:] in field for r in pos])
    ab = np.array([meta[r]["abstract"] for r in pos])
    rng = np.random.default_rng(0)
    sample = [int(x) for x in rng.choice(sorted(field, key=int), 400, replace=False)]
    det = _details(ctx, sample)
    texts = []
    for w in sample:
        dd = det.get(w)
        nd = int(ctx.cit.nodes([w])[0])
        texts.append(J.paper_text(dd.get("title"), dd.get("abstract")) if dd else J.paper_text(ctx.cit.title(nd) if nd >= 0 else "", None))
    t = J.Teacher(1.0)
    p = t.label(texts, "habitat fragmentation", None)
    out[name] = {"papers": len(field), "recall": round(float(found.mean()), 3),
                 "recall_with_abstract": round(float(found[ab].mean()), 3), "recall_title_only": round(float(found[~ab].mean()), 3),
                 "precision": round(float(np.nanmean(p >= 0.5)), 3), "precision_n": int((~np.isnan(p)).sum()), "kimi_usd": round(t.usd, 3)}
    print(name, json.dumps(out[name]), flush=True)
(Path(__file__).parent / "validate_kimi.json").write_text(json.dumps(out, indent=1))

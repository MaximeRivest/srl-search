"""The test sets, in one file: papers of the two held-out topics with the reference model's answers (the yardstick), their
sampling weights (stratified samples of each topic's whole-field candidates: weights give population precision and
recall), Jev's answers (the checker to beat), the meta-analyses' on-topic references (habitat fragmentation) and the
44 landmark papers (LLM post-training).

    ../../.venv/bin/python build_tests.py        (reference-model/Jev keys in the environment; writes tests.jsonl; ~$0.10 of Jev)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
EXP = Path(__file__).resolve().parents[1] / "topicmap" / "teacher"
FRAG = ("habitat fragmentation", None)
LLM = ("LLM post-training", "training large language models after pretraining: supervised fine-tuning and instruction "
       "tuning, reinforcement learning from human or AI feedback, preference optimization such as DPO, reinforcement "
       "learning for reasoning, and alignment training")
ctx = F.Ctx()


def pairs(ws):
    det = F._details(ctx, ws)
    out = []
    for w in ws:
        d = det.get(int(w))
        if d:
            out.append((d.get("title"), d.get("abstract")))
        else:
            nd = int(ctx.cit.nodes([int(w)])[0])
            out.append((ctx.cit.title(nd) if nd >= 0 else "", None))
    return out


rows = []


def add(test, topic, s, ids, ys, strata=None, N=None, n=None, jev=None):
    ps = pairs(ids)
    if jev is None:
        jev = J.Jev(1.0).label(ps, *topic)
    for k, (w, (t, a)) in enumerate(zip(ids, ps)):
        r = {"test": test, "set": s, "topic": topic[0], "definition": topic[1], "id": int(w), "text": J.paper_text(t, a),
             "reference": ys[k], "jev": None if jev[k] is None or jev[k] != jev[k] else float(jev[k]), "title_only": not a}
        if strata:
            r.update({"stratum": strata[k], "weight": N[strata[k]] / n[strata[k]]})
        rows.append(r)


e = json.loads((EXP / "eval_population.json").read_text())
add("frag", FRAG, "population", e["ids"], e["y"], e["stratum"], e["N"], e["n"])
e = json.loads((EXP / "eval_classics.json").read_text())
add("frag", FRAG, "cited50", e["ids"], e["y"], e["stratum"], e["N"], e["n"])
refs = [json.loads(l) for l in open(Path.home() / "Projects/tiny-classifiers/data/tasks/frag/refs_labelled.jsonl")]
pos = [int(r["id"][1:]) for r in refs if r["label"] == "on topic"]
add("frag", FRAG, "meta_refs", pos, [1.0] * len(pos))
e = json.loads((EXP / "eval_population_llm_posttraining.json").read_text())
add("llm", LLM, "population", e["ids"], e["y"], e["stratum"], e["N"], e["n"], jev=e["jev"])
lm = [int(l.split("\t")[0][1:]) for l in open(EXP / "round2" / "landmarks.tsv") if l.split("\t")[2] == "OK"] + \
     [4367000491, 4387322976, 4386437475, 4414281281]
add("llm", LLM, "landmarks", lm, [1.0] * len(lm))
(D / "tests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
import collections
print(collections.Counter((r["test"], r["set"]) for r in rows))

"""The meaning-vector classifier as a prefilter: trained on the budget's Kimi labels (375) alone, or on them plus the
0.8B decider's 10,000 labels, how many papers can it rule out before any text is read, at a cut-off that keeps
98% / 99% / 99.5% of the on-topic papers (Kimi's labels, test + refs, papers with a vector)?

    search/.venv/bin/python search-dev/experiments/topicmap/teacher/prefilter.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search.field import Ctx, _vectors  # noqa: E402

T = Path.home() / "Projects/tiny-classifiers/data/tasks/frag"
L = Path.home() / "Projects/tiny-classifiers/labels"
load = lambda n: [json.loads(l) for l in open(T / f"{n}.jsonl")]
ctx = Ctx()


def V(ids):
    X, h = _vectors(ctx, np.array([int(i[1:]) for i in ids], np.int64))
    return X, h


kimi = {}
for f in ("frag-all.jsonl", "frag-more.jsonl"):
    for l in open(L / f):
        r = json.loads(l)
        kimi[r["id"]] = r["pred"] == "on topic"
train = [r["id"] for r in load("train") + load("more") if r["id"] in kimi]
random.Random(0).shuffle(train)
train = train[:375]
pseudo = json.load(open(HERE / "preds" / "pool20k__vllm__frag-qwen08-n375.json"))
pseudo_ids = [r["id"] for r in load("pool20k")][:10000]
ev = load("test") + load("refs_labelled")
Xe, he = V([r["id"] for r in ev])
ye = np.array([r["label"] == "on topic" for r in ev])
pool_ids = [r["id"] for r in load("pool")]
Xp, hp = V(pool_ids)
out = {}
for name, ids, lab in (("kimi375", train, lambda i: kimi[i]),
                       ("kimi375+decider10k", train + pseudo_ids, lambda i: kimi[i] if i in kimi else pseudo[i] >= 0.5)):
    X, h = V(ids)
    y = np.array([lab(i) for i in ids])
    clf = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(X[h], y[h])
    pe = clf.predict_proba(Xe[he])[:, 1]
    pp = clf.predict_proba(Xp[hp])[:, 1]
    yy = ye[he]
    res = {"auc": round(float(roc_auc_score(yy, pe)), 4), "evaluated": int(he.sum()), "of": len(ev)}
    for keep in (0.98, 0.99, 0.995):
        t = float(np.quantile(pe[yy], 1 - keep))           # the cut-off that keeps this share of on-topic papers
        res[f"keep_{keep}"] = {"cutoff": round(t, 4), "eval_kept": round(float((pe >= t).mean()), 3),
                               "pool_kept": round(float((pp >= t).mean()), 3)}
    out[name] = res
    print(name, json.dumps(res), flush=True)
out["pool_with_vector"] = round(float(hp.mean()), 3)
(HERE / "prefilter.json").write_text(json.dumps(out, indent=1))

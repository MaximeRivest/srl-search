"""The last step of the staircase: logistic regression on the papers' meaning vectors (instant over all 125M papers),
trained on N teacher labels alone, or on them plus the pool papers labelled by the 150M student.
Evaluated on the papers of test / refs that have a vector (it cannot see the others: reported apart).

    search/.venv/bin/python search-dev/experiments/topicmap/teacher/lr_staircase.py
Appends to curve_results.jsonl (model "vectors-lr").
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
vec_cache: dict = {}


def vecs(ids):
    need = [i for i in ids if i not in vec_cache]
    if need:
        V, h = _vectors(ctx, np.array([int(i[1:]) for i in need], np.int64))
        for i, v, hh in zip(need, V, h):
            vec_cache[i] = v if hh else None
    return ids


kimi = {}
for f in ("frag-all.jsonl", "frag-more.jsonl"):
    for l in open(L / f):
        r = json.loads(l)
        kimi[r["id"]] = r["pred"] == "on topic"
train_pool = [r["id"] for r in load("train") + load("more") if r["id"] in kimi]
evals = {"test": load("test"), "refs": load("refs_labelled")}
for rows in evals.values():
    vecs([r["id"] for r in rows])
pool = [r["id"] for r in load("pool")]
vecs(pool)
vecs(train_pool)

for n in (250, 1000, 4000, 17580):
    ids = list(train_pool)
    random.Random(0).shuffle(ids)
    ids = ids[:n]
    for pseudo in (None, "150m"):
        X = [vec_cache[i] for i in ids if vec_cache[i] is not None]
        y = [kimi[i] for i in ids if vec_cache[i] is not None]
        if pseudo:
            f = HERE / "preds" / f"pool__ettin-encoder-150m__n{n}__s0.json"
            if not f.exists():
                continue
            pp = json.loads(f.read_text())
            X += [vec_cache[i] for i in pool if vec_cache[i] is not None]
            y += [pp[i] >= 0.5 for i in pool if vec_cache[i] is not None]
        clf = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(np.array(X), np.array(y))
        res = {"model": "vectors-lr", "n": n, "seed": 0, "pseudo": len(X) - sum(1 for i in ids if vec_cache[i] is not None) if pseudo else 0,
               "pseudo_from": f.name if pseudo else None, "tag": "staircase" if pseudo else "", "train_rows": len(X)}
        for name, rows in evals.items():
            keep = [r for r in rows if vec_cache[r["id"]] is not None]
            yy = np.array([r["label"] == "on topic" for r in keep])
            p = clf.predict_proba(np.array([vec_cache[r["id"]] for r in keep]))[:, 1]
            k = p >= 0.5
            res[name] = {"auc": round(float(roc_auc_score(yy, p)), 4), "acc": round(float((k == yy).mean()), 4),
                         "recall": round(float(k[yy].mean()), 4), "precision": round(float(yy[k].mean()), 4),
                         "recall_title_only": None, "evaluated": len(keep), "of": len(rows)}
        with open(HERE / "curve_results.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)

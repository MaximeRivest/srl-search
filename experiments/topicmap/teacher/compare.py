"""Teacher experiment: Ettin fine-tuned on Kimi K3's labels vs logistic regression on meaning vectors trained on the
same labels; Kimi vs Jev agreement. Writes compare.json beside this file.

    search/.venv/bin/python search-dev/experiments/topicmap/teacher/compare.py TASK_DIR ETTIN_SCORES.json
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from srl_search.field import Ctx, _vectors  # noqa: E402
T, ettin = Path(sys.argv[1]), json.load(open(sys.argv[2]))
meta = json.load(open(T / "meta.json"))
load = lambda n: [json.loads(l) for l in open(T / f"{n}.jsonl")]
train, test, refs = load("train"), load("test"), load("refs_labelled")
ctx = Ctx()
def vec(rows):
    return _vectors(ctx, np.array([int(r["id"][1:]) for r in rows], np.int64))
Xtr, htr = vec(train)
ytr = np.array([r["label"] == "on topic" for r in train])
lr = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xtr[htr], ytr[htr])
out = {"train": len(train), "train_with_vector": int(htr.sum())}
for name, rows in (("test", test), ("refs", refs)):
    X, h = vec(rows)
    y = np.array([r["label"] == "on topic" for r in rows])
    p_lr = np.where(h, lr.predict_proba(X)[:, 1], 0.0)          # no vector: the vector model cannot see it
    key = "test" if name == "test" else "refs_labelled"
    p_et = np.array([ettin[key][r["id"]] for r in rows])
    has_abs = np.array([meta[r["id"]]["abstract"] for r in rows])
    sets = np.array([meta[r["id"]]["set"] for r in rows])
    res = {"n": len(rows), "on_topic": int(y.sum()), "without_abstract": int((~has_abs).sum())}
    for m, p in (("ettin17", p_et), ("vectors_lr", p_lr)):
        r = {"auc": round(float(roc_auc_score(y, p)), 3), "accuracy": round(float(((p >= 0.5) == y).mean()), 3),
             "recall": round(float((p[y] >= 0.5).mean()), 3), "precision": round(float(y[p >= 0.5].mean()), 3),
             "recall_with_abstract": round(float((p[y & has_abs] >= 0.5).mean()), 3),
             "recall_without_abstract": round(float((p[y & ~has_abs] >= 0.5).mean()), 3) if (y & ~has_abs).any() else None}
        if name == "test":
            r["accuracy_by_set"] = {s: round(float(((p[sets == s] >= 0.5) == y[sets == s]).mean()), 3) for s in ("judged", "spread", "cited")}
        res[m] = r
    out[name] = res
# Kimi vs Jev, on the papers the pipeline had Jev judge
kimi = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(T.parent.parent.parent / "labels" / "frag-all.jsonl")}
both = [(kimi[k] == "on topic", m["jev"] >= 0.5) for k, m in meta.items() if m.get("jev") is not None and k in kimi]
a = np.array(both)
out["kimi_vs_jev"] = {"n": len(a), "agree": round(float((a[:, 0] == a[:, 1]).mean()), 3),
                      "kimi_yes_jev_no": int((a[:, 0] & ~a[:, 1]).sum()), "kimi_no_jev_yes": int((~a[:, 0] & a[:, 1]).sum())}
(HERE / "compare.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

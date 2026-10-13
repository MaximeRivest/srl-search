"""Does a classifier on the papers' meaning vectors, trained on Jev's labels, rank on-topic papers better than distance to
the archetypes? 5-fold cross-validation on calibrate_scope.json's sample.

    search/.venv/bin/python search-dev/experiments/topicmap/probe_scope.py search-dev/experiments/topicmap/calibrate_scope.json
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search.engine import Engine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
P = {k: Path(v) for k, v in rel["paths"].items()}
eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=8, encoder=False)
d = json.load(open(sys.argv[1]))
S = d["sample"]
pos = np.array([eng.position_of(int(s["id"][1:])) for s in S])
X = eng.vectors.get(pos); X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.array([s["jev"] >= 0.5 for s in S]); sim = np.array([s["similarity"] for s in S])
far = np.array([s["rank"] >= 10000 for s in S])
pred = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
for name, s in (("distance to archetypes", sim), ("classifier on meaning, cross-validated", pred)):
    print(f"{name:42} AUC all {roc_auc_score(y, s):.3f} | AUC beyond rank 10k {roc_auc_score(y[far], s[far]):.3f} | avg precision beyond 10k {average_precision_score(y[far], s[far]):.3f}")
print("positives", y.sum(), "beyond 10k", y[far].sum(), "of", far.sum())

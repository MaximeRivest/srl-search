"""Fit the junk model (srl_search/quality.py) on judged search results of the TRAINING questions.

    uv run python fit_quality.py --quality /mnt/fast/scholarsreadinglist/search/quality-2026-10-05

Labels: experiments/quality/judgments.jsonl, questions t* only (train_queries.jsonl). A paper is junk when
most of its judgments say spam or non_research. Every paper that also appears in the results of an
evaluation question (q*) is dropped, so the evaluation never scores papers the model was fitted on.
Prints cross-validated quality (folds grouped by question) for record-only, text-only and both, then
writes model.npz next to the features.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from srl_search.quality import FEATURES, Quality
from srl_search.vectors import Vectors

HERE = Path(__file__).resolve().parent / "experiments" / "quality"
RELEASE = Path("/mnt/fast/scholarsreadinglist/release/current")


def labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    votes, first_q = defaultdict(list), {}
    eval_docs = set()
    for line in (HERE / "judgments.jsonl").read_text().splitlines():
        d = json.loads(line)
        w = int(d["id"][1:])
        if d["qid"].startswith("q"):
            eval_docs.add(w)
            continue
        votes[w].append(d["kind"])
        first_q.setdefault(w, d["qid"])
    ws = sorted(w for w in votes if w not in eval_docs)
    y = np.array([sum(k != "research" for k in votes[w]) * 2 > len(votes[w]) for w in ws], dtype=np.int8)
    groups = np.array([int(first_q[w][1:]) for w in ws])
    return np.array(ws, dtype=np.int64), y, groups, sum(1 for w in votes if w in eval_docs)


def vectors_of(ws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index = Path(json.loads((RELEASE / "manifest.json").read_text())["paths"]["index"])
    man = json.loads((index / "manifest.json").read_text())
    work_ids = np.load(index / "work_ids.npy", mmap_mode="r")
    by_work = np.load(index / "by_work.npy", mmap_mode="r")
    sorted_ids = np.asarray(work_ids[np.asarray(by_work)])
    k = np.minimum(np.searchsorted(sorted_ids, ws), len(sorted_ids) - 1)
    ok = sorted_ids[k] == ws
    pos = np.asarray(by_work)[k[ok]]
    vec = Vectors(man["shards"]).get(pos)
    return np.asarray(vec, np.float32), ok


def cv(x, y, groups, c) -> tuple[float, float]:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(x[tr], y[tr])
        p[te] = m.predict_proba(x[te])[:, 1]
    return roc_auc_score(y, p), average_precision_score(y, p)


def main(quality: Path) -> None:
    ws, y, groups, dropped = labels()
    vec, ok = vectors_of(ws)
    ws, y, groups = ws[ok], y[ok], groups[ok]
    q = Quality(quality, model=Path("/nonexistent"))
    meta = q.features(ws)
    mean, scale = meta.mean(0), meta.std(0)
    scale[scale == 0] = 1
    xm = (meta - mean) / scale
    print(f"{len(ws):,} training papers ({y.mean():.1%} junk), {dropped:,} dropped for appearing in evaluation results")
    best = {}
    for name, x in (("record", xm), ("text", vec), ("both", np.hstack([xm, vec]))):
        for c in (0.01, 0.1, 1.0, 3.0, 10.0):
            auc, ap = cv(x, y, groups, c)
            print(f"  {name:6s} C={c:<5} ROC AUC {auc:.3f}  average precision {ap:.3f}")
            if ap > best.get(name, (0,))[0]:
                best[name] = (ap, c, auc)
    m = LogisticRegression(C=best["both"][1], max_iter=5000, class_weight="balanced").fit(np.hstack([xm, vec]), y)
    mo = LogisticRegression(C=best["record"][1], max_iter=5000, class_weight="balanced").fit(xm, y)
    nm = len(FEATURES)
    np.savez(quality / "model.npz", features=np.array(FEATURES), mean=mean, scale=scale,
             w_meta=m.coef_[0][:nm].astype(np.float32), w_vec=m.coef_[0][nm:].astype(np.float32),
             b=np.float32(m.intercept_[0]), w_meta_only=mo.coef_[0].astype(np.float32),
             b_meta_only=np.float32(mo.intercept_[0]))
    info = {"training_papers": int(len(ws)), "junk_share": float(y.mean()), "dropped_eval_overlap": dropped,
            "cv": {k: {"average_precision": round(v[0], 4), "roc_auc": round(v[2], 4), "C": v[1]} for k, v in best.items()},
            "note": "class_weight=balanced: probabilities are relative scores, not calibrated rates"}
    (quality / "model.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    for f, w in sorted(zip(FEATURES, mo.coef_[0]), key=lambda t: -abs(t[1]))[:10]:
        print(f"    record-only weight {f:20s} {w:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=Path, required=True)
    main(ap.parse_args().quality)

"""Recall of a whole-field selection (srl_search/field.py's field.json) on the studies cited by the 19 fragmentation
meta-analyses of recall_reviews.py, judged with the same hand-written question; and its precision on a random
sample of the selection, judged with that question too.

    search/.venv/bin/python search-dev/experiments/topicmap/validate_field.py FIELD_DIR   (needs TYPESAFE_API_KEY)

The selection was made with field.py's general question (no hand-written definition), so the hand-written one
is a second opinion, not the same judgment. Writes validate_field.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from calibrate_scope import jev  # noqa: E402  (the hand-written fragmentation question)
from recall_reviews import REVIEWS  # noqa: E402
from srl_search.citations import Citations  # noqa: E402
from srl_search.engine import Engine  # noqa: E402


def main(field_dir: Path):
    sel = json.loads((field_dir / "field.json").read_text())
    field = {int(w): f["via"] for w, f in sel["papers"].items()}
    rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
    P = {k: Path(v) for k, v in rel["paths"].items()}
    eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=8, encoder=False)
    cit = Citations(P["citations"])

    def state(w):
        d = eng.details([w]).get(w)
        if d:
            return {"title": d.get("title") or "", "abstract": (d.get("abstract") or "")[:3000]}, True
        nd = int(cit.nodes([w])[0])
        return {"title": cit.title(nd) if nd >= 0 else "", "abstract": ""}, False

    refs = {}
    for rw in REVIEWS:
        nd = int(cit.nodes([rw])[0])
        for r in (cit.refs(nd) if nd >= 0 else []):
            w = cit.work(int(r))
            if w not in REVIEWS:
                refs.setdefault(w, set()).add(rw)
    ids = sorted(refs)
    st = [state(w) for w in ids]
    with ThreadPoolExecutor(48) as ex:
        lab = np.array(list(ex.map(lambda s: jev(s[0]), st)))
    on = lab >= 0.5
    found = np.array([w in field for w in ids])
    has_abs = np.array([bool(s[0]["abstract"]) for s in st])
    rng = np.random.default_rng(0)

    def ci(mask, sel_):
        rv = list(REVIEWS)
        vals = []
        for _ in range(2000):
            pick = rng.choice(rv, len(rv))
            idx = [i for r in pick for i, w in enumerate(ids) if r in refs[w] and sel_[i]]
            if idx:
                vals.append(mask[idx].mean())
        return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

    years = np.array([int(cit.year[int(cit.nodes([w])[0])]) if cit.nodes([w])[0] >= 0 else 0 for w in ids])
    via = {}
    for w, f, o in zip(ids, found, on):
        if o and f:
            via[field[w]] = via.get(field[w], 0) + 1
    out = {"field_papers": len(field), "references": len(ids), "on_topic": int(on.sum()),
           "recall": round(float(found[on].mean()), 3), "ci95_by_review": ci(found, on),
           "recall_with_abstract": round(float(found[on & has_abs].mean()), 3),
           "recall_without_abstract": round(float(found[on & ~has_abs].mean()), 3) if (on & ~has_abs).any() else None,
           "recall_by_year": {k: round(float(found[on & m].mean()), 3) for k, m in
                              {"before 2000": years < 2000, "2000-2009": (years >= 2000) & (years < 2010),
                               "2010 on": years >= 2010}.items() if (on & m).any()},
           "found_via": via,
           "per_review": {REVIEWS[r]: round(float(np.mean([w in field for w in ids if r in refs[w] and on[ids.index(w)]])), 3)
                          for r in REVIEWS}}
    # precision: a random sample of the selection, judged with the hand-written question
    sample = rng.choice(sorted(field), 400, replace=False)
    with ThreadPoolExecutor(48) as ex:
        ps = np.array(list(ex.map(lambda w: jev(state(int(w))[0]), sample)))
    out["precision_sample"] = {"n": 400, "on_topic": round(float((ps >= 0.5).mean()), 3),
                               "by_route": {v: round(float((ps[[field[int(w)] == v for w in sample]] >= 0.5).mean()), 3)
                                            for v in sorted(set(field[int(w)] for w in sample))}}
    out["missed_examples"] = [{"id": f"W{w}", "year": int(y), "title": s[0]["title"][:140]}
                              for w, y, s, o, f in zip(ids, years, st, on, found) if o and not f][:40]
    (HERE / "validate_field.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "missed_examples"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

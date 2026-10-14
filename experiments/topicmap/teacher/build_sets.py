"""Papers for the teacher experiment (habitat fragmentation): a task folder for tiny-classifiers' label.py / train.py.

    search/.venv/bin/python search-dev/experiments/topicmap/teacher/build_sets.py FIELD_DIR OUT_DIR

Sets (each paper once; text = title + abstract, or the title alone):
  judged   the papers the whole-field pipeline had Jev label (archetypes, near-to-far sample, uncertain ones)
  spread   3,000 more from the 300k papers closest to the field's centre, evenly over the distance
  cited    2,500 works cited >= 3 times by the field's own papers (many have no abstract)
  refs     the references of the 19 meta-analyses (recall_reviews.REVIEWS): for testing only
Writes OUT_DIR/task.json, all.jsonl (judged + spread + cited), refs.jsonl, meta.json (set, abstract, vector, Jev).
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from recall_reviews import REVIEWS  # noqa: E402
from srl_search.engine import passes  # noqa: E402
from srl_search.field import Ctx, _details, _vectors, unit  # noqa: E402

TOPIC = "habitat fragmentation"
INSTRUCTION = (
    f"You judge scientific papers for a map of one research topic: {TOPIC}. Is {TOPIC} a central subject of the "
    "paper below (its title and abstract, or its title alone)? On topic: the paper studies the topic itself as a main "
    "subject: the phenomenon, its causes, consequences, mechanisms, measurement, methods, or how it is managed, in "
    "any system or organism. Not on topic: the topic is absent or only mentioned in passing; or the words mean "
    "something else in this paper; or the paper studies a neighbouring subject without studying the topic itself.")
LABELS = ["on topic", "not on topic"]


def text(d: dict | None, title_only: str = "") -> tuple[str, bool]:
    import re
    if d:
        t = re.sub(r"<[^>]+>", "", d.get("title") or "")
        a = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", d.get("abstract") or "")).strip()
        if a:
            return f"Title: {t}\nAbstract: {a[:1500]}", True
        return f"Title: {t}\n(no abstract)", False
    return f"Title: {title_only}\n(no abstract)", False


def main(field_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((field_dir / "field.json").read_text())
    ctx = Ctx()
    eng, cit = ctx.engine, ctx.cit
    rng = np.random.default_rng(0)
    jev = {int(w): p for w, p in sel["labels"].items()}
    chosen: dict[int, str] = {w: "judged" for w in jev}

    # spread: around the centre of the on-topic judged papers
    pos = np.array([w for w, p in jev.items() if p >= 0.5], np.int64)
    V, h = _vectors(ctx, pos)
    c = unit(V[h].mean(0, keepdims=True))
    faiss.omp_set_num_threads(16)
    _, rows = eng.index.search(c.astype(np.float32), 300_000, params=faiss.SearchParametersIVF(nprobe=2048))
    rows = rows[0][rows[0] >= 0]
    ws = np.asarray(eng.work_ids[rows], dtype=np.int64)
    sims = unit(eng.vectors.get(rows)) @ c[0]
    ws = ws[np.argsort(-sims)]
    picks = [int(ws[i]) for i in np.linspace(0, len(ws) - 1, 4500).astype(int)]
    det = _details(ctx, picks)
    picks = [w for w in picks if w in det and passes(det[w], {}) and w not in chosen][:3000]
    chosen.update({w: "spread" for w in picks})

    # cited: works cited >= 3 times by the field papers found by meaning, phrase or judgment
    core = [int(w) for w, f in sel["papers"].items() if not f["via"].startswith("cited")]
    nodes = cit.nodes(np.array(core, np.int64))
    allrefs = np.concatenate([cit.refs(int(x)) for x in nodes[nodes >= 0]])
    u, cnt = np.unique(allrefs, return_counts=True)
    cand = [cit.work(int(x)) for x in u[cnt >= 3]]
    cand = [w for w in cand if w not in chosen]
    pick = [cand[i] for i in rng.choice(len(cand), 2500, replace=False)]
    chosen.update({w: "cited" for w in pick})

    refs = set()
    for rw in REVIEWS:
        nd = int(cit.nodes([rw])[0])
        refs |= {cit.work(int(r)) for r in (cit.refs(nd) if nd >= 0 else [])}
    refs -= set(REVIEWS)

    def rows_of(ids, kind_of):
        det = _details(ctx, ids)
        _, hv = _vectors(ctx, ids)
        rs, meta = [], {}
        for w, has_v in zip(ids, hv):
            nd = int(cit.nodes([w])[0])
            t, has_abs = text(det.get(w), cit.title(nd) if nd >= 0 else "")
            if len(t) < 20:
                continue
            rs.append({"id": f"W{w}", "text": t})
            meta[f"W{w}"] = {"set": kind_of(w), "abstract": has_abs, "vector": bool(has_v),
                             "jev": jev.get(w), "in_field": str(w) in sel["papers"]}
        return rs, meta

    all_rows, meta = rows_of(sorted(chosen), lambda w: chosen[w])
    ref_rows, meta_r = rows_of(sorted(refs), lambda w: "refs")
    rng.shuffle(all_rows)
    (out / "task.json").write_text(json.dumps({"name": "frag", "instruction": INSTRUCTION, "labels": LABELS}, indent=1))
    for name, rs in (("all", all_rows), ("refs", ref_rows)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({**meta, **{k: v for k, v in meta_r.items() if k not in meta}}))
    print({s: sum(1 for m in meta.values() if m["set"] == s) for s in ("judged", "spread", "cited")},
          "refs", len(ref_rows), "without abstract:", sum(1 for m in meta.values() if not m["abstract"]),
          sum(1 for m in meta_r.values() if not m["abstract"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

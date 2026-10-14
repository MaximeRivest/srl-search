"""More papers for the teacher experiment: 12,000 more for Kimi K3 to label (more.jsonl), and 150,000 unlabelled
ones (pool.jsonl) for students to label (the staircase). Same mix as build_sets.py: spread over the 300k papers
closest to the field's centre, and works cited >= 3 times by the field; none already in all.jsonl or refs.jsonl.

    search/.venv/bin/python search-dev/experiments/topicmap/teacher/build_more.py FIELD_DIR TASK_DIR
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from build_sets import text  # noqa: E402
from srl_search.engine import passes  # noqa: E402
from srl_search.field import Ctx, _details, _vectors, unit  # noqa: E402


def main(field_dir: Path, T: Path):
    sel = json.loads((field_dir / "field.json").read_text())
    have = {json.loads(l)["id"] for n in ("all", "refs") for l in open(T / f"{n}.jsonl")}
    ctx = Ctx()
    eng, cit = ctx.engine, ctx.cit
    rng = np.random.default_rng(1)
    jev = {int(w): p for w, p in sel["labels"].items()}
    pos = np.array([w for w, p in jev.items() if p >= 0.5], np.int64)
    V, h = _vectors(ctx, pos)
    c = unit(V[h].mean(0, keepdims=True))
    faiss.omp_set_num_threads(16)
    _, rows = eng.index.search(c.astype(np.float32), 300_000, params=faiss.SearchParametersIVF(nprobe=2048))
    near = [int(w) for w in np.asarray(eng.work_ids[rows[0][rows[0] >= 0]], dtype=np.int64) if f"W{int(w)}" not in have]
    near_set = set(near)
    core = [int(w) for w, f in sel["papers"].items() if not f["via"].startswith("cited")]
    nodes = cit.nodes(np.array(core, np.int64))
    u, cnt = np.unique(np.concatenate([cit.refs(int(x)) for x in nodes[nodes >= 0]]), return_counts=True)
    cited = [cit.work(int(x)) for x in u[cnt >= 3]]
    cited = [w for w in cited if f"W{w}" not in have and w not in near_set]
    a = list(rng.permutation(near))
    b = list(rng.permutation(cited))
    plan = {"more": a[:7000] + b[:5000], "pool": a[7000:7000 + 100_000] + b[5000:5000 + 50_000]}
    meta = {}
    for name, ids in plan.items():
        ids = [int(x) for x in ids]
        out = []
        for s in range(0, len(ids), 20000):
            chunk = ids[s:s + 20000]
            det = _details(ctx, chunk)
            _, hv = _vectors(ctx, chunk)
            for w, has_v in zip(chunk, hv):
                d = det.get(w)
                if d is not None and not passes(d, {}):
                    continue
                nd = int(cit.nodes([w])[0])
                t, has_abs = text(d, cit.title(nd) if nd >= 0 else "")
                if len(t) < 20:
                    continue
                out.append({"id": f"W{w}", "text": t})
                meta[f"W{w}"] = {"set": "spread" if w in near_set else "cited", "abstract": has_abs, "vector": bool(has_v)}
        with open(T / f"{name}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(out), "without abstract", sum(1 for r in out if not meta[r["id"]]["abstract"]), flush=True)
    m = json.loads((T / "meta.json").read_text())
    m.update({k: v for k, v in meta.items() if k not in m})
    (T / "meta.json").write_text(json.dumps(m))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

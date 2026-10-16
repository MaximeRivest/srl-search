"""Step 9b: the reading-list rules as shipped (topics.Options.lens_min_score, field.LENS_MIN_SCORE) on the round-2
replay maps (fields-r4, 0.8B arm): Kimi on every list paper (cached labels from listgate/readinglists reused)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import field as F, judges as J, topics as T
from srl_search.topic_service import finish
import listgate as L                                     # its maps' selection, texts and labels (cached: no GPU)

out = {}
for t in L.TOPICS:
    a = "new08-only"
    Z = np.load(L.R4 / f"{t}-{a}" / "work" / "scores.npz")
    ids, final, cut = Z["ids"], Z["final"], float(Z["cut"])
    yes = np.flatnonzero(final >= cut)
    ws = np.array(sorted(int(w) for w in ids[yes][L.publications(ids[yes])]), np.int64)
    sel = {"topic": L.TOPICS[t][0], "definition": L.TOPICS[t][1], "made": F._now(), "release": L.ctx.release["id"],
           "report": {}, "papers": {str(int(w)): {"p": round(float(p), 3)} for w, p in zip(ids, final) if int(w) in set(ws.tolist())}}
    f = L.WORK / f"{t}-{a}-shipped.json"
    if not f.exists():
        det = F._details(L.ctx, ws); V, has = F._vectors(L.ctx, ws)
        details = [det.get(int(w)) or {"title": L.cit.title(int(L.cit.nodes([int(w)])[0]))} for w in ws]
        seed = T.Seed(ws, V, has, details, np.array([sel["papers"][str(int(w))]["p"] for w in ws], np.float32))
        seed = seed.subset(T.collapse_versions(seed, L.cit))
        T.PARALLEL_MIN = 1500
        raw = T.build(seed, L.cit, sel["topic"], T.Options(refs_top=30, workers=F.LEIDEN_WORKERS, lens_min_score=F.LENS_MIN_SCORE),
                      background=L.ctx.keywords.phrase_counts if L.ctx.keywords is not None else None)
        f.write_text(json.dumps(finish(L.ctx, L.cit, raw, seed, L.ctx.release["id"])))
    m = json.loads(f.read_text())
    rows = [(n, p) for g in m["groups"] for n in L.LISTS + ("foundations",) for p in (g["lenses"].get(n) or [])]
    lab = L.labels[t]
    new = sorted({int(p["id"][1:]) for _, p in rows} - set(lab))
    if new:
        y = J.Teacher(3.0).label(L.text_of(new), *L.TOPICS[t])
        lab.update(zip(new, y.tolist()))
    on = lambda n: [lab[int(p["id"][1:])] for nn, p in rows if nn == n and lab.get(int(p["id"][1:])) == lab.get(int(p["id"][1:]))]
    lists = [v for n in L.LISTS for v in on(n)]
    out[t] = {"on_topic": round(float(np.mean(lists)), 3), "shown": len(lists), "new_kimi": len(new),
              "by_list": {n: round(float(np.mean(on(n))), 3) if on(n) else None for n in L.LISTS + ("foundations",)},
              "classics_in_lists": len({int(p["id"][1:]) for n, p in rows if n != "foundations"} & L.classics[t])}
    print(t, json.dumps(out[t]), flush=True)
    for g in m["groups"]:
        print("  ", g["name"][:40], "| core:", " / ".join(p["title"][:45] for p in g["lenses"]["core"][:3]),
              "| foundations:", " / ".join(p["title"][:40] for p in (g["lenses"].get("foundations") or [])[:3]), flush=True)
L.WORK.joinpath("kimi.json").write_text(json.dumps({t: {str(k): v for k, v in d.items()} for t, d in L.labels.items()}))
(Path(__file__).parent / "listcheck.json").write_text(json.dumps(out, indent=1))

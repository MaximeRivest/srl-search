"""Recall of "the whole field" methods, measured on studies chosen by people: the reference lists of meta-analyses and
systematic reviews of habitat fragmentation (not among the archetypes). Their authors searched the literature with
explicit rules and listed the studies they used; our methods never saw them.

    search/.venv/bin/python search-dev/experiments/topicmap/recall_reviews.py   (needs TYPESAFE_API_KEY)

Each reference is labelled by Jev with calibrate_scope.py's question (on topic = a fragmentation study, not
background reading such as statistics methods). Methods compared on the on-topic references:
  phrases       exact fragmentation phrases (calibrate_scope.KEYWORD_Q, 36.5k papers)
  near-K        the K papers closest in meaning to the archetypes' centroid
  classifier    logistic regression on meaning vectors, trained on calibrate_scope.json's 850 Jev labels; a paper is
                in when P >= t and it is in the candidate pool (300k nearest the centroid + 300k nearest the
                classifier's direction), a publication and not junk. Also "anywhere": P >= t without the pool.
  union         classifier (0.5) or phrases
References in the classifier's training sample, the archetypes and the reviews themselves are left out.
Writes recall_reviews.json.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from calibrate_scope import ARCHETYPES, KEYWORD_Q, jev  # noqa: E402
from srl_search.citations import Citations  # noqa: E402
from srl_search.engine import Engine, passes  # noqa: E402
from srl_search.keyword import KeywordIndex  # noqa: E402
from srl_search.quality import Quality  # noqa: E402

REVIEWS = {
    2149427681: "plant reproduction (Aguilar 2006)", 4414026499: "insect pollinators (2025)",
    2582456127: "insect herbivores and herbivory (2017)", 2119776782: "herbivorous insects (2014)",
    4200024369: "seagrass edges and fragmentation (2021)", 2898210967: "genetic effects (2018)",
    1905487223: "woody plants, genetic diversity (2011)", 2114975481: "nest predators (2002)",
    1811969986: "interspecific interactions (2014)", 4387764691: "mutualisms and antagonisms (2023)",
    3154832160: "herpetofauna of Brazil (2021)", 2315712689: "mammal life histories (2014)",
    2112037272: "bird reproductive traits (2011)", 2141365420: "patch size effect (Bender 1998)",
    1482764309: "matrix composition (Prevedello 2010)", 2104734141: "climate x habitat loss (Mantyka-Pringle 2011)",
    2026695720: "wetland vertebrates, habitat loss (2014)", 1977549989: "trophic disruption (2014)",
    4410144033: "dam-induced river fragmentation (2025)",
}
K_POOL = 300_000
THRESHOLDS = (0.3, 0.5, 0.7)


def main():
    T = {}
    t0 = time.perf_counter()
    rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
    P = {k: Path(v) for k, v in rel["paths"].items()}
    eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=16, encoder=False,
                 quality=Quality(P["corpus"].parent / "quality-2026-10-05"))
    cit = Citations(P["citations"])
    kw = KeywordIndex([P["corpus"] / "keywords"])
    faiss.omp_set_num_threads(16)
    T["load_s"] = round(time.perf_counter() - t0, 1)

    def unit(V):
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    def vec(ws):
        pos = [eng.position_of(int(w)) for w in ws]
        has = np.array([p is not None for p in pos])
        V = np.zeros((len(ws), 768), np.float32)
        if has.any():
            V[has] = unit(eng.vectors.get(np.array([p for p in pos if p is not None], np.int64)))
        return V, has

    # the classifier, trained on the calibration sample's Jev labels
    cal = json.loads((HERE / "calibrate_scope.json").read_text())
    train_ids = [int(s["id"][1:]) for s in cal["sample"]]
    Xtr, has_tr = vec(train_ids)
    ytr = np.array([s["jev"] >= 0.5 for s in cal["sample"]])
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xtr[has_tr], ytr[has_tr])
    w, b = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    prob = lambda X: 1 / (1 + np.exp(-(X @ w + b)))

    # candidate pool: nearest the archetypes' centroid, and nearest the classifier's direction
    t = time.perf_counter()
    A, _ = vec(list(ARCHETYPES))
    c = unit(A.mean(0, keepdims=True))[0]
    _, ca = eng.index.search(c.reshape(1, -1), K_POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    _, cb = eng.index.search(unit(w.reshape(1, -1)), K_POOL, params=faiss.SearchParametersIVF(nprobe=2048))
    rows_a, rows_b = ca[0][ca[0] >= 0], cb[0][cb[0] >= 0]
    rows = np.union1d(rows_a, rows_b)
    V = unit(eng.vectors.get(rows))
    pool_w = np.asarray(eng.work_ids[rows], dtype=np.int64)
    pool_p = prob(V)
    pool_junk = eng.quality.junk(pool_w, V)
    sim_a = np.sort(unit(eng.vectors.get(rows_a)) @ c)[::-1]          # centroid similarities, best first
    del V
    T["pool_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    pub = np.zeros(len(pool_w), bool)
    for s in range(0, len(pool_w), 5000):
        d = eng.details(pool_w[s:s + 5000].tolist())
        pub[s:s + 5000] = [d.get(int(x)) is not None and passes(d[int(x)], {}) for x in pool_w[s:s + 5000]]
    T["pool_details_s"] = round(time.perf_counter() - t, 1)
    ok_pool = pub & (pool_junk < 0.5)
    sizes = {str(th): int((ok_pool & (pool_p >= th)).sum()) for th in THRESHOLDS}
    pool_set = set(pool_w[ok_pool].tolist())

    hits, matches, _ = kw.search(KEYWORD_Q, 60000, {}, "all")
    phrase = {h[0] for h in hits}

    # the reviews' references
    skip = set(ARCHETYPES) | set(train_ids) | set(REVIEWS)
    per_review, refs = {}, {}
    for rw, label in REVIEWS.items():
        nd = int(cit.nodes([rw])[0])
        ws = [cit.work(int(r)) for r in cit.refs(nd)] if nd >= 0 else []
        per_review[rw] = {"label": label, "references_indexed": len(ws)}
        for x in ws:
            if x not in skip:
                refs.setdefault(x, set()).add(rw)
    ref_ids = sorted(refs)
    det = {}
    for s in range(0, len(ref_ids), 5000):
        det.update(eng.details(ref_ids[s:s + 5000]))
    Xr, has_r = vec(ref_ids)
    pr = np.where(has_r, prob(Xr), 0.0)
    jr = np.where(has_r, eng.quality.junk(np.array(ref_ids, np.int64), Xr), 1.0)
    sim_r = np.where(has_r, Xr @ c, -1.0)
    nodes = cit.nodes(ref_ids)

    def state(x, nd):
        d = det.get(x) or {}
        title = d.get("title") or (cit.title(int(nd)) if nd >= 0 else "")
        return {"title": title or "", "abstract": (d.get("abstract") or "")[:3000]}
    t = time.perf_counter()
    with ThreadPoolExecutor(24) as ex:
        lab = np.array(list(ex.map(jev, [state(x, nd) for x, nd in zip(ref_ids, nodes)])))
    T["jev_s"] = round(time.perf_counter() - t, 1)
    year = np.array([(det.get(x) or {}).get("year") or (int(cit.year[nd]) if nd >= 0 else 0) for x, nd in zip(ref_ids, nodes)])
    is_pub = np.array([x in det and passes(det[x], {}) for x in ref_ids])

    member = {
        "phrases": np.array([x in phrase for x in ref_ids]),
        **{f"near-{k // 1000}k": has_r & (sim_r >= sim_a[min(k, len(sim_a)) - 1]) for k in (10_000, 30_000, 100_000, 300_000)},
        **{f"classifier>={th}": np.array([x in pool_set for x in ref_ids]) & (pr >= th) for th in THRESHOLDS},
        "classifier>=0.5 anywhere": has_r & (pr >= 0.5) & is_pub & (jr < 0.5),
    }
    member["union: classifier>=0.5 or phrases"] = member["classifier>=0.5"] | member["phrases"]
    # citation expansion: works cited by >= k papers of the found field (classifier>=0.5 or phrases), the reviews
    # themselves never counted as citers. Reaches studies without an abstract, which no text search can see.
    t = time.perf_counter()
    field = (set(pool_w[ok_pool & (pool_p >= 0.5)].tolist()) | phrase) - set(REVIEWS)
    fn = cit.nodes(np.array(sorted(field), np.int64))
    fn = fn[fn >= 0]
    allrefs = np.concatenate([cit.refs(int(x)) for x in fn]) if len(fn) else np.zeros(0, np.uint32)
    u, cnt = np.unique(allrefs, return_counts=True)
    ref_nodes = np.array([int(x) for x in nodes])
    pos = np.searchsorted(u, ref_nodes)
    pos_c = np.minimum(pos, len(u) - 1)
    cited_by = np.where((ref_nodes >= 0) & (u[pos_c] == ref_nodes), cnt[pos_c], 0)
    expansion_sizes = {}
    for k in (2, 3, 5, 10):
        member[f"union + cited by >={k} field papers"] = member["union: classifier>=0.5 or phrases"] | (cited_by >= k)
        expansion_sizes[str(k)] = int((cnt >= k).sum())
    T["expansion_s"] = round(time.perf_counter() - t, 1)
    on = lab >= 0.5

    def recall(mask, sel):
        return round(float(mask[sel].mean()), 3) if sel.any() else None

    def boot(mask, sel, n=2000):      # resample reviews (clusters), not single references
        rng = np.random.default_rng(0)
        rv = list(REVIEWS)
        vals = []
        for _ in range(n):
            pick = rng.choice(rv, len(rv))
            idx = np.concatenate([[i for i, x in enumerate(ref_ids) if r in refs[x] and sel[i]] for r in pick]).astype(int)
            if len(idx):
                vals.append(mask[idx].mean())
        return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]

    bins = {"before 2000": year < 2000, "2000-2009": (year >= 2000) & (year < 2010), "2010 on": year >= 2010}
    res = {}
    for name, m in member.items():
        res[name] = {"recall_on_topic": recall(m, on), "ci95_by_review": boot(m, on),
                     "recall_on_topic_with_abstract_vector": recall(m, on & has_r),
                     "recall_all_publication_references": recall(m, is_pub),
                     "by_year": {k: recall(m, on & v) for k, v in bins.items()}}
    for rw in REVIEWS:
        sel = on & np.array([rw in refs[x] for x in ref_ids])
        per_review[rw].update({"on_topic": int(sel.sum()),
                               **{n: recall(member[n], sel) for n in ("phrases", "near-300k", "classifier>=0.5",
                                                                     "union: classifier>=0.5 or phrases")}})
    missed = [{"id": f"W{x}", "year": int(year[i]), "jev": round(float(lab[i]), 2), "p": round(float(pr[i]), 3),
               "has_vector": bool(has_r[i]), "in_phrases": bool(member["phrases"][i]),
               "title": state(x, nodes[i])["title"][:160]}
              for i, x in enumerate(ref_ids) if on[i] and not member["union + cited by >=3 field papers"][i]]
    out = {"reviews": per_review, "references": len(ref_ids), "on_topic": int(on.sum()),
           "on_topic_with_vector": int((on & has_r).sum()), "publication_references": int(is_pub.sum()),
           "phrase_matches": matches, "pool": int(len(pool_w)), "classifier_output_size": sizes,
           "field_set": len(field), "expansion_added_size": expansion_sizes,
           "near_k_similarity": {k: round(float(sim_a[k - 1]), 4) for k in (10_000, 30_000, 100_000, 300_000)},
           "recall": res, "missed_by_union": missed, "timing": T}
    (HERE / "recall_reviews.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("missed_by_union",)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""How far does a field reach? Archetype papers -> everything by closeness in meaning -> a judge (Jev) labels a
stratified sample along the distance -> a calibrated curve P(on topic | similarity) -> a principled cut-off, the
expected number of on-topic papers inside and beyond it, and what exact-word matching misses.

    search/.venv/bin/python search-dev/experiments/topicmap/calibrate_scope.py   (needs TYPESAFE_API_KEY)

Read-only on the release (its own process). Writes calibrate_scope.json (every sampled paper with its label).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from srl_search.engine import Engine, passes  # noqa: E402
from srl_search.keyword import KeywordIndex  # noqa: E402
from srl_search.quality import Quality  # noqa: E402

TOPIC = "habitat fragmentation"
ARCHETYPES = {  # W-numbers looked up in the corpus (the classics of the field)
    2149752076: "Fahrig 2003, Effects of habitat fragmentation on biodiversity",
    2127597920: "Haddad 2015, Habitat fragmentation and its lasting impact on Earth's ecosystems",
    2616962235: "Fahrig 2017, Ecological responses to habitat fragmentation per se",
    2007749020: "Turner 1996, Species loss in fragments of tropical rain forest",
    2129775337: "Saunders 1991, Biological consequences of ecosystem fragmentation",
    2000779440: "Andren 1994, Effects of habitat fragmentation on birds and mammals",
    2109263076: "Ewers & Didham 2006, Confounding factors in species responses to habitat fragmentation",
    3003011554: "Watling 2020, Support for the habitat amount hypothesis",
}
KEYWORD_Q = ('"habitat fragmentation" OR "forest fragmentation" OR "landscape fragmentation" OR "fragmented landscape" '
             'OR "fragmented landscapes" OR "habitat fragments" OR "forest fragments"')
K = 300_000                       # candidates by closeness
EDGES = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 120000, 160000,
         220000, K]
PER_BUCKET = 50
QUESTION = {
    "type": "noul",
    "instructions": "Is habitat fragmentation a central subject of the paper described in `title` and `abstract`?",
    "criteria": {
        "true": "The paper studies habitat fragmentation or its consequences as a main subject: natural habitat broken "
                "into patches or remnants; patch size, isolation or edge effects; connectivity or corridors between "
                "habitat patches; how species, populations, genes, interactions or ecosystems respond in fragmented "
                "landscapes or habitat fragments; measuring or mapping fragmentation; conservation or planning for "
                "fragmented habitat.",
        "false": "Fragmentation of habitat is absent or only mentioned in passing; or 'fragmentation' means something "
                 "else (DNA, cells, rocks, markets, software, social groups); or the paper is about land use, habitat "
                 "loss, climate or species ecology without studying habitat being broken into pieces.",
    },
}


def jev(state: dict) -> float:
    body = json.dumps({"state": state, "model": "jev-1.13.0", "questions": {"q": QUESTION}}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=body, headers={
                "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return float(json.load(r)["answers"]["q"]["noul"])
        except Exception as e:  # 429 / transient
            time.sleep(2 ** attempt)
            last = e
    raise last


def isotonic(x, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)


def main():
    T = {}
    rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
    P = {k: Path(v) for k, v in rel["paths"].items()}
    t = time.perf_counter()
    eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=16,
                 quality=Quality(P["corpus"].parent / "quality-2026-10-05"))
    kw = KeywordIndex([P["corpus"] / "keywords"])
    T["load_s"] = round(time.perf_counter() - t, 1)

    # 1. archetypes -> centroid
    pos = [eng.position_of(w) for w in ARCHETYPES]
    missing = [w for w, p in zip(ARCHETYPES, pos) if p is None]
    S = eng.vectors.get(np.array([p for p in pos if p is not None], np.int64))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    c = S.mean(0)
    c /= np.linalg.norm(c)

    # 2. everything close, exact similarity
    t = time.perf_counter()
    faiss.omp_set_num_threads(16)
    _, cand = eng.index.search(c.reshape(1, -1).astype(np.float32), K, params=faiss.SearchParametersIVF(nprobe=2048))
    cand = cand[0][cand[0] >= 0]
    T["ivf_s"] = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    V = eng.vectors.get(cand)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ c
    maxsim = (V @ S.T).max(1)
    wids = np.asarray(eng.work_ids[cand], dtype=np.int64)
    junk = eng.quality.junk(wids, V)
    T["rerank_s"] = round(time.perf_counter() - t, 1)

    # 3. publications only, not junk (details for every candidate: the production path)
    t = time.perf_counter()
    keep = np.zeros(len(wids), bool)
    det = {}
    for s in range(0, len(wids), 5000):
        d = eng.details(wids[s:s + 5000].tolist())
        det.update({w: {"title": x.get("title"), "abstract": x.get("abstract"), "year": x.get("year"),
                        "type": x.get("type")} for w, x in d.items()})
        for i in range(s, min(s + 5000, len(wids))):
            x = d.get(int(wids[i]))
            keep[i] = x is not None and junk[i] < 0.5 and passes(x, {})
    T["details_s"] = round(time.perf_counter() - t, 1)
    order = np.argsort(-sim[keep], kind="stable")
    idx = np.flatnonzero(keep)[order]
    wids_k, sim_k, max_k = wids[idx], sim[idx], maxsim[idx]
    n = len(wids_k)

    # 4. exact-word set
    t = time.perf_counter()
    hits, matches, _ = kw.search(KEYWORD_Q, 60000, {}, "all")
    kwset = {h[0] for h in hits}
    T["keyword_s"] = round(time.perf_counter() - t, 1)
    in_kw = np.array([int(w) in kwset for w in wids_k])

    # 5. stratified sample along rank, labelled by Jev
    rng = np.random.default_rng(0)
    edges = [e for e in EDGES if e < n] + [n]
    sample = []
    for a, b in zip(edges, edges[1:]):
        pick = rng.choice(np.arange(a, b), size=min(PER_BUCKET, b - a), replace=False)
        sample += [(int(i), a, b) for i in sorted(pick)]
    states = [{"title": det[int(wids_k[i])]["title"] or "", "abstract": (det[int(wids_k[i])]["abstract"] or "")[:3000]}
              for i, _, _ in sample]
    t = time.perf_counter()
    with ThreadPoolExecutor(24) as ex:
        labels = list(ex.map(jev, states))
    T["jev_s"] = round(time.perf_counter() - t, 1)
    y = np.array(labels)
    xs = np.array([sim_k[i] for i, _, _ in sample])
    xm = np.array([max_k[i] for i, _, _ in sample])

    # 6. calibrated curve P(on topic | similarity), estimates of what lies where
    from sklearn.metrics import roc_auc_score
    hard = y >= 0.5
    auc_c, auc_m = roc_auc_score(hard, xs), roc_auc_score(hard, xm)
    f = isotonic(xs, y)
    p_all = f.predict(sim_k)               # P(on topic) for every candidate, from its similarity
    total = float(p_all.sum())
    tail_p = float(y[[j for j, (_, a, b) in enumerate(sample) if a == edges[-2]]].mean())
    cut = {}
    for target in (0.9, 0.5, 0.25):
        above = np.flatnonzero(p_all >= target)
        k = int(above.max()) + 1 if len(above) else 0
        inside = float(p_all[:k].sum())
        cut[str(target)] = {"rank": k, "similarity": round(float(sim_k[k - 1]), 4) if k else None,
                            "expected_on_topic_inside": round(inside), "precision_inside": round(inside / max(k, 1), 3),
                            "recall_of_retrieved": round(inside / total, 3)}
    # exact words vs meaning, per bucket: share of on-topic papers the phrase search misses
    buckets = []
    for a, b in zip(edges, edges[1:]):
        js = [j for j, (i, aa, _) in enumerate(sample) if aa == a]
        on = [j for j in js if y[j] >= 0.5]
        buckets.append({"ranks": [a, b], "similarity": [round(float(sim_k[a]), 4), round(float(sim_k[b - 1]), 4)],
                        "mean_p": round(float(y[js].mean()), 3), "on_topic_share": round(len(on) / len(js), 3),
                        "on_topic_not_in_keyword_set": round(sum(1 for j in on if not in_kw[sample[j][0]]) / max(len(on), 1), 3),
                        "keyword_share": round(float(in_kw[a:b].mean()), 3),
                        "expected_on_topic": round(float(np.mean(y[js]) * (b - a)))})
    est_missed_by_words = sum(bk["expected_on_topic"] * bk["on_topic_not_in_keyword_set"] for bk in buckets)
    out = {"topic": TOPIC, "archetypes": ARCHETYPES, "archetypes_missing_vectors": missing, "candidates": int(len(wids)),
           "publications_kept": n, "keyword_matches": matches, "keyword_set_in_candidates": int(in_kw.sum()),
           "auc_centroid": round(auc_c, 3), "auc_max_archetype": round(auc_m, 3),
           "expected_on_topic_in_candidates": round(total), "p_at_last_bucket": round(tail_p, 3),
           "cutoffs": cut, "expected_on_topic_missed_by_exact_words": round(est_missed_by_words),
           "buckets": buckets, "timing": T,
           "sample": [{"id": f"W{int(wids_k[i])}", "rank": i, "similarity": round(float(sim_k[i]), 4),
                       "in_keyword_set": bool(in_kw[i]), "jev": round(float(y[j]), 3),
                       "title": det[int(wids_k[i])]["title"], "abstract": (det[int(wids_k[i])]["abstract"] or "")[:600]}
                      for j, (i, _, _) in enumerate(sample)]}
    path = Path(__file__).parent / "calibrate_scope.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "sample"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

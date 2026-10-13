"""Topic map prototype: Scimeetr's idea on today's data.

    PYTHONPATH=/tmp/tmlibs search/.venv/bin/python search-dev/experiments/topicmap/prototype.py N "question" ["question" ...]

A question -> the N papers closest in meaning -> groups of papers that share references and wording
-> a name for each group -> reading lists per group, through several lenses.

Throwaway measurement script: it loads the live release read-only, in its own process, and changes nothing.
"""
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import igraph
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from srl_search.engine import Engine  # noqa: E402
from srl_search.quality import Quality  # noqa: E402

REL = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
P = {k: Path(v) for k, v in REL["paths"].items()}
CIT = P["citations"]
YEAR_NOW = 2026
T = {}


def tick(name, t0):
    T[name] = round((time.perf_counter() - t0) * 1000)
    return time.perf_counter()


# ------------------------------------------------------------------ citation index (memory-mapped, read-only)
wn = np.load(CIT / "wn.npy", mmap_mode="r")
fwd_off = np.load(CIT / "fwd_off.npy", mmap_mode="r")
fwd = np.load(CIT / "fwd.npy", mmap_mode="r")
c_year = np.load(CIT / "year.npy", mmap_mode="r")
c_cites = np.load(CIT / "cites.npy", mmap_mode="r")
c_kind = np.load(CIT / "kind.npy", mmap_mode="r")
title_off = np.load(CIT / "title_off.npy", mmap_mode="r")
title_row = np.load(CIT / "title_row.npy", mmap_mode="r")
titles = open(CIT / "titles.bin", "rb")
KINDS = json.loads((CIT / "meta.json").read_text())["types"]


def node_of(w):
    i = int(np.searchsorted(wn, w))
    return i if i < len(wn) and wn[i] == w else -1


def title_of(node):
    r = int(title_row[node])
    titles.seek(int(title_off[r]))
    return titles.read(int(title_off[r + 1] - title_off[r])).decode("utf-8", "replace")


def load():
    t = time.perf_counter()
    quality = Quality(P["corpus"].parent / "quality-2026-10-05")
    eng = Engine(P["index"], P["corpus"] / "works.sqlite", threads=8, quality=quality)
    print(f"engine loaded in {time.perf_counter() - t:.1f}s", file=sys.stderr)
    return eng, quality


def main(eng, quality, question, n_seed=1000):
    T.clear()

    # 1. the papers closest in meaning (junk left out, publications only)
    t = time.perf_counter()
    q = eng.encoder.encode(question)[0]
    import faiss
    params = faiss.SearchParametersIVF(nprobe=512)
    _, cand = eng.index.search(q.reshape(1, -1), n_seed * 4, params=params)
    cand = cand[0][cand[0] >= 0]
    vecs = eng.vectors.get(cand)
    sims = vecs @ q
    wids = np.asarray(eng.work_ids[cand], dtype=np.int64)
    junk = quality.junk(wids, vecs)
    order = np.argsort(-(sims - 0.3 * junk))
    keep = [i for i in order if junk[i] < 0.5][: n_seed * 2]
    det = eng.details([int(wids[i]) for i in keep])
    keep = [i for i in keep if int(wids[i]) in det and det[int(wids[i])]["type"] in
            {"article", "review", "preprint", "conference-paper", "book-chapter", "book", "dissertation"}][:n_seed]
    seen, dedup = set(), []
    for i in keep:   # keep is best-first; a later copy of the same title is the preprint/duplicate
        key = re.sub(r"[^a-z0-9]", "", (det[int(wids[i])]["title"] or "").lower())[:120]
        if key and key in seen:
            continue
        seen.add(key)
        dedup.append(i)
    keep = dedup
    wids, vecs, sims = wids[keep], vecs[keep], sims[keep]
    n = len(wids)
    t = tick("1 search+details", t)

    # 2. references of each paper (citation index), as a sparse paper x reference matrix
    nodes = np.searchsorted(wn, wids)          # one vectorized binary search over the mmap
    nodes[nodes >= len(wn)] = 0
    nodes = np.where(np.asarray(wn[nodes]) == wids, nodes, -1)
    t = tick("2a node ids", t)
    rows, cols = [], []
    for i, nd in enumerate(nodes):
        if nd >= 0:
            r = np.asarray(fwd[int(fwd_off[nd]):int(fwd_off[nd + 1])])
            rows.append(np.full(len(r), i))
            cols.append(r)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    uref, cidx = np.unique(cols, return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cidx)), shape=(n, len(uref)))
    nref = np.asarray(A.sum(1)).ravel()
    t = tick("2 references", t)

    # 3. two similarities, then one graph:
    #    bibliographic coupling (shared references, Salton cosine) + meaning (k nearest by embedding cosine)
    C = (A @ A.T).tocoo()
    m = C.row < C.col
    bc_r, bc_c = C.row[m], C.col[m]
    bc_w = C.data[m] / np.sqrt(nref[bc_r] * nref[bc_c])
    S = vecs @ vecs.T
    np.fill_diagonal(S, -1)
    K = 10
    nn = np.argpartition(-S, K, axis=1)[:, :K]
    tx_r = np.repeat(np.arange(n), K)
    tx_c = nn.ravel()
    tx_s = S[tx_r, tx_c]
    floor = np.median(tx_s) - 0.1   # similarity relative to this set: cosine values are all high inside a topic
    tx_w = np.clip((tx_s - floor) / (1 - floor), 0, None)
    # equal total weight for each source, so neither logic swamps the other (Boyack & Klavans hybrids)
    bc_w = bc_w / bc_w.sum()
    tx_w = tx_w / tx_w.sum()
    W = sparse.coo_matrix((np.concatenate([bc_w, tx_w]), (np.concatenate([bc_r, tx_r]), np.concatenate([bc_c, tx_c]))),
                          shape=(n, n)).tocsr()
    W = (W + W.T).tocoo()
    m = W.row < W.col
    g = igraph.Graph(n=n, edges=list(zip(W.row[m].tolist(), W.col[m].tolist())), edge_attrs={"w": W.data[m]})
    t = tick("3 graph", t)

    # 4. Leiden (modularity), seeded; small groups folded into "other"
    best = None
    for res in (0.6, 0.8, 1.0, 1.2):
        cl = g.community_leiden(objective_function="modularity", weights="w", resolution=res, n_iterations=-1)
        sizes = Counter(cl.membership)
        big = [c for c, s in sizes.items() if s >= max(20, n // 60)]
        if best is None or (4 <= len(big) <= 12 and abs(len(big) - 8) < abs(best[2] - 8)):
            best = (res, cl, len(big))
    res, cl, _ = best

    def consensus(runs=20):
        """Leiden many times; link papers by how often they land together (on the graph's own edges); Leiden again."""
        E = np.array(g.get_edgelist())
        together = np.zeros(len(E))
        for _ in range(runs):
            mm = np.array(g.community_leiden(objective_function="modularity", weights="w", resolution=res,
                                             n_iterations=-1).membership)
            together += mm[E[:, 0]] == mm[E[:, 1]]
        keep_e = together >= runs / 2
        h = igraph.Graph(n=n, edges=E[keep_e].tolist(), edge_attrs={"w": (together[keep_e] / runs) * np.asarray(g.es["w"])[keep_e]})
        return h.community_leiden(objective_function="modularity", weights="w", resolution=res, n_iterations=-1)

    t_c = time.perf_counter()
    c1, c2, c3 = consensus(), consensus(), consensus()
    out_cons = [round(igraph.compare_communities(c1, c2, method="ari"), 3), round(igraph.compare_communities(c1, c3, method="ari"), 3)]
    T["consensus_x1"] = round((time.perf_counter() - t_c) * 1000 / 3)
    cl = c1
    # stability: do other random starts give the same groups? (adjusted Rand index vs this run, 1 = identical)
    out_stab = [round(igraph.compare_communities(cl, g.community_leiden(objective_function="modularity", weights="w",
                resolution=res, n_iterations=-1), method="ari"), 3) for _ in range(5)]
    sizes = Counter(cl.membership)
    groups = [c for c, s in sizes.most_common() if s >= max(20, n // 60)]
    memb = np.array(cl.membership)
    t = tick("4 leiden", t)

    # 5. names: OpenAlex topics and title words over-represented in the group (Scimeetr's relative frequency)
    STOP = set("a an the of in on for and or to with by from at as is are be was were its their this that these "
               "using use based study analysis effect effects role new via between among under during after into "
               "review case evidence model approach toward towards".split())

    def terms(w):
        d = det[int(w)]
        out = {"topic:" + d["topic"]} if d.get("topic") else set()
        words = [x.strip(".,:;()[]?!'\"").lower() for x in (d.get("title") or "").split()]
        words = [x for x in words if len(x) > 2 and x not in STOP and not x.isdigit()]
        out |= {f"{a} {b}" for a, b in zip(words, words[1:])}
        out |= set(words)
        return out

    T_ = [terms(w) for w in wids]
    df_all = Counter(x for s in T_ for x in s)

    def name(c):
        idx = np.where(memb == c)[0]
        dfc = Counter(x for i in idx for x in T_[i])
        sc = {x: (f / len(idx)) * math.log((f / len(idx)) / (df_all[x] / n)) for x, f in dfc.items() if f >= 3}
        top = sorted(sc, key=sc.get, reverse=True)
        topics = [x[6:] for x in top if x.startswith("topic:")][:2]
        words = []
        for x in top:
            if x.startswith("topic:"):
                continue
            if any(x in y or y in x for y in words):
                continue
            words.append(x)
            if len(words) == 5:
                break
        return topics, words

    # 6. lenses
    group_of_ref = {}
    out = {"question": question, "papers": n, "resolution": res, "groups": []}
    cited_by_group = {}
    for c in groups:
        idx = np.where(memb == c)[0]
        sub = A[idx]
        local = np.asarray(sub.sum(0)).ravel()           # citations from inside the group, per reference
        cited_by_group[c] = local
    total_local = sum(cited_by_group.values())

    def ref_line(j, local, extra=""):
        nd = int(uref[j])
        return {"id": f"W{int(wn[nd])}", "year": int(c_year[nd]), "title": title_of(nd)[:120],
                "cited_by_group": int(local[j]), "cited_total": int(c_cites[nd]), **({"why": extra} if extra else {})}

    def paper_line(i, extra=""):
        d = det[int(wids[i])]
        return {"id": f"W{int(wids[i])}", "year": d["year"], "type": d["type"], "title": (d["title"] or "")[:120],
                "cited_total": d["citations"], **({"why": extra} if extra else {})}

    gi = {c: k for k, c in enumerate(groups)}
    G = sparse.csr_matrix((np.ones(n), (np.arange(n), [gi.get(m_, len(groups)) for m_ in memb])), shape=(n, len(groups) + 1))
    WG = (sparse.csr_matrix(W) @ G).toarray()[:, :len(groups)]
    for c in groups:
        idx = np.where(memb == c)[0]
        local = cited_by_group[c]
        topics, words = name(c)
        # core: what this group cites most (also finds classics with no abstract, outside the search set)
        core = np.argsort(-local)[:8]
        # distinctive core: cited by this group much more than by the other groups (lift)
        share = local / np.maximum(total_local, 1)
        dist = np.argsort(-(local * share))[:6]
        # rising: recent references, local citations per year since publication, beyond what age predicts
        yrs = np.array([int(c_year[int(uref[j])]) for j in range(len(uref))])
        age = np.clip(YEAR_NOW - yrs, 1, None)
        recent = (yrs >= YEAR_NOW - 5) & (local >= 3)
        rising = [j for j in np.argsort(-(local / age)) if recent[j]][:5]
        # start here: group papers citing the most of the group's core (reviews and syntheses rise)
        top_core = np.argsort(-local)[:40]
        cov = np.asarray(A[idx][:, top_core].sum(1)).ravel()
        start = idx[np.argsort(-cov)[:5]]
        # bridges: group papers whose coupling is spread across groups (participation coefficient)
        Sg = WG[idx]                                   # weight from each group paper to each group
        tot = Sg.sum(1)
        pc = 1 - ((Sg / np.maximum(tot[:, None], 1e-12)) ** 2).sum(1)
        other = Sg.copy(); other[:, gi[c]] = -1
        sc = pc * np.sqrt(tot)
        bridges = [(sc[k], idx[k], groups[int(np.argmax(other[k]))]) for k in np.argsort(-sc)[:4]]
        newest = sorted(idx, key=lambda i: (-(det[int(wids[i])]["year"] or 0), -sims[i]))[:4]
        out["groups"].append({
            "group": int(c), "size": int(len(idx)), "topics": topics, "words": words,
            "years": [int(np.percentile([det[int(wids[i])]["year"] or 0 for i in idx], p)) for p in (10, 50, 90)],
            "core": [ref_line(j, local) for j in core],
            "distinctive": [ref_line(j, local) for j in dist],
            "rising": [ref_line(j, local, f"{local[j] / age[j]:.1f}/yr") for j in rising],
            "start_here": [paper_line(i, f"cites {int(cov[list(idx).index(i)])} of the group's 40 most-cited") for i in start],
            "bridges": [paper_line(i, f"links to group {o}") for _, i, o in bridges],
            "newest": [paper_line(i) for i in newest],
        })
    t = tick("5-6 names+lenses", t)
    out["coverage"] = {"with_references_in_index": int((nref > 0).sum()), "median_refs": float(np.median(nref)),
                       "coupled_pairs": int(len(bc_w)), "in_groups": int(np.isin(memb, groups).sum()),
                       "similarity_range": [round(float(sims.min()), 3), round(float(sims.max()), 3)]}
    out["stability_ari"] = out_stab
    out["consensus_ari"] = out_cons
    out["timing_ms"] = T
    return out


def report(r, q):
    slug = "".join(ch if ch.isalnum() else "-" for ch in q.lower())[:50]
    path = Path(__file__).parent / f"out-{slug}-{r['papers']}.json"
    path.write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps({k: r[k] for k in ("papers", "resolution", "stability_ari", "consensus_ari", "timing_ms")}))
    for gr in r["groups"]:
        print(f"\n## group {gr['group']} · {gr['size']} papers · {gr['years']} · {gr['topics']} · {gr['words']}")
        for lens in ("core", "distinctive", "rising", "start_here", "bridges", "newest"):
            print(f"  {lens}:")
            for x in gr[lens][:4]:
                print(f"    {x['year']} {x['id']:>12} {x.get('cited_by_group', ''):>4} {x['title'][:95]}  {x.get('why', '')}")
    print("\nwrote", path)


if __name__ == "__main__":
    n = int(sys.argv[1])
    eng, quality = load()
    for q in sys.argv[2:]:
        report(main(eng, quality, q, n), q)

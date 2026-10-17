"""Topic maps: the research groups inside a topic, and what to read in each (Scimeetr, 2016, rebuilt on this data).

A set of papers (the closest in meaning to a question, or the matches of an exact-words query) becomes a graph:
two papers are linked when they cite the same works (bibliographic coupling) and when they say similar things
(their embeddings), each source carrying half the total weight. Leiden finds the groups; a consensus of many runs
keeps them steady; each group is named by the phrases over-represented in it; and each group gets reading lists
("lenses") computed from the citations its own papers make.

Why these choices (experiments/topicmap/README.md has the measurements):
- Coupling + text: for clustering current literature, bibliographic coupling is the most accurate pure citation
  link and a citation-text hybrid beats it (Boyack & Klavans 2010, doi:10.1002/asi.21419; Boyack et al. 2020,
  doi:10.1162/qss_a_00085). Text also carries fields where OpenAlex has few references.
- Coupling is the cosine of the papers' reference lists, each reference weighted by how rare it is in the set
  (idf): two papers sharing an obscure source are closer than two papers both citing the field's textbook.
- Text links: each paper's k nearest neighbours by embedding, weighted as UMAP's fuzzy graph does (distance beyond
  the nearest neighbour, scaled per paper), so dense and sparse corners of a topic get comparable links
  (McInnes et al. 2018, arXiv:1802.03426); the usual input to Leiden in single-cell biology.
- Leiden, not Louvain: groups are guaranteed connected (Traag et al. 2019, doi:10.1038/s41598-019-41695-z).
  One run is not reproducible enough (measured adjusted Rand index 0.51-0.87 between runs), so the final groups
  are a consensus of `runs` runs (Lancichinetti & Fortunato 2012, doi:10.1038/srep00336). Papers whose group
  changes between runs are marked "between".
- Resolution: the one whose runs give closest to 7 groups (a person scans 5-9; any group can be split further),
  one step coarser if the consensus still has more than 9.

Everything here is deterministic for the same input (seeded); the API caches answers per data release.
"""
from __future__ import annotations

import math
import random
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import igraph
import numpy as np
from scipy import sparse

YEAR_NOW = date.today().year
SURVEY = re.compile(r"\b(survey|review|overview|tutorial|primer)\b", re.I)
BLOCK = 2000          # rows per block for the big products (memory stays ~linear in n: whole fields are 50k+ papers)
KNN_EXACT_MAX = 200_000   # exact text neighbours on the CPU up to here (~n^2: 300k papers took 10 min, 2M would take
                          #   ~7 h); beyond, exact on the field worker's GPU (2M: ~4.5 min), else an inverted-file
                          #   index (0.97 of the exact neighbours at 300k)
KNN_NPROBE = 32
KNN_GPU_MIN = 30_000      # the whole-field worker (KNN_WORK set) uses its GPU from here (112k papers: 90 s -> ~5 s)
TERMS_MAX = 300_000       # papers whose words name the groups and make the clouds (a random sample beyond)
PAIR_BUDGET = 2_000_000_000   # paper pairs that coupling may go through (sum over references of cited-by-in-set^2);
                              #   300k papers needed ~1e9 (45 s). Beyond, the most cited references are left out
# References that are not reading material: left out of the lenses (they still link papers in the graph).
NOT_READING = frozenset({"dataset", "paratext", "supplementary-materials", "software", "peer-review", "libguides",
                         "erratum", "retraction"})

# igraph's random numbers come from one global generator: clustering runs one map at a time, reseeded each time,
# so the same input always gives the same groups whatever else the server is doing. (~1-2 s per map.)
_RNG = random.Random(0)
_CLUSTER_LOCK = threading.Lock()
igraph.set_random_number_generator(_RNG)


@dataclass
class Options:
    links: str = "hybrid"          # hybrid | refs | text
    cluster_sample: int = 500_000  # more papers than this: groups found on a random sample of this many (the graph
                                   #   among them), every other paper joins the group its links weigh most toward.
                                   #   A 2.2M-paper map's 35 runs took 4 min each; 500k: ~4 s each
    terms_max: int = TERMS_MAX     # papers whose words name the groups (a random sample beyond)
    iterations: int = -1           # Leiden iterations per run; -1: until nothing changes. On a 100k-paper field 2
                                   #   iterations gave the same groups as -1 (ARI 0.97, modularity -0.001) in 1/20 of
                                   #   the time; two seeds differ far more (ARI 0.54), which the consensus handles
    idf: bool = True               # weight shared references by their rarity in the set
    refs_top: int = 0              # keep each paper's strongest coupling links only (0 = all)
    k_text: int = 15               # neighbours per paper in the text graph
    runs: int = 20                 # Leiden runs in the consensus
    resolutions: tuple = (0.5, 0.75, 1.0, 1.25, 1.5)
    scan_runs: int = 3             # runs per resolution when choosing it
    min_groups: int = 4
    target_groups: int = 7
    max_groups: int = 9            # a person scans 5-9 groups; any group can be split further (group=N)
    min_size: int | None = None    # smallest group shown (default max(12, n/50)); smaller ones are "other papers"
    seed: int = 20161028           # Scimeetr's first commit day
    workers: int = 1               # Leiden runs side by side (forked processes) for maps of PARALLEL_MIN+ papers
    lens_size: int = 8              # per reading list (experts too): sent with the map
    # ... and computed this deep: the rest is kept beside the map and read a page at a time (topic_service.py
    # finish: "more"; GET /topics read=N&part=more). Each list also ends where its own rule does (cited by 2+ papers
    # of the group, ...): most stop well before this in small groups.
    lens_depth: int = 150
    # Whole-field maps (seed.score = the judge's P(on topic)): the reading lists show only papers of the field scored
    # at least this; the works cited most from outside the field go to their own list, "foundations". None (topic
    # maps): lists as before. Measured on two replayed fields (experiments/fieldjudge/listgate.py): papers on topic in
    # the lists 80% -> 94% (habitat fragmentation), 50% -> 80% (LLM post-training, with sparse_refs below)
    lens_min_score: float | None = None
    # ... and a group where fewer than this share of papers have reference lists (OpenAlex has few for arXiv
    # preprints: 45 of 3,111 in an RLHF group) takes core, rising and start here from its own papers, ranked by
    # citations from all of science: its few lists name 1950s statistics and child language, not InstructGPT
    sparse_refs: float = 0.3


@dataclass
class Seed:
    """The papers to map, best first. vecs: unit vectors (rows of zeros where has_vec is False)."""
    wids: np.ndarray
    vecs: np.ndarray
    has_vec: np.ndarray
    details: list[dict]
    score: np.ndarray | None = None

    def subset(self, idx) -> "Seed":
        idx = np.asarray(idx)
        return Seed(self.wids[idx], self.vecs[idx], self.has_vec[idx], [self.details[i] for i in idx],
                    None if self.score is None else self.score[idx])


@dataclass
class Graph:
    A: sparse.csr_matrix           # paper x reference (binary)
    refs: np.ndarray               # citation-index node of each reference column
    nodes: np.ndarray              # citation-index node of each paper (-1: not in the snapshot)
    W: sparse.csr_matrix           # symmetric paper x paper link weights
    timing: dict = field(default_factory=dict)


def first_author(d: dict) -> str:
    a = (d.get("authors") or "").split("|")[0].strip()
    return a.split()[-1].lower() if a else ""


def collapse_versions(seed: "Seed", cit, threshold: float = 0.95) -> np.ndarray:
    """Rows to keep: one per study, when a preprint and its journal version (or two repository copies) both made
    it in under different titles. Same first author and nearly the same abstract (cosine >= threshold); kept: the
    one that is not a preprint, then the one with references in the citation index, then the better ranked."""
    n = len(seed.wids)
    if n == 0 or not seed.has_vec.any():
        return np.arange(n)
    V = seed.vecs
    pairs = []
    if n > KNN_EXACT_MAX:                             # n x n is hours here: copies are among each other's neighbours
        idx = np.flatnonzero(seed.has_vec)
        nn, d = knn(np.ascontiguousarray(V[idx], dtype=np.float32), 8)
        r, c = np.nonzero(1 - d >= threshold)
        a, b = idx[r], idx[nn[r, c]]
        pairs = sorted({(int(min(x, y)), int(max(x, y))) for x, y in zip(a.tolist(), b.tolist()) if x != y})
    for start in range(0, n if n <= KNN_EXACT_MAX else 0, BLOCK):   # block by block: never the full n x n matrix
        B = V[start:start + BLOCK] @ V.T
        r, c = np.nonzero(B >= threshold)
        r = r + start
        m = c > r
        pairs += list(zip(r[m].tolist(), c[m].tolist()))
    nodes = cit.nodes(seed.wids)
    has_refs = np.array([nd >= 0 and int(cit.fwd_off[nd + 1]) > int(cit.fwd_off[nd]) for nd in nodes])
    fa = [first_author(d) for d in seed.details]
    rank = lambda i: (seed.details[i].get("type") == "preprint", not has_refs[i], i)
    gone = np.zeros(n, bool)
    for i, j in sorted(pairs):
        if gone[i] or gone[j] or not fa[i] or fa[i] != fa[j]:
            continue
        gone[max(i, j, key=rank)] = True
    return np.flatnonzero(~gone)


# ------------------------------------------------------------------------------------------------- the graph
def reference_matrix(cit, wids: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    nodes = cit.nodes(wids)
    rows, cols = [], []
    for i, nd in enumerate(nodes):
        if nd >= 0:
            r = cit.refs(int(nd))
            if len(r):
                rows.append(np.full(len(r), i, np.int32))
                cols.append(r.astype(np.int64))
    n = len(wids)
    if not rows:
        return sparse.csr_matrix((n, 0), dtype=np.float32), np.zeros(0, np.int64), nodes
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    refs, cidx = np.unique(cols, return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cidx)), shape=(n, len(refs)))
    A.data[:] = 1.0      # a reference listed twice counts once
    return A, refs, nodes


def _ref_vectors(A: sparse.csr_matrix, idf: bool) -> sparse.csr_matrix:
    n = A.shape[0]
    df = np.asarray(A.sum(0)).ravel()
    X = A.astype(np.float64)
    if idf:
        X = X @ sparse.diags(np.log(n / df))          # a reference every paper cites carries nothing
    norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norm[norm == 0] = 1
    return (sparse.diags(1 / norm) @ X).tocsr()


def coupling(A: sparse.csr_matrix, idf: bool, top: int = 0) -> sparse.csr_matrix:
    """Cosine similarity of the papers' reference vectors (Salton's measure when idf is False). top > 0: each paper's
    `top` strongest links only, symmetrized by the larger direction, computed block by block (whole fields: the full
    product of 50k papers that share references is hundreds of GB)."""
    n = A.shape[0]
    if A.shape[1] == 0:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    X = _ref_vectors(A, idf)
    c = np.diff(A.tocsc().indptr).astype(np.int64)            # papers in the set citing each reference
    if top and int((c * c).sum()) > PAIR_BUDGET:
        # the work grows with the square of how many papers share a reference: a field of 2M papers has references
        # cited by 10^4-10^5 of them. Those carry the least about who works with whom (idf weighs them least), so the
        # most cited are left out until the budget holds; their papers keep their other references and text links
        o = np.sort(c)[::-1]
        excess = np.cumsum((o * o)[::-1])[::-1]                # pairs from this reference and all less cited ones
        cap = int(o[np.argmax(excess <= PAIR_BUDGET)]) if (excess <= PAIR_BUDGET).any() else 1
        X = X @ sparse.diags((c <= cap).astype(np.float64))
        X.eliminate_zeros()
        COUPLING_INFO.update({"references_left_out": int((c > cap).sum()), "cited_by_more_than": cap})
    if not top:
        S = (X @ X.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        return S
    XT = X.T.tocsr()                                   # csr: scipy would convert a csc one for every block
    rows, cols, vals = [], [], []
    for start in range(0, n, BLOCK):
        B = (X[start:start + BLOCK] @ XT).tocsr()
        for r in range(B.shape[0]):
            i = start + r
            a, b = B.indptr[r], B.indptr[r + 1]
            d, c = B.data[a:b], B.indices[a:b]
            m = c != i
            d, c = d[m], c[m]
            if len(d) > top:
                keep = np.argpartition(-d, top)[:top]
                d, c = d[keep], c[keep]
            rows.append(np.full(len(c), i)); cols.append(c); vals.append(d)
    if not rows:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    K = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return K.maximum(K.T).tocsr()


COUPLING_INFO: dict = {}
KNN_WORK: dict = {}       # {"dir": a scratch folder, "say": progress} lets big maps use the GPU (srl_search/field.py)


def knn(V: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row's k nearest other rows by cosine (unit rows): (neighbours, distances). Exact in blocks up to
    KNN_EXACT_MAX rows; beyond, an inverted-file index with exact distances in the lists probed (~0.95+ of the exact
    neighbours, measured)."""
    m = len(V)
    if KNN_WORK.get("dir") and m >= KNN_GPU_MIN:              # the whole-field worker: exact, on its GPU
        from srl_search.judges import knn_on_gpu
        got = knn_on_gpu(V, k, Path(KNN_WORK["dir"]), say=KNN_WORK.get("say", lambda s: None))
        if got is not None:
            KNN_WORK["used"] = "gpu-exact"
            return got
    nn = np.empty((m, k), np.int64)
    d = np.empty((m, k), np.float64)
    if m <= KNN_EXACT_MAX:
        for start in range(0, m, BLOCK):
            D = 1 - V[start:start + BLOCK] @ V.T
            D[np.arange(D.shape[0]), start + np.arange(D.shape[0])] = np.inf
            part = np.argpartition(D, k, axis=1)[:, :k]
            nn[start:start + BLOCK] = part
            d[start:start + BLOCK] = np.take_along_axis(D, part, 1)
        return nn, d
    KNN_WORK["used"] = f"ivf-nprobe{KNN_NPROBE}"
    import faiss
    nlist = int(4 * math.sqrt(m))
    ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(V.shape[1]), V.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(0)
    ix.train(V[np.sort(rng.choice(m, size=min(m, 64 * nlist), replace=False))])
    ix.add(V)
    ix.nprobe = KNN_NPROBE
    for start in range(0, m, 100_000):
        S, I = ix.search(V[start:start + 100_000], k + 1)
        own = I == (start + np.arange(len(I)))[:, None]
        S[own | (I < 0)] = -np.inf                              # itself (and empty slots) last
        o = np.argsort(-S, axis=1)[:, :k]
        nn[start:start + len(I)] = np.take_along_axis(I, o, 1)
        d[start:start + len(I)] = 1 - np.take_along_axis(S, o, 1)
    bad = nn < 0
    if bad.any():                                               # too few in the lists probed: a far stand-in
        nn[bad], d[bad] = np.nonzero(bad)[0], 2.0
    return nn, d


def text_graph(vecs: np.ndarray, has_vec: np.ndarray, k: int) -> sparse.csr_matrix:
    """UMAP's fuzzy k-nearest-neighbour graph over cosine distance, symmetrized (a + b - ab). Exact neighbours,
    computed in blocks of rows (never the full n x n matrix)."""
    n = len(vecs)
    idx = np.flatnonzero(has_vec)
    m = len(idx)
    if m <= 2:
        return sparse.csr_matrix((n, n), dtype=np.float64)
    k = min(k, m - 1)
    V = np.ascontiguousarray(vecs[idx], dtype=np.float32)
    nn, d = knn(V, k)
    rho = d.min(1)
    target = math.log2(k)
    lo, hi = np.full(m, 1e-6), np.full(m, 10.0)
    for _ in range(50):                                   # per-paper scale: sum of weights = log2(k)
        sig = (lo + hi) / 2
        s = np.exp(-np.maximum(d - rho[:, None], 0) / sig[:, None]).sum(1)
        big = s > target
        hi = np.where(big, sig, hi)
        lo = np.where(big, lo, sig)
    w = np.exp(-np.maximum(d - rho[:, None], 0) / ((lo + hi) / 2)[:, None])
    P = sparse.csr_matrix((w.ravel(), (np.repeat(idx, k), idx[nn.ravel()])), shape=(n, n))
    return (P + P.T - P.multiply(P.T)).tocsr()


def build_graph(cit, seed: Seed, opt: Options) -> Graph:
    t = time.perf_counter()
    A, refs, nodes = reference_matrix(cit, seed.wids)
    timing = {"references_ms": round((time.perf_counter() - t) * 1000)}
    t = time.perf_counter()
    parts = []
    if opt.links in ("hybrid", "refs"):
        parts.append(coupling(A, opt.idf, opt.refs_top))
    if opt.links in ("hybrid", "text"):
        parts.append(text_graph(seed.vecs, seed.has_vec, opt.k_text))
    W = sparse.csr_matrix((len(seed.wids),) * 2, dtype=np.float64)
    for P in parts:
        total = P.sum()
        if total > 0:
            W = W + P / total                             # each source carries the same total weight
    timing["graph_ms"] = round((time.perf_counter() - t) * 1000)
    return Graph(A, refs, nodes, W.tocsr(), timing)


# ------------------------------------------------------------------------------------------------ grouping
def _graph(n: int, E: np.ndarray, w: np.ndarray) -> igraph.Graph:
    g = igraph.Graph(n=n, edges=E, directed=False)         # an (m, 2) array: no Python list of m pairs
    g.es["weight"] = w
    return g


def _igraph(W: sparse.csr_matrix) -> igraph.Graph:
    U = sparse.triu(W, k=1).tocoo()
    return _graph(W.shape[0], np.column_stack([U.row, U.col]).astype(np.int64), U.data.astype(np.float64))


_ITERATIONS = -1                     # set by cluster() for its runs (forked children inherit it)


def _leiden(g: igraph.Graph, res: float) -> np.ndarray:
    return np.array(g.community_leiden(objective_function="modularity", weights="weight", resolution=res,
                                       n_iterations=_ITERATIONS).membership)


# Leiden runs, each with its own seed (so the result is the same run in sequence or side by side). Big maps (the
# whole-field worker) run them in forked processes: one run of a 33k-paper map takes ~10 s on one core, and a map
# needs ~35. The search API (maps of <= 2,000 papers) never forks.
PARALLEL_MIN = 5000
_SHARED: dict = {}


def _seeded_run(args) -> np.ndarray:
    res, seed = args
    _RNG.seed(seed)
    return _leiden(_SHARED["g"], res)


RUN_BYTES_PER_EDGE = 60            # a Leiden run's own memory, measured; side-by-side runs are limited to RUN_MEMORY
RUN_MEMORY = 48 * 2**30


def _runs(g: igraph.Graph, jobs: list[tuple[float, int]], workers: int) -> list[np.ndarray]:
    workers = min(workers, max(1, RUN_MEMORY // max(1, RUN_BYTES_PER_EDGE * (g.ecount() + g.vcount()))))
    if workers <= 1 or g.vcount() < PARALLEL_MIN:
        return [_seeded_run_on(g, r, sd) for r, sd in jobs]
    import multiprocessing as mp
    _SHARED["g"] = g                              # forked children see it without copying
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        out = pool.map(_seeded_run, jobs, chunksize=1)
    _SHARED.clear()
    return out


def _seeded_run_on(g, res, seed):
    _RNG.seed(seed)
    return _leiden(g, res)


def _ari(a, b) -> float:
    if np.array_equal(a, b):
        return 1.0                     # identical, even when one group (where the index itself is undefined)
    v = float(igraph.compare_communities(a.tolist(), b.tolist(), method="adjusted_rand"))
    return v if math.isfinite(v) else 0.0


def _big(m: np.ndarray, min_size: int) -> int:
    return sum(1 for c in Counter(m.tolist()).values() if c >= min_size)


def cluster(W: sparse.csr_matrix, opt: Options, min_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(group of each paper, 0..G-1 by size, -1 = other papers; agreement of each paper with its group across the
    consensus runs, 0-1; facts about the choice)."""
    n = W.shape[0]
    if opt.cluster_sample and n > opt.cluster_sample:
        return _cluster_sampled(W, opt, min_size)
    return _cluster(W, opt, min_size)


def _cluster_sampled(W: sparse.csr_matrix, opt: Options, min_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Groups of a random sample (the graph among its papers: each keeps ~sample/n of its links, ~20 of ~90 at 2M),
    then every other paper joins the group its links weigh most toward, twice (the second time counting papers placed
    the first). Agreement of a placed paper: the share of its link weight going to its group."""
    n = W.shape[0]
    rng = np.random.default_rng(opt.seed)
    S = np.sort(rng.choice(n, size=opt.cluster_sample, replace=False))
    gs, ags, info = _cluster(W[S][:, S].tocsr(), opt, max(2, round(min_size * len(S) / n)))
    G = int(gs.max()) + 1 if (gs >= 0).any() else 0
    group = np.full(n, -1, np.int64)
    agree = np.zeros(n)
    group[S], agree[S] = gs, ags
    rest = np.setdiff1d(np.arange(n), S)
    if G:
        for _ in range(2):
            known = np.flatnonzero(group >= 0)
            onehot = sparse.csr_matrix((np.ones(len(known)), (known, group[known])), shape=(n, G))
            V = (W[rest] @ onehot).toarray()
            tot = V.sum(1)
            best = V.argmax(1)
            ok = tot > 0
            group[rest] = np.where(ok, best, -1)
            agree[rest] = np.where(ok, V[np.arange(len(rest)), best] / np.maximum(tot, 1e-12), 0.0)
    # groups by size again, the small ones ("other papers") as in _cluster
    sizes = Counter(group[group >= 0].tolist())
    order = [c for c, s_ in sorted(sizes.items(), key=lambda x: (-x[1], x[0])) if s_ >= min_size]
    remap = np.full(max(G, 1), -1, np.int64)
    remap[order] = np.arange(len(order))
    group = np.where(group >= 0, remap[np.maximum(group, 0)], -1)
    info = {**info, "sample": int(len(S)), "placed_by_links": int((group[rest] >= 0).sum()),
            "consensus_agreement": round(float(agree[group >= 0].mean()), 3) if (group >= 0).any() else None}
    return group, agree, info


def _cluster(W: sparse.csr_matrix, opt: Options, min_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    global _ITERATIONS
    n = W.shape[0]
    U = sparse.triu(W, k=1).tocoo()
    E = np.column_stack([U.row, U.col]).astype(np.int64)
    w = U.data.astype(np.float64)
    del U
    g = _graph(n, E, w)
    workers = opt.workers
    with _CLUSTER_LOCK:
        _ITERATIONS = opt.iterations
        _RNG.seed(opt.seed)
        scan = []
        jobs = [(res, opt.seed * 1000 + 10 * i + k) for i, res in enumerate(opt.resolutions) for k in range(opt.scan_runs)]
        scan_runs = _runs(g, jobs, workers)
        for i, res in enumerate(opt.resolutions):
            ms = scan_runs[i * opt.scan_runs:(i + 1) * opt.scan_runs]
            stab = float(np.mean([_ari(ms[i], ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]))
            scan.append({"resolution": res, "groups": int(np.median([_big(m, min_size) for m in ms])),
                         "stability": round(stab, 3)})
        # The group count changes smoothly with the resolution; a stability estimate from a few runs does not
        # (measured: choosing by it made the groups depend on the random seed, ARI 0.50 between seeds).
        pick = min(scan, key=lambda s: (abs(s["groups"] - opt.target_groups), s["resolution"]))
        ri = list(opt.resolutions).index(pick["resolution"])
        while True:
            res = opt.resolutions[ri]
            runs = _runs(g, [(res, opt.seed * 1000 + 500 + 50 * ri + k) for k in range(opt.runs)], workers)
            together = np.zeros(len(E))
            for m in runs:
                together += m[E[:, 0]] == m[E[:, 1]]
            frac = together / len(runs)
            keep = frac >= 0.5
            h = _graph(n, E[keep], frac[keep] * w[keep])
            final = _seeded_run_on(h, res, opt.seed * 1000 + 999)
            if _big(final, min_size) <= opt.max_groups or ri == 0:
                break
            ri -= 1                                  # too many groups for one screen: one step coarser
        pick = next(s for s in scan if s["resolution"] == res)
        _ITERATIONS = -1
    sizes = Counter(final.tolist())
    order = [c for c, s in sorted(sizes.items(), key=lambda x: (-x[1], x[0])) if s >= min_size]
    remap = {c: i for i, c in enumerate(order)}
    group = np.array([remap.get(c, -1) for c in final.tolist()])
    # agreement: in how many runs the paper sat in the run-cluster that holds most of its final group
    agree = np.zeros(n)
    G_ = int(group.max()) + 1 if (group >= 0).any() else 0
    for m in runs:
        best_of = np.full(max(G_, 1), -1, np.int64)
        for c in range(G_):
            vals, cnt = np.unique(m[group == c], return_counts=True)
            best_of[c] = vals[np.argmax(cnt)]
        agree += (group >= 0) & (m == best_of[np.maximum(group, 0)])
    agree /= len(runs)
    info = {"resolution": res, "scan": scan, "runs": opt.runs, "seed": opt.seed, "iterations": opt.iterations,
            "stability": pick["stability"], "consensus_agreement": round(float(agree[group >= 0].mean()), 3) if (group >= 0).any() else None}
    return group, agree, info


# ------------------------------------------------------------------------------------------------- names
STOP = frozenset("""a about above across after again against all almost along also although always am among an and
another any are around as at be because been before being below between both but by can could did do does doing
done down during each either else even ever every few for from further had has have having here how however i if in
into is it its itself just less like made make many may might more most much must my near neither no nor not now of
off often on once one only or other others our out over own per perhaps rather same several shall should since so
some such than that the their them then there these they this those though through thus to too toward towards under
until up upon us use used uses using via was we were what when where whether which while who whom whose why will
with within without would yet you your based study studies analysis effect effects new approach results result
show shows showed shown paper review case evidence role impact impacts data model models method methods using
during toward among associated related potential different important significant significantly increased
increase increases higher lower high low large small first second two three four five also found find findings
present provide provides provided including include includes included compared well well-known however therefore
total level levels number numbers way ways current recent novel research investigation investigate investigated
examine examined examines assess assessed assessment evaluate evaluated evaluation aim aims objective objectives
conclusion conclusions background purpose introduction discussion here herein report reports reported among
whereas respectively mainly overall i.e e.g et al vs versus remain remains remained suggest suggests suggested
indicate indicates indicated reveal reveals revealed become becomes became need needs needed help helps seem seems
appear appears lead leads allow allows require requires required affect affected affecting include involve involves
regarding according particularly especially
abstract highlight highlights graphical copyright elsevier springer wiley ltd inc rights reserved""".split())
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
TAG = re.compile(r"<[^>]+>")


@lru_cache(maxsize=200_000)
def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "sis")):
        return w[:-1]
    return w


SPLIT = re.compile(r"[.;:!?()\[\]]")


def doc_terms(title: str | None, abstract: str | None, surfaces: dict) -> set:
    """Phrases of one paper as stem keys: 1-3 words from the title, 1-2 from the abstract. A phrase never starts
    or ends with a stopword or a number and has no stopword inside; its spellings are counted in `surfaces`."""
    keys = set()
    for text, nmax in ((title, 3), (abstract, 2)):
        if not text:
            continue
        for sentence in SPLIT.split(TAG.sub(" ", text)):
            toks = TOKEN.findall(sentence)
            if not toks:
                continue
            low = [t.lower() for t in toks]
            stems = [stem(t) for t in toks]
            stop = [x in STOP for x in low]
            edge = [not stop[k] and low[k][0].isalpha() and len(low[k]) >= 2 for k in range(len(toks))]
            for i in range(len(toks)):
                if not edge[i]:
                    continue
                for j in range(i, min(i + nmax, len(toks))):
                    if j > i + 1 and stop[j - 1]:          # a stopword inside ends the phrases from i
                        break
                    if not edge[j] or (j == i and len(low[i]) < 3):
                        continue
                    key = " ".join(stems[i:j + 1])
                    keys.add(key)
                    c = surfaces.get(key)
                    if c is None:
                        c = surfaces[key] = Counter()
                    c[" ".join(toks[i:j + 1])] += 1
    return keys


def display(surfaces: Counter) -> str:
    s = surfaces.most_common(1)[0][0]
    # Title Case from titles reads as shouting in a label; keep acronyms (COVID-19, NAFLD, GNN)
    words = s.split()
    return " ".join(w if sum(ch.isupper() for ch in w) >= 2 else w.lower() for w in words)


def group_counts(terms: list[set], group: np.ndarray, G: int) -> tuple[Counter, list[Counter]]:
    """How many papers carry each phrase: in the whole map, and in each group."""
    return Counter(k for s in terms for k in s), [Counter(k for i in np.flatnonzero(group == c) for k in terms[i])
                                                  for c in range(G)]


def name_groups(terms: list[set], group: np.ndarray, G: int, query: str, surfaces: dict,
                counts: tuple[Counter, list[Counter]] | None = None) -> list[list[str]]:
    """Per group, up to 4 phrases over-represented in it compared with the other papers (the KL divergence term
    p log(p/q), Scimeetr's relative frequency in a form that also weighs how common the phrase is). A phrase is
    skipped when it repeats a word of a chosen one, or mostly names the same papers (Jaccard >= 0.6)."""
    qstems = {stem(w) for w in TOKEN.findall(query or "")}
    n = len(terms)
    df_all, dfs = counts or group_counts(terms, group, G)
    names = []
    for c in range(G):
        idx = np.flatnonzero(group == c)
        ng = len(idx)
        df = dfs[c]
        need = max(3, math.ceil(0.05 * ng))
        score = {}
        for k, f in df.items():
            if f < need or all(t in qstems for t in k.split()):
                continue
            p = f / ng
            q = (df_all[k] - f + 0.5) / (n - ng + 1)
            if p > q:
                score[k] = p * math.log(p / q)
        chosen: list[str] = []
        used: set = set()
        docs = {}

        def papers_with(x):
            if x not in docs:
                docs[x] = {i for i in idx if x in terms[i]}
            return docs[x]

        for k in sorted(score, key=lambda x: (-score[x], x))[:60]:
            toks = set(k.split())
            if toks & used:
                continue
            if any(len(papers_with(k) & papers_with(c)) >= 0.6 * len(papers_with(k) | papers_with(c)) for c in chosen):
                continue
            # a longer phrase that carries most of this word's papers reads better ("time-restricted eating")
            longer = [x for x in score if len(x.split()) > len(k.split()) and f" {k} " in f" {x} "
                      and df[x] >= 0.6 * df[k] and not set(x.split()) & used]
            if longer:
                k = max(longer, key=lambda x: (df[x], score[x]))
                toks = set(k.split())
            chosen.append(k)
            used |= toks
            if len(chosen) == 4:
                break
        names.append([display(surfaces[k]) for k in chosen])
    return names


# -------------------------------------------------------------------------------------------------- word clouds
# Scimeetr's word clouds (2016-17, ScholarsReadingList's uploader/views.py): a group's most frequent words, sized by
# how often its papers use them and coloured by relative frequency, blue when the group uses a word more than the
# rest, red when less, grey when about as much. Relative frequency here: p / q, p the share of the group's papers
# that use the phrase, q the share among the map's other papers (the whole map against all of science, counted in
# the keyword index). Most of the words are the most frequent (what the group talks about); a third are its most
# telling, by LDAvis's relevance lambda * log p + (1 - lambda) * log(p / q) (Sievert & Shirley 2014,
# doi:10.3115/v1/w14-3110) with Scimeetr's lambda 0.4 (at 0.6 everyday words outranked a group's own), so rarer
# words that set it apart show too, small and deep blue.
CLOUD_LAMBDA = 0.4
CLOUD_SIZE = 30
CLOUD_TELLING = 10             # of CLOUD_SIZE: the most telling words not already among the most frequent
CLOUD_MIN_LIFT = 1.5           # a "telling" word is used at least this many times as often as elsewhere
CLOUD_CANDIDATES = 160         # the whole map: its most common phrases, counted in all of science (cached)


def relevance(p: float, q: float, lam: float = CLOUD_LAMBDA) -> float:
    return lam * math.log(p) + (1 - lam) * math.log(p / q)


def _cloud(cands: dict[str, tuple[float, float]], df: Counter, surfaces: dict, k: int) -> list[dict]:
    """cands: phrase key -> (p, q). Up to k words, most frequent first: k - CLOUD_TELLING by frequency, the rest by
    relevance among those used at least CLOUD_MIN_LIFT times as often as elsewhere. A phrase stands for its words:
    a word whose papers are mostly those of a longer phrase holding it ("fragmentation" under "habitat
    fragmentation") is left out."""
    def ok(key, pool):
        return not any(len(x) > len(key) and f" {key} " in f" {x} " and df[x] >= 0.7 * df[key] for x in pool)
    by_freq = sorted(cands, key=lambda x: (-cands[x][0], x))
    pool = set(by_freq[:6 * k])
    chosen: list[str] = [x for x in by_freq[:6 * k] if ok(x, pool)][:k - CLOUD_TELLING]
    telling = sorted((x for x in cands if cands[x][0] > CLOUD_MIN_LIFT * cands[x][1]),
                     key=lambda x: (-relevance(*cands[x]), x))[:6 * k]
    tpool = set(telling) | pool
    for x in telling:
        if len(chosen) == k:
            break
        if x not in chosen and ok(x, tpool):
            chosen.append(x)
    for x in by_freq[:6 * k]:                          # few telling words: more of the frequent ones
        if len(chosen) == k:
            break
        if x not in chosen and ok(x, pool):
            chosen.append(x)
    chosen.sort(key=lambda x: (-cands[x][0], x))
    return [{"t": display(surfaces[x]), "share": round(cands[x][0], 4), "lift": round(cands[x][0] / cands[x][1], 2),
             "score": round(relevance(*cands[x]), 3)} for x in chosen]


def group_clouds(terms: list[set], group: np.ndarray, G: int, query: str, surfaces: dict,
                 counts: tuple[Counter, list[Counter]], k: int = CLOUD_SIZE) -> list[list[dict]]:
    """Each group's words against the rest of the map. The question's own words are left out: every group uses them."""
    qstems = {stem(w) for w in TOKEN.findall(query or "")}
    n = len(terms)
    df_all, dfs = counts
    out = []
    for c in range(G):
        df = dfs[c]
        ng = int((group == c).sum())
        need = max(5, math.ceil(0.03 * ng))      # fewer: often copies of one text (a book's chapters)
        cands = {}
        for key, f in df.items():
            if f < need or all(t in qstems for t in key.split()):
                continue
            cands[key] = (f / ng, (df_all[key] - f + 0.5) / (n - ng + 1))
        out.append(_cloud(cands, df, surfaces, k))
    return out


def topic_cloud(terms: list[set], query: str, surfaces: dict, df_all: Counter, background,
                k: int = CLOUD_SIZE) -> list[dict]:
    """The whole map's words against all of science. background(phrases) -> ({phrase: papers in the corpus that
    use it}, corpus size), or None (no cloud: against itself, every word has lift 1)."""
    if background is None or not terms:
        return []
    qstems = {stem(w) for w in TOKEN.findall(query or "")}
    n = len(terms)
    need = max(3, math.ceil(0.01 * n))
    common = [key for key, f in sorted(df_all.items(), key=lambda x: (-x[1], x[0]))
              if f >= need and not all(t in qstems for t in key.split())][:CLOUD_CANDIDATES]
    shown = {key: display(surfaces[key]) for key in common}
    try:
        counts, N = background(sorted(set(shown.values())))
    except Exception as e:  # noqa: BLE001 (the cloud is a nicety: a map without it beats no map)
        print(f"topic cloud: no background ({type(e).__name__}: {e})", flush=True)
        return []
    cands = {}
    for key in common:
        c = counts.get(shown[key])
        if c is None or not N:
            continue
        # the corpus holds the map's papers (and stems more, so it may count more)
        cands[key] = (df_all[key] / n, max(c, df_all[key]) / N)
    return _cloud(cands, df_all, surfaces, k)


# -------------------------------------------------------------------------------------------------- lenses
def credit(pos: np.ndarray, nauth: int) -> np.ndarray:
    """Harmonic credit relative to the first author: first 1, last 1/2, the k-th of the others 1/(k+1).
    (Scimeetr's rule: the last author ranks second. Its R code inverted the weights; this does what it described.)"""
    pos = pos.astype(np.float64)
    rank = np.where(pos == 1, 1.0, np.where((pos == nauth) & (nauth >= 2), 2.0, pos + 1))
    return 1 / rank


CHAPTER_DOI = re.compile(r"(\.\d{3}\.\d{4}|_\d+|[./]ch\d+|\.c\d+)$", re.I)


def book_of(d: dict) -> str | None:
    """The book a chapter belongs to (a book counts as its own), else None: the DOI without its chapter suffix
    (OUP .003.0005, Springer _5, Wiley .ch5), or the venue of a chapter."""
    if d.get("type") not in ("book-chapter", "book"):
        return None
    doi = (d.get("doi") or "").lower()
    if doi:
        return CHAPTER_DOI.sub("", doi)
    return (d.get("venue") or "").lower() or None


def h_index(values) -> int:
    v = sorted(values, reverse=True)
    h = 0
    for i, x in enumerate(v, 1):
        if x >= i:
            h = i
        else:
            break
    return h


def distinct_works(js, gr: Graph, cit, k: int) -> list:
    """The first k reference columns of js with different titles: OpenAlex can hold two records for one work (a book
    entered twice: "The Theory of Island Biogeography" filled two places of one list)."""
    seen, out = set(), []
    for j in js:
        key = re.sub(r"[^a-z0-9]", "", (cit.title(int(gr.refs[j])) or "").lower())[:150]
        if key and key in seen:
            continue
        seen.add(key)
        out.append(j)
        if len(out) == k:
            break
    return out


def lenses_for(c: int, idx: np.ndarray, gr: Graph, seed: Seed, group: np.ndarray, cit, opt: Options,
               ref_year: np.ndarray, ref_ok: np.ndarray, total_local: np.ndarray | None, unit: str = "group",
               ref_seed: np.ndarray | None = None) -> dict:
    """Reading lists of one group, or of the whole map (unit="topic", idx = every paper, total_local None: no
    "only here" list, which compares a group with the others). Items: {"w": work number, "why": sentence, ...}.
    With opt.lens_min_score and ref_seed (the row in seed of each reference column, -1 outside the map): whole-field
    rules, see Options.lens_min_score and Options.sparse_refs."""
    k = opt.lens_depth
    ng = len(idx)
    A = gr.A
    local = np.asarray(A[idx].sum(0)).ravel() if A.shape[1] else np.zeros(0)
    out: dict[str, list] = {}
    judged = opt.lens_min_score is not None and ref_seed is not None and seed.score is not None
    if judged:
        score = np.asarray(seed.score, np.float64)
        ok = ref_ok & (ref_seed >= 0) & (score[np.maximum(ref_seed, 0)] >= opt.lens_min_score)
        outside = ref_ok & (ref_seed < 0)
        members = idx[score[idx] >= opt.lens_min_score]
    else:
        ok, outside, members = ref_ok, None, idx

    core = distinct_works((j for j in np.argsort(-local, kind="stable") if local[j] >= 2 and ok[j]), gr, cit, k)
    out["core"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]),
                    "why": f"cited by {int(local[j])} of the {ng:,} papers in this {unit}"} for j in core]
    if outside is not None:
        found = distinct_works((j for j in np.argsort(-local, kind="stable") if local[j] >= 3 and outside[j]), gr, cit, k)
        out["foundations"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]),
                               "why": f"cited by {int(local[j])} of the {ng:,} papers in this {unit}"} for j in found]

    if total_local is not None:
        lift = np.where(total_local > 0, local / np.maximum(total_local, 1), 0)
        core_set = set(core)
        dist = distinct_works((j for j in np.argsort(-(local * lift), kind="stable")
                               if local[j] >= 3 and ok[j] and j not in core_set and lift[j] >= 0.5), gr, cit, k)
        out["distinctive"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]), "cited_in_map": int(total_local[j]),
                               "why": f"{int(local[j])} of its {int(total_local[j])} citations in this map come from this group"}
                              for j in dist]

    recent = (ref_year >= YEAR_NOW - 5) & (local >= 3) & ok
    rate = np.where(recent, local / np.maximum(YEAR_NOW - ref_year + 1, 1), 0)
    out["rising"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]), "year": int(ref_year[j]),
                      "why": f"from {int(ref_year[j])}, already cited by {int(local[j])} papers of this {unit}"}
                     for j in distinct_works((j for j in np.argsort(-rate, kind="stable") if rate[j] > 0), gr, cit, k)]

    top = [j for j in np.argsort(-local, kind="stable")[:30] if local[j] >= 2]
    if top:
        cov = np.asarray(A[idx][:, top].sum(1)).ravel()
        if judged:
            cov = np.where(score[idx] >= opt.lens_min_score, cov, 0)
        years = np.array([seed.details[i].get("year") or 0 for i in idx])
        pool = np.flatnonzero((years >= YEAR_NOW - 8) & (cov >= 3))
        if len(pool) < 3:
            pool = np.flatnonzero(cov >= 3)
        best, books = [], set()
        for p in sorted(pool, key=lambda p: (-cov[p], -years[p])):
            b = book_of(seed.details[idx[p]])           # one entry per book: its chapters all cite the same works
            if b and b in books:
                continue
            books.add(b) if b else None
            best.append(p)
            if len(best) == k:
                break
        out["start_here"] = [{"w": int(seed.wids[idx[p]]), "covers": int(cov[p]),
                              "why": f"cites {int(cov[p])} of the {unit}'s {len(top)} most-cited works"
                                     + (" · a review" if seed.details[idx[p]].get("type") == "review" else "")}
                             for p in best]
    else:
        out["start_here"] = []

    newest = sorted(members, key=lambda i: (-(seed.details[i].get("year") or 0),
                                            -(seed.score[i] if seed.score is not None else 0)))[:k]
    out["newest"] = [{"w": int(seed.wids[i]), "why": ""} for i in newest]

    with_refs = int((np.asarray(A[idx].sum(1)).ravel() > 0).sum()) if A.shape[1] else 0
    if judged and len(members) and with_refs < opt.sparse_refs * ng:
        out.update(own_lists(members, gr, seed, cit, k, unit, out["start_here"]))
    return out, local


def own_lists(members: np.ndarray, gr: Graph, seed: Seed, cit, k: int, unit: str, start_here: list) -> dict:
    """Core, rising and start here from a group's own papers, ranked by how often all of science cites them: for a
    group whose papers mostly lack reference lists (Options.sparse_refs)."""
    nd = gr.nodes[members]
    cites = np.where(nd >= 0, np.asarray(cit.cites)[np.maximum(nd, 0)], 0).astype(np.int64)
    # one place per work (a preprint and its proceedings copy): the most cited copy
    key = lambda i: re.sub(r"[^a-z0-9]", "", (seed.details[i].get("title") or "").lower())[:150] or f"#{i}"
    by_cites = np.argsort(-cites, kind="stable")
    _, first = np.unique([key(members[p]) for p in by_cites], return_index=True)
    keep = np.sort(by_cites[first])
    members, cites = members[keep], cites[keep]
    years = np.array([seed.details[i].get("year") or 0 for i in members])
    why = f"cited {{:,}} times across science (few papers of this {unit} list their references here)"
    out = {"core": [{"w": int(seed.wids[members[p]]), "citations_all": int(cites[p]), "why": why.format(int(cites[p]))}
                    for p in np.argsort(-cites, kind="stable")[:k] if cites[p] > 0]}
    rec = np.flatnonzero((years >= YEAR_NOW - 5) & (cites > 0))
    rate = cites[rec] / np.maximum(YEAR_NOW - years[rec] + 1, 1)
    out["rising"] = [{"w": int(seed.wids[members[p]]), "year": int(years[p]), "citations_all": int(cites[p]),
                      "why": f"from {int(years[p])}, already cited {int(cites[p]):,} times"}
                     for p in rec[np.argsort(-rate, kind="stable")][:k]]
    reviews = [p for p in np.argsort(-cites, kind="stable") if years[p] >= YEAR_NOW - 8 and cites[p] > 0 and (
        seed.details[members[p]].get("type") == "review" or SURVEY.search(seed.details[members[p]].get("title") or ""))]
    first = [{"w": int(seed.wids[members[p]]), "citations_all": int(cites[p]),
              "why": f"a review of the {unit}, cited {int(cites[p]):,} times"} for p in reviews[:k]]
    have = {it["w"] for it in first}
    out["start_here"] = (first + [it for it in start_here if it["w"] not in have])[:k]
    return out


def bridges_for(c: int, idx: np.ndarray, WG: np.ndarray, G: int, k: int, seed: Seed,
                min_score: float | None = None) -> list[dict]:
    S = WG[idx]
    tot = S.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot[:, None] > 0, S / np.maximum(tot[:, None], 1e-12), 0)
    other = share.copy()
    other[:, c] = -1
    best = other.argmax(1)
    bshare = other.max(1)
    pc = 1 - (share ** 2).sum(1)
    # a bridge others have used: broad reviews in obscure venues link everything and are cited by no one
    cited = np.array([float(seed.details[i].get("citations") or 0) for i in idx])
    sc = np.where(bshare >= 0.2, pc * np.sqrt(tot) * (1 + np.log1p(cited)), -1)
    if min_score is not None and seed.score is not None:           # whole-field maps: papers judged on topic only
        sc = np.where(np.asarray(seed.score)[idx] >= min_score, sc, -1)
    out = []
    for p in np.argsort(-sc, kind="stable")[:k]:
        if sc[p] <= 0:
            break
        out.append({"w": int(seed.wids[idx[p]]), "to_group": int(best[p]), "share": round(float(bshare[p]), 2),
                    "why": f"{round(100 * float(bshare[p]))}% of its links go to group {int(best[p]) + 1}"})
    return out


def experts_for(idx: np.ndarray, local: np.ndarray, gr: Graph, seed: Seed, cit, k: int, unit: str = "group") -> list[dict]:
    """Authors ranked by a harmonic local h-index: h of their papers are each cited by at least h papers of this
    group, after position credit. Only papers cited by >= 2 group papers can count toward h >= 2."""
    au = cit.authors
    if au is None:
        return []
    credited: dict[int, list[float]] = defaultdict(list)
    papers: dict[int, list[tuple[int, int, int]]] = defaultdict(list)   # author -> (local, year, work number)
    for j in np.flatnonzero(local >= 2):
        node = int(gr.refs[j])
        rows, pos, n = au.of(node)
        if not len(rows):
            continue
        cr = credit(pos, n) * local[j]
        y = int(cit.year[node])
        for r, v in zip(rows.tolist(), cr.tolist()):
            credited[r].append(v)
            papers[r].append((int(local[j]), y, cit.work(node)))
    if not credited:
        return []
    mine: dict[int, list[tuple[int, int]]] = defaultdict(list)          # author -> their papers in the group
    for i in idx:
        nd = int(gr.nodes[i])
        if nd < 0:
            continue
        rows, _, _ = au.of(nd)
        for r in rows.tolist():
            mine[r].append((int(seed.details[i].get("year") or 0), int(seed.wids[i])))
    # OpenAlex sometimes splits one person into several author records (Ilkka Hanski twice in one group): within a
    # group, records with the same name are one expert. (Two different people with one name in one niche: rare.)
    by_name: dict[str, list[int]] = defaultdict(list)
    for r in credited:
        by_name[re.sub(r"[^a-z]", "", au.name(r).lower()) or str(r)].append(r)
    for rs in by_name.values():
        if len(rs) > 1:
            keep = max(rs, key=lambda r: sum(credited[r]))
            for r in rs:
                if r != keep:
                    credited[keep] += credited.pop(r)
                    papers[keep] += papers.pop(r)
                    mine[keep] += mine.pop(r, [])
    ranked = sorted(credited, key=lambda r: (-h_index(credited[r]), -sum(credited[r]), r))[:k]
    out = []
    for r in ranked:
        h = h_index(credited[r])
        if h < 2:
            break
        top = max(papers[r])
        latest = max(mine[r]) if mine[r] else None
        out.append({"author": au.openalex_id(r), "name": au.name(r), "h": h,
                    "cited_works": len(papers[r]), "papers_in_group": len(mine[r]),
                    "top": top[2], "top_cited_here": top[0], "latest": latest[1] if latest else None,
                    "why": f"{h} of their papers are each cited by at least {h} papers of this {unit} "
                           f"(first author full credit, last author half, others less)"})
    return out


def centrality(WG: np.ndarray, group: np.ndarray) -> list[float]:
    out = np.zeros(len(group))
    for c in set(group.tolist()) - {-1}:
        idx = np.flatnonzero(group == c)
        s = WG[idx, c]
        out[idx] = s / s.max() if s.max() > 0 else 0
    return [round(float(x), 3) for x in out]


# -------------------------------------------------------------------------------------------------- the map
def build(seed: Seed, cit, query: str = "", opt: Options | None = None, background=None) -> dict:
    """The topic map of these papers: groups, names, lenses, links between groups, and the same reading lists for
    the whole map ("overall"). Works are given as OpenAlex work numbers ("w"); the caller adds titles, links and
    positions. background: counts phrases in all of science, for the whole map's word cloud (topic_cloud)."""
    opt = opt or Options()
    t0 = time.perf_counter()
    n = len(seed.wids)
    gr = build_graph(cit, seed, opt)
    timing = dict(gr.timing)
    min_size = opt.min_size or max(12, n // 50)
    t = time.perf_counter()
    if n < 2 * min_size or gr.W.nnz == 0:
        group, agree, info = np.full(n, -1), np.zeros(n), {"resolution": None, "note": "too few papers to group"}
        if n and gr.W.nnz:
            group[:] = 0
            agree[:] = 1
    else:
        group, agree, info = cluster(gr.W, opt, min_size)
    timing["grouping_ms"] = round((time.perf_counter() - t) * 1000)
    G = int(group.max()) + 1 if (group >= 0).any() else 0

    t = time.perf_counter()
    surfaces: dict = {}
    # the words of up to TERMS_MAX papers (a random sample beyond: shares of 300k papers are as good as of 2M,
    #   and reading every abstract of 2M papers takes ~15 min on one core)
    ti = np.arange(n) if n <= opt.terms_max else np.sort(np.random.default_rng(opt.seed).choice(n, opt.terms_max, replace=False))
    terms = [doc_terms(seed.details[i].get("title"), seed.details[i].get("abstract"), surfaces) for i in ti]
    gt = group[ti]
    counts = group_counts(terms, gt, G)
    names = name_groups(terms, gt, G, query, surfaces, counts)
    clouds = group_clouds(terms, gt, G, query, surfaces, counts)
    timing["names_ms"] = round((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    cloud = topic_cloud(terms, query, surfaces, counts[0], background)
    timing["cloud_ms"] = round((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    A = gr.A
    ref_year = np.asarray(cit.year[gr.refs]).astype(np.int64) if len(gr.refs) else np.zeros(0, np.int64)
    kinds = np.asarray(cit.kind[gr.refs]) if len(gr.refs) else np.zeros(0, np.uint8)
    bad = np.array([i for i, tname in enumerate(cit.types) if tname in NOT_READING], dtype=np.uint8)
    ref_ok = ~np.isin(kinds, bad)
    in_groups = np.flatnonzero(group >= 0)
    total_local = np.asarray(A[in_groups].sum(0)).ravel() if A.shape[1] else np.zeros(0)
    onehot = sparse.csr_matrix((np.ones(len(in_groups)), (in_groups, group[in_groups])), shape=(n, max(G, 1)))
    WG = (gr.W @ onehot).toarray()

    # direct citations between the mapped papers: which group builds on which
    seed_col = {int(nd): i for i, nd in enumerate(gr.nodes) if nd >= 0}
    flows = np.zeros((max(G, 1), max(G, 1)), np.int64)
    ref_is_seed = np.array([seed_col.get(int(r), -1) for r in gr.refs]) if len(gr.refs) else np.zeros(0, int)
    if len(gr.refs):
        C = A.tocoo()
        tgt = ref_is_seed[C.col]
        m = (tgt >= 0) & (group[C.row] >= 0)
        m[m] &= group[tgt[m]] >= 0
        np.add.at(flows, (group[C.row[m]], group[tgt[m]]), 1)

    ref_seed = ref_is_seed if opt.lens_min_score is not None else None
    groups = []
    for c in range(G):
        idx = np.flatnonzero(group == c)
        lens, local = lenses_for(c, idx, gr, seed, group, cit, opt, ref_year, ref_ok, total_local, ref_seed=ref_seed)
        lens["bridges"] = bridges_for(c, idx, WG, G, opt.lens_depth, seed, opt.lens_min_score)
        lens["experts"] = experts_for(idx, local, gr, seed, cit, opt.lens_depth)
        strength = WG[idx, c]
        years = [d.get("year") for d in (seed.details[i] for i in idx) if d.get("year")]
        topics = Counter(seed.details[i].get("topic") for i in idx if seed.details[i].get("topic"))
        uses = {int(o) + 1: int(flows[c, o]) for o in np.argsort(-flows[c]) if o != c and flows[c, o] > 0}
        groups.append({
            "id": c + 1, "size": int(len(idx)), "name": " · ".join(names[c][:3]) or f"group {c + 1}",
            "phrases": names[c],
            "years": [int(np.percentile(years, p)) for p in (10, 50, 90)] if years else None,
            "typical": int(seed.wids[idx[int(np.argmax(strength))]]),
            "openalex_topics": [{"topic": tname, "share": round(v / len(idx), 2)} for tname, v in topics.most_common(3)
                                if v / len(idx) >= 0.2],
            "with_references": int((np.asarray(A[idx].sum(1)).ravel() > 0).sum()) if A.shape[1] else 0,
            "between": int((agree[idx] < 0.6).sum()),
            "cites_groups": dict(list(uses.items())[:4]),
            "cites_inside": int(flows[c, c]),
            "cloud": clouds[c],
            "lenses": lens,
        })
    # the whole map read as one group: its own reading lists, from every mapped paper's citations
    every = np.arange(n)
    lens = {}
    if n:
        lens, local = lenses_for(-1, every, gr, seed, group, cit, opt, ref_year, ref_ok, None, unit="topic",
                                 ref_seed=ref_seed)
        lens["experts"] = experts_for(every, local, gr, seed, cit, opt.lens_depth, unit="topic")
    years = [d.get("year") for d in seed.details if d.get("year")]
    overall = {"id": 0, "size": int(n), "years": [int(np.percentile(years, p)) for p in (10, 50, 90)] if years else None,
               "cloud": cloud, "lenses": lens}
    timing["lenses_ms"] = round((time.perf_counter() - t) * 1000)

    nrefs = np.asarray(A.sum(1)).ravel() if A.shape[1] else np.zeros(n)
    timing["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return {
        "groups": groups,
        "overall": overall,
        "papers": {"w": [int(w) for w in seed.wids], "group": [int(g) + 1 if g >= 0 else 0 for g in group],
                   "between": [bool(group[i] >= 0 and agree[i] < 0.6) for i in range(n)],
                   # how central each paper is in its group: its link weight into the group / the group's highest
                   "centrality": centrality(WG, group)},
        "coverage": {"papers": n, "with_references": int((nrefs > 0).sum()),
                     "median_references": float(np.median(nrefs)) if n else 0.0,
                     "without_text_vector": int((~seed.has_vec).sum()),
                     "in_groups": int(len(in_groups)), "other": int(n - len(in_groups)),
                     "references_thin": bool(n and (nrefs > 0).mean() < 0.65)},
        "method": {"links": opt.links, "idf": opt.idf, "k_text": opt.k_text, "min_group_size": min_size,
                   **info, "citations": cit.generation,
                   "authors": bool(cit.authors is not None)},
        "timing": timing,
    }

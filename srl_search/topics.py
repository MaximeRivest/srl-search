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

import igraph
import numpy as np
from scipy import sparse

YEAR_NOW = date.today().year
BLOCK = 2000          # rows per block for the big products (memory stays ~linear in n: whole fields are 50k+ papers)
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
    lens_size: int = 8              # per reading list (experts too)


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
    for start in range(0, n, BLOCK):                  # block by block: never the full n x n matrix
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
    if not top:
        S = (X @ X.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        return S
    XT = X.T.tocsc()
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
    nn = np.empty((m, k), np.int64)
    d = np.empty((m, k), np.float64)
    for start in range(0, m, BLOCK):
        D = 1 - V[start:start + BLOCK] @ V.T
        D[np.arange(D.shape[0]), start + np.arange(D.shape[0])] = np.inf
        part = np.argpartition(D, k, axis=1)[:, :k]
        nn[start:start + BLOCK] = part
        d[start:start + BLOCK] = np.take_along_axis(D, part, 1)
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
def _igraph(W: sparse.csr_matrix) -> igraph.Graph:
    U = sparse.triu(W, k=1).tocoo()
    g = igraph.Graph(n=W.shape[0], edges=np.column_stack([U.row, U.col]).tolist(), directed=False)
    g.es["weight"] = U.data.tolist()
    return g


def _leiden(g: igraph.Graph, res: float) -> np.ndarray:
    return np.array(g.community_leiden(objective_function="modularity", weights="weight", resolution=res,
                                       n_iterations=-1).membership)


# Leiden runs, each with its own seed (so the result is the same run in sequence or side by side). Big maps (the
# whole-field worker) run them in forked processes: one run of a 33k-paper map takes ~10 s on one core, and a map
# needs ~35. The search API (maps of <= 2,000 papers) never forks.
PARALLEL_MIN = 5000
_SHARED: dict = {}


def _seeded_run(args) -> np.ndarray:
    res, seed = args
    _RNG.seed(seed)
    return _leiden(_SHARED["g"], res)


def _runs(g: igraph.Graph, jobs: list[tuple[float, int]], workers: int) -> list[np.ndarray]:
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
    g = _igraph(W)
    workers = opt.workers
    with _CLUSTER_LOCK:
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
        E = np.array(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        w = np.asarray(g.es["weight"], dtype=np.float64)
        while True:
            res = opt.resolutions[ri]
            runs = _runs(g, [(res, opt.seed * 1000 + 500 + 50 * ri + k) for k in range(opt.runs)], workers)
            together = np.zeros(len(E))
            for m in runs:
                together += m[E[:, 0]] == m[E[:, 1]]
            frac = together / len(runs)
            keep = frac >= 0.5
            h = igraph.Graph(n=n, edges=E[keep].tolist(), directed=False)
            h.es["weight"] = (frac[keep] * w[keep]).tolist()
            final = _seeded_run_on(h, res, opt.seed * 1000 + 999)
            if _big(final, min_size) <= opt.max_groups or ri == 0:
                break
            ri -= 1                                  # too many groups for one screen: one step coarser
        pick = next(s for s in scan if s["resolution"] == res)
    sizes = Counter(final.tolist())
    order = [c for c, s in sorted(sizes.items(), key=lambda x: (-x[1], x[0])) if s >= min_size]
    remap = {c: i for i, c in enumerate(order)}
    group = np.array([remap.get(c, -1) for c in final.tolist()])
    # agreement: in how many runs the paper sat in the run-cluster that holds most of its final group
    agree = np.zeros(n)
    for m in runs:
        best_of = {}
        for c in set(group.tolist()) - {-1}:
            vals, cnt = np.unique(m[group == c], return_counts=True)
            best_of[c] = vals[np.argmax(cnt)]
        agree += np.array([group[i] >= 0 and m[i] == best_of[group[i]] for i in range(n)])
    agree /= len(runs)
    info = {"resolution": res, "scan": scan, "runs": opt.runs, "seed": opt.seed,
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
regarding according particularly especially""".split())
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


def name_groups(terms: list[set], group: np.ndarray, G: int, query: str, surfaces: dict) -> list[list[str]]:
    """Per group, up to 4 phrases over-represented in it compared with the other papers (the KL divergence term
    p log(p/q), Scimeetr's relative frequency in a form that also weighs how common the phrase is). A phrase is
    skipped when it repeats a word of a chosen one, or mostly names the same papers (Jaccard >= 0.6)."""
    qstems = {stem(w) for w in TOKEN.findall(query or "")}
    n = len(terms)
    df_all = Counter(k for s in terms for k in s)
    names = []
    for c in range(G):
        idx = np.flatnonzero(group == c)
        ng = len(idx)
        df = Counter(k for i in idx for k in terms[i])
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


def lenses_for(c: int, idx: np.ndarray, gr: Graph, seed: Seed, group: np.ndarray, cit, opt: Options,
               ref_year: np.ndarray, ref_ok: np.ndarray, total_local: np.ndarray) -> dict:
    """Reading lists of one group. Items: {"w": work number, "why": sentence, ...numbers}."""
    k = opt.lens_size
    ng = len(idx)
    A = gr.A
    local = np.asarray(A[idx].sum(0)).ravel() if A.shape[1] else np.zeros(0)
    out: dict[str, list] = {}

    order = [j for j in np.argsort(-local, kind="stable") if local[j] >= 2 and ref_ok[j]]
    core = order[:k]
    out["core"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]),
                    "why": f"cited by {int(local[j])} of the {ng} papers in this group"} for j in core]

    lift = np.where(total_local > 0, local / np.maximum(total_local, 1), 0)
    dist = [j for j in np.argsort(-(local * lift), kind="stable")
            if local[j] >= 3 and ref_ok[j] and j not in set(core) and lift[j] >= 0.5][:k]
    out["distinctive"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]), "cited_in_map": int(total_local[j]),
                           "why": f"{int(local[j])} of its {int(total_local[j])} citations in this map come from this group"}
                          for j in dist]

    recent = (ref_year >= YEAR_NOW - 5) & (local >= 3) & ref_ok
    rate = np.where(recent, local / np.maximum(YEAR_NOW - ref_year + 1, 1), 0)
    out["rising"] = [{"w": cit.work(gr.refs[j]), "cited_here": int(local[j]), "year": int(ref_year[j]),
                      "why": f"from {int(ref_year[j])}, already cited by {int(local[j])} papers of this group"}
                     for j in np.argsort(-rate, kind="stable")[:k] if rate[j] > 0]

    top = [j for j in np.argsort(-local, kind="stable")[:30] if local[j] >= 2]
    if top:
        cov = np.asarray(A[idx][:, top].sum(1)).ravel()
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
                              "why": f"cites {int(cov[p])} of the group's {len(top)} most-cited works"
                                     + (" · a review" if seed.details[idx[p]].get("type") == "review" else "")}
                             for p in best]
    else:
        out["start_here"] = []

    newest = sorted(idx, key=lambda i: (-(seed.details[i].get("year") or 0),
                                        -(seed.score[i] if seed.score is not None else 0)))[:k]
    out["newest"] = [{"w": int(seed.wids[i]), "why": ""} for i in newest]
    return out, local


def bridges_for(c: int, idx: np.ndarray, WG: np.ndarray, G: int, k: int, seed: Seed) -> list[dict]:
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
    out = []
    for p in np.argsort(-sc, kind="stable")[:k]:
        if sc[p] <= 0:
            break
        out.append({"w": int(seed.wids[idx[p]]), "to_group": int(best[p]), "share": round(float(bshare[p]), 2),
                    "why": f"{round(100 * float(bshare[p]))}% of its links go to group {int(best[p]) + 1}"})
    return out


def experts_for(idx: np.ndarray, local: np.ndarray, gr: Graph, seed: Seed, cit, k: int) -> list[dict]:
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
                    "why": f"{h} of their papers are each cited by at least {h} papers of this group "
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
def build(seed: Seed, cit, query: str = "", opt: Options | None = None) -> dict:
    """The topic map of these papers: groups, names, lenses, links between groups. Works are given as OpenAlex
    work numbers ("w"); the caller adds titles, links and positions."""
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
    terms = [doc_terms(d.get("title"), d.get("abstract"), surfaces) for d in seed.details]
    names = name_groups(terms, group, G, query, surfaces)
    timing["names_ms"] = round((time.perf_counter() - t) * 1000)

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

    groups = []
    for c in range(G):
        idx = np.flatnonzero(group == c)
        lens, local = lenses_for(c, idx, gr, seed, group, cit, opt, ref_year, ref_ok, total_local)
        lens["bridges"] = bridges_for(c, idx, WG, G, opt.lens_size, seed)
        lens["experts"] = experts_for(idx, local, gr, seed, cit, opt.lens_size)
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
            "lenses": lens,
        })
    timing["lenses_ms"] = round((time.perf_counter() - t) * 1000)

    nrefs = np.asarray(A.sum(1)).ravel() if A.shape[1] else np.zeros(n)
    timing["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return {
        "groups": groups,
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

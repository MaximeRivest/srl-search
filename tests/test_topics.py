"""Topic maps (srl_search/topics.py, topic_service.py) on a made-up literature with three planted research groups:
the groups are found, named by their own words, their reading lists come from their own citations, and the same
input always gives the same answer. No index is loaded."""
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from srl_search import topics as T
from srl_search.topic_service import TopicService, as_text, read_text

TYPES = ["article", "dataset", "review", "software"]
WORDS = {0: ("coral bleaching", "symbiont loss"), 1: ("reef restoration", "coral gardening"),
         2: ("carbonate budget", "reef accretion")}


class FakeAuthors:
    def __init__(self, table, names):
        self.table, self.names = table, names     # node -> [(author row, position)], n listed

    def of(self, node):
        rows = self.table.get(node, [])
        return (np.array([r for r, _ in rows], np.uint32), np.array([p for _, p in rows], np.uint16),
                max([p for _, p in rows], default=0))

    def name(self, row):
        return self.names[row]

    def openalex_id(self, row):
        return f"A{1000 + row}"


class FakeCitations:
    """Works 1..N as nodes 0..N-1 (work number = node + 1)."""
    generation = "test"
    types = TYPES

    def __init__(self, refs, years, kinds, authors=None):
        self._refs = refs
        self.wn = np.arange(1, len(years) + 1, dtype=np.int64)
        self.year = np.array(years)
        self.kind = np.array(kinds, np.uint8)
        self.cites = np.full(len(years), 10)
        off = np.zeros(len(years) + 1, np.int64)
        for i in range(len(years)):
            off[i + 1] = off[i] + len(refs.get(i, []))
        self.fwd_off = off
        self.authors = authors

    def nodes(self, w):
        w = np.asarray(w, np.int64)
        return np.where((w >= 1) & (w <= len(self.wn)), w - 1, -1)

    def refs(self, node):
        return np.array(self._refs.get(int(node), []), np.uint32)

    def work(self, node):
        return int(node) + 1

    def title(self, node):
        return f"Work {node + 1}"

    def type(self, node):
        return TYPES[self.kind[node]]


def literature(seed=0, per=60, with_vectors=True):
    """Three groups of `per` papers (nodes 0..3*per-1). Each group cites its own pool of 40 works (popular ones
    more), and all cite one textbook and one software package; titles use the group's words; vectors sit near
    the group's direction. References are nodes 1000+."""
    rng = np.random.default_rng(seed)
    n = 3 * per
    refs, years = {}, [2015 + int(rng.integers(0, 11)) for _ in range(n)]
    POOL = 1000
    years += [0] * (POOL - n) + [2000 + (i % 26) for i in range(130)]
    kinds = [0] * len(years)
    textbook, software = POOL + 120, POOL + 121
    kinds[software] = TYPES.index("software")
    details, vecs = [], []
    centres = rng.normal(size=(3, 64))
    for i in range(n):
        g = i // per
        pool = np.arange(POOL + 40 * g, POOL + 40 * g + 40)
        p = 1 / np.arange(1, 41)
        pick = rng.choice(pool, size=12, replace=False, p=p / p.sum())
        refs[i] = sorted(set(pick.tolist()) | {textbook, software})
        a, b = WORDS[g]
        details.append({"title": f"{a.title()} and {b} in reef {i}", "abstract": f"We studied {a} and {b}. Coral reefs matter.",
                        "year": years[i], "type": "review" if i % 17 == 0 else "article", "topic": f"Topic {g}"})
        v = centres[g] + 0.6 * rng.normal(size=64)
        vecs.append(v / np.linalg.norm(v))
    # recent works in group 0's pool, so "rising" has something to find
    for j in range(POOL, POOL + 5):
        years[j] = 2024
    authors = {}
    for j in range(POOL, POOL + 120):           # group g's pool: author row g is first author of every work
        g = (j - POOL) // 40
        authors[j] = [(g, 1), (10 + j, 2)]
    cit = FakeCitations(refs, years, kinds, FakeAuthors(authors, {0: "Ana", 1: "Ben", 2: "Cleo", **{10 + j: f"X{j}" for j in range(POOL, POOL + 120)}}))
    V = np.array(vecs, np.float32) if with_vectors else np.zeros((n, 64), np.float32)
    seed_ = T.Seed(np.arange(1, n + 1, dtype=np.int64), V, np.full(n, with_vectors), details,
                   np.linspace(1, 0, n).astype(np.float32))
    return seed_, cit


def planted(out, per=60):
    return np.array(out["papers"]["group"]), np.repeat([0, 1, 2], per)


def test_finds_the_planted_groups_and_names_them_by_their_own_words():
    seed, cit = literature()
    out = T.build(seed, cit, "coral reefs")
    group, truth = planted(out)
    assert len(out["groups"]) == 3
    import igraph
    assert igraph.compare_communities(group.tolist(), truth.tolist(), method="adjusted_rand") > 0.95
    for g in out["groups"]:
        members = np.flatnonzero(group == g["id"])
        planted_g = np.bincount(truth[members]).argmax()
        words = " ".join(g["phrases"]).lower()
        a, b = WORDS[planted_g]
        assert a in words or b in words, (g["phrases"], WORDS[planted_g])
        assert "coral reefs" not in words and "reef" not in g["name"].split(" · ")   # question words say nothing


def test_reading_lists_come_from_the_groups_own_citations():
    seed, cit = literature()
    out = T.build(seed, cit, "coral reefs")
    group, truth = planted(out)
    for g in out["groups"]:
        planted_g = np.bincount(truth[group == g["id"]]).argmax()
        pool = range(1000 + 40 * planted_g + 1, 1000 + 40 * planted_g + 41)   # work numbers = node + 1
        core = [it["w"] for it in g["lenses"]["core"]]
        assert core and all(w in pool or w == 1121 for w in core)
        assert 1122 not in core                                   # software is not reading material
        assert all(w in pool for w in (it["w"] for it in g["lenses"]["distinctive"]))   # textbook is cited by all
        assert all(1 <= it["w"] <= 180 for it in g["lenses"]["start_here"] + g["lenses"]["newest"])
        assert [e["name"] for e in g["lenses"]["experts"]][:1] == [["Ana", "Ben", "Cleo"][planted_g]]
        assert g["lenses"]["experts"][0]["h"] >= 3
    rising = [it for g in out["groups"] for it in g["lenses"]["rising"]]
    assert rising and all(it["year"] >= T.YEAR_NOW - 5 for it in rising)
    assert any(it["w"] == 1001 for it in rising)       # 2024, cited by most of its group


def test_same_input_same_answer_and_options_change_it():
    seed, cit = literature()
    a, b = T.build(seed, cit, "x"), T.build(seed, cit, "x")
    a.pop("timing"); b.pop("timing")
    assert a == b
    assert a["method"]["resolution"] in T.Options().resolutions
    assert 0.9 <= a["method"]["consensus_agreement"] <= 1


def test_text_alone_or_references_alone_still_group():
    seed, cit = literature()
    for links in ("refs", "text"):
        out = T.build(seed, cit, "x", T.Options(links=links))
        group, truth = planted(out)
        import igraph
        assert igraph.compare_communities(group.tolist(), truth.tolist(), method="adjusted_rand") > 0.8, links


def test_papers_without_vectors_group_by_references():
    seed, cit = literature(with_vectors=False)
    out = T.build(seed, cit, "x")
    assert out["coverage"]["without_text_vector"] == 180
    assert len(out["groups"]) == 3


def test_too_few_papers_is_no_groups_not_an_error():
    seed, cit = literature()
    out = T.build(seed.subset(np.arange(10)), cit, "x")
    assert out["groups"] == [] or len(out["groups"]) == 1


def test_flows_between_groups():
    seed, cit = literature()
    # every paper of the second planted group also cites a paper of the first
    for i in range(60, 120):
        cit._refs[i] = cit._refs[i] + [5]
    cit.fwd_off = np.r_[0, np.cumsum([len(cit._refs.get(i, [])) for i in range(len(cit.year))])]
    out = T.build(seed, cit, "x")
    group, _ = planted(out)
    g_first, g_second = group[0], group[60]
    second = next(g for g in out["groups"] if g["id"] == g_second)
    assert second["cites_groups"].get(int(g_first)) == 60


def test_credit_and_h_index():
    assert T.credit(np.array([1, 2, 3, 4]), 4).tolist() == [1, 1 / 3, 1 / 4, 1 / 2]   # last author ranks second
    assert T.credit(np.array([1, 2]), 2).tolist() == [1, 1 / 2]
    assert T.h_index([10, 8, 5, 4, 3]) == 4
    assert T.h_index([0.5]) == 0 and T.h_index([]) == 0


def test_idf_coupling_ignores_what_everyone_cites():
    from scipy import sparse
    A = sparse.csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [1, 0, 1]], np.float32))
    S = T.coupling(A, idf=True).toarray()
    assert S[0, 1] == 0 and S[1, 2] > 0.99
    S = T.coupling(A, idf=False).toarray()
    assert S[0, 1] > 0


def test_text_graph_is_symmetric_and_bounded():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(50, 8)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    has = np.ones(50, bool); has[3] = False; V[3] = 0
    W = T.text_graph(V, has, 10).toarray()
    assert np.allclose(W, W.T) and W.min() >= 0 and W.max() <= 1 and W[3].sum() == 0


def test_labels_skip_question_words_and_keep_acronyms():
    surfaces = {}
    terms = [T.doc_terms("COVID-19 and remote work productivity", None, surfaces) for _ in range(10)] + \
            [T.doc_terms("Telework and family conflict", None, surfaces) for _ in range(10)]
    names = T.name_groups(terms, np.repeat([0, 1], 10), 2, "remote work", surfaces)
    assert "COVID-19" in " ".join(names[0]) and not any("remote work" == p for p in names[0])
    assert any("family conflict" in p or "telework" in p for p in names[1])


# ------------------------------------------------------------------------------------------ the service
class FakeApi:
    RELEASE = {"id": "r1"}
    KEYWORDS = None
    ATLAS = None

    def __init__(self):
        self.engine = SimpleNamespace(details=lambda ws: {w: {"title": f"T{w}", "year": 2020, "type": "article",
                                                               "authors": "A | B | C", "authors_count": 3,
                                                               "abstract": f"<p>Abstract of {w}.</p>"} for w in ws})

    @staticmethod
    def looks_boolean(q):
        return '"' in q

    def readable(self, w):
        return w == 1

    def place_work(self, w):
        return (0.5, 0.5)


def test_service_caches_per_release_and_computes_once(tmp_path, monkeypatch):
    (tmp_path / "old-release").mkdir()
    svc = TopicService(FakeApi(), tmp_path, FakeCitations({}, [2020], [0]))
    assert not (tmp_path / "old-release").exists()
    calls = []

    def compute(q, papers, mode, yf, yt):
        calls.append(q)
        time.sleep(0.2)
        return {"question": q, "mode": mode, "groups": [], "timing": {}, "papers": {"id": [], "group": []},
                "filters": {}, "asked_papers": papers}

    monkeypatch.setattr(svc, "compute", compute)
    outs = []
    th = [threading.Thread(target=lambda: outs.append(svc.get("coral  reefs"))) for _ in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert calls == ["coral reefs"] and len(outs) == 3
    assert svc.get("coral reefs")["cached"] is True
    svc.get("coral reefs", papers=500)
    assert len(calls) == 2


def test_service_refuses_when_full(tmp_path, monkeypatch):
    import srl_search.topic_service as S
    monkeypatch.setattr(S, "WAIT_S", 0.1)
    svc = TopicService(FakeApi(), tmp_path, FakeCitations({}, [2020], [0]))
    for _ in range(S.MAX_CONCURRENT):
        svc.slots.acquire()
    with pytest.raises(HTTPException) as e:
        svc.get("anything")
    assert e.value.status_code == 503 and e.value.headers["Retry-After"]


def test_finish_and_text_on_the_planted_map(tmp_path):
    seed, cit = literature()
    svc = TopicService(FakeApi(), tmp_path, cit)
    out = svc.finish(T.build(seed, cit, "coral reefs"), seed)
    out.update({"question": "coral reefs", "mode": "semantic", "filters": {}, "asked_papers": 1000})
    g = out["groups"][0]
    assert g["typical"]["id"].startswith("W") and g["centre"] == {"x": 0.5, "y": 0.5}
    it = g["lenses"]["core"][0]
    assert it["title"] == f"T{it['id'][1:]}" and "why" in it and it["in_map"] is False
    assert g["lenses"]["experts"][0]["top"]["id"].startswith("W")
    assert len(out["papers"]["id"]) == len(out["papers"]["x"]) == 180
    text = as_text(out, "https://api.example")
    assert text.startswith("# Research groups in: coral reefs") and "## Group 1:" in text and "Start here" in text
    assert "https://api.example/topics/coral%20reefs?group=N" in text
    # complete: every item of every list and every expert is in the text (small pages: across all of them)
    import srl_search.topic_service as S0
    keep = S0.PAGE_CHARS
    S0.PAGE_CHARS = 9000 + 2500
    try:
        p1 = as_text(out, "https://api.example")
        pages = [p1]
        while "The map continues" in pages[-1]:
            pages.append(as_text(out, "https://api.example", len(pages) + 1))
        assert len(pages) > 1 and "groups at a glance" in pages[0]
        assert all(f"Group {g['id']}:" in pages[0] for g in out["groups"])
        with pytest.raises(HTTPException):
            as_text(out, "x", len(pages) + 1)
        text = "".join(pages)
    finally:
        S0.PAGE_CHARS = keep
    for g in out["groups"]:
        for name, items in g["lenses"].items():
            for it in items:
                assert (it["name"] if name == "experts" else it["title"]) in text
    # one group in depth: the lists with abstracts, then every paper, page by page, most central first
    import srl_search.topic_service as S
    r = svc.read(out, 1, "lists")
    assert r["pages"] == 1 and all(p.get("abstract", "").startswith("Abstract of") for p in r["lenses"]["core"])
    given = [p["id"] for items in r["lenses"].values() for p in items if "id" in p and not p.get("given_above")]
    assert len(given) == len(set(given))                     # each paper's abstract once
    t = read_text(r, "https://api.example")
    assert "Abstract of" in t and "<p>" not in t and "part=papers, page=1" in t
    first = svc.read(out, 1, "papers", 1)
    members = sum(1 for g in out["papers"]["group"] if g == 1)
    assert first["total"] == members
    cen = [p["centrality"] for p in first["papers"]]
    assert cen == sorted(cen, reverse=True) and cen[0] == 1.0
    seen = [p["id"] for pg in range(1, first["pages"] + 1) for p in svc.read(out, 1, "papers", pg)["papers"]]
    assert len(seen) == len(set(seen)) == members              # every paper, once
    # small pages: everything is still there, on more pages, each within the budget
    old_budget = S.PAGE_CHARS
    S.PAGE_CHARS = 3000
    try:
        p1 = svc.read(out, 1, "papers", 1)
        assert p1["pages"] > 1 and "Next: page=2" in read_text(p1, "x")
        all_pages = [svc.read(out, 1, "papers", pg) for pg in range(1, p1["pages"] + 1)]
        assert sum(len(x["papers"]) for x in all_pages) == members
        assert all(len(read_text(x, "x")) < 3000 + 1500 for x in all_pages)
        assert "last page" in read_text(all_pages[-1], "x")
        lp = svc.read(out, 1, "lists", 1)
        n_items = sum(len(v) for v in out["groups"][0]["lenses"].values())
        got = sum(len(v) for pg in range(1, lp["pages"] + 1) for v in svc.read(out, 1, "lists", pg)["lenses"].values())
        assert lp["pages"] > 1 and got == n_items                # every list item, across pages
    finally:
        S.PAGE_CHARS = old_budget
    with pytest.raises(HTTPException):
        svc.read(out, 99, "lists")


def test_a_preprint_and_its_journal_version_count_once():
    seed, cit = literature()
    seed.details[5] = {**seed.details[5], "authors": "Cathie A. Page | B", "type": "article"}
    seed.details[7] = {**seed.details[7], "authors": "Cathie Page | C", "type": "preprint"}
    seed.details[9] = {**seed.details[9], "authors": "Someone Else", "type": "article"}
    seed.vecs[7] = seed.vecs[5]
    seed.vecs[9] = seed.vecs[5]                     # same words, another first author: a different study
    keep = T.collapse_versions(seed, cit).tolist()
    assert 5 in keep and 7 not in keep and 9 in keep and len(keep) == len(seed.wids) - 1


def test_a_group_that_does_not_split_is_still_a_valid_answer():
    import json
    seed, cit = literature()
    one = seed.subset(np.arange(60))                    # one planted group: nothing to split
    out = T.build(one, cit, "x", T.Options(min_size=8, min_groups=2, max_groups=8))
    json.dumps(out, allow_nan=False)
    assert all(math.isfinite(s["stability"]) for s in out["method"]["scan"])


def test_service_answers_even_when_the_cache_cannot_be_written(tmp_path, monkeypatch):
    import shutil
    svc = TopicService(FakeApi(), tmp_path, FakeCitations({}, [2020], [0]))
    monkeypatch.setattr(svc, "compute", lambda *a: {"question": "q", "groups": [], "timing": {}, "papers": {}})
    shutil.rmtree(svc.cache)
    assert svc.get("again")["cached"] is False and svc.cache.exists()      # the folder comes back
    svc.cache.chmod(0o500)
    try:
        assert svc.get("read only")["cached"] is False
    finally:
        svc.cache.chmod(0o700)


def test_reading_a_group_never_makes_a_map(tmp_path, monkeypatch):
    svc = TopicService(FakeApi(), tmp_path, FakeCitations({}, [2020], [0]))
    monkeypatch.setattr(svc, "compute", lambda *a: (_ for _ in ()).throw(AssertionError("computed")))
    with pytest.raises(HTTPException) as e:
        svc.get("never mapped", cached_only=True)
    assert e.value.status_code == 409 and "first" in e.value.detail


def test_one_person_split_in_two_author_records_is_one_expert():
    seed, cit = literature()
    au = cit.authors
    au.names[3] = "Ana"                        # group 0's pool: half of the works now by "another" Ana record
    for j in range(1000, 1020):
        au.table[j] = [(3, 1), (10 + j, 2)]
    out = T.build(seed, cit, "coral reefs")
    names = [e["name"] for g in out["groups"] for e in g["lenses"]["experts"]]
    assert names.count("Ana") == 1


def test_a_book_and_its_chapters_count_once():
    oup = "10.1093/oso/9780198838388"
    assert T.book_of({"type": "book-chapter", "doi": oup + ".003.0005"}) == T.book_of({"type": "book", "doi": oup + ".001.0001"})
    assert T.book_of({"type": "book-chapter", "doi": "10.1007/978-3-030-1_5"}) == "10.1007/978-3-030-1"
    assert T.book_of({"type": "article", "doi": "10.1/x"}) is None
    seed, cit = literature()
    for i in range(0, 60, 2):                       # half of group 0 becomes chapters of one book
        seed.details[i] = {**seed.details[i], "type": "book-chapter", "doi": f"{oup}.003.{i:04d}"}
    out = T.build(seed, cit, "x")
    for g in out["groups"]:
        chapters = [it for it in g["lenses"]["start_here"] if seed.details[it["w"] - 1]["type"] == "book-chapter"]
        assert len(chapters) <= 1


def test_the_whole_topic_has_its_own_reading_lists():
    seed, cit = literature()
    out = T.build(seed, cit, "coral reefs")
    o = out["overall"]
    assert o["size"] == 180 and set(o["lenses"]) == {"core", "rising", "start_here", "newest", "experts"}
    core = [it["w"] for it in o["lenses"]["core"]]
    assert core[0] == 1121                                  # the textbook every paper cites leads the whole topic
    assert 1122 not in core                                 # software is still not reading material
    assert "180 papers in this topic" in o["lenses"]["core"][0]["why"]
    # every group's best-known first author is an expert of the whole topic
    assert {"Ana", "Ben", "Cleo"} <= {e["name"] for e in o["lenses"]["experts"]}
    assert all("of this topic" in e["why"] for e in o["lenses"]["experts"])
    assert o["cloud"] == []                                 # no background corpus: no cloud for the whole topic


def test_each_group_has_a_cloud_of_its_own_words():
    seed, cit = literature()
    out = T.build(seed, cit, "coral reefs")
    group, truth = planted(out)
    for g in out["groups"]:
        planted_g = np.bincount(truth[group == g["id"]]).argmax()
        words = [w["t"].lower() for w in g["cloud"]]
        assert any(w in words for w in WORDS[planted_g]), (words, WORDS[planted_g])
        assert all(w["lift"] >= T.CLOUD_MIN_LIFT and 0 < w["share"] <= 1 for w in g["cloud"])
        assert [w["score"] for w in g["cloud"]] == sorted((w["score"] for w in g["cloud"]), reverse=True)
        assert "coral reefs" not in words and "reef" not in words            # the question's words
        others = {w for k, v in WORDS.items() if k != planted_g for w in v}
        assert not others & set(words)                                         # nothing of the other groups
        # a word standing inside one of its phrases is not shown twice ("bleaching" under "coral bleaching")
        assert not any(w != x and f" {w} " in f" {x} " for w in words for x in words)


def test_the_whole_topic_cloud_is_against_all_of_science():
    seed, cit = literature()
    asked = []

    def background(phrases):
        asked.append(phrases)
        # everyday science words are common everywhere; the topic's own phrases are rare outside it
        return {p: (5_000_000 if p.lower() in ("coral", "matter", "studied") else 300) for p in phrases}, 200_000_000
    out = T.build(seed, cit, "coral reefs", background=background)
    words = [w["t"].lower() for w in out["overall"]["cloud"]]
    assert asked and len(asked[0]) <= T.CLOUD_CANDIDATES
    assert words and "matter" not in words and "coral" not in words
    assert any(w in words for v in WORDS.values() for w in v)
    # a failing background costs the cloud, not the map
    out = T.build(seed, cit, "coral reefs", background=lambda p: (_ for _ in ()).throw(OSError("index gone")))
    assert out["overall"]["cloud"] == [] and len(out["groups"]) == 3


def test_relevance_weighs_frequency_and_lift():
    assert T.relevance(0.5, 0.05) > T.relevance(0.05, 0.005)    # same lift, more papers
    assert T.relevance(0.2, 0.01) > T.relevance(0.2, 0.1)        # same share, more distinctive
    assert T.relevance(0.3, 0.3, lam=0) == 0


def test_the_whole_topic_reads_as_group_zero(tmp_path):
    seed, cit = literature()
    svc = TopicService(FakeApi(), tmp_path, cit)
    out = svc.finish(T.build(seed, cit, "coral reefs"), seed)
    out.update({"question": "coral reefs", "mode": "semantic", "filters": {}, "asked_papers": 1000})
    it = out["overall"]["lenses"]["core"][0]
    assert it["id"] == "W1121" and it["title"] == "T1121" and "w" not in it
    assert out["overall"]["lenses"]["experts"][0]["top"]["id"].startswith("W")
    text = as_text(out, "https://api.example")
    assert "## Reading lists of the whole topic (all 180 papers" in text
    assert text.index("Reading lists of the whole topic") < text.index("## Group 1:")
    assert "group=0" in text and "?read=0" in text and "its words, against the other groups:" in text
    r = svc.read(out, 0, "lists")
    assert r["group"]["id"] == 0 and r["lenses"]["core"][0]["abstract"].startswith("Abstract of")
    t = read_text(r, "https://api.example")
    assert t.startswith("# Reading lists of the whole topic of “coral reefs” (180 papers") and "Start here: recent papers of the topic" in t
    assert "part=papers" not in t and "?read=1" in t
    with pytest.raises(HTTPException) as e:
        svc.read(out, 0, "papers")
    assert e.value.status_code == 400
    # maps made before the whole topic had lists: group 0 does not exist, the text is as before
    old = {k: v for k, v in out.items() if k != "overall"}
    with pytest.raises(HTTPException) as e:
        svc.read(old, 0, "lists")
    assert e.value.status_code == 404
    assert "Reading lists of the whole topic" not in as_text(old, "https://api.example")

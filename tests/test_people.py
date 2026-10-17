"""Author and institution profiles (srl_search/people.py) on a made-up literature: the fields are the boxes,
the lists are theirs, co-authors and partners are counted right, and the name index finds people. No index loaded."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from srl_search import people as P

TYPES = ["article", "dataset", "review"]
TOPICS = [{"id": "T1", "name": "Coral Bleaching", "subfield": "Marine Biology", "field": "Agricultural and Biological Sciences"},
          {"id": "T2", "name": "Reef Fish", "subfield": "Marine Biology", "field": "Agricultural and Biological Sciences"},
          {"id": "T3", "name": "Ocean Carbon", "subfield": "Oceanography", "field": "Earth and Planetary Sciences"}]


def csr(rows: dict[int, list], n: int):
    off = np.zeros(n + 1, np.uint64)
    vals = []
    for i in range(n):
        off[i + 1] = off[i] + len(rows.get(i, []))
        vals += rows.get(i, [])
    return off, vals


class FakeAuthors:
    """6 papers; authors 0 (Ann Reef), 1 (Bo Fish), 2 (Cy Sea); paper 5 is a dataset (not a publication)."""
    by = True

    def __init__(self):
        table = {0: [(0, 1), (1, 2)], 1: [(0, 1), (1, 2), (2, 3)], 2: [(0, 2), (2, 1)], 3: [(1, 1)], 4: [(0, 1)], 5: [(0, 1)]}
        self.off, pairs = csr(table, 6)
        self.auth = np.array([a for a, _ in pairs], np.uint32)
        self.pos = np.array([p for _, p in pairs], np.uint16)
        self.nauth = np.array([2, 3, 2, 1, 1, 1], np.uint16)
        self.an = np.array([101, 102, 103])
        self._names = ["Ann Reef", "Bo Fish", "Cy Sea"]
        mine = {0: [(0, 1), (1, 1), (2, 2), (4, 1), (5, 1)], 1: [(0, 2), (1, 2), (3, 1)], 2: [(1, 3), (2, 1)]}
        self.by_off, mp = csr(mine, 3)
        self.by_node = np.array([n for n, _ in mp], np.uint32)
        self.by_pos = np.array([p for _, p in mp], np.uint16)
        self.works = np.array([4, 3, 2], np.uint32)
        self.cites = np.array([100, 50, 10], np.uint32)
        self.h = np.array([3, 2, 1], np.uint16)
        self.year0 = np.array([2010, 2011, 2012], np.uint16)
        self.year1 = np.array([2024, 2020, 2016], np.uint16)
        self.topic = np.array([0, 1, 2], np.int16)

    def of(self, node):
        a, b = int(self.off[node]), int(self.off[node + 1])
        return self.auth[a:b], self.pos[a:b], int(self.nauth[node])

    def papers(self, row):
        a, b = int(self.by_off[row]), int(self.by_off[row + 1])
        return self.by_node[a:b], self.by_pos[a:b]

    def name(self, row):
        return self._names[row]

    def openalex_id(self, row):
        return f"A{self.an[row]}"

    def row_of(self, an):
        i = int(np.searchsorted(self.an, an))
        return i if i < len(self.an) and self.an[i] == an else None


class FakeInstitutions:
    """Two organisations: 0 (Reef University, papers 0 1 2 4), 1 (Sea Lab, papers 1 3)."""

    def __init__(self):
        self.w_off, wi = csr({0: [0], 1: [0, 1], 2: [0], 3: [1], 4: [0]}, 6)
        self.w_inst = np.array(wi, np.uint32)
        self.i_off, ip = csr({0: [0, 1, 2, 4], 1: [1, 3]}, 2)
        self.i_node = np.array(ip, np.uint32)
        self.inst_in = np.array([7, 9])

    def papers(self, row):
        return self.i_node[int(self.i_off[row]):int(self.i_off[row + 1])]

    def of_paper(self, node):
        return self.w_inst[int(self.w_off[node]):int(self.w_off[node + 1])]

    def of_author(self, row):
        return [(0, 3, 2010, 2024)] if row == 0 else []

    def name(self, row):
        return ["Reef University", "Sea Lab"][row]

    def openalex_id(self, row):
        return f"I{self.inst_in[row]}"

    def row_of(self, i):
        return {7: 0, 9: 1}.get(i)

    def describe(self, row):
        return {"id": self.openalex_id(row), "name": self.name(row), "country": "AU", "kind": "education", "ror": None}


class FakeCitations:
    types = TYPES

    def __init__(self, tmp: Path):
        self.root = tmp
        self.kind = np.array([0, 0, 2, 0, 0, 1], np.uint8)
        self.year = np.array([2010, 2015, 2016, 2020, 2024, 2024], np.uint16)
        self.cites = np.array([40, 30, 20, 5, 1, 0], np.uint32)
        self.wn = np.array([1000, 1001, 1002, 1003, 1004, 1005])
        self.authors = FakeAuthors()
        self.institutions = FakeInstitutions()
        np.save(tmp / "topic.npy", np.array([0, 0, 1, 2, -1, 0], np.int16))
        (tmp / "topics.json").write_text(__import__("json").dumps(TOPICS))

    def work(self, node):
        return int(self.wn[node])

    def nodes(self, ws):
        return np.array([int(np.searchsorted(self.wn, w)) if w in self.wn else -1 for w in ws])

    def title(self, node):
        return f"Paper {node}"

    def type(self, node):
        return TYPES[self.kind[node]]


FAKE_API = SimpleNamespace(engine=SimpleNamespace(details=lambda ws: {}), place_work=lambda w: (0.1, 0.2), readable=lambda w: None)


@pytest.fixture
def svc(tmp_path):
    return P.People(FAKE_API, FakeCitations(tmp_path), None)


def test_author_profile_groups_lists_and_coauthors(svc):
    raw = svc.raw("author", 0)
    assert raw["n"] == 4                                     # the dataset is not a publication
    assert raw["level"] == "subfields"                       # 3 of 4 labelled papers in one field
    assert [g["name"] for g in raw["groups"]] == ["Marine Biology", "Oceanography"] or \
        [g["name"] for g in raw["groups"]][0] == "Marine Biology"
    assert raw["other"] == 1                                 # paper 4 has no topic
    ov = raw["overall"]["lenses"]
    assert ov["cited"][:2] == [0, 1] and ov["newest"][0] == 4
    assert ov["first"] == [0, 1, 4]                          # as first author, most cited first
    co = ov["coauthors"]
    assert [(c["name"], c["n"]) for c in co] == [("Bo Fish", 2), ("Cy Sea", 2)]
    assert co[0]["top"] == 0                                 # the most cited shared paper
    p = raw["person"]
    assert p["name"] == "Ann Reef" and p["positions"] == {"first": 3, "second": 1, "other": 0}
    assert p["years"] == [2010, 2024]
    assert p["institutions"][0]["name"] == "Reef University"


def test_profile_resolves_heads_and_more_pages(svc):
    d = svc.profile("author", 0)
    assert d["kind"] == "author" and d["mode"] == "author" and d["question"] == "Ann Reef"
    assert d["papers"]["id"] == ["W1000", "W1001", "W1002", "W1004"] and d["papers"]["x"] == [0.1] * 4
    assert d["overall"]["lenses"]["cited"][0]["id"] == "W1000"
    assert d["overall"]["lenses"]["coauthors"][0]["top"]["id"] == "W1000"
    page = svc.more("author", 0, "", "cited", 1, 2)
    assert [x["id"] for x in page["items"]] == ["W1001", "W1002"] and page["total"] == 4
    page = svc.more("author", 0, "1", "coauthors", 0, 5)
    assert page["items"] and page["total"] >= 1
    with pytest.raises(Exception):
        svc.more("author", 0, "9", "cited", 0, 5)


def test_institution_profile_people_and_partners(svc):
    raw = svc.raw("institution", 0)
    assert raw["n"] == 4 and raw["person"]["name"] == "Reef University"
    assert raw["person"]["authors"] == 3
    ov = raw["overall"]["lenses"]
    assert [(x["name"], x["n"]) for x in ov["people"]][0] == ("Ann Reef", 4)
    assert [(x["name"], x["n"]) for x in ov["partners"]] == [("Sea Lab", 1)]
    d = svc.profile("institution", 0)
    assert d["overall"]["lenses"]["partners"][0]["institution"] == "I9"
    assert d["overall"]["cloud"][0]["author"] == "A101"


def test_consortium_papers_do_not_count_for_coauthors(svc):
    svc.au.nauth[1] = 500                                     # paper 1 lists 500 authors
    raw = svc._raw_author(0)
    assert [(c["name"], c["n"]) for c in raw["overall"]["lenses"]["coauthors"]] == [("Bo Fish", 1), ("Cy Sea", 1)]


def test_gather_is_the_csr_expansion():
    off = np.array([0, 2, 2, 5], np.uint64)
    owner, at = P.gather(off, np.array([2, 0]))
    assert owner.tolist() == [0, 0, 0, 1, 1] and at.tolist() == [2, 3, 4, 0, 1]
    owner, at = P.gather(off, np.array([1]))
    assert owner.tolist() == [] and at.tolist() == []


def test_name_index_finds_people(tmp_path):
    import tantivy
    (tmp_path / "index").mkdir()
    S = P.schema(tantivy)
    ix = tantivy.Index(S, path=str(tmp_path / "index"))
    ix.register_tokenizer("names", P.analyzer(tantivy))
    w = ix.writer(30_000_000)
    for row, (name, works) in enumerate([("Vincent Larivière", 300), ("Vincent Lariviere", 5), ("V. Larivière", 2),
                                         ("Wei Wang", 1000), ("Wei Wang", 20), ("José García-López", 40)]):
        w.add_document(tantivy.Document(name=name, works=works, row=row))
    w.commit()
    (tmp_path / "meta.json").write_text("{}")
    ni = P.NameIndex(tmp_path)
    assert ni.search("lariviere", 10) == ([0, 1, 2], "exact")          # accents and case folded, most published first
    assert ni.search("v larivière", 10) == ([0, 1, 2], "exact")        # a one-letter word is an initial
    assert ni.search("wang wei", 10) == ([3, 4], "exact")              # order of words does not matter
    assert ni.search("garcia", 10)[0] == [5] and ni.search("lopez", 10)[0] == [5]
    assert ni.search("larivier", 10) == ([0, 1, 2], "fuzzy")           # one letter off, when nothing matches
    assert ni.search("nobody here", 10) == ([], "none")
    assert ni.search("...", 10) == ([], "empty")

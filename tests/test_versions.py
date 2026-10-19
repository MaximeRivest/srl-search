"""Versions in results: a paper's records are listed once, as its readable or canonical version."""
import sqlite3
import numpy as np
from srl_search.engine import Engine
from srl_search.versions import Versions


def make_versions(tmp_path):
    # group 0: 10 (canonical journal article), 11 (preprint), 12 (mirror copy); 30 alone
    np.save(tmp_path / "work.npy", np.array([10, 11, 12], np.uint64))
    np.save(tmp_path / "cluster.npy", np.array([0, 0, 0], np.uint32))
    np.save(tmp_path / "rank.npy", np.array([0, 1, 2], np.uint16))
    c = sqlite3.connect(tmp_path / "versions.sqlite")
    c.execute("CREATE TABLE members (cluster INTEGER, rank INTEGER, work INTEGER, type TEXT, year INTEGER, doi TEXT, venue TEXT, has_abstract INTEGER)")
    c.executemany("INSERT INTO members VALUES (?,?,?,?,?,?,?,?)", [(0, 0, 10, "article", 2021, "10.1/j", "J", 0),
                  (0, 1, 11, "preprint", 2020, "10.48550/x", "arXiv", 1), (0, 2, 12, "article", 2021, "10.60692/y", "GSIS", 1)])
    c.commit(); c.close()
    return Versions(tmp_path)


def fake_engine(versions, held):
    e = Engine.__new__(Engine)
    e.versions = versions
    rec = lambda w, smr=0: {"title": f"t{w}", "abstract": f"a{w}", "year": 2021, "type": "article", "smr": smr, "is_retracted": 0}
    e.details = lambda ids: {w: rec(w, held[w]) for w in ids if w in held}
    return e


def test_listed_once_as_canonical(tmp_path):
    e = fake_engine(make_versions(tmp_path), {10: 0, 11: 0, 12: 0, 30: 0})
    out = e.shape([(12, 0.9), (30, 0.8), (11, 0.7)], k=10, filters={}, abstracts=True)
    assert [r["id"] for r in out] == ["W10", "W30"]          # the mirror hit shows the journal article; preprint folded
    assert out[0]["score"] == 0.9 and out[0]["versions_count"] == 3
    assert {v["id"] for v in out[0]["versions"]} == {"W11", "W12"}
    assert "versions" not in out[1]


def test_readable_version_shown(tmp_path):
    e = fake_engine(make_versions(tmp_path), {10: 0, 11: 1, 12: 0})
    out = e.shape([(12, 0.9)], k=10, filters={}, abstracts=True)
    assert out[0]["id"] == "W11"


def test_canonical_not_held(tmp_path):
    # the version of record is not in the corpus (no abstract): the best held version is shown
    e = fake_engine(make_versions(tmp_path), {11: 0, 12: 0})
    out = e.shape([(12, 0.9)], k=10, filters={}, abstracts=True)
    assert out[0]["id"] == "W11"


def test_long_cache_names(tmp_path):
    from srl_search.people import People
    p = People.__new__(People)
    p.size_of = lambda kind, row: 10 ** 6
    p.ident = lambda kind, row: ",".join(f"A{5100000000 + i}" for i in range(20))
    path = p._cached(tmp_path, "author", (1,))
    assert len(path.name) < 100 and path.name.startswith("A5100000000+19-")

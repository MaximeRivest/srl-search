import json

import numpy as np

from srl_search.engine import passes
from srl_search.vectors import Vectors, select_shards


def make_shards(root, sizes, date="2024-05-01"):
    rng = np.random.default_rng(0)
    shards, all_v, all_w = [], [], []
    w = 100
    for i, n in enumerate(sizes):
        d = root / f"year=2024/rg-000000/part-{i:03d}"
        d.mkdir(parents=True)
        v = rng.standard_normal((n, 768)).astype(np.float16)
        ids = np.arange(w, w + n, dtype=np.uint64); w += n
        np.save(d / "vectors.npy", v); np.save(d / "work_ids.npy", ids)
        (d / "commit.json").write_text(json.dumps({"task": {"publication_date": date, "rows": n}}))
        all_v.append(v); all_w.append(ids)
    return np.concatenate(all_v), np.concatenate(all_w)


def test_positions_map_to_the_right_rows(tmp_path):
    v, w = make_shards(tmp_path, [3, 5, 2])
    shards = select_shards("2024-01-01", tmp_path)
    vec = Vectors(shards, tmp_path)
    assert vec.total == 10
    pos = np.array([9, 0, 4, 3, 2])
    assert np.array_equal(vec.get(pos), v[pos].astype(np.float32))
    assert np.array_equal(vec.work_ids(pos), w[pos])
    firsts = [f for f, _, _ in vec.iter_shards()]
    assert firsts == [0, 3, 8]


def test_date_cutoff(tmp_path):
    make_shards(tmp_path, [2], date="2023-12-31")
    assert select_shards("2024-01-01", tmp_path) == []
    assert len(select_shards("2023-01-01", tmp_path)) == 1


def test_filters():
    d = {"year": 2025, "type": "article", "is_oa": 1, "license": "cc-by", "pmc_oa": 1, "is_retracted": 0}
    assert passes(d, {})
    assert not passes(d, {"year_from": 2026})
    assert not passes(d, {"year_to": 2024})
    assert passes(d, {"type": "article", "open_access": True, "license": "cc-by", "pmc_open_access": True})
    assert not passes({**d, "pmc_oa": None}, {"pmc_open_access": True})
    assert not passes({**d, "is_retracted": 1}, {})
    assert not passes({**d, "type": "paratext"}, {})
    assert passes({**d, "type": "paratext"}, {"type": "paratext"})


def test_exact_collection(tmp_path):
    import sqlite3
    import faiss
    from build_metadata import SCHEMA
    from srl_search.engine import ExactEngine
    rng = np.random.default_rng(1)
    v = rng.standard_normal((50, 768)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v.astype(np.float16).astype(np.float32)
    ids = np.arange(1000, 1050, dtype=np.int64)
    idx = faiss.IndexIDMap2(faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT))
    idx.add_with_ids(v, ids)
    faiss.write_index(idx, str(tmp_path / "index.faiss"))
    db = sqlite3.connect(tmp_path / "meta.sqlite"); db.executescript(SCHEMA)
    for i in ids:
        db.execute("INSERT INTO works (id, doi, title, year, type, is_oa, is_retracted, pmcid, pmc_oa, abstract) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (int(i), f"10.1/{i}", f"paper {i}", 2020 + int(i) % 5, "article", 1, 0, f"PMC{i}", 1, f"abstract {i}"))
    db.commit(); db.close()
    (tmp_path / "manifest.json").write_text(json.dumps({"papers": 50, "oldest_shard_date": "2020-01-01",
                                                        "created_utc": "x"}))
    e = ExactEngine(tmp_path)
    q = v[7]
    res = e._finish(q, 5, 0, 300, None, True, {}, 0.0)["results"]
    exact = ids[np.argsort(-(v @ q))[:5]]
    assert [r["id"] for r in res] == [f"W{i}" for i in exact]
    sim = e.similar(1007, k=3)["results"]
    assert "W1007" not in [r["id"] for r in sim] and len(sim) == 3
    assert e.similar(99999) is None
    only = e._finish(q, 5, 0, 300, {"year_from": 2024}, True, {}, 0.0)["results"]
    assert only and all(r["year"] >= 2024 for r in only)
    assert e.resolve("PMC1007") == 1007 and e.resolve("10.1/1007") == 1007


def test_details_from_base_then_updates(tmp_path):
    """Details and DOI / PMCID lookups fall through from the base corpus to the daily updates corpus."""
    import sqlite3
    import threading
    from srl_search.engine import Engine

    def make(path, rows):
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE works (id INTEGER PRIMARY KEY, doi TEXT, pmcid TEXT, title TEXT)")
        db.executemany("INSERT INTO works VALUES (?,?,?,?)", rows)
        db.commit(); db.close()
        return path

    e = Engine.__new__(Engine)       # only the details part: no index needed
    e.meta_path = make(tmp_path / "base.sqlite", [(1, "10.1/a", "PMC1", "base one"), (2, None, None, "base two")])
    e.more_meta = [make(tmp_path / "upd.sqlite", [(10, "10.1/New", None, "new one")])]
    e.local = threading.local()
    d = e.details([1, 10, 99])
    assert {k: v["title"] for k, v in d.items()} == {1: "base one", 10: "new one"}
    assert e.resolve("10.1/new") == 10 and e.resolve("https://doi.org/10.1/A") == 1
    assert e.resolve("PMC1") == 1 and e.resolve("W10") == 10 and e.resolve("10.9/none") is None

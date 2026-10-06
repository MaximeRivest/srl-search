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

"""The source index (srl_search/sources.py) on a made-up one: lookup by S-number and ISSN, figures, the publisher, a
paper's source, and the name search (every word, the last a prefix, the exact name first)."""
import json

import numpy as np

from srl_search.sources import Sources


def strings(d, stem, values):
    data = [v.encode() for v in values]
    off = np.zeros(len(data) + 1, np.uint64)
    np.cumsum([len(x) for x in data], out=off[1:])
    np.save(d / f"{stem}_off.npy", off)
    (d / f"{stem}s.bin").write_bytes(b"".join(data))


def make(tmp_path):
    d = tmp_path / "sources"
    d.mkdir()
    names = ["Journal of Ecology", "Ecology", "Ecology Letters", "arXiv (Cornell University)"]
    np.save(d / "src_in.npy", np.array([10, 20, 30, 40], np.int64))
    strings(d, "name", names)
    strings(d, "issn", ["0022-0477", "0012-9658", "1461-023X", ""])
    strings(d, "pub_name", ["Wiley", "Cornell University"])
    np.save(d / "pub_ids.npy", np.array(["P1", "I2"], dtype="U16"))
    np.save(d / "pub.npy", np.array([0, 0, 0, 1], np.int32))
    np.save(d / "kind.npy", np.array([0, 0, 0, 1], np.uint8))
    np.save(d / "w_src.npy", np.array([0, 1, 1, -1, 3], np.int32))
    np.save(d / "s_off.npy", np.array([0, 1, 3, 3, 4], np.uint64))
    np.save(d / "s_node.npy", np.array([0, 1, 2, 4], np.uint32))
    np.save(d / "oa.npy", np.array([True, False, True, False, True]))
    for k, v in {"s_works": [1, 2, 0, 1], "s_cites": [5, 30, 0, 2], "s_h": [1, 2, 0, 1], "s_year0": [2001, 1990, 0, 2020],
                 "s_year1": [2001, 2010, 0, 2020], "s_oa": [1, 1, 0, 1]}.items():
        np.save(d / f"{k}.npy", np.array(v))
    (d / "meta.json").write_text(json.dumps({"kinds": ["journal", "repository"]}))
    return Sources(d)


def test_lookup_and_figures(tmp_path):
    s = make(tmp_path)
    assert s.row("S20") == 1 and s.row("https://openalex.org/S40") == 3 and s.row("S99") is None
    assert s.row("0012-9658") == 1 and s.row("1461023x") == 2
    d = s.describe(1)
    assert d == {"id": "S20", "name": "Ecology", "type": "journal", "issn": "0012-9658",
                 "publisher": {"id": "P1", "name": "Wiley"}, "works": 2, "citations": 30, "h": 2, "years": [1990, 2010],
                 "open_access": 0.5}
    assert s.describe(3)["type"] == "repository" and s.describe(3)["issn"] is None
    assert s.of_paper(2) == 1 and s.of_paper(3) is None
    assert s.papers(1).tolist() == [1, 2]


def test_name_search(tmp_path):
    s = make(tmp_path)
    assert [s.name(r) for r in s.search("ecology", 5)] == ["Ecology", "Journal of Ecology", "Ecology Letters"]
    assert [s.name(r) for r in s.search("ecology lett", 5)] == ["Ecology Letters"]
    assert [s.name(r) for r in s.search("Écology", 1)] == ["Ecology"]
    assert s.search("arxiv cornell", 5) == [3]
    assert s.search("botany", 5) == []

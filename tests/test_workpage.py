"""A paper's surroundings (srl_search/workpage.py) on a made-up literature: references and citing papers in the right
order and counted right, authors in order with the organisations they signed from, and the field maps holding a
paper with its group and subgroup. No real index loaded."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from srl_search import workpage as WP


def csr(rows: dict[int, list], n: int):
    off = np.zeros(n + 1, np.uint64)
    vals = []
    for i in range(n):
        off[i + 1] = off[i] + len(rows.get(i, []))
        vals += rows.get(i, [])
    return off, np.array(vals, np.uint32)


class FakeCitations:
    """5 papers (works 10..14). 0 cites 1 and 2; 3 and 4 cite 0; 4 also cites 1."""

    def __init__(self, root: Path):
        self.root = root
        refs = {0: [1, 2], 3: [0], 4: [0, 1]}
        self.fwd_off, self.fwd = csr(refs, 5)
        back: dict[int, list] = {}
        for s, ts in refs.items():
            for t in ts:
                back.setdefault(t, []).append(s)
        boff, bwd = csr(back, 5)
        np.save(root / "bwd_off.npy", boff)
        np.save(root / "bwd.npy", bwd)
        np.save(root / "nrefs.npy", np.array([3, 0, 0, 1, 2], np.int32))
        self.wn = np.array([10, 11, 12, 13, 14])
        self.year = np.array([2010, 2001, 1990, 2015, 2020], np.int16)
        self.cites = np.array([2, 2, 1, 5, 40], np.int32)
        self.titles = ["Focus", "Old classic", "Older classic", "Later work", "Newest work"]
        self.authors = FakeAuthors()
        self.institutions = FakeInstitutions(root)

    def nodes(self, ws):
        return np.array([int(np.searchsorted(self.wn, w)) if w in self.wn else -1 for w in ws])

    def refs(self, nd):
        return self.fwd[int(self.fwd_off[nd]):int(self.fwd_off[nd + 1])]

    def work(self, nd):
        return int(self.wn[nd])

    def title(self, nd):
        return self.titles[nd]

    def type(self, nd):
        return "review" if nd == 2 else "article"


class FakeAuthors:
    """Paper 0 by Ann (row 0, first) and Bo (row 1, second) of 3 listed (the third tied to no person)."""
    by = True

    def __init__(self):
        self.works, self.h = np.array([12, 3]), np.array([5, 1])
        self.names = ["Ann Reef", "Bo Fish"]

    def of(self, nd):
        if nd == 0:
            return np.array([1, 0]), np.array([2, 1]), 3
        return np.array([0]), np.array([1]), 1

    def name(self, r):
        return self.names[r]

    def openalex_id(self, r):
        return f"A{100 + r}"


class FakeInstitutions:
    """Paper 0 lists organisations 0 and 1; Ann signed from 1, Bo from 0 and 1 (the per-authorship index)."""

    def __init__(self, root: Path):
        self.root = root / "institutions"
        d = self.root / "authorships"
        d.mkdir(parents=True)
        off, _ = csr({0: [0, 0, 0]}, 5)
        np.save(d / "aw_off.npy", off)
        np.save(d / "aw_arow.npy", np.array([0, 1, 1], np.uint32))
        np.save(d / "aw_inst.npy", np.array([1, 0, 1], np.uint32))
        (d / "meta.json").write_text("{}")

    def of_paper(self, nd):
        return np.array([0, 1]) if nd == 0 else np.array([], int)

    def describe(self, r):
        return {"id": f"I{r}", "name": ["Reef Lab", "Sea University"][r], "country": "AU", "kind": "education", "ror": None}


APP = SimpleNamespace(place_work=lambda w: (0.5, 0.5) if w != 12 else None, readable=lambda w: w == 13)


def test_context_counts_orders_and_signs(tmp_path):
    wp = WP.WorkPage(APP, FakeCitations(tmp_path), None)
    c = wp.context(10)
    assert c["title"] == "Focus" and c["year"] == 2010 and c["references_listed"] == 3
    assert c["references"]["count"] == 2 and [x["id"] for x in c["references"]["top"]] == ["W11", "W12"]
    assert c["cited_by"]["count"] == 2 and [x["id"] for x in c["cited_by"]["top"]] == ["W14", "W13"]
    assert [x["id"] for x in c["cited_by"]["newest"]] == ["W14", "W13"]
    assert c["cited_by"]["years"] == {"2015": 1, "2020": 1}
    assert sorted(p[2] for p in c["references"]["plot"]) == ["W11", "W12"]
    # authors in their order; organisations numbered in the order the authors sign (Ann first: Sea University)
    assert [a["name"] for a in c["authors"]] == ["Ann Reef", "Bo Fish"] and c["authors_count"] == 3
    assert c["affiliations"] == "exact"
    assert [o["name"] for o in c["institutions"]] == ["Sea University", "Reef Lab"]
    assert c["authors"][0]["at"] == [0] and sorted(c["authors"][1]["at"]) == [0, 1]
    # list items carry what a row shows
    later = next(x for x in c["cited_by"]["top"] if x["id"] == "W13")
    assert later["readable"] and later["first_author"] == "Ann Reef" and later["x"] == 0.5
    older = next(x for x in c["references"]["top"] if x["id"] == "W12")
    assert older["type"] == "review" and "x" not in older


def test_unknown_paper_and_paging(tmp_path):
    wp = WP.WorkPage(APP, FakeCitations(tmp_path), None)
    assert wp.context(99) == {"id": "W99", "in_index": False, "maps": []}
    page = wp.links(10, "cited_by", "oldest", 1, 5)
    assert page["total"] == 2 and [x["id"] for x in page["items"]] == ["W14"]


def test_field_maps_holding_a_paper(tmp_path):
    def field(key, topic, papers, groups, cited=(), status="done"):
        d = tmp_path / "fields" / key
        (d / "maps" / "v1").mkdir(parents=True)
        (d / "job.json").write_text(json.dumps({"status": status, "request": {"q": topic}}))
        (d / "field.json").write_text(json.dumps({"papers": {str(w): {"via": "cited" if w in cited else "meaning"}
                                                             for w, _, _ in papers}}))
        (d / "maps" / "v1" / "main.json").write_text(json.dumps({
            "groups": groups,
            "papers": {"id": [f"W{w}" for w, _, _ in papers], "group": [g for _, g, _ in papers], "sub": [s for _, _, s in papers]}}))
    g = [{"id": 1, "name": "a · b", "size": 2, "label": {"name": "Bleaching", "about": "They study heat."},
          "subgroups": [{"id": 2, "name": "x", "size": 1, "label": {"name": "Heatwaves"}}]}]
    field("small", "coral reefs", [(10, 1, 2), (11, 1, 0)], g)
    field("big", "biodiversity", [(10, 1, 0), (12, 1, 0), (13, 1, 0)], [{"id": 1, "name": "life", "size": 3}], cited=(10,))
    field("gone", "old topic", [(10, 1, 0)], [{"id": 1, "name": "z", "size": 1}], status="merged")
    fi = WP.FieldIndex(tmp_path / "fields")
    for _ in range(100):
        if fi.ready:
            break
        time.sleep(0.05)
    maps = fi.of(10)
    assert [m["topic"] for m in maps] == ["coral reefs", "biodiversity"]      # on topic first, then where it is cited
    assert maps[0]["via"] == "topic" and maps[1]["via"] == "cited"
    assert maps[0]["group"] == {"id": 1, "name": "Bleaching", "about": "They study heat.", "size": 2}
    assert maps[0]["sub"]["name"] == "Heatwaves"
    assert fi.of(99) == []

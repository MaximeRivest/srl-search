"""Authors and institutions as pages: a profile shaped like a topic map (the site draws both with the same
overview: boxes of fields, the globe, reading lists), and a name search.

Data: the citation index's authorship index the other way round (citewalk/build_author_works.py: each author's
papers, with per-author figures) and its institution index (citewalk/build_institutions.py). Fields are OpenAlex's
labels: each paper's primary topic, in a subfield, in a field. A profile is computed on request (a few ms for most
authors, ~0.3 s for one with 5,000 papers, a few seconds for a large university) and kept in memory for the next
pages of its lists.

Routes are in app.py: /authors?q= (search), /authors/{id} (profile; read=&list=&offset=&limit= for a list's
next page), /works/{ref}/authors (who wrote a paper, with their ids); /institutions?q=, /institutions/{id} the same
way.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np

from srl_search.scope import SCHOLARLY_TYPES

HEAD = 8                   # items of each list sent with the profile (the site's Options.lens_size)
HEAD_SUB = 3               # ... of a subgroup's lists (a large university has hundreds of subgroups): read on when opened
LIST_DEPTH = 150           # items a list goes on to, read page by page
MAX_DOTS = 1200            # papers placed on the map (the most cited and the newest when there are more)
MAX_COAUTHOR_PAPER = 100   # papers listing more authors than this don't count for co-authors (consortia)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
PAPER_LISTS = ("cited", "newest", "first")


def fold(s: str) -> str:
    """Lower-case, accents removed: what the names index holds."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def gather(off: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For CSR offsets and a set of rows: (which row each entry belongs to, the entries' positions), vectorised."""
    a = np.asarray(off[nodes]).astype(np.int64)
    b = np.asarray(off[nodes + 1]).astype(np.int64)
    cnt = b - a
    total = int(cnt.sum())
    if not total:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    owner = np.repeat(np.arange(len(nodes)), cnt)
    starts = np.repeat(a, cnt)
    within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return owner, starts + within


class People:
    def __init__(self, api, cit, names_index: Path | None, inst_index: Path | None = None):
        self.api = api
        self.cit = cit
        self.au = cit.authors
        self.inst = getattr(cit, "institutions", None)
        self.ok = self.au is not None and getattr(self.au, "by", False)
        self.topic = None
        self.topics: list[dict] = []
        if (cit.root / "topic.npy").exists() and (cit.root / "topics.json").exists():
            self.topic = np.load(cit.root / "topic.npy", mmap_mode="r")
            self.topics = json.loads((cit.root / "topics.json").read_text())
        self.field_of = np.array([t["field"] for t in self.topics] + [""])
        self.sub_of = np.array([t["subfield"] for t in self.topics] + [""])
        self.top_of = np.array([t["name"] for t in self.topics] + [""])
        self.is_pub = np.array([t in SCHOLARLY_TYPES for t in cit.types], bool)
        self.names = NameIndex(names_index) if names_index and (names_index / "meta.json").exists() else None
        self.inst_names = NameIndex(inst_index) if inst_index and (inst_index / "meta.json").exists() else None
        self.recent: OrderedDict[tuple[str, int], dict] = OrderedDict()   # (kind, row) -> raw profile
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ lookup
    def row(self, ref: str) -> int | None:
        r = ref.strip()
        if r.lower().startswith("https://openalex.org/"):
            r = r.rsplit("/", 1)[-1]
        if r[:1] in "Aa" and r[1:].isdigit():
            return self.au.row_of(int(r[1:]))
        return None

    def inst_row(self, ref: str) -> int | None:
        r = ref.strip()
        if r.lower().startswith("https://openalex.org/"):
            r = r.rsplit("/", 1)[-1]
        if self.inst is not None and r[:1] in "Ii" and r[1:].isdigit():
            return self.inst.row_of(int(r[1:]))
        return None

    def authors_of_work(self, w: int) -> list[dict]:
        """Who wrote a paper, in order, with ids (positions OpenAlex could not tie to a person are left out)."""
        nd = int(self.cit.nodes([w])[0])
        if nd < 0:
            return []
        rows, pos, n = self.au.of(nd)
        order = np.argsort(pos, kind="stable")
        out = []
        for r, p in zip(rows[order].tolist(), pos[order].tolist()):
            d = {"id": self.au.openalex_id(r), "name": self.au.name(r), "position": int(p)}
            if self.au.by:
                d["works"] = int(self.au.works[r])
            out.append(d)
        return out

    def institutions_of_work(self, w: int) -> list[dict]:
        nd = int(self.cit.nodes([w])[0])
        if nd < 0 or self.inst is None:
            return []
        return [self.inst.describe(int(r)) for r in self.inst.of_paper(nd)]

    # ------------------------------------------------------------------ the profile
    def raw(self, kind: str, row: int) -> dict:
        key = (kind, row)
        with self.lock:
            hit = self.recent.get(key)
            if hit is not None:
                self.recent.move_to_end(key)
                return hit
        out = self._raw_author(row) if kind == "author" else self._raw_institution(row)
        with self.lock:
            self.recent[key] = out
            while len(self.recent) > 64:
                self.recent.popitem(last=False)
        return out

    def _publications(self, nodes: np.ndarray) -> np.ndarray:
        return self.is_pub[np.asarray(self.cit.kind[nodes])] if len(nodes) else np.zeros(0, bool)

    def _raw_author(self, row: int) -> dict:
        au = self.au
        nodes, pos = au.papers(row)
        pub = self._publications(nodes)
        nodes, pos = nodes[pub], pos[pub]
        # everyone on their papers but them: the co-authors (consortia papers left out)
        owner, at = gather(au.off, nodes)
        listed = np.asarray(au.nauth[nodes])[owner] if len(owner) else owner
        co = np.asarray(au.auth[at]).astype(np.int64) if len(at) else at
        keep = (co != row) & (listed <= MAX_COAUTHOR_PAPER)
        people = (owner[keep], co[keep])
        out = self._raw(nodes, pos, people)
        person = {"id": au.openalex_id(row), "name": au.name(row), "works": int(au.works[row]),
                  "citations": int(au.cites[row]), "h": int(au.h[row]),
                  "years": self._first_last(out["_years"]), "topic": self._topic(int(au.topic[row]))}
        if self.inst is not None:
            person["institutions"] = [dict(self.inst.describe(r), papers=k, years=[y0, y1] if y0 else None)
                                      for r, k, y0, y1 in self.inst.of_author(row)[:8]]
        hist = np.bincount(np.minimum(pos, 4).astype(np.int64), minlength=5) if len(pos) else np.zeros(5, np.int64)
        person["positions"] = {"first": int(hist[1]), "second": int(hist[2]), "other": int(hist[3] + hist[4])}
        out["person"] = person
        out["kind"] = "author"
        return out

    def _raw_institution(self, row: int) -> dict:
        ins, au = self.inst, self.au
        nodes = ins.papers(row)
        pub = self._publications(nodes)
        nodes = nodes[pub]
        # the people: every identified author on its papers (position-weighted lists come later)
        owner, at = gather(au.off, nodes)
        listed = np.asarray(au.nauth[nodes])[owner] if len(owner) else owner
        co = np.asarray(au.auth[at]).astype(np.int64) if len(at) else at
        keep = listed <= MAX_COAUTHOR_PAPER
        people = (owner[keep], co[keep])
        # partner organisations: the other institutions on its papers
        o2, at2 = gather(ins.w_off, nodes)
        other = np.asarray(ins.w_inst[at2]).astype(np.int64) if len(at2) else at2
        k2 = other != row
        partners = (o2[k2], other[k2])
        out = self._raw(nodes, None, people, partners)
        d = ins.describe(row)
        cites = out["_cites"]
        d.update({"works": int(len(nodes)), "citations": int(cites.sum()), "h": self._h(cites),
                  "years": self._first_last(out["_years"]), "authors": int(len(np.unique(people[1]))) if len(people[1]) else 0})
        out["person"] = d
        out["kind"] = "institution"
        return out

    @staticmethod
    def _h(cites: np.ndarray) -> int:
        if not len(cites):
            return 0
        c = np.sort(cites)[::-1]
        return int(np.sum(c >= np.arange(1, len(c) + 1)))

    @staticmethod
    def _first_last(years: np.ndarray) -> list[int] | None:
        y = years[years > 0]
        return [int(y.min()), int(y.max())] if len(y) else None

    def _raw(self, nodes: np.ndarray, pos: np.ndarray | None, people: tuple, partners: tuple | None = None) -> dict:
        """The common part: a set of papers in groups by field, with lists and clouds, and the dots."""
        cit = self.cit
        t0 = time.perf_counter()
        n = len(nodes)
        years = np.asarray(cit.year[nodes]).astype(np.int64) if n else np.zeros(0, np.int64)
        cites = np.asarray(cit.cites[nodes]).astype(np.int64) if n else np.zeros(0, np.int64)
        tp = np.asarray(self.topic[nodes]).astype(np.int64) if self.topic is not None and n else np.full(n, -1)
        if pos is None:
            pos = np.zeros(n, np.int64)
        ctx = dict(nodes=nodes, pos=pos, years=years, cites=cites, people=people, partners=partners)

        # ---- fields: OpenAlex field > subfield > topic of each paper. Boxes are fields with subfields inside when
        # the work spans fields; subfields with topics inside when most of it is in one field.
        at = np.where(tp >= 0, tp, len(self.topics))
        f, s, t = self.field_of[at], self.sub_of[at], self.top_of[at]
        fu, fc = np.unique(f[f != ""], return_counts=True)
        level = "fields"
        if len(fu) and fc.max() >= 0.7 * max(1, (f != "").sum()):
            level = "subfields"
        big, small = (f, s) if level == "fields" else (s, t)
        bu, bc = np.unique(big[big != ""], return_counts=True)
        order = np.argsort(-bc, kind="stable")
        bu, bc = bu[order], bc[order]
        # fields with under 1% of the papers, past the eighth, fall into "other" (too small to be read as boxes)
        keep = np.ones(len(bu), bool)
        if len(bu) > 8:
            keep[8:] = bc[8:] >= max(2, 0.01 * n)
        gid = np.zeros(n, np.int64)            # 0: in no group
        sid = np.zeros(n, np.int64)
        groups = []
        for k, (name, c) in enumerate(zip(bu[keep], bc[keep]), 1):
            m = big == name
            gid[m] = k
            su, sc = np.unique(small[m & (small != "")], return_counts=True)
            so = np.argsort(-sc, kind="stable")
            subs = []
            for j, (sn, scn) in enumerate(zip(su[so], sc[so]), 1):
                mm = m & (small == sn)
                sid[mm] = j
                subs.append(self._unit(np.flatnonzero(mm), ctx, 40, id=j, name=str(sn)))
            g = self._unit(np.flatnonzero(m), ctx, 60, id=k, name=str(name))
            g.update({"typical": int(cit.work(int(nodes[np.flatnonzero(m)[np.argmax(cites[m])]]))) if c else None,
                      "cites_groups": {}, "between": 0, "with_references": int(c)})
            if len(subs) > 1:
                g["subgroups"] = subs
                g["subgroups_other"] = int((m & (small == "")).sum())
            groups.append(g)
        other = int((gid == 0).sum())
        overall = self._unit(np.arange(n), ctx, LIST_DEPTH)

        # ---- the dots: every paper, or the most cited and newest when there are too many
        if n > MAX_DOTS:
            by_c = np.argsort(-cites, kind="stable")[: MAX_DOTS * 2 // 3]
            by_y = np.argsort(-years, kind="stable")
            chosen = np.unique(np.concatenate([by_c, by_y[: MAX_DOTS - len(by_c)]]))
        else:
            chosen = np.arange(n)
        dots = {"id": [f"W{cit.work(int(nodes[i]))}" for i in chosen], "group": gid[chosen].tolist(),
                "sub": sid[chosen].tolist(), "node": nodes[chosen].tolist()}
        yu, yc = np.unique(years[years > 0], return_counts=True)
        return {"level": level, "groups": groups, "overall": overall, "other": other, "dots": dots, "n": n,
                "per_year": {int(y): int(c) for y, c in zip(yu, yc)}, "_years": years, "_cites": cites,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _unit(self, idx: np.ndarray, ctx: dict, depth: int, **head) -> dict:
        """A group, subgroup or the whole: its size, years, lists (as nodes) and cloud."""
        nodes, pos, years, cites = ctx["nodes"], ctx["pos"], ctx["years"], ctx["cites"]
        lenses: dict[str, list] = {}
        if len(idx):
            c, y = cites[idx], years[idx]
            lenses["cited"] = [int(nodes[i]) for i in idx[np.lexsort((-y, -c))][:LIST_DEPTH]]
            lenses["newest"] = [int(nodes[i]) for i in idx[np.lexsort((-c, -y))][:LIST_DEPTH]]
            fi = idx[pos[idx] == 1]
            if len(fi):
                lenses["first"] = [int(nodes[i]) for i in fi[np.lexsort((-years[fi], -cites[fi]))][:LIST_DEPTH]]
        else:
            lenses["cited"] = lenses["newest"] = []
        people = self._ranked(ctx["people"], idx, nodes, years, cites, depth)
        lenses["coauthors" if ctx["partners"] is None else "people"] = people
        if ctx["partners"] is not None:
            lenses["partners"] = self._ranked(ctx["partners"], idx, nodes, years, cites, depth, who="institution")
        size = int(len(idx))
        cloud = [{"t": p["name"], "share": round(p["n"] / max(1, size), 4), "lift": 1.0, "score": p["n"],
                  "author" if "author" in p else "institution": p.get("author") or p.get("institution")} for p in people[:30]]
        return dict(head, size=size, years=self._span(years[idx]), lenses=lenses, cloud=cloud)

    def _ranked(self, pairs, idx: np.ndarray, nodes, years, cites, depth: int, who: str = "author") -> list[dict]:
        """The people (or organisations) on the most of the papers idx, with the most cited shared paper."""
        pi, co = pairs
        if not len(pi) or not len(idx):
            return []
        inside = np.zeros(len(nodes), bool)
        inside[idx] = True
        m = inside[pi]
        pi, co = pi[m], co[m]
        if not len(pi):
            return []
        u, inv, cnt = np.unique(co, return_inverse=True, return_counts=True)
        csum = np.bincount(inv, weights=cites[pi], minlength=len(u))
        order = np.lexsort((u, -csum, -cnt))[:depth]
        # the papers of the chosen ones only (not every co-author's): sort the pairs by person once
        want = np.zeros(len(u), bool)
        want[order] = True
        sel = want[inv]
        pi_s, inv_s = pi[sel], inv[sel]
        by = np.argsort(inv_s, kind="stable")
        pi_s, inv_s = pi_s[by], inv_s[by]
        bounds = np.searchsorted(inv_s, np.arange(len(u) + 1))
        out = []
        for j in order.tolist():
            ix = pi_s[bounds[j]:bounds[j + 1]]
            top = int(ix[np.argmax(cites[ix])])
            latest = int(ix[np.argmax(years[ix])])
            r = int(u[j])
            d = {"name": "", "n": int(cnt[j]), "years": [int(years[ix].min()), int(years[ix].max())],
                 "top": int(nodes[top]), "latest": int(nodes[latest]) if latest != top else None}
            if who == "author":
                d.update(author=self.au.openalex_id(r), name=self.au.name(r), works=int(self.au.works[r]))
            else:
                d.update(institution=self.inst.openalex_id(r), name=self.inst.name(r),
                         country=(self.inst.describe(r)["country"]))
            out.append(d)
        return out

    def _span(self, y: np.ndarray) -> list[int] | None:
        y = y[y > 0]
        if not len(y):
            return None
        q = np.percentile(y, [10, 50, 90])
        return [int(q[0]), int(q[1]), int(q[2])]

    def _topic(self, t: int) -> dict | None:
        if t < 0 or t >= len(self.topics):
            return None
        x = self.topics[t]
        return {"name": x["name"], "subfield": x["subfield"], "field": x["field"]}

    # ------------------------------------------------------------------ what the site gets
    def profile(self, kind: str, row: int) -> dict:
        from srl_search.topic_service import resolve
        raw = self.raw(kind, row)
        cit = self.cit
        blocks = [(g, HEAD) for g in raw["groups"] + [raw["overall"]]] + \
                 [(s, HEAD_SUB) for g in raw["groups"] for s in g.get("subgroups", [])]
        wanted = set()
        for g, head in blocks:
            if g.get("typical"):
                wanted.add(g["typical"])
            for name, items in g["lenses"].items():
                for it in items[:head]:
                    if isinstance(it, dict):
                        wanted.add(cit.work(it["top"]))
                        if it.get("latest"):
                            wanted.add(cit.work(it["latest"]))
                    else:
                        wanted.add(cit.work(it))
        t = time.perf_counter()
        papers = resolve(self.api, cit, wanted)
        resolve_ms = round((time.perf_counter() - t) * 1000, 1)
        def lists(g, head=HEAD):
            out = {}
            for name, items in g["lenses"].items():
                out[name] = [self._person_item(it, papers) if isinstance(it, dict) else papers[cit.work(it)] for it in items[:head]]
                if len(items) > head:
                    out.setdefault("_total", {})[name] = len(items)
            return out
        unit = lambda g, keys, head=HEAD: dict({k: g[k] for k in keys}, lenses=lists(g, head))   # noqa: E731
        groups = []
        for g in raw["groups"]:
            gg = unit(g, ("id", "name", "size", "years", "cites_groups", "between", "with_references", "cloud"))
            gg["typical"] = papers.get(g["typical"]) if g.get("typical") else None
            if g.get("subgroups"):
                gg["subgroups"] = [unit(s, ("id", "name", "size", "years", "cloud"), HEAD_SUB) for s in g["subgroups"]]
                gg["subgroups_other"] = g["subgroups_other"]
            groups.append(gg)
        overall = unit(raw["overall"], ("size", "years", "cloud"))
        t = time.perf_counter()
        xs, ys = [], []
        for nd in raw["dots"]["node"]:
            p = self.api.place_work(cit.work(nd))
            xs.append(round(p[0], 5) if p else None)
            ys.append(round(p[1], 5) if p else None)
        d = raw["dots"]
        person = dict(raw["person"], per_year=raw["per_year"])
        return {"kind": raw["kind"], "person": person, "question": person["name"], "mode": raw["kind"], "level": raw["level"],
                "groups": groups, "overall": overall,
                "papers": {"id": d["id"], "group": d["group"], "sub": d["sub"], "between": [False] * len(d["id"]), "x": xs, "y": ys},
                "coverage": {"papers": raw["n"], "with_references": raw["n"], "in_groups": raw["n"] - raw["other"],
                             "other": raw["other"], "references_thin": False},
                "timing": {"profile_ms": raw["ms"], "resolve_ms": resolve_ms, "resolved": len(wanted),
                           "map_ms": round((time.perf_counter() - t) * 1000, 1)}}

    def _person_item(self, it: dict, papers: dict) -> dict:
        cit = self.cit
        d = {k: v for k, v in it.items() if k not in ("top", "latest")}
        d["top"] = papers[cit.work(it["top"])]
        if it.get("latest"):
            d["latest"] = papers[cit.work(it["latest"])]
        return d

    def more(self, kind: str, row: int, at: str, name: str, offset: int, limit: int) -> dict:
        """The next items of one list: at = "" (the whole), "3" (a group) or "3.2" (a subgroup)."""
        from fastapi import HTTPException
        from srl_search.topic_service import resolve
        raw = self.raw(kind, row)
        unit = raw["overall"]
        if at:
            g, _, s = at.partition(".")
            unit = next((x for x in raw["groups"] if x["id"] == int(g)), None)
            if unit is None:
                raise HTTPException(404, "no such group")
            if s:
                unit = next((x for x in unit.get("subgroups", []) if x["id"] == int(s)), None)
                if unit is None:
                    raise HTTPException(404, "no such subgroup")
        items = unit["lenses"].get(name)
        if items is None:
            raise HTTPException(404, "no such list here")
        page = items[offset: offset + limit]
        cit = self.cit
        if page and isinstance(page[0], dict):
            wanted = {cit.work(it["top"]) for it in page} | {cit.work(it["latest"]) for it in page if it.get("latest")}
            papers = resolve(self.api, cit, wanted)
            return {"items": [self._person_item(it, papers) for it in page], "total": len(items)}
        papers = resolve(self.api, cit, {cit.work(nd) for nd in page})
        return {"items": [papers[cit.work(nd)] for nd in page], "total": len(items)}

    # ------------------------------------------------------------------ search
    def search(self, q: str, k: int) -> dict:
        if self.names is None:
            from fastapi import HTTPException
            raise HTTPException(503, "the names index is not built")
        t0 = time.perf_counter()
        hits, how = self.names.search(q, k)
        out = []
        qtok = set(WORD.findall(fold(q)))
        for row in hits:
            name = self.au.name(row)
            ntok = set(WORD.findall(fold(name)))
            full = all(t in ntok or (len(t) == 1 and any(x.startswith(t) for x in ntok)) for t in qtok)
            y0, y1 = int(self.au.year0[row]), int(self.au.year1[row])
            d = {"id": self.au.openalex_id(row), "name": name, "works": int(self.au.works[row]),
                 "citations": int(self.au.cites[row]), "h": int(self.au.h[row]),
                 "years": [y0, y1] if y0 else None,
                 "topic": self._topic(int(self.au.topic[row])), "match": "full" if full else "part"}
            if self.inst is not None:
                where = self.inst.of_author(row)[:1]
                if where:
                    d["institution"] = self.inst.describe(where[0][0])
            out.append(d)
        return {"q": q, "results": out, "how": how, "timing": {"ms": round((time.perf_counter() - t0) * 1000, 1)}}

    def search_institutions(self, q: str, k: int) -> dict:
        if self.inst_names is None or self.inst is None:
            from fastapi import HTTPException
            raise HTTPException(503, "the institutions index is not built")
        t0 = time.perf_counter()
        hits, how = self.inst_names.search(q, k)
        qtok = set(WORD.findall(fold(q)))
        out = []
        for row in hits:
            d = self.inst.describe(row)
            d["works"] = int(self.inst.i_off[row + 1] - self.inst.i_off[row])
            d["match"] = "full" if qtok <= set(WORD.findall(fold(d["name"]))) else "part"
            out.append(d)
        return {"q": q, "results": out, "how": how, "timing": {"ms": round((time.perf_counter() - t0) * 1000, 1)}}


class NameIndex:
    """Names in tantivy (search/build_names_index.py): every token of the query must match (one-letter tokens as
    initials: a prefix), results ordered by the count in the "works" field; when nothing matches, each token is
    tried with one letter of difference."""

    def __init__(self, root: Path):
        import tantivy
        self.tv = tantivy
        self.meta = json.loads((root / "meta.json").read_text())
        self.index = tantivy.Index.open(str(root / "index"))
        self.index.register_tokenizer("names", analyzer(tantivy))
        self.schema = self.index.schema

    def search(self, q: str, k: int) -> tuple[list[int], str]:
        Q, O = self.tv.Query, self.tv.Occur
        toks = [t for t in WORD.findall(fold(q)) if t][:6]
        if not toks:
            return [], "empty"
        def run(clauses):
            s = self.index.searcher()
            r = s.search(Q.boolean_query(clauses), k, count=False, order_by_field="works")
            return [int(s.doc(a).to_dict()["row"][0]) for _, a in r.hits]
        exact = [(O.Must, Q.regex_query(self.schema, "name", f"{re.escape(t)}.*") if len(t) == 1
                  else Q.term_query(self.schema, "name", t)) for t in toks]
        rows = run(exact)
        if rows:
            return rows, "exact"
        if all(len(t) >= 4 for t in toks):
            fuzzy = [(O.Must, Q.fuzzy_term_query(self.schema, "name", t, distance=1, prefix=True)) for t in toks]
            rows = run(fuzzy)
            if rows:
                return rows, "fuzzy"
        return [], "none"


def analyzer(tantivy):
    return (tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple()).filter(tantivy.Filter.lowercase())
            .filter(tantivy.Filter.ascii_fold()).build())


def schema(tantivy):
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("name", stored=False, tokenizer_name="names", index_option="basic")
    sb.add_integer_field("works", stored=False, indexed=True, fast=True)
    sb.add_integer_field("row", stored=True, fast=True)
    return sb.build()

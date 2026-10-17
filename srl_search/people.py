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

import gzip
import json
import os
import re
import threading
import time
import unicodedata
import urllib.parse
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
EDITIONS_UP_TO = 20000     # papers: up to this, a page lists each work once (editions folded; titles read one by one)
MAX_TOGETHER = 20          # author records seen together on one page
CACHE_FROM = 5000          # papers: profiles this big are kept on disk (computed once per snapshot; see warm())
LIST_TITLES = {"cited": "Most cited", "newest": "Newest", "first": "As first author", "coauthors": "Co-authors",
               "people": "People", "partners": "Partner organisations"}


DATED = re.compile(r"[(\[]\s*(1[5-9]\d\d|20[0-2]\d)\s*[)\]]\s*\.?\s*$")
NUMBERING = re.compile(r"^\s*((chapter|part|book|section)\s+)?([0-9]+|[ivxlc]+|[a-z]+(-[a-z]+)?)\s*[.:)]\s+", re.I)


def edition_key(title: str) -> str:
    """A title as editions share it: folded, without a leading chapter number ("18.", "TWENTY-EIGHT.", "CHAPTER
    XII.") or a trailing year ("(1921)"); '' when too short to be telling."""
    t = DATED.sub("", NUMBERING.sub("", title or ""))
    k = re.sub(r"[^a-z0-9]+", " ", fold(t)).strip()
    return k if len(k) >= 25 and len(k.split()) >= 4 else ""


def rows_of(row) -> tuple[int, ...]:
    return tuple(int(r) for r in row) if isinstance(row, (tuple, list)) else (int(row),)


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
    def __init__(self, api, cit, names_index: Path | None, inst_index: Path | None = None, cache_root: Path | None = None):
        self.api = api
        self.cit = cit
        # profiles of big units on disk: raw (lists as nodes) per snapshot; the page itself per snapshot + atlas
        atlas = getattr(getattr(api, "ATLAS", None), "id", None) or "noatlas"
        self.cache_raw = cache_root / cit.generation / "raw" if cache_root else None
        self.cache_page = cache_root / f"{cit.generation}-{atlas}" if cache_root else None
        if cache_root:
            Path(cache_root).mkdir(parents=True, exist_ok=True)
            for d in Path(cache_root).iterdir():          # other snapshots' answers are stale
                if d.is_dir() and d.name not in (cit.generation, f"{cit.generation}-{atlas}"):
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
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
    def size_of(self, kind: str, row) -> int:
        """How many papers (any type) the unit has: known before computing anything. row: one record, or a tuple of
        author records seen together."""
        if kind == "author":
            return sum(int(self.au.by_off[r + 1] - self.au.by_off[r]) for r in rows_of(row))
        return int(self.inst.i_off[row + 1] - self.inst.i_off[row])

    def ident(self, kind: str, row) -> str:
        return ",".join(self.au.openalex_id(r) for r in rows_of(row)) if kind == "author" else self.inst.openalex_id(row)

    def _cached(self, root: Path | None, kind: str, row: int) -> Path | None:
        if root is None or self.size_of(kind, row) < CACHE_FROM:
            return None
        return root / kind / f"{self.ident(kind, row).replace(',', '+')}.json.gz"

    @staticmethod
    def _read(path: Path | None) -> dict | None:
        if path and path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return None
        return None

    @staticmethod
    def _write(path: Path | None, d: dict) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(d, fh, separators=(",", ":"))
        os.replace(tmp, path)

    def raw(self, kind: str, row: int) -> dict:
        key = (kind, row)
        with self.lock:
            hit = self.recent.get(key)
            if hit is not None:
                self.recent.move_to_end(key)
                return hit
        path = self._cached(self.cache_raw, kind, row)
        out = self._read(path)
        if out is None:
            out = self._raw_author(row) if kind == "author" else self._raw_institution(row)
            out.pop("_years", None)
            out.pop("_cites", None)
            self._write(path, out)
        with self.lock:
            self.recent[key] = out
            while len(self.recent) > 64:
                self.recent.popitem(last=False)
        return out

    def _publications(self, nodes: np.ndarray) -> np.ndarray:
        return self.is_pub[np.asarray(self.cit.kind[nodes])] if len(nodes) else np.zeros(0, bool)

    def _raw_author(self, row) -> dict:
        au = self.au
        rows = rows_of(row)
        parts = [au.papers(r) for r in rows]
        nodes = np.concatenate([x[0] for x in parts]).astype(np.int64)
        pos = np.concatenate([x[1] for x in parts]).astype(np.int64)
        if len(rows) > 1:                                   # records seen together: each paper once, oldest first
            order = np.lexsort((pos, nodes))
            nodes, pos = nodes[order], pos[order]
            first = np.r_[True, nodes[1:] != nodes[:-1]] if len(nodes) else np.zeros(0, bool)
            nodes, pos = nodes[first], pos[first]
            order = np.lexsort((nodes, np.asarray(self.cit.year[nodes])))
            nodes, pos = nodes[order], pos[order]
        pub = self._publications(nodes)
        nodes, pos = nodes[pub], pos[pub]
        # everyone on their papers but them: the co-authors (consortia papers left out)
        owner, at = gather(au.off, nodes)
        listed = np.asarray(au.nauth[nodes])[owner] if len(owner) else owner
        co = np.asarray(au.auth[at]).astype(np.int64) if len(at) else at
        keep = ~np.isin(co, rows) & (listed <= MAX_COAUTHOR_PAPER)
        merged = self._merge_namesakes(co[keep])
        pair = np.unique((owner[keep] << 32) | merged)          # one count per paper once two records became one
        people = (pair >> 32, pair & 0xFFFFFFFF)
        out = self._raw(nodes, pos, people, editions=len(nodes) <= EDITIONS_UP_TO)
        cites = out["_cites"]
        main = rows[0]
        person = {"id": self.ident("author", row), "name": au.name(main), "works": int(len(nodes)),
                  "citations": int(cites.sum()), "h": self._h(cites),
                  "years": self._first_last(out["_years"]), "cited_years": self._cited_span(out["_years"], cites),
                  "topic": self._topic(int(au.topic[main])) if len(rows) == 1 else self._topic(out["_topic"])}
        if len(rows) > 1:
            person["records"] = [{"id": au.openalex_id(r), "name": au.name(r), "works": int(au.works[r]),
                                  "citations": int(au.cites[r])} for r in rows]
        if self.inst is not None:
            where: dict[int, list] = {}
            for r in rows:
                for i, k, y0, y1 in self.inst.of_author(r):
                    w = where.setdefault(i, [0, y0 or 9999, y1])
                    w[0] += k
                    w[1] = min(w[1], y0 or 9999)
                    w[2] = max(w[2], y1)
            top = sorted(where.items(), key=lambda kv: (-kv[1][0], -kv[1][2]))[:8]
            person["institutions"] = [dict(self.inst.describe(i), papers=k, years=[y0, y1] if y0 < 9999 else None)
                                      for i, (k, y0, y1) in top]
        hist = np.bincount(np.minimum(pos, 4).astype(np.int64), minlength=5) if len(pos) else np.zeros(5, np.int64)
        person["positions"] = {"first": int(hist[1]), "second": int(hist[2]), "other": int(hist[3] + hist[4])}
        out["person"] = person
        out["kind"] = "author"
        return out

    def _merge_namesakes(self, co: np.ndarray) -> np.ndarray:
        """Among one author's co-authors, records with the same name are one person (OpenAlex sometimes splits a
        person in two; two different people with one name among one author's collaborators is rare). The same
        rule as the topic maps' experts. The record with the most publications stands for the others."""
        if not len(co):
            return co
        u = np.unique(co)
        names = [fold(re.sub(r"[^\w]", "", self.au.name(int(r)))) for r in u]
        best: dict[str, int] = {}
        for r, nm in zip(u.tolist(), names):
            if nm and (nm not in best or int(self.au.works[r]) > int(self.au.works[best[nm]])):
                best[nm] = r
        canon = np.array([best.get(nm, r) for r, nm in zip(u.tolist(), names)], np.int64)
        if np.array_equal(canon, u):
            return co
        return canon[np.searchsorted(u, co)]

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
    def _cited_span(years: np.ndarray, cites: np.ndarray) -> list[int] | None:
        """The years of the work that carries most of the citations (10th to 80th percentile, weighted by
        citations: the top end left wide of later editions and papers wrongly attributed): for a historical figure, when they did it, not when it was last reprinted."""
        m = (years > 0) & (cites > 0)
        if m.sum() < 3:
            return None
        y, c = years[m], cites[m].astype(np.float64)
        o = np.argsort(y, kind="stable")
        y, cum = y[o], np.cumsum(c[o]) / c.sum()
        return [int(y[np.searchsorted(cum, 0.1)]), int(y[min(len(y) - 1, np.searchsorted(cum, 0.8))])]

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

    def _editions(self, nodes: np.ndarray, years: np.ndarray, cites: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
        """Papers that are another edition of one on the same page: the same title (folded, chapter numbers and a
        trailing year left out, 25+ characters, 4+ words) as one more cited (or as cited and older). A reprint, a
        translation under the same title, a chapter in collected works. Returns (is another edition, node of the
        kept one -> how many editions in all, is a reprint by its own title: kept in the lists, not among the newest)."""
        titles = [self.cit.title(int(nd)) for nd in nodes]
        keys = [edition_key(t) for t in titles]
        # a title that dates itself well before the paper ("Geometry and Experience (1921)" printed in 2025): a reprint
        reprint = np.array([bool((m := DATED.search(t)) and int(m.group(1)) <= y - 10) for t, y in zip(titles, years)], bool)
        best: dict[str, int] = {}
        count: dict[str, int] = {}
        for i, k in enumerate(keys):
            if not k:
                continue
            count[k] = count.get(k, 0) + 1
            j = best.get(k)
            if j is None or (cites[i], -years[i]) > (cites[j], -years[j]):
                best[k] = i
        other = np.zeros(len(nodes), bool)
        kept: dict[int, int] = {}
        for i, k in enumerate(keys):
            if count.get(k, 0) > 1:
                if best[k] == i:
                    kept[int(nodes[i])] = count[k]
                else:
                    other[i] = True
        return other, kept, reprint

    def _raw(self, nodes: np.ndarray, pos: np.ndarray | None, people: tuple, partners: tuple | None = None,
             editions: bool = False) -> dict:
        """The common part: a set of papers in groups by field, with lists and clouds, and the dots."""
        cit = self.cit
        t0 = time.perf_counter()
        n = len(nodes)
        years = np.asarray(cit.year[nodes]).astype(np.int64) if n else np.zeros(0, np.int64)
        cites = np.asarray(cit.cites[nodes]).astype(np.int64) if n else np.zeros(0, np.int64)
        tp = np.asarray(self.topic[nodes]).astype(np.int64) if self.topic is not None and n else np.full(n, -1)
        if pos is None:
            pos = np.zeros(n, np.int64)
        other, kept, reprint = self._editions(nodes, years, cites) if editions and n else (np.zeros(n, bool), {}, np.zeros(n, bool))
        ctx = dict(nodes=nodes, pos=pos, years=years, cites=cites, people=people, partners=partners, other=other, reprint=reprint)

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
        tv = tp[tp >= 0]
        return {"editions": {str(k): v for k, v in kept.items()}, "_topic": int(np.bincount(tv).argmax()) if len(tv) else -1,
                "level": level, "groups": groups, "overall": overall, "other": other, "dots": dots, "n": n,
                "per_year": {int(y): int(c) for y, c in zip(yu, yc)}, "_years": years, "_cites": cites,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _unit(self, idx: np.ndarray, ctx: dict, depth: int, **head) -> dict:
        """A group, subgroup or the whole: its size, years, lists (as nodes) and cloud."""
        nodes, pos, years, cites = ctx["nodes"], ctx["pos"], ctx["years"], ctx["cites"]
        every = idx
        lenses: dict[str, list] = {}
        if ctx.get("other") is not None and len(idx):
            idx = idx[~ctx["other"][idx]] if ctx["other"][idx].any() else idx   # lists: one edition of each work
        if len(idx):
            c, y = cites[idx], years[idx]
            lenses["cited"] = [int(nodes[i]) for i in idx[np.lexsort((-y, -c))][:LIST_DEPTH]]
            fresh = idx[~ctx["reprint"][idx]] if ctx.get("reprint") is not None else idx
            lenses["newest"] = [int(nodes[i]) for i in fresh[np.lexsort((-cites[fresh], -years[fresh]))][:LIST_DEPTH]]
            fi = idx[pos[idx] == 1]
            if len(fi):
                lenses["first"] = [int(nodes[i]) for i in fi[np.lexsort((-years[fi], -cites[fi]))][:LIST_DEPTH]]
        else:
            lenses["cited"] = lenses["newest"] = []
        idx = every
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
        path = self._cached(self.cache_page, kind, row)
        out = self._read(path)
        if out is None:
            out = self._profile(kind, row)
            self._write(path, out)
            out["cached"] = False
        else:
            out["cached"] = True
        return out

    def _profile(self, kind: str, row: int) -> dict:
        from srl_search.topic_service import resolve
        raw = self.raw(kind, row)
        cit = self.cit
        eds = raw.get("editions") or {}
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
        for nd, k in eds.items():                         # "3 editions" where the same work was printed again
            w = cit.work(int(nd))
            if w in papers:
                papers[w] = dict(papers[w], editions=k)
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
        eds = raw.get("editions") or {}
        return {"items": [dict(papers[cit.work(nd)], editions=eds[str(nd)]) if str(nd) in eds else papers[cit.work(nd)]
                          for nd in page], "total": len(items)}

    def namesakes(self, row: int, k: int = 50) -> dict:
        """Other author records with exactly the same name (folded): one person is often split across several
        (historical figures most of all: reprints, letters, collected works). Not merged: listed, with what tells
        them apart, for the reader to choose which to see together."""
        if self.names is None:
            from fastapi import HTTPException
            raise HTTPException(503, "the names index is not built")
        au = self.au
        key = lambda r: " ".join(WORD.findall(fold(au.name(r))))   # noqa: E731
        me = key(row)
        rows, _ = self.names.search(au.name(row), 200)
        same = sorted({r for r in rows if r != row and key(r) == me}, key=lambda r: -int(au.cites[r]))[:k]
        from srl_search.topic_service import resolve
        tops = {}
        for r in same:
            nodes, _ = au.papers(r)
            nodes = nodes[self._publications(nodes)]
            if len(nodes):
                tops[r] = self.cit.work(int(nodes[np.argmax(np.asarray(self.cit.cites[nodes]))]))
        papers = resolve(self.api, self.cit, set(tops.values()))
        out = []
        for r in same:
            y0, y1 = int(au.year0[r]), int(au.year1[r])
            d = {"id": au.openalex_id(r), "name": au.name(r), "works": int(au.works[r]), "citations": int(au.cites[r]),
                 "h": int(au.h[r]), "years": [y0, y1] if y0 else None, "topic": self._topic(int(au.topic[r]))}
            if r in tops:
                d["top"] = papers[tops[r]]
            if self.inst is not None:
                where = self.inst.of_author(r)[:1]
                if where:
                    d["institution"] = self.inst.describe(where[0][0])
            out.append(d)
        return {"id": au.openalex_id(row), "name": au.name(row), "results": out}

    # ------------------------------------------------------------------ search
    def search(self, q: str, k: int) -> dict:
        if self.names is None:
            from fastapi import HTTPException
            raise HTTPException(503, "the names index is not built")
        t0 = time.perf_counter()
        hits, how = self.names.search(q, max(k, min(200, 4 * k)))
        qtok = set(WORD.findall(fold(q)))
        # the best known first (citations), and for a one-word name, people with it as their surname first
        surname = list(qtok)[0] if len(qtok) == 1 else None
        last = lambda r: (WORD.findall(fold(self.au.name(r))) or [""])[-1]   # noqa: E731
        hits = sorted(hits, key=lambda r: (surname is not None and last(r) != surname, -int(self.au.cites[r])))[:k]
        out = []
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


# ---------------------------------------------------------------------------------------------------- as text
def _n(n: int, one: str = "paper") -> str:
    return f"{n:,} {one}" + ("" if n == 1 else "s")


def _paper_line(p: dict, note: str = "") -> str:
    who = "; ".join(p.get("authors") or [])
    if (p.get("authors_count") or 0) > len(p.get("authors") or []):
        who += f"; et al. ({p['authors_count']} authors)"
    facts = [str(p.get("year") or "n.d."), p.get("venue") or "", who,
             f"cited {p['citations']:,} times" if p.get("citations") else ""]
    link = f"https://doi.org/{p['doi']}" if p.get("doi") else f"https://openalex.org/{p['id']}"
    return f"- **{p.get('title') or '(no title)'}** · " + " · ".join(f for f in facts if f) + f" · {p['id']} · {link}" + \
        (f" · {note}" if note else "")


def _person_line(it: dict, base: str, k: str) -> str:
    if it.get("author"):
        ref = f"{base}/authors/{it['author']}"
        who = f"**{it['name']}** ({it['author']}, {_n(it.get('works', 0))} in all)"
    else:
        ref = f"{base}/institutions/{it['institution']}"
        who = f"**{it['name']}**" + (f" ({it['country']})" if it.get("country") else "") + f" ({it['institution']})"
    here = "together" if k == "coauthors" else "here"
    top = it.get("top") or {}
    return (f"- {who}: {_n(it['n'])} {here}, {it['years'][0]}–{it['years'][1]}; most cited {here}: "
            f"{top.get('title', '')} ({top.get('year', 'n.d.')}, {top.get('id', '')}) · {ref}")


def _unit_text(u: dict, base: str, ref: str, at: str, head: int) -> list[str]:
    lines = []
    total = u["lenses"].get("_total") or {}
    for k, title in LIST_TITLES.items():
        items = (u["lenses"].get(k) or [])[:head]
        if not items:
            continue
        lines += [f"### {title}" + (f" ({total[k]} in all)" if total.get(k) else ""), ""]
        for it in items:
            lines.append(_person_line(it, base, k) if isinstance(it, dict) and ("author" in it or "institution" in it)
                         else _paper_line(it))
        if total.get(k, 0) > len(items):
            q = urllib.parse.urlencode({"read": at, "list": k, "offset": len(items), "limit": 25})
            lines.append(f"  … the next 25: {base}/{ref}{'&' if '?' in ref else '?'}{q}")
        lines.append("")
    return lines


def profile_text(d: dict, base: str) -> str:
    """A profile as Markdown for a model: the figures, the fields at a glance, then every field's lists."""
    p, kind = d["person"], d["kind"]
    ids = p["id"].split(",")
    ref = f"{'authors' if kind == 'author' else 'institutions'}/{ids[0]}" + (f"?with={','.join(ids[1:])}" if len(ids) > 1 else "")
    yrs = f"{p['years'][0]}–{p['years'][1]}" if p.get("years") else ""
    facts = [_n(p['works'], "publication"), _n(p['authors'], "author") if p.get("authors") else "",
             f"cited {p['citations']:,} times", f"h-index {p['h']}", yrs]
    lines = [f"# {p['name']} ({p['id']})", " · ".join(f for f in facts if f)]
    if p.get("records"):
        lines.append(f"{len(p['records'])} OpenAlex records seen together as one person (chosen, not verified): " +
                     "; ".join(f"{r['id']} ({_n(r['works'])})" for r in p["records"]))
    if p.get("cited_years") and p.get("years") and p["cited_years"] != p["years"]:
        lines.append(f"Most of the citations go to work from {p['cited_years'][0]}–{p['cited_years'][1]}"
                     " (later years: often reprints and collected editions).")
    if kind == "author":
        if p.get("topic"):
            lines.append(f"Commonest topic: {p['topic']['name']} ({p['topic']['subfield']}, {p['topic']['field']})")
        if p.get("institutions"):
            lines.append("Has published from: " + "; ".join(
                f"{i['name']} ({i['papers']} papers" + (f", {i['years'][0]}–{i['years'][1]}" if i.get("years") else "") + f", {base}/institutions/{i['id']})"
                for i in p["institutions"][:5]))
        pos = p.get("positions") or {}
        lines.append(f"Position on papers: first author on {pos.get('first', 0)}, second on {pos.get('second', 0)}, other on {pos.get('other', 0)}.")
    else:
        lines.append(" · ".join(x for x in (p.get("kind"), p.get("country"), p.get("ror")) if x))
    site = f"author={ids[0]}" + (f"&with={','.join(ids[1:])}" if len(ids) > 1 else "") if kind == "author" else f"institution={p['id']}"
    lines += [f"On ScienceMadeReadable: https://sciencemadereadable.com/?{site} · "
              f"OpenAlex: https://openalex.org/{p['id']}", "",
              f"Caveat: this is one OpenAlex {'author' if kind == 'author' else 'institution'} record. "
              + ("A person can be split across several records, and about a quarter of author names on recent papers are "
                 "tied to no record, so some of their work can be missing; two different people with one name are rarely "
                 "merged. Co-authors on papers with over 100 authors are not counted."
                 if kind == "author" else
                 "Papers count for it when OpenAlex matched an author's affiliation to it (most recent papers, far fewer "
                 "old ones): counts are lower bounds."), ""]
    level = d.get("level", "fields")
    lines += [f"## Its {level} ({d['coverage']['in_groups']:,} papers in {len(d['groups'])} {level}; "
              f"{d['coverage']['other']:,} unlabelled or in {level} too small to list)", ""]
    for g in d["groups"]:
        yrs = g.get("years")
        subs = g.get("subgroups") or []
        lines.append(f"{g['id']}. **{g['name']}**: {_n(g['size'])}" + (f", mostly {yrs[0]}–{yrs[2]}" if yrs else "") +
                     (" — " + "; ".join(f"{g['id']}.{x['id']} {x['name']} ({x['size']:,})" for x in subs[:12]) +
                      (f"; … {len(subs)} in all" if len(subs) > 12 else "") if subs else ""))
    lines += ["", "## All its work" if kind == "institution" else "## All their work", ""]
    lines += _unit_text(d["overall"], base, ref, "", HEAD)
    for g in d["groups"]:
        lines += [f"## {g['id']}. {g['name']} ({_n(g['size'])})", ""]
        lines += _unit_text(g, base, ref, str(g["id"]), HEAD)
        for x in g.get("subgroups") or []:
            lines += [f"### {g['id']}.{x['id']} {x['name']} ({_n(x['size'])})", ""]
            lines += _unit_text(x, base, ref, f"{g['id']}.{x['id']}", HEAD_SUB)
    return "\n".join(lines)


def search_text(d: dict, base: str, kind: str) -> str:
    """Name-search results as Markdown."""
    lines = [f"# {'Authors' if kind == 'author' else 'Institutions'} named “{d['q']}” ({len(d['results'])} found, the most published first)", ""]
    if d.get("how") == "fuzzy":
        lines += ["No name holds every word exactly; these are one letter off.", ""]
    for r in d["results"]:
        yrs = f"{r['years'][0]}–{r['years'][1]}" if r.get("years") else ""
        if kind == "author":
            facts = [f"{r['works']:,} papers", f"cited {r['citations']:,} times" if r.get("citations") else "", f"h {r['h']}" if r.get("h") else "", yrs,
                     r["topic"]["name"] if r.get("topic") else "", r["institution"]["name"] if r.get("institution") else ""]
            lines.append(f"- **{r['name']}** ({r['id']}): " + " · ".join(f for f in facts if f) + f" · {base}/authors/{r['id']}")
        else:
            facts = [f"{r['works']:,} papers", r.get("kind") or "", r.get("country") or ""]
            lines.append(f"- **{r['name']}** ({r['id']}): " + " · ".join(f for f in facts if f) + f" · {base}/institutions/{r['id']}")
    if not d["results"]:
        lines.append("Nobody by that name. Try the surname alone, without initials or accents.")
    lines += ["", "Records are OpenAlex's: the same person can appear under several; tell them apart by field, years and institution."]
    return "\n".join(lines)


def more_text(d: dict, name: str, base: str, ref: str, at: str, offset: int) -> str:
    lines = [f"# {LIST_TITLES.get(name, name)}: items {offset + 1}–{offset + len(d['items'])} of {d['total']}", ""]
    for it in d["items"]:
        lines.append(_person_line(it, base, name) if isinstance(it, dict) and ("author" in it or "institution" in it) else _paper_line(it))
    if offset + len(d["items"]) < d["total"]:
        q = urllib.parse.urlencode({"read": at, "list": name, "offset": offset + len(d["items"]), "limit": 25})
        lines += ["", f"Next: {base}/{ref}{'&' if '?' in ref else '?'}{q}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- warming
def warm(api_base: str, cit_root: Path, min_inst: int = 10000, min_author: int = 2000, cache_root: Path | None = None) -> None:
    """Ask the running API for every big institution's and author's page, so the first visitor finds it on disk
    (python -m srl_search.people warm). Skips the ones already there; idempotent, so it can run every night."""
    import urllib.request
    cit_root = Path(cit_root)
    gen = json.loads((cit_root / "meta.json").read_text())["generation"]
    jobs = []
    if (cit_root / "institutions" / "i_off.npy").exists():
        n = np.diff(np.load(cit_root / "institutions" / "i_off.npy")).astype(np.int64)
        ids = np.load(cit_root / "institutions" / "inst_in.npy", mmap_mode="r")
        rows = np.flatnonzero(n >= min_inst)
        jobs += [("institutions", f"I{int(ids[r])}", int(n[r])) for r in rows[np.argsort(-n[rows])]]
    if (cit_root / "authors" / "a_works.npy").exists():
        w = np.load(cit_root / "authors" / "a_works.npy").astype(np.int64)
        an = np.load(cit_root / "authors" / "author_an.npy", mmap_mode="r")
        rows = np.flatnonzero(w >= min_author)
        jobs += [("authors", f"A{int(an[r])}", int(w[r])) for r in rows[np.argsort(-w[rows])]]
    done = skipped = failed = 0
    t0 = time.time()
    print(f"{time.strftime('%H:%M:%S')} warming {len(jobs)} pages for {gen} through {api_base}", flush=True)
    for i, (kind, ident, n) in enumerate(jobs, 1):
        if cache_root:
            atlas_dirs = [d for d in Path(cache_root).glob(f"{gen}-*") if d.is_dir()]
            sub = "author" if kind == "authors" else "institution"
            if any((d / sub / f"{ident}.json.gz").exists() for d in atlas_dirs):
                skipped += 1
                continue
        try:
            t = time.time()
            with urllib.request.urlopen(f"{api_base}/{kind}/{ident}", timeout=600) as r:
                r.read()
            done += 1
            if done % 50 == 0 or time.time() - t > 20:
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(jobs)} {kind}/{ident} ({n:,} papers) {time.time() - t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{time.strftime('%H:%M:%S')} {kind}/{ident}: {type(e).__name__} {e}", flush=True)
            time.sleep(5)
        time.sleep(0.2)                                   # the public API keeps first claim
    print(f"{time.strftime('%H:%M:%S')} warmed {done}, {skipped} already there, {failed} failed, {(time.time() - t0) / 60:.0f} min", flush=True)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="python -m srl_search.people warm: cache the big institutions' and authors' pages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm")
    w.add_argument("--api", default="http://127.0.0.1:8810")
    w.add_argument("--citations", type=Path, default=None, help="default: the current release's")
    w.add_argument("--cache", type=Path, default=Path("/mnt/fast/scholarsreadinglist/search/people-cache"))
    w.add_argument("--min-institution", type=int, default=10000)
    w.add_argument("--min-author", type=int, default=2000)
    a = ap.parse_args(argv)
    cit = a.citations
    if cit is None:
        rel = json.loads(Path("/mnt/fast/scholarsreadinglist/release/current/manifest.json").read_text())
        cit = Path(rel["paths"]["citations"])
    warm(a.api, cit, a.min_institution, a.min_author, a.cache)



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
        self.order = self.meta.get("order", "works")       # author names since 2026-10-17: by citations

    def search(self, q: str, k: int) -> tuple[list[int], str]:
        Q, O = self.tv.Query, self.tv.Occur
        toks = [t for t in WORD.findall(fold(q)) if t][:6]
        if not toks:
            return [], "empty"
        def run(clauses):
            s = self.index.searcher()
            r = s.search(Q.boolean_query(clauses), k, count=False, order_by_field=self.order)
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
    sb.add_integer_field("cites", stored=False, indexed=True, fast=True)
    sb.add_integer_field("row", stored=True, fast=True)
    return sb.build()


if __name__ == "__main__":
    main()

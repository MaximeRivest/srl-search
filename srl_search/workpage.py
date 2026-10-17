"""A paper's surroundings, for its page: who wrote it and from where, what it cites, what cites it, and which
whole-field maps hold it (and in which of their groups). Everything comes from indexes already on disk, memory-mapped;
nothing is computed ahead and no AI is involved.

- Authors and their affiliations on this paper: the authorship index (citewalk/build_authors.py) and, when built,
  the per-authorship affiliations (citewalk/build_authorships.py). Without that index the paper's organisations are
  listed without saying who signed from which: guessing it was measured at 70 % precision, too wrong to show.
- References and citations: the citation index's two directions (fwd, bwd: citewalk/build_index.py). Only links
  whose both ends are in the OpenAlex snapshot; papers added daily since have none yet.
- Field maps: every whole-field map made (srl_search/field.py), its newest map's every paper with its group and
  subgroup (main.json, or papers.npz for maps of more than 25,000 papers), kept in memory as sorted arrays (~3.5M
  papers across 29 maps: ~45 MB) and reloaded when a map changes.

Kept apart from citations.py and topics.py on purpose: those are hashed into the topic maps' version, and changing
them would mark every map made as older.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

LIST_MAX = 100          # items of one list in one answer
PLOT_MAX = 400          # papers of each direction drawn on the page's timeline (the most cited)


def _arr(x) -> np.ndarray:
    return np.asarray(x)


class WorkPage:
    def __init__(self, app, cit, fields_root: Path | None):
        self.app = app                      # srl_search.app: place_work, readable
        self.cit = cit
        root = Path(cit.root)
        self.bwd_off = np.load(root / "bwd_off.npy", mmap_mode="r")
        self.bwd = np.load(root / "bwd.npy", mmap_mode="r")
        self.au = cit.authors
        self.inst = cit.institutions
        self.aff = None
        self._aff_checked = 0.0
        self.fields = FieldIndex(fields_root) if fields_root else None

    # ------------------------------------------------------------------ per-authorship affiliations (optional)
    def _affiliations(self):
        """The per-authorship index, opened when it appears (it can be built while the API runs)."""
        if self.aff is None and self.inst is not None and time.time() - self._aff_checked > 60:
            self._aff_checked = time.time()
            d = Path(self.inst.root) / "authorships"
            if (d / "meta.json").exists():
                self.aff = (np.load(d / "aw_off.npy", mmap_mode="r"), np.load(d / "aw_arow.npy", mmap_mode="r"),
                            np.load(d / "aw_inst.npy", mmap_mode="r"))
        return self.aff

    # ------------------------------------------------------------------ one paper as a list item
    def item(self, nd: int) -> dict:
        c = self.cit
        w = c.work(nd)
        d = {"id": f"W{w}", "title": c.title(nd), "year": int(c.year[nd]) or None, "citations": int(c.cites[nd])}
        t = c.type(nd)
        if t and t != "article":
            d["type"] = t
        if self.au is not None:
            rows, pos, n = self.au.of(nd)
            if len(rows):
                first = int(rows[int(np.argmin(pos))])
                d["first_author"] = self.au.name(first)
            d["authors_count"] = int(n)
        p = self.app.place_work(w)
        if p is not None:
            d["x"], d["y"] = round(p[0], 5), round(p[1], 5)
        r = self.app.readable(w)
        if r:
            d["readable"] = True
        return d

    def _refs(self, nd: int) -> np.ndarray:
        return _arr(self.cit.refs(nd)).astype(np.int64)

    def _cited_by(self, nd: int) -> np.ndarray:
        return _arr(self.bwd[int(self.bwd_off[nd]):int(self.bwd_off[nd + 1])]).astype(np.int64)

    def _order(self, nodes: np.ndarray, sort: str) -> np.ndarray:
        if not len(nodes):
            return nodes
        cites = _arr(self.cit.cites)[nodes].astype(np.int64)
        year = _arr(self.cit.year)[nodes].astype(np.int64)
        if sort == "newest":
            o = np.lexsort((-cites, -year))
        elif sort == "oldest":
            o = np.lexsort((-cites, np.where(year > 0, year, 9999)))
        else:                                            # most cited
            o = np.lexsort((-year, -cites))
        return nodes[o]

    def _years(self, nodes: np.ndarray) -> dict[str, int]:
        if not len(nodes):
            return {}
        y = _arr(self.cit.year)[nodes]
        u, n = np.unique(y[y > 0], return_counts=True)
        return {str(int(a)): int(b) for a, b in zip(u, n)}

    def _plot(self, nodes: np.ndarray) -> list[list]:
        """[year, citations, W-number, title cut to 100 characters] for the most cited PLOT_MAX of them (the
        timeline's dots: the title is shown when a dot is pointed at, without another request)."""
        top = self._order(nodes, "cited")[:PLOT_MAX]
        y = _arr(self.cit.year)[top]
        cs = _arr(self.cit.cites)[top]
        cut = lambda t: t if len(t) <= 100 else t[:99].rsplit(" ", 1)[0] + "…"
        return [[int(a), int(b), f"W{self.cit.work(int(n))}", cut(self.cit.title(int(n)))]
                for a, b, n in zip(y, cs, top) if a > 0]

    # ------------------------------------------------------------------ the answer
    def context(self, w: int) -> dict:
        c = self.cit
        nd = int(c.nodes([w])[0])
        out: dict = {"id": f"W{w}", "in_index": nd >= 0}
        out["maps"] = self.fields.of(w) if self.fields else []
        if nd < 0:
            return out
        out["year"] = int(c.year[nd]) or None
        # for papers without a record in the search corpus (no English abstract): the page still has a title
        out["title"], out["type"], out["citations"] = c.title(nd), c.type(nd), int(c.cites[nd])
        nrefs = Path(c.root) / "nrefs.npy"
        if nrefs.exists():                              # what the paper lists, against the part we can link
            out["references_listed"] = int(np.load(nrefs, mmap_mode="r")[nd])
        out.update(self._people(nd))
        src = getattr(getattr(self.app, "PEOPLE", None), "src", None)     # where it was published (sources.py)
        r = src.of_paper(nd) if src is not None else None
        if r is not None:
            out["source"] = src.describe(r)
        refs, citing = self._refs(nd), self._cited_by(nd)
        out["references"] = {"count": int(len(refs)), "years": self._years(refs), "plot": self._plot(refs),
                             "top": [self.item(int(n)) for n in self._order(refs, "cited")[:12]]}
        out["cited_by"] = {"count": int(len(citing)), "years": self._years(citing), "plot": self._plot(citing),
                           "top": [self.item(int(n)) for n in self._order(citing, "cited")[:12]],
                           "newest": [self.item(int(n)) for n in self._order(citing, "newest")[:6]]}
        return out

    def _people(self, nd: int) -> dict:
        out: dict = {"authors": [], "institutions": [], "affiliations": "none"}
        if self.au is None:
            return out
        rows, pos, n = self.au.of(nd)
        order = np.argsort(pos, kind="stable")
        rows, pos = rows[order], pos[order]
        out["authors_count"] = int(n)
        aff = self._affiliations()
        by_author: dict[int, list[int]] = {}
        if aff is not None:
            a, b = int(aff[0][nd]), int(aff[0][nd + 1])
            for r, i in zip(_arr(aff[1][a:b]).tolist(), _arr(aff[2][a:b]).tolist()):
                by_author.setdefault(int(r), []).append(int(i))
            out["affiliations"] = "exact"
        inst_rows = _arr(self.inst.of_paper(nd)).tolist() if self.inst is not None else []
        # organisations in the order their first author signs (then the rest)
        seen: list[int] = []
        for r in rows.tolist():
            for i in by_author.get(int(r), []):
                if i not in seen:
                    seen.append(i)
        seen += [int(i) for i in inst_rows if int(i) not in seen]
        index = {i: k for k, i in enumerate(seen)}
        for r, p in zip(rows.tolist(), pos.tolist()):
            d = {"id": self.au.openalex_id(int(r)), "name": self.au.name(int(r)), "position": int(p)}
            if self.au.by:
                d["works"] = int(self.au.works[int(r)])
                d["h"] = int(self.au.h[int(r)])
            if aff is not None:
                d["at"] = [index[i] for i in by_author.get(int(r), []) if i in index]
            out["authors"].append(d)
        if self.inst is not None:
            out["institutions"] = [self.inst.describe(i) for i in seen]
        return out

    def links(self, w: int, direction: str, sort: str, offset: int, limit: int) -> dict:
        nd = int(self.cit.nodes([w])[0])
        if nd < 0:
            return {"id": f"W{w}", "direction": direction, "total": 0, "items": []}
        nodes = self._refs(nd) if direction == "references" else self._cited_by(nd)
        nodes = self._order(nodes, sort)
        limit = max(1, min(limit, LIST_MAX))
        return {"id": f"W{w}", "direction": direction, "sort": sort, "total": int(len(nodes)), "offset": offset,
                "items": [self.item(int(n)) for n in nodes[offset:offset + limit]]}


class FieldIndex:
    """Which whole-field maps hold a paper, and in which group and subgroup. Each map's newest version, as sorted
    arrays; a map whose main.json changed (remade, relabelled) is reloaded, at most once a minute."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.maps: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.checked = 0.0
        self.ready = False
        self._refresh()                 # loads in the background (~20 s for 29 maps): the API starts at once

    def _refresh(self) -> None:
        """Reload changed maps in a background thread; answers meanwhile use the maps already loaded."""
        if time.time() - self.checked < 60 or self.lock.locked():
            return
        self.checked = time.time()
        threading.Thread(target=self._reload, name="field-index", daemon=True).start()

    def _reload(self) -> None:
        with self.lock:
            live = set()
            for job in self.root.glob("*/job.json"):
                try:
                    st = json.loads(job.read_text())
                except (OSError, ValueError):
                    continue
                # the public library's rule (FieldStore.made): made, or being remade by newer map code; not merged
                if not (job.parent / "field.json").exists() or st.get("status") == "merged" \
                        or (st.get("status") != "done" and st.get("kind") != "map"):
                    continue
                mains = list(job.parent.glob("maps/*/main.json"))
                if not mains:
                    continue
                main = max(mains, key=lambda m: m.stat().st_mtime)
                key = job.parent.name
                live.add(key)
                mt = main.stat().st_mtime
                if key in self.maps and self.maps[key]["mtime"] == mt:
                    continue
                try:
                    self.maps[key] = self._load(main, st)
                except Exception as e:  # noqa: BLE001 (one unreadable map must not hide the others)
                    print(f"field index: {key}: {type(e).__name__}: {e}", flush=True)
            for k in set(self.maps) - live:
                del self.maps[k]
            self.checked = time.time()
            self.ready = True

    @staticmethod
    def _load(main: Path, st: dict) -> dict:
        m = json.loads(main.read_text())
        req = st.get("request") or {}
        shown = m.get("papers_shown")
        if shown and (main.parent / shown["file"]).exists():
            z = np.load(main.parent / shown["file"])
            w, g, s = z["w"].astype(np.int64), z["group"].astype(np.int16), z["sub"].astype(np.int16)
        else:
            P = m["papers"]
            w = np.array([int(x[1:]) for x in P["id"]], np.int64)
            g = np.array(P["group"], np.int16)
            s = np.array(P.get("sub") or [0] * len(w), np.int16)
        o = np.argsort(w, kind="stable")
        # how each paper came in: close in meaning, the topic's exact phrase, or cited by the field's own papers (a
        # classic or a method the field leans on, which may belong to another field: said so on the page)
        cited = np.zeros(len(w), bool)
        fj = main.parent.parent.parent / "field.json"
        if fj.exists():
            papers = json.loads(fj.read_text()).get("papers") or {}
            cw = np.array(sorted(int(k) for k, v in papers.items() if v.get("via") == "cited"), np.int64)
            cited = np.isin(w[o], cw)
            del papers
        name = lambda u: ((u.get("label") or {}).get("name") or u.get("name") or "").strip()
        about = lambda u: ((u.get("label") or {}).get("about") or "").strip() or None   # AI-written, from its papers
        groups = {int(gr["id"]): {"name": name(gr), "about": about(gr), "size": int(gr.get("size") or 0),
                                  "subs": {int(sg["id"]): {"name": name(sg), "about": about(sg), "size": int(sg.get("size") or 0)}
                                           for sg in gr.get("subgroups") or []}}
                  for gr in m.get("groups", [])}
        return {"mtime": main.stat().st_mtime, "w": w[o], "g": g[o], "s": s[o], "cited": cited, "groups": groups,
                "topic": req.get("q") or (m.get("field") or {}).get("topic") or m.get("question"),
                "definition": req.get("definition"), "papers": int(len(w))}

    def of(self, w: int) -> list[dict]:
        self._refresh()
        out = []
        for key, m in list(self.maps.items()):
            i = int(np.searchsorted(m["w"], w))
            if i >= len(m["w"]) or int(m["w"][i]) != w:
                continue
            g, s = int(m["g"][i]), int(m["s"][i])
            d = {"key": key, "topic": m["topic"], "definition": m["definition"], "papers": m["papers"],
                 "via": "cited" if bool(m["cited"][i]) else "topic"}
            gr = m["groups"].get(g)
            if gr:
                d["group"] = {"id": g, "name": gr["name"], "about": gr["about"], "size": gr["size"]}
                sg = gr["subs"].get(s)
                if sg:
                    d["sub"] = {"id": s, "name": sg["name"], "about": sg["about"], "size": sg["size"]}
            out.append(d)
        # maps it is on topic in before those that only cite it; then the most specific: a paper in "coral reefs"
        # and in "biodiversity" is best described by the first
        return sorted(out, key=lambda d: (d["via"] == "cited", d["papers"]))


# ---------------------------------------------------------------------------------------------- a word of a cloud
WORD_SCAN = 3000        # a unit's most central papers read for a word (a group of a whole-field map can have 60,000)
WORD_MAX = 50


def word_papers(svc, out: dict, group: int, word: str, limit: int = 20) -> dict:
    """The papers of one group of a map (0: the whole map) that use a word of its cloud, most central first. The
    word is matched as the clouds count it (srl_search/topics.py doc_terms: stemmed phrases of the title and
    abstract), so a word shown in a cloud finds the papers it was counted in. Only the group's WORD_SCAN most central
    papers are read; `scanned` and `matched` say how many, and `share` is the cloud's own figure for the whole group."""
    from srl_search import topics as T
    key = " ".join(T.stem(t) for t in T.TOKEN.findall(word))
    if not key:
        return {"word": word, "group": group, "scanned": 0, "matched": 0, "papers": []}
    shown = out.get("papers_shown") or {}
    if shown.get("source"):                        # a big whole-field map: every paper is in its .npz
        z = np.load(svc.field_store().root / shown["source"])
        wids, cen = z["w"], z["centrality"]
        members = np.flatnonzero(z["group"] == group) if group else np.arange(len(wids))
        members = members[np.lexsort((members, -cen[members]))]
        ws = [int(wids[i]) for i in members[:WORD_SCAN]]
        total = int(len(members))
    else:
        P = out["papers"]
        cen = P.get("centrality") or [0] * len(P["id"])
        members = [i for i, g in enumerate(P["group"]) if (g == group if group else True)]
        members.sort(key=lambda i: (-(cen[i] or 0), i))
        ws = [int(P["id"][i][1:]) for i in members[:WORD_SCAN]]
        total = len(members)
    eng = svc.api.engine
    hits: list[int] = []
    matched = 0
    surfaces: dict = {}
    for s0 in range(0, len(ws), 500):
        chunk = ws[s0:s0 + 500]
        det = eng.details(chunk)
        for w in chunk:
            r = det.get(w)
            if r and key in T.doc_terms(r.get("title"), r.get("abstract"), surfaces):
                matched += 1
                if len(hits) < WORD_MAX:
                    hits.append(w)
    wp = svc.api.WORKPAGE
    cit = wp.cit if wp else None
    papers = []
    for w in hits[:max(1, min(limit, WORD_MAX))]:
        nd = int(cit.nodes([w])[0]) if cit is not None else -1
        if nd >= 0:
            papers.append(wp.item(nd))
        else:                                       # not in the citation snapshot (added since): the record
            r = eng.details([w]).get(w) or {}
            papers.append({"id": f"W{w}", "title": r.get("title") or "", "year": r.get("year"),
                           "citations": int(r.get("citations") or 0)})
    return {"word": word, "group": group, "total": total, "scanned": len(ws), "matched": matched, "papers": papers}

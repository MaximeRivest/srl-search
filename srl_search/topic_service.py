"""Topic maps as a service: which papers, the cache, the load limits, and the finished answer (titles, links,
map positions). The algorithm is srl_search/topics.py; the routes are in app.py (/topics, /topics/{q}) and the
MCP tool map_topic.

- Papers: the `papers` closest in meaning to the question (the same ranking as /search: similarity minus 0.3 x
  P(junk); records more likely junk than not are left out), or the best matches of an exact-words query.
  Publications only, retracted papers out, one copy per title (the one with references in the citation index).
- Cache: one JSON file per (question, options) per data release, so a repeat or a shared link is instant and
  identical, and the next nightly release recomputes. Other releases' files are removed at start.
- Load: at most MAX_CONCURRENT maps computed at once; the same request asked twice at once is computed once.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import faiss
import numpy as np
from fastapi import HTTPException

from srl_search import topics as T
from srl_search.engine import JUNK_PENALTY, passes

# The cache key includes the method's code: any change to it makes new answers (no stale maps after a promote).
VERSION = hashlib.sha256(b"".join((Path(__file__).parent / f).read_bytes()
                                  for f in ("topics.py", "topic_service.py", "citations.py"))).hexdigest()[:12]
MAX_CONCURRENT = int(os.environ.get("SRL_TOPICS_CONCURRENT", "2"))
WAIT_S = 25                    # a request waits this long for a free slot, then gets a 503 with Retry-After
NPROBE = 512                   # the 1000th-closest paper needs a wider scan than the top 10
JUNK_MAX = 0.5
CACHE_KEEP = 20000             # files per release (~150 KB each)
# One answer about a group stays under ~64k characters (~16-20k tokens): AI apps cap a tool's answer (Claude Code at
# 25k tokens by default) and cut what is over, so pages carry the rest instead.
PAGE_CHARS = 64_000
# Groups of a 1,000-paper map at least this big are split when the map is made: the overview shows how each group
# divides, and "split this group" opens at once (the same split, cached under its own request).
SPLIT_FROM = 60
# What the map carries of each subgroup (its papers: papers.sub; the split's own map has everything)
SUB_KEYS = ("id", "name", "label", "phrases", "size", "years", "typical", "centre", "cites_groups", "between", "cloud",
            "lenses")                     # label: an AI-written name and description (whole-field maps, srl_search/suggest.py)


def clean_abstract(a: str | None) -> str | None:
    a = " ".join(T.TAG.sub(" ", a).split()) if a else ""
    return a or None


def shorten(a: str | None, mode: str) -> str | None:
    if not a or mode == "none":
        return None
    if mode == "short" and len(a) > 320:
        return a[:320].rsplit(" ", 1)[0] + "…"
    return a


def size_of(p: dict) -> int:
    """Characters this paper takes in a text answer (title, facts, link, abstract)."""
    return 220 + len(p.get("title") or "") + len(p.get("venue") or "") + len(p.get("why") or "") + len(p.get("abstract") or "")


def paginate(sizes: list[int], budget: int | None = None) -> list[list[int]]:
    """Consecutive items into pages of at most `budget` characters (an item bigger than that alone on its page)."""
    budget = budget or PAGE_CHARS
    pages, cur, used = [], [], 0
    for i, sz in enumerate(sizes):
        if cur and used + sz > budget:
            pages.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += sz
    if cur or not pages:
        pages.append(cur)
    return pages


def norm_title(t: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"<[^>]+>", "", (t or "").lower()))[:150]


class TopicService:
    def __init__(self, api, cache_root: Path, citations):
        self.api = api
        self.cit = citations
        self.release = api.RELEASE["id"] if api.RELEASE else "none"
        self.cache = Path(cache_root) / self.release
        self.cache.mkdir(parents=True, exist_ok=True)
        for d in Path(cache_root).iterdir():           # answers of other releases are stale
            if d.is_dir() and d.name != self.release:
                shutil.rmtree(d, ignore_errors=True)
        self.slots = threading.BoundedSemaphore(MAX_CONCURRENT)
        self.locks: dict[str, threading.Lock] = {}
        self.locks_guard = threading.Lock()
        self.writes = 0

    # ------------------------------------------------------------------------------------------ the papers
    def _filters(self, year_from, year_to) -> dict:
        return {k: v for k, v in {"year_from": year_from, "year_to": year_to}.items() if v is not None}

    def _keep(self, ws: list[int], f: dict, n: int) -> tuple[list[int], dict]:
        """Best first: details, filters, one copy per title (preferring a copy with references), at most n."""
        eng = self.api.engine
        out, seen, det_all, refs_of = [], {}, {}, {}
        for start in range(0, len(ws), 1000):
            page = ws[start:start + 1000]
            det = eng.details(page)
            det_all.update(det)
            nodes = self.cit.nodes(np.array(page, np.int64))
            has_refs = {w: nd >= 0 and int(self.cit.fwd_off[nd + 1]) > int(self.cit.fwd_off[nd]) for w, nd in zip(page, nodes)}
            for w in page:
                d = det.get(w)
                if d is None or not passes(d, f):
                    continue
                key = norm_title(d.get("title"))
                if len(key) >= 12 and key in seen:
                    i = seen[key]
                    if has_refs[w] and not refs_of.get(out[i]):
                        out[i] = w                    # the published version with references, at the best rank
                        refs_of[w] = True
                    continue
                if len(key) >= 12:
                    seen[key] = len(out)
                refs_of[w] = has_refs[w]
                out.append(w)
                if len(out) == n:
                    return out, det_all
        return out, det_all

    def seed_semantic(self, q: str, n: int, f: dict):
        eng = self.api.engine
        qv = eng.encoder.encode(q)[0]
        faiss.omp_set_num_threads(eng.threads)
        _, cand = eng.index.search(qv.reshape(1, -1), max(3 * n, 3000), params=faiss.SearchParametersIVF(nprobe=NPROBE))
        cand = cand[0][cand[0] >= 0]
        vecs = eng.vectors.get(cand)
        sims = vecs @ qv
        wids = np.asarray(eng.work_ids[cand], dtype=np.int64)
        junk = eng.quality.junk(wids, vecs) if eng.quality is not None else np.zeros(len(wids))
        order = np.lexsort((-sims, -(sims - JUNK_PENALTY * junk)))
        order = [i for i in order if junk[i] < JUNK_MAX]
        by_w = {int(wids[i]): i for i in order}
        kept, det = self._keep([int(wids[i]) for i in order], f, n)
        rows = [by_w[w] for w in kept]
        V = vecs[rows]
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        return T.Seed(np.array(kept, np.int64), V.astype(np.float32), np.ones(len(kept), bool),
                      [det[w] for w in kept], sims[rows].astype(np.float32)), {"matches": None}

    def seed_keyword(self, q: str, n: int, f: dict):
        api = self.api
        hits, matches, understood = api.KEYWORDS.search(q, 3 * n, f, "all")
        kept, det = self._keep([h[0] for h in hits], f, n)
        return self.seed_of(kept, det), {"matches": matches, "query": understood}

    def seed_of(self, ws: list[int], det: dict | None = None) -> T.Seed:
        """Papers given by id (a group being split): details and vectors, in this order."""
        eng = self.api.engine
        det = det if det is not None else eng.details(ws)
        ws = [w for w in ws if w in det]
        pos = [eng.position_of(w) for w in ws]
        has = np.array([p is not None for p in pos], bool)
        V = np.zeros((len(ws), 768), np.float32)
        if has.any():
            got = eng.vectors.get(np.array([p for p in pos if p is not None], np.int64))
            V[has] = got / np.maximum(np.linalg.norm(got, axis=1, keepdims=True), 1e-12)
        return T.Seed(np.array(ws, np.int64), V, has, [det[w] for w in ws], None)

    # ------------------------------------------------------------------------------------------ the answer
    def get(self, q: str, papers: int = 1000, mode: str = "auto", year_from=None, year_to=None,
            group: int | None = None, cached_only: bool = False, scope: str = "closest",
            definition: str | None = None) -> dict:
        """The map (cached or made now). cached_only: reading a group uses a map already made (it is priced as
        one request, so it must never cost a map's work): 409 if there is none. scope="field": the whole-field map
        (srl_search/field.py), made by a background job: raises field.Pending while it is being made."""
        q = " ".join(q.split())
        if scope == "field":
            return self.field_store().get(q, (definition or "").strip() or None, None, VERSION, group,
                                          start=not cached_only)
        if mode == "auto":
            mode = "keyword" if self.api.KEYWORDS and self.api.looks_boolean(q) else "semantic"
        if mode == "keyword" and not self.api.KEYWORDS:
            raise HTTPException(503, "keyword search is not built yet")
        path = self._path(q, papers, mode, year_from, year_to, group)
        if path.exists():
            out = json.loads(path.read_text())
            out["cached"] = True
            return out
        if cached_only:
            what = f"group {group} split into its own groups" if group else "this topic"
            raise HTTPException(409, f"Map {what} first (map_topic, or /topics with the same question and options): "
                                     "reading a group uses that map. Maps are kept until the next nightly data update.")
        with self.locks_guard:
            lock = self.locks.setdefault(path.name, threading.Lock())
        with lock:                                     # the same map asked twice at once: computed once
            if path.exists():
                out = json.loads(path.read_text())
                out["cached"] = True
                return out
            if group is not None:
                parent = self.get(q, papers, mode, year_from, year_to, None)   # cached, or computed now
            if not self.slots.acquire(timeout=WAIT_S):
                raise HTTPException(503, "Many topic maps are being made right now. Try again in a minute.",
                                    headers={"Retry-After": "30"})
            try:
                t0 = time.perf_counter()
                if group is None:
                    out = self.compute(q, papers, mode, year_from, year_to)
                else:
                    out = self.compute_group(q, parent, group)
                out["timing"]["all_ms"] = round((time.perf_counter() - t0) * 1000)
            finally:
                self.slots.release()
            self._store(path, out)
        with self.locks_guard:
            self.locks.pop(path.name, None)
        self._name_later(path, q, group, self._path(q, papers, mode, year_from, year_to, None))
        out["cached"] = False
        return out

    def _name_later(self, path: Path, q: str, group: int | None, main_path: Path) -> None:
        """AI names and descriptions for a new map's groups and subgroups (srl_search/suggest.py), in the
        background: the map is answered at once; asked again a few seconds later it carries the names. Never fails
        a map: without names, groups keep their phrases."""
        def work():
            try:
                from srl_search import suggest
                m = json.loads(path.read_text())
                parent = None
                if group is not None:            # a split: named within its group, as the main map names it
                    time.sleep(8)                # (the main map's names are usually being written right now)
                    main = json.loads(main_path.read_text()) if main_path.exists() else {}
                    g = next((x for x in main.get("groups", []) if x.get("id") == group), {})
                    parent = (g.get("label") or {}).get("name") or (m.get("parent") or {}).get("name")
                    if g.get("label"):
                        m.setdefault("parent", {})["label"] = g["label"]
                cap = suggest.TOPIC_LABELS_USD           # the groups first (saved at once), then their subgroups
                usd = suggest.label_groups(q, m.get("groups", []), parent, lane="topic_labels", cap=cap)
                if any(g.get("label") for g in m.get("groups", [])):
                    m.setdefault("timing", {})["labels_usd"] = round(usd, 4)
                    self._store(path, m)
                    usd += suggest.label_map(q, m, parent, lane="topic_labels", cap=cap)
                    m["timing"]["labels_usd"] = round(usd, 4)
                    self._store(path, m)
            except Exception as e:  # noqa: BLE001
                print(f"topics: groups not named ({type(e).__name__})", file=sys.stderr, flush=True)
        threading.Thread(target=work, daemon=True).start()

    def _path(self, q, papers, mode, year_from, year_to, group) -> Path:
        key_obj = [VERSION, q, papers, mode, year_from, year_to, group]
        return self.cache / f"{hashlib.sha256(json.dumps(key_obj).encode()).hexdigest()[:32]}.json"

    def _store(self, path: Path, out: dict) -> None:
        body = json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)   # never a broken answer
        try:                                           # a full disk or a removed folder costs the cache, not the answer
            self.cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(body)
            os.replace(tmp, path)
            self._prune()
        except OSError as e:
            print(f"topics: not cached ({e})", file=sys.stderr, flush=True)

    def compute(self, q, papers, mode, year_from, year_to) -> dict:
        f = self._filters(year_from, year_to)
        t = time.perf_counter()
        seed, how = (self.seed_keyword if mode == "keyword" else self.seed_semantic)(q, papers, f)
        t_seed = round((time.perf_counter() - t) * 1000)
        n0 = len(seed.wids)
        seed = seed.subset(T.collapse_versions(seed, self.cit))
        how["versions_merged"] = n0 - len(seed.wids)
        if len(seed.wids) < 30:
            raise HTTPException(404, f"Only {len(seed.wids)} papers found for this; a map needs at least 30. "
                                     "Try other or fewer words, or search by meaning.")
        raw = T.build(seed, self.cit, q, background=self.background())
        raw["timing"]["papers_ms"] = t_seed
        out = self.finish(raw, seed)
        out.update({"question": q, "mode": mode, "filters": f, "asked_papers": papers, **how})
        t = time.perf_counter()
        group_of = np.array(out["papers"]["group"])
        splits = {}
        for g in out["groups"]:
            if g["size"] >= SPLIT_FROM:
                sub = self.compute_group(q, out, g["id"], seed.subset(np.flatnonzero(group_of == g["id"])))
                self._store(self._path(q, papers, mode, year_from, year_to, g["id"]), sub)
                splits[g["id"]] = sub
        embed_splits(out, splits)
        out["timing"]["subgroups_ms"] = round((time.perf_counter() - t) * 1000)
        return out

    def compute_group(self, q, parent: dict, group: int, seed: T.Seed | None = None) -> dict:
        """Group `group` of the map split into its own groups. seed: its papers, already at hand (the map being
        made): the same as seed_of gives (details and unit vectors), without the closeness to the question."""
        g = next((x for x in parent["groups"] if x["id"] == group), None)
        if g is None:
            raise HTTPException(404, f"no group {group}; this map has groups 1-{len(parent['groups'])}")
        ws = [int(p[1:]) for p, gg in zip(parent["papers"]["id"], parent["papers"]["group"]) if gg == group]
        t = time.perf_counter()
        if seed is not None:
            seed = T.Seed(seed.wids, seed.vecs, seed.has_vec, seed.details, None)
        else:
            seed = self.seed_of(ws)
        t_seed = round((time.perf_counter() - t) * 1000)
        opt = T.Options(min_size=max(8, len(ws) // 25), min_groups=2, max_groups=8)
        raw = T.build(seed, self.cit, q, opt, background=self.background())
        raw["timing"]["papers_ms"] = t_seed
        out = self.finish(raw, seed)
        out.update({"question": q, "mode": parent["mode"], "filters": parent["filters"],
                    "asked_papers": parent["asked_papers"],
                    "parent": {"group": group, "name": g["name"], "size": g["size"]}})
        return out

    def background(self):
        """Phrase counts in all of science, for the whole map's word cloud (None without a keyword index)."""
        kw = getattr(self.api, "KEYWORDS", None)
        return kw.phrase_counts if kw is not None else None

    def finish(self, raw: dict, seed: T.Seed) -> dict:
        return finish(self.api, self.cit, raw, seed, self.release)

    def field_store(self):
        if getattr(self, "_fields", None) is None:
            from srl_search.field import FieldStore
            from srl_search.same import Matcher
            root = Path(os.environ.get("SRL_FIELDS", "/mnt/fast/scholarsreadinglist/search/fields"))
            # one map per topic: a request for a topic mapped or waiting under other words opens that map
            self._fields = FieldStore(root, same=Matcher(root, encode=lambda t: self.api.engine.encoder.encode(t)[0]))
        return self._fields

    def read(self, out: dict, group: int, part: str = "lists", page: int = 1, abstracts: str = "full") -> dict:
        """One group of a map in depth, with abstracts: its reading lists ("lists"), or all of its papers, most
        central first ("papers"). Pages are filled up to PAGE_CHARS of text (abstracts vary 10x in length), so
        every answer fits what AI apps accept from one tool call, and nothing is cut: the rest is on the next page.
        A paper on several lists is given in full once. Group 0 is the whole topic (its reading lists only: its
        papers are those of every group)."""
        g = overall_group(out) if group == 0 else next((x for x in out["groups"] if x["id"] == group), None)
        if g is None:
            raise HTTPException(404, f"no group {group}; this map has groups 1-{len(out['groups'])}"
                                     + (" (and 0, the whole topic)" if out.get("overall") else ""))
        if group == 0 and part != "lists":
            raise HTTPException(400, "group 0 (the whole topic) has reading lists only: read its groups' papers "
                                     "(read=1, 2, ...)")
        eng = self.api.engine
        res = {"question": out["question"], "group": {k: g.get(k) for k in ("id", "name", "phrases", "size", "years",
                                                                         "cites_groups", "between", "openalex_topics")},
               "groups": {x["id"]: x["name"] for x in out["groups"]}, "part": part, "parent": out.get("parent"),
               "abstracts": abstracts}
        if part == "lists":
            entries = []                               # (lens, item) in reading order
            for key, _ in LENS_TITLES:
                entries += [(key, p) for p in g["lenses"].get(key, [])]
            entries += [("experts", e) for e in g["lenses"].get("experts", [])]
            ids = {int(p["id"][1:]) for k, p in entries if k != "experts"}
            ids |= {int(e[x]["id"][1:]) for k, e in entries if k == "experts" for x in ("top", "latest") if e.get(x)}
            det = eng.details(sorted(ids))
            seen: set = set()
            items = []
            for k, p in entries:
                if k == "experts":
                    e = {**p, "top": self._with_abstract(p["top"], det, abstracts, seen),
                         "latest": self._with_abstract(p["latest"], det, abstracts, seen) if p.get("latest") else None}
                    items.append((k, e, size_of(e["top"]) + (size_of(e["latest"]) if e["latest"] else 0) + 200))
                else:
                    q = self._with_abstract(p, det, abstracts, seen)
                    items.append((k, q, size_of(q)))
            pages = paginate([sz for _, _, sz in items])
            if not 1 <= page <= len(pages):
                raise HTTPException(404, f"the lists of group {group} have {len(pages)} pages")
            lenses: dict = {}
            for k, q, _ in (items[i] for i in pages[page - 1]):
                lenses.setdefault(k, []).append(q)
            res.update({"page": page, "pages": len(pages), "lenses": lenses})
            return res
        shown = out.get("papers_shown") or {}
        if shown.get("source"):                        # a big whole-field map: every paper is in its .npz
            z = np.load(self.field_store().root / shown["source"])
            wids, cen_a, btw_a = z["w"], z["centrality"], z["between"]
            members = np.flatnonzero(z["group"] == group)
            members = members[np.lexsort((members, -cen_a[members]))].tolist()
            wid, cen, btw = (lambda i: int(wids[i])), (lambda i: float(cen_a[i])), (lambda i: bool(btw_a[i]))
        else:
            P = out["papers"]
            members = [i for i, gg in enumerate(P["group"]) if gg == group]
            cen_l = P.get("centrality") or [0] * len(P["id"])
            members.sort(key=lambda i: (-cen_l[i], i))
            wid, cen, btw = (lambda i: int(P["id"][i][1:])), (lambda i: cen_l[i]), (lambda i: P["between"][i])
        # pages filled in order (as paginate), the details read 500 papers at a time and only as far as the page
        # asked for: a group of a big map has hundreds of thousands
        done, cur, used, rank, n_rows, finished = [], [], 0, 0, 0, True
        for s0 in range(0, len(members), 500):
            chunk = members[s0:s0 + 500]
            det = eng.details([wid(i) for i in chunk])
            for i in chunk:
                rank += 1
                if wid(i) not in det:
                    continue
                r = {**self._paper(wid(i), det, abstracts), "centrality": cen(i), "between": btw(i), "rank": rank}
                sz = size_of(r)
                if cur and used + sz > PAGE_CHARS:
                    done.append(cur)
                    cur, used = [], 0
                cur.append(r)
                used += sz
                n_rows += 1
            if len(done) >= page and s0 + 500 < len(members):
                finished = False
                break
        if cur or not done:
            done.append(cur)
        pages = len(done) if finished else max(page + 1, math.ceil(len(members) / max(1, n_rows / len(done))))
        if not 1 <= page <= len(done):
            raise HTTPException(404, f"group {group} has {pages} pages of papers")
        res.update({"page": page, "pages": pages, "total": n_rows if finished else len(members),
                    "papers": done[page - 1]})
        if not finished:
            res["pages_estimated"] = True
        return res

    def _paper(self, w: int, det: dict, abstracts: str = "full") -> dict:
        d = det[w]
        names = [a.strip() for a in (d.get("authors") or "").split("|") if a.strip()]
        out = {"id": f"W{w}", "title": T.TAG.sub("", d.get("title") or "") or "(no title)", "year": d.get("year"),
               "type": d.get("type"), "venue": d.get("venue"), "authors": names[:3],
               "authors_count": d.get("authors_count") or len(names), "citations": d.get("citations"),
               "doi": d.get("doi"), "pmcid": d.get("pmcid"), "abstract": shorten(clean_abstract(d.get("abstract")), abstracts)}
        if self.api.readable(w):
            out["readable"] = True
        return {k: v for k, v in out.items() if v not in (None, [], "")}

    def _with_abstract(self, p: dict, det: dict, abstracts: str = "full", seen: set | None = None) -> dict:
        """The item with its abstract, the first time it is given (later: marked as given above)."""
        if seen is not None and p["id"] in seen:
            return {**p, "given_above": True}
        if seen is not None:
            seen.add(p["id"])
        d = det.get(int(p["id"][1:]))
        a = shorten(clean_abstract(d.get("abstract")) if d else None, abstracts)
        return {**p, "abstract": a} if a else dict(p)

    def _prune(self):
        self.writes += 1
        if self.writes % 200:
            return
        files = sorted(self.cache.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - CACHE_KEEP)]:
            p.unlink(missing_ok=True)


def overall_group(out: dict) -> dict | None:
    """The whole map as a group (id 0) for reading: maps made before 2026-10-15 have none."""
    o = out.get("overall")
    if not o:
        return None
    name = f"the whole of group {out['parent']['group']}" if out.get("parent") else "the whole topic"
    return {**o, "id": 0, "name": name, "phrases": [x["t"] for x in o.get("cloud", [])[:4]]}


def embed_splits(out: dict, splits: dict[int, dict]) -> None:
    """Each split group's subgroups into the map (SUB_KEYS: what the overview draws and reads, with no other
    request), and each paper's subgroup: papers.sub (0: in no subgroup, or its group was not split)."""
    sub_of = {}
    for sp in splits.values():
        sub_of.update(zip(sp["papers"]["id"], sp["papers"]["group"]))
    for g in out["groups"]:
        sp = splits.get(g["id"])
        if sp:
            g["subgroups"] = [{k: x.get(k) for k in SUB_KEYS} for x in sp["groups"]]
            g["subgroups_other"] = sp["coverage"]["other"]
    out["papers"]["sub"] = [sub_of.get(pid, 0) for pid in out["papers"]["id"]]


def finish(api, cit, raw: dict, seed: T.Seed, release: str) -> dict:
    """Work numbers -> papers a person can open: titles, years, venues, links, map positions. (Shared with the
    whole-field worker, srl_search/field.py, which is not this service.)"""
    t = time.perf_counter()
    inside = {int(w) for w in seed.wids}
    wanted = set()
    blocks = raw["groups"] + ([raw["overall"]] if raw.get("overall") else [])
    for g in blocks:
        if "typical" in g:
            wanted.add(g["typical"])
        for name, items in g["lenses"].items():
            for it in items:
                for k in ("w", "top", "latest"):
                    if it.get(k):
                        wanted.add(int(it[k]))
    det = api.engine.details(sorted(wanted))
    by_seed = {int(w): d for w, d in zip(seed.wids, seed.details)}

    def paper(w: int) -> dict:
        d = det.get(w) or by_seed.get(w)
        if d is None:                              # in the citation index only (no English abstract)
            nd = int(cit.nodes([w])[0])
            d = {"title": cit.title(nd) if nd >= 0 else None, "year": int(cit.year[nd]) if nd >= 0 else None,
                 "type": cit.type(nd) if nd >= 0 else None, "citations": int(cit.cites[nd]) if nd >= 0 else None}
        names = [a.strip() for a in (d.get("authors") or "").split("|") if a.strip()]
        out = {"id": f"W{w}", "title": T.TAG.sub("", d.get("title") or "") or "(no title)", "year": d.get("year"),
               "type": d.get("type"), "venue": d.get("venue"), "authors": names[:3],
               "authors_count": d.get("authors_count") or len(names), "citations": d.get("citations"),
               "doi": d.get("doi"), "pmcid": d.get("pmcid"), "in_map": w in inside}
        r = api.readable(w)
        if r:
            out["readable"] = True
        return {k: v for k, v in out.items() if v not in (None, [], "")}

    for g in blocks:
        if "typical" in g:
            g["typical"] = paper(g["typical"])
        for name, items in g["lenses"].items():
            if name == "experts":
                for it in items:
                    it["top"] = paper(it["top"])
                    if it.get("latest"):
                        it["latest"] = paper(it["latest"])
                continue
            for it in items:
                it.update(paper(int(it.pop("w"))))

    # map positions: the release's table (one row per embedded paper)
    xs, ys = [], []
    for w in seed.wids:
        p = api.place_work(int(w))
        xs.append(round(p[0], 5) if p else None)
        ys.append(round(p[1], 5) if p else None)
    papers = raw["papers"]
    papers["id"] = [f"W{w}" for w in papers.pop("w")]
    papers["x"], papers["y"] = xs, ys
    for g in raw["groups"]:
        pts = np.array([(x, y) for x, y, gg in zip(xs, ys, papers["group"]) if gg == g["id"] and x is not None])
        g["centre"] = {"x": round(float(np.median(pts[:, 0])), 5), "y": round(float(np.median(pts[:, 1])), 5)} if len(pts) else None
    raw["timing"]["finish_ms"] = round((time.perf_counter() - t) * 1000)
    raw["release"] = release
    raw["atlas"] = api.ATLAS.id if api.ATLAS is not None else None
    return raw


# ------------------------------------------------------------------------------------------------ as text
# Written for a model to read and act on. Complete: every group, every reading list, every item; what is not
# here (abstracts, every member paper) is one call away, and each answer says which call.
SITE = "https://sciencemadereadable.com"


def _link(p: dict) -> str:
    return f"https://doi.org/{p['doi']}" if p.get("doi") else f"https://openalex.org/{p['id']}"


def _line(p: dict, why: str = "", full: bool = True) -> str:
    """One paper on one line. full: with first author and venue (reading a group); else the map's lean line."""
    a = p.get("authors", [])
    who = (a[0] + (" et al." if p.get("authors_count", 0) > 1 else "")) if a and full else ""
    review = "review" if p.get("type") == "review" and "review" not in why else ""
    cites = f"cited {p['citations']:,}×" if p.get("citations") else ""
    facts = [str(p.get("year") or "n.d."), (p.get("venue") or "") if full else "", who, review, cites, why]
    return f"- **{p['title']}** · " + " · ".join(f for f in facts if f) + f" · {_link(p)}"


def _abstract(p: dict) -> list[str]:
    if p.get("given_above"):
        return ["  (abstract given above)"]
    return [f"  {p['abstract']}"] if p.get("abstract") else []


LENS_TITLES = [
    ("start_here", "Start here: recent papers of the group citing most of its key works (often reviews)"),
    ("core", "Core: what the group's papers cite most (its foundations; may be older or outside the mapped set; in "
             "whole-field maps only works of the field, and its most cited papers where few list their references)"),
    ("rising", "Rising: works of the last 5 years already cited by many of the group's papers"),
    ("distinctive", "Only here: cited mostly by this group, not the others (what sets it apart)"),
    ("foundations", "Foundations from outside the field (whole-field maps): what the group's papers cite most that is not "
                    "itself part of the field: methods, tools, neighbouring classics"),
    ("bridges", "Bridges: papers of the group with many links to another group"),
    ("newest", "Newest papers of the group"),
]


def _lens_title(title: str, gid: int) -> str:
    return title.replace("the group", "the topic").replace("the others", "the rest") if gid == 0 else title


def _ref(out: dict, enc) -> str:
    q = enc(out["question"])
    return q + (f", within group {out['parent']['group']}" if out.get("parent") else "")


def _gname(g: dict) -> str:
    """A group's AI-written name with its telltale phrases, or the phrases alone (srl_search/suggest.py)."""
    lab = (g.get("label") or {}).get("name")
    return f"{lab} [{g['name']}]" if lab else g["name"]


def as_text(out: dict, base: str, page: int = 1) -> str:
    """The whole map as Markdown: the groups at a glance, then every group with every reading list and expert,
    whole groups per page (pages of ~PAGE_CHARS). Raises 404 for a page that does not exist."""
    enc = urllib.parse.quote
    q = out["question"]
    cov = out["coverage"]
    names = {x["id"]: x["name"] for x in out["groups"]}
    whole = overall_group(out)
    # the whole topic's lists come first (block 0 when there are any); then one block per group
    blocks = ([_group_block(whole, names)] if whole else []) + [_group_block(g, names) for g in out["groups"]]
    first = 1 if whole else 0
    pages = paginate([len(b) for b in blocks], PAGE_CHARS - 9000) if blocks else [[]]   # room for the glance + next
    if not 1 <= page <= len(pages):
        raise HTTPException(404, f"this map has {len(pages)} pages")
    head = (f"# Research groups in: {out['parent']['name']} (group {out['parent']['group']} of “{q}”, split further)"
            if out.get("parent") else f"# Research groups in: {q}")
    how = ("closest in meaning to the question" if out["mode"] == "semantic"
           else "of the whole field (found by meaning, judged one by one by an AI classifier, by the topic's exact "
                "phrase, and by following the studies the field cites, which reaches papers without abstracts)"
           if out["mode"] == "field"
           else f"best matches for the exact words (`{out.get('query') or q}`)")
    lines = [head, ""]
    if page == 1:
        lines += [f"{cov['papers']:,} papers {how}, grouped by the works they cite and the words they use: "
                  f"{cov['in_groups']:,} in {len(out['groups'])} groups, {cov['other']:,} in small groups or alone. "
                  "Group names are the phrases most over-represented in each group (automatic, sometimes rough: judge "
                  "a group by its papers). Each reading list comes from the citations made by the group's own papers, "
                  "so it shows what matters to that group, not what is famous overall.", ""]
        if cov.get("references_thin"):
            lines += [f"Note: only {cov['with_references']:,} of these papers have references in OpenAlex, so here the "
                      "groups lean more on wording than on citations.", ""]
        fr = out.get("field")
        if fr:
            lines += [f"How the field was found: {fr['papers']:,} papers judged on topic, made {fr['made'][:10]}. Checked "
                      "once on habitat fragmentation against 19 meta-analyses: it found ~90% of the studies they cite, "
                      "and ~80% of its papers are clearly on topic. Groups of 300+ papers are split into subgroups: read "
                      "those (map with group=N) rather than thousands of papers at once.", ""]
            if cov.get("without_text_vector"):
                lines += [f"{cov['without_text_vector']:,} papers have no English abstract (older papers, books, "
                          "chapters): they are here because the field cites them; their titles are all there is.", ""]
        if whole and whole.get("cloud"):
            lines += ["Words that set the topic apart from the rest of science (most telling first; ×N = how many "
                      "times more often its papers use it): " + _words(whole["cloud"]), ""]
        lines.append("## The groups at a glance")
        for g in out["groups"]:
            yrs = g.get("years")
            uses = ", ".join(f"{k}" for k in list((g.get("cites_groups") or {}))[:3])
            lines.append(f"- Group {g['id']}: {_gname(g)} · {g['size']} papers"
                         + (f" · mostly {yrs[0]}–{yrs[2]}" if yrs else "")
                         + (f" · builds most on groups {uses}" if uses else "")
                         + f" (details on page {next(i for i, pg in enumerate(pages, 1) if g['id'] - 1 + first in pg)})")
            if g.get("subgroups"):
                lines.append("  divides into: " + "; ".join(f"{g['id']}.{x['id']} {_gname(x)} ({x['size']})"
                                                           for x in g["subgroups"])
                             + f" (map_topic group={g['id']} for their reading lists)")
        lines.append("")
    else:
        lines += [f"Page {page} of {len(pages)} of the map (page 1 has the groups at a glance).", ""]
    for i in pages[page - 1]:
        lines.append(blocks[i])
    lines.append("## Next")
    if page < len(pages):
        lines.append(f"- **The map continues: page {page + 1} of {len(pages)}** (map_topic with page={page + 1}, or "
                     f"{base}/topics/{enc(q)}?page={page + 1}). Read every page.")
    if whole:
        lines.append(f"- The whole topic's reading lists with full abstracts: read_topic_group with group=0, or "
                     f"{base}/topics/{enc(q)}?read=0")
    lines += ["- Then read every group that bears on the question in depth, with full abstracts and authors: its "
              "reading lists (read_topic_group, part=lists), then all of its papers, most central first "
              "(part=papers), page by page until the last. Over HTTP: "
              f"{base}/topics/{enc(q)}?read=N&part=lists, then &part=papers&page=1, 2, …",
              f"- Split a large or mixed group into its own groups: map_topic with group=N, or {base}/topics/{enc(q)}?group=N",
              f"- Everything as JSON (every mapped paper with its group, centrality and map position): {base}/topics?q={enc(q)}",
              f"- On the map of science, for people: {SITE}/?topics={enc(q)}",
              "Limits: citations come from OpenAlex and only count links between works it holds; papers from the last "
              "few weeks have no references yet. A group that is missing was not among these papers; it may still exist."]
    return "\n".join(lines) + "\n"


def _words(cloud: list[dict], k: int = 20) -> str:
    return ", ".join(f"{w['t']} (×{w['lift']:g})" for w in cloud[:k])


def _group_block(g: dict, names: dict) -> str:
    yrs = g.get("years")
    span = f", mostly {yrs[0]}–{yrs[2]} (median {yrs[1]})" if yrs else ""
    if g["id"] == 0:
        lines = [f"## Reading lists of {g['name']} (all {g['size']:,} papers{span}; group=0)"]
    else:
        lines = [f"## Group {g['id']}: {_gname(g)} ({g['size']} papers{span})"]
        if (g.get("label") or {}).get("about"):
            lines.append(g["label"]["about"] + " (name and description written by an AI)")
    extra = []
    if g["id"] and g.get("cloud"):
        extra.append("its words, against the other groups: " + _words(g["cloud"], 12))
    if g["id"] and g.get("phrases") and len(g["phrases"]) > 3:
        extra.append("also: " + ", ".join(g["phrases"][3:]))
    uses = g.get("cites_groups") or {}
    if uses:
        extra.append("builds on " + "; ".join(f"group {k} ({names.get(int(k), '')}, {v} citations)" for k, v in uses.items()))
    if g.get("openalex_topics"):
        extra.append("OpenAlex topics: " + ", ".join(f"{t['topic']} {round(100 * t['share'])}%" for t in g["openalex_topics"]))
    if g.get("between"):
        extra.append(f"{g['between']} papers sit between this group and another")
    if extra:
        lines.append("; ".join(extra) + ".")
    L = g["lenses"]
    shown: set = set()
    for key, title in LENS_TITLES:
        if L.get(key):
            lines.append(f"{_lens_title(title, g['id'])}:")
            for p in L[key]:
                if p["id"] in shown:                   # on an earlier list of this group: title and reason only
                    lines.append(f"- {p['title']} (above) · {p.get('why', '')}".rstrip(" ·"))
                else:
                    shown.add(p["id"])
                    lines.append(_line(p, p.get("why", ""), full=False))
    if L.get("experts"):
        unit = "topic" if g["id"] == 0 else "group"
        lines.append(f"Experts here (h-index counted inside the {unit}: h of their works each cited by ≥ h of the {unit}'s "
                     "papers; first author full credit, last author half, others less):")
        for e in L["experts"]:
            def ref(p):
                return f"{p['title']} ({p.get('year') or 'n.d.'})" + ("" if p["id"] in shown else f" {_link(p)}")
            lines.append(f"- {e['name']} (h = {e['h']}, https://openalex.org/{e['author']}) · most cited here: "
                         + ref(e["top"]) + (f" · latest here: {ref(e['latest'])}"
                                            if e.get("latest") and e["latest"]["id"] != e["top"]["id"] else ""))
    return "\n".join(lines) + "\n"


def read_text(r: dict, base: str, abstracts: str | None = None) -> str:
    """One group in depth, one page: its reading lists with abstracts, or its papers with abstracts."""
    enc = urllib.parse.quote
    g = r["group"]
    yrs = g.get("years")
    span = f", mostly {yrs[0]}–{yrs[2]}" if yrs else ""
    where = f" (within group {r['parent']['group']} of “{r['question']}”)" if r.get("parent") else f" of “{r['question']}”"
    title = (f"Reading lists of {g['name']} of “{r['question']}”" if g["id"] == 0
             else f"Group {g['id']}{where}: {g['name']}")
    lines = [f"# {title} ({g['size']:,} papers{span})", ""]
    qs = enc(r["question"])
    more = r["page"] < r["pages"]
    if r["part"] == "lists":
        nxt = (f"Next: page={r['page'] + 1} (part=lists)." if more else
               f"Next: each group in depth, its lists then its papers ({base}/topics/{qs}?read=1, then read=2, …)."
               if g["id"] == 0 else
               f"Next: all of the group's papers, most central first: part=papers, page=1 "
               f"({base}/topics/{qs}?read={g['id']}&part=papers&page=1).")
        lines += [f"Its reading lists, each paper with its abstract (given once when it is on several lists). "
                  f"Page {r['page']} of {r['pages']}. {nxt}", ""]
        for key, title in LENS_TITLES:
            items = r["lenses"].get(key) or []
            if items:
                lines += [f"## {_lens_title(title, g['id'])}", ""]
                for p in items:
                    lines += [_line(p, p.get("why", ""))] + _abstract(p) + [""]
        ex = r["lenses"].get("experts") or []
        if ex:
            lines += ["## Experts here (h-index counted inside the group, position-weighted)", ""]
            for e in ex:
                lines.append(f"- {e['name']} (h = {e['h']}, https://openalex.org/{e['author']}"
                             + (f", author of {e['papers_in_group']} of the group's papers" if e.get("papers_in_group") else "") + ")")
                lines += ["  Most cited here: " + _line(e["top"])[2:]] + [f"  {x.strip()}" for x in _abstract(e["top"])]
                if e.get("latest") and e["latest"]["id"] != e["top"]["id"]:
                    lines += ["  Latest here: " + _line(e["latest"])[2:]] + [f"  {x.strip()}" for x in _abstract(e["latest"])]
                lines.append("")
        if more:
            lines.append(f"(Continued on page {r['page'] + 1}.)")
        return "\n".join(lines) + "\n"
    ps = r["papers"]
    lines += [f"Papers {ps[0]['rank']}–{ps[-1]['rank']} of {r['total']}, most central first (most linked to the rest of "
              f"the group by shared references and meaning). Page {r['page']} of {r['pages']}. "
              + (f"Next: page={r['page'] + 1}." if more else "This is the last page: you have read the whole group."), ""]
    for p in ps:
        mark = " · between groups" if p.get("between") else ""
        lines += [f"{p['rank']}. " + _line(p)[2:] + mark] + _abstract(p) + [""]
    return "\n".join(lines) + "\n"

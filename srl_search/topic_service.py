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
            group: int | None = None) -> dict:
        q = " ".join(q.split())
        if mode == "auto":
            mode = "keyword" if self.api.KEYWORDS and self.api.looks_boolean(q) else "semantic"
        if mode == "keyword" and not self.api.KEYWORDS:
            raise HTTPException(503, "keyword search is not built yet")
        key_obj = [VERSION, q, papers, mode, year_from, year_to, group]
        key = hashlib.sha256(json.dumps(key_obj).encode()).hexdigest()[:32]
        path = self.cache / f"{key}.json"
        if path.exists():
            out = json.loads(path.read_text())
            out["cached"] = True
            return out
        with self.locks_guard:
            lock = self.locks.setdefault(key, threading.Lock())
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
            body = json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)   # never a broken answer
            try:                                       # a full disk or a removed folder costs the cache, not the answer
                self.cache.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
                tmp.write_text(body)
                os.replace(tmp, path)
                self._prune()
            except OSError as e:
                print(f"topics: not cached ({e})", file=sys.stderr, flush=True)
        with self.locks_guard:
            self.locks.pop(key, None)
        out["cached"] = False
        return out

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
        raw = T.build(seed, self.cit, q)
        raw["timing"]["papers_ms"] = t_seed
        out = self.finish(raw, seed)
        out.update({"question": q, "mode": mode, "filters": f, "asked_papers": papers, **how})
        return out

    def compute_group(self, q, parent: dict, group: int) -> dict:
        g = next((x for x in parent["groups"] if x["id"] == group), None)
        if g is None:
            raise HTTPException(404, f"no group {group}; this map has groups 1-{len(parent['groups'])}")
        ws = [int(p[1:]) for p, gg in zip(parent["papers"]["id"], parent["papers"]["group"]) if gg == group]
        t = time.perf_counter()
        seed = self.seed_of(ws)
        t_seed = round((time.perf_counter() - t) * 1000)
        opt = T.Options(min_size=max(8, len(ws) // 25), min_groups=2, max_groups=8)
        raw = T.build(seed, self.cit, q, opt)
        raw["timing"]["papers_ms"] = t_seed
        out = self.finish(raw, seed)
        out.update({"question": q, "mode": parent["mode"], "filters": parent["filters"],
                    "asked_papers": parent["asked_papers"],
                    "parent": {"group": group, "name": g["name"], "size": g["size"]}})
        return out

    def finish(self, raw: dict, seed: T.Seed) -> dict:
        """Work numbers -> papers a person can open: titles, years, venues, links, map positions."""
        api, cit = self.api, self.cit
        t = time.perf_counter()
        inside = {int(w) for w in seed.wids}
        wanted = set()
        for g in raw["groups"]:
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

        for g in raw["groups"]:
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
        raw["release"] = self.release
        raw["atlas"] = api.ATLAS.id if api.ATLAS is not None else None
        return raw

    def _prune(self):
        self.writes += 1
        if self.writes % 200:
            return
        files = sorted(self.cache.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for p in files[:max(0, len(files) - CACHE_KEEP)]:
            p.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------ as text
SITE = "https://sciencemadereadable.com"


def _link(p: dict) -> str:
    return f"https://doi.org/{p['doi']}" if p.get("doi") else f"https://openalex.org/{p['id']}"


def _line(p: dict, why: str = "") -> str:
    a = p.get("authors", [])
    who = (a[0] + (" et al." if p.get("authors_count", 0) > 1 else "")) if a else ""
    review = "review" if p.get("type") == "review" and "review" not in why else ""
    title = p["title"] if len(p["title"]) <= 150 else p["title"][:147].rsplit(" ", 1)[0] + "…"
    facts = [str(p.get("year") or "n.d."), p.get("venue") or "", who, review, why]
    return f"- **{title}** · " + " · ".join(f for f in facts if f) + f" · {_link(p)}"


def as_text(out: dict, base: str, per_lens: int = 2) -> str:
    """A topic map as Markdown: every group with its name, size, years, and the top of each reading list."""
    enc = urllib.parse.quote
    q = out["question"]
    cov = out["coverage"]
    head = (f"# Research groups in: {out['parent']['name']} (group {out['parent']['group']} of “{q}”)"
            if out.get("parent") else f"# Research groups in: {q}")
    how = ("closest in meaning to the question" if out["mode"] == "semantic"
           else f"best matches for the exact words (`{out.get('query') or q}`)")
    lines = [head, "",
             f"{cov['papers']:,} papers {how}, grouped by the works they cite and the words they use "
             f"({cov['in_groups']:,} in {len(out['groups'])} groups, {cov['other']:,} in small groups or alone). "
             "Group names are the phrases most over-represented in each group (automatic). Each list is computed from "
             "the citations made by the group's own papers, so it shows what matters to that group.",
             ""]
    if cov.get("references_thin"):
        lines += [f"Note: only {cov['with_references']:,} of these papers have references in OpenAlex, so the groups "
                  "lean more on wording than on citations here.", ""]
    for g in out["groups"]:
        yrs = g.get("years")
        span = f", mostly {yrs[0]}–{yrs[2]}" if yrs else ""
        lines += [f"## Group {g['id']}: {g['name']} ({g['size']} papers{span})"]
        uses = g.get("cites_groups") or {}
        if uses:
            names = {x["id"]: x["name"] for x in out["groups"]}
            top = list(uses.items())[:2]
            lines.append("Builds on: " + "; ".join(f"group {k} ({names.get(int(k), '')}, {v} citations)" for k, v in top))
        L = g["lenses"]
        for key, title, k in (("start_here", "Start here (recent papers citing most of the group's key works)", per_lens),
                              ("core", "Core (what this group cites most)", per_lens + 1),
                              ("rising", "Rising (recent works gathering citations fast here)", per_lens)):
            if L.get(key):
                lines += [f"{title}:"] + [_line(p, p.get("why", "")) for p in L[key][:k]]
        if L.get("experts"):
            lines.append("Experts here (h-index inside the group, position-weighted): " + "; ".join(
                f"{e['name']} (h = {e['h']}, https://openalex.org/{e['author']})" for e in L["experts"][:3]))
        if L.get("bridges"):
            lines += ["Bridge to other groups:"] + [_line(p, p.get("why", "")) for p in L["bridges"][:1]]
        lines.append("")
    qs = f"{enc(q)}"
    lines += [f"Split one group further: {base}/topics/{qs}?group=N · the same map as JSON (every paper, all lists): "
              f"{base}/topics?q={qs} · on the map of science: {SITE}/?topics={qs}",
              "Citations come from OpenAlex and only count links between works it holds; a missing group means it was "
              "not among these papers, not that it does not exist."]
    return "\n".join(lines) + "\n"

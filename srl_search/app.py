"""HTTP API: one search service for every project (the test page, ScienceMadeReadable, programs and agents via llms.txt).

  GET /search?q=...&k=10
       &mode=auto|semantic|keyword   auto (default): keyword when the query uses search syntax
                                     (quotes, AND/OR/NOT, brackets, word*, -word, title:), else semantic
       &collection=all|smr           smr: papers ScienceMadeReadable can open (PMC full text, CC BY,
                                     research article)
       &year_from=&year_to=&type=article&open_access=true&license=cc-by&pmc_open_access=true
       &abstracts=true&nprobe=&depth=
       &junk_penalty=                    by meaning: rank by similarity - junk_penalty * P(junk), where P(junk) is
                                         how likely a record is spam or not scholarship (srl_search/quality.py);
                                         0 = pure similarity. Default 0.3 for `all`, 0 for `smr` (curated: the
                                         penalty measured no gain there). Each result carries "junk". Not applied
                                         to exact-word (keyword) results.
       &map=true                         add atlas positions: "x","y" on each result we hold a vector
                                         for, and "map": {"x","y"} where the question lands (semantic)
  GET /similar/{ref}?k=10           papers closest in meaning to a paper (W123, DOI or PMCID)
  GET /works/{ref}                  one paper's details
  GET /map/place?ids=W1,W2&q=...    atlas positions of papers and/or a question
  GET /map/regions                  atlas field names, colours and label positions
  GET /map/tiles/{release}/...      the release's map tiles (index.json, z/x/y.pts, z/x/y.ids); immutable;
                                    v2/: the map engine's data (map.json, labels.json, d|p|i/z/x/y.bin)
  GET /map/counts                   exact publications per year and subfield (whole corpus)
  GET /walk?id=W...|DOI             citation walk (citewalk server) with map positions and "readable"
                                    flags on every paper
  GET /topics?q=...&papers=1000     topic map: the research groups inside a topic, named, with reading lists
                                    per group (srl_search/topics.py); &group=N splits one group; cached per release
  GET /topics/{q}                   the same as plain text (Markdown)
Without an explicit type=, search leaves out records that are not publications (datasets,
figures, covers...; srl_search/scope.py), in both modes, and retracted papers (flag or title).
In auto mode, a query that looked like search syntax but matched no exact words is searched by
meaning instead ("fallback" says so).
  GET /health                       what each mode and collection covers
  GET /                             a small test page (web/index.html)
  GET /llms.txt                     the guide for programs and AI agents (web/llms.txt + live coverage)
  GET /docs, /openapi.json          interactive reference and spec of the public routes only
  POST /mcp                         the same search as an MCP server for AI apps (srl_search/mcp_server.py)

Run: see README.md. With a published data release (pipeline/release, DATA.md) every path comes
from SRL_RELEASE (default .../release/current); otherwise SRL_CORPUS, SRL_INDEX, SRL_SMR, SRL_ATLAS.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

import numpy as np

from srl_search.atlas import Atlas, question_spot
from srl_search.citations import Citations
from srl_search.field import Pending, pending_text
from srl_search.engine import EMBEDDING_PLAN, JUNK_PENALTY, Engine, ExactEngine
from srl_search.keyword import KeywordIndex, looks_boolean
from srl_search.public import PUBLIC_BASE, PUBLIC_ROUTES
from srl_search.quality import Quality
from srl_search.topic_service import TopicService, as_text as topic_text, read_text as topic_read_text

ROOT = Path("/mnt/fast/scholarsreadinglist/search")
RELEASE_LINK = Path(os.environ.get("SRL_RELEASE", "/mnt/fast/scholarsreadinglist/release/current"))
RELEASE = json.loads((RELEASE_LINK / "manifest.json").read_text()) if (RELEASE_LINK / "manifest.json").exists() else None
P = {k: Path(v) for k, v in RELEASE["paths"].items()} if RELEASE else {}
CORPUS = P.get("corpus") or Path(os.environ.get("SRL_CORPUS", ROOT / "corpus-2026-10-05"))
INDEX = P.get("index") or Path(os.environ.get("SRL_INDEX", ROOT / "index-2024plus"))
TILES = P.get("tiles")
CITEWALK = os.environ.get("SRL_CITEWALK", "http://127.0.0.1:8798")
WORKS = CORPUS / "works.sqlite"
# Works OpenAlex added after the base snapshot, fetched daily (pipeline/updates): a small corpus beside the base one.
UPDATES = P.get("updates")
UPDATES = UPDATES if UPDATES and (UPDATES / "DONE.json").exists() else None
UPDATES_INFO = json.loads((UPDATES / "manifest.json").read_text()) if UPDATES else None
META = WORKS if WORKS.exists() else Path(os.environ.get("SRL_META", ROOT / "meta-2024plus.sqlite"))
NPROBE = int(os.environ.get("SRL_NPROBE", "128"))  # eval 2026-10-06: recall@10 0.89, ~37 ms search
DEPTH = int(os.environ.get("SRL_DEPTH", "300"))
# Record-quality signals and the junk model (build_quality.py, fit_quality.py); per OpenAlex snapshot, like CORPUS.
QUALITY_DIR = P.get("quality") or Path(os.environ.get("SRL_QUALITY", ROOT / "quality-2026-10-05"))

# The built-in /docs and /openapi.json are replaced below: the published spec lists only the routes the public
# door passes (sciencemadereadable/app/server.py DATA_ROUTES), and the docs page loads it by a relative URL so
# it works both here and behind the door's /api/data/ prefix.
app = FastAPI(title="Scholars Reading List search", version="0.3", docs_url=None, redoc_url=None, openapi_url=None,
              description="Search ~200 million scholarly papers by meaning or exact words. Guide for programs "
                          "and agents: [llms.txt](llms.txt).")
# Read-only GETs, and POSTs to /mcp; lets the test page and browser-based MCP clients work from another address.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["Mcp-Session-Id"])
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"
GUIDE = Path(__file__).resolve().parent.parent / "web" / "llms.txt"


@app.middleware("http")
async def field_asker(request, call_next):
    """The site's door sends an anonymous per-address code (X-SRL-Asker) so one reader can't fill the free line of
    whole-field maps; direct callers (tailnet) have none and no such limit."""
    from srl_search.field import ASKER
    tok = ASKER.set(request.headers.get("x-srl-asker") or None)
    try:
        return await call_next(request)
    finally:
        ASKER.reset(tok)

# Parameter documentation, shared by /search and /similar
D_K = "Number of results, 1-100. No paging: ask for up to 100 at once."
D_COLLECTION = "all (default) or smr: only papers ScienceMadeReadable can open (PubMed Central, CC BY, research articles)."
D_YEAR_FROM, D_YEAR_TO = "Earliest publication year.", "Latest publication year."
D_TYPE = ("Only this publication type (article, review, preprint, book, book-chapter, conference-paper, dissertation, "
          "report, letter, editorial, dataset, ...). Without it, non-publications (datasets, figures, covers) are left out.")
D_OA = "Only open-access papers."
D_LICENSE = "Only this licence, e.g. cc-by."
D_PMC = "Only papers whose full text is in PubMed Central's open-access set."
D_ABSTRACTS = "Include abstracts. false makes answers about 10x smaller."
D_NPROBE = "Index lists scanned (search by meaning): more = slower and slightly more complete. Rarely needed."
D_DEPTH = "Candidates re-scored exactly before filtering (search by meaning). Rarely needed."
D_JUNK = ("Search by meaning ranks by similarity - junk_penalty x junk (how likely a record is spam or not "
          "scholarship). Default 0.3 for all, 0 for smr; 0 = pure similarity.")

QUALITY = Quality(QUALITY_DIR, more=[UPDATES / "quality"] if UPDATES else []) \
    if (QUALITY_DIR / "model.npz").exists() else None
# smr holds PMC research articles under CC BY only; on the 60 evaluation questions it had no junk in the top 10
# and the penalty changed nothing measurable (useful nDCG -0.007, 95% interval -0.017 … +0.004), so it is off.
DEFAULT_JUNK_PENALTY = {"all": JUNK_PENALTY, "smr": 0.0}
engine = Engine(INDEX, META, threads=int(os.environ.get("SRL_THREADS", "4")), quality=QUALITY,
                more_meta=[UPDATES / "works.sqlite"] if UPDATES else [])
SEMANTIC = {"all": engine}
SMR = str(P["smr"]) if "smr" in P else os.environ.get("SRL_SMR")
if SMR:
    # 512 of 4,096 groups: 99% of the exact top 15 for plain questions, ~45 ms (experiments/smr_hnsw.log)
    SEMANTIC["smr"] = ExactEngine(Path(SMR), threads=4, encoder=engine.encoder,
                                  nprobe=int(os.environ.get("SRL_SMR_NPROBE", "512")),
                                  meta_path=WORKS if WORKS.exists() else None, quality=QUALITY)
KEYWORDS = KeywordIndex([CORPUS / "keywords"] + ([UPDATES / "keywords"] if UPDATES else [])) \
    if (CORPUS / "keywords" / "DONE.json").exists() else None
KEYWORD_COLLECTIONS = {"all", "smr"}
ATLAS_BUILD = P.get("atlas") or Path(os.environ.get("SRL_ATLAS", "/mnt/fast/science-atlas/build/20261008T131030-tsne"))
ATLAS = Atlas(ATLAS_BUILD, cache_dir=ROOT) if (ATLAS_BUILD / "reference.npz").exists() else None
# The release's position table (one row per embedded paper, NaN = not placed yet) and its
# "readable on the site" column, both aligned with the index's rows (engine.position_of).
XY = np.load(TILES / "xy.npy", mmap_mode="r") if TILES and (TILES / "xy.npy").exists() else None
READABLE = np.load(P["rows"] / "smr.npy", mmap_mode="r") if "rows" in P else None


def place_work(work_id: int) -> tuple[float, float] | None:
    """Map position of an embedded paper: the release's table, else placed from its vector."""
    if XY is not None:
        row = engine.position_of(int(work_id))
        if row is not None and not np.isnan(XY[row, 0]):
            return float(XY[row, 0]), float(XY[row, 1])
    v = vector_of(int(work_id)) if ATLAS is not None else None
    if v is None:
        return None
    x, y = ATLAS.place(v[None], np.array([work_id], np.uint64))[0]
    return float(x), float(y)


def readable(work_id: int) -> bool | None:
    if READABLE is None:
        return None
    row = engine.position_of(int(work_id))
    return bool(READABLE[row]) if row is not None else None


def vector_of(work_id: int) -> np.ndarray | None:
    """The paper's stored fp16 embedding, from whichever loaded index holds it."""
    smr = SEMANTIC.get("smr")
    if smr is not None:
        try:
            return np.asarray(smr.index.reconstruct(int(work_id)), dtype=np.float32)
        except RuntimeError:
            pass
    pos = engine.position_of(int(work_id))
    return engine.vectors.get(np.array([pos]))[0] if pos is not None else None


def add_positions(out: dict, q: str | None) -> dict:
    if ATLAS is None:
        return out
    t = time.perf_counter()
    found = 0
    for r in out["results"]:
        p = place_work(int(r["id"][1:]))
        if p is not None:
            r["x"], r["y"] = round(p[0], 5), round(p[1], 5)
            found += 1
    if q and out.get("mode") == "semantic":
        pts = [(r["x"], r["y"]) for r in out["results"] if "x" in r]
        if len(pts) >= 3:   # where its results are (atlas.question_spot); the atlas rule only without results
            x, y = question_spot(np.array(pts, np.float64))
        else:
            x, y = ATLAS.place(engine.encoder.encode(q))[0]
        out["map"] = {"x": round(float(x), 5), "y": round(float(y), 5)}
    out["map_placed"] = found
    out.setdefault("timing", {})["map_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-cache"})


def semantic_engine(collection: str):
    e = SEMANTIC.get(collection)
    if e is None:
        raise HTTPException(404, f"unknown collection {collection!r} for semantic search; have {sorted(SEMANTIC)}")
    return e


def filters(year_from, year_to, type, open_access, license, pmc_open_access):
    return {"year_from": year_from, "year_to": year_to, "type": type,
            "open_access": open_access or None, "license": license,
            "pmc_open_access": pmc_open_access or None}


@app.get("/health")
def health():
    def sem(e):
        m = e.manifest
        return {"papers": m["papers"], "published_from": m["min_date"], "built_utc": m["created_utc"]}
    out = {"semantic": {name: sem(e) for name, e in SEMANTIC.items()},
           "model": "lightonai/DenseOn", "searched_text": "English title + abstract (first 512 tokens)",
           "details": str(META.name)}
    if KEYWORDS:
        out["keyword"] = {"papers": KEYWORDS.num_docs, "collections": sorted(KEYWORD_COLLECTIONS),
                          "searched_text": "title, abstract, OpenAlex keywords + MeSH (English stemming)"}
    # older clients read these top-level fields
    out.update({"papers": engine.manifest["papers"], "published_from": engine.manifest["min_date"],
                "collections": sorted(SEMANTIC)})
    if RELEASE:
        out["release"] = {k: RELEASE.get(k) for k in ("id", "built_utc", "published_utc", "counts", "embeddings")}
    if UPDATES_INFO:
        out["updates"] = {"papers": UPDATES_INFO["works"], "published": UPDATES_INFO.get("dates"),
                          "fetched_utc": UPDATES_INFO["created_utc"], "base": UPDATES_INFO["base"],
                          "what": "papers OpenAlex added after the base snapshot, fetched daily"}
    out["ranking"] = ({"junk_model": str(QUALITY_DIR), "default_junk_penalty": DEFAULT_JUNK_PENALTY,
                       "records": QUALITY.manifest["works"], "built_utc": QUALITY.manifest["built_utc"]}
                      if QUALITY else {"junk_model": None})
    return out


@app.get("/llms.txt", include_in_schema=False)
def guide():
    """How to use this API, for programs and AI agents, with what is covered right now."""
    return PlainTextResponse(GUIDE.read_text().replace("{{COVERAGE}}", coverage_text()),
                             headers={"Cache-Control": "public, max-age=3600"})


def coverage_text() -> str:
    lines = []
    # the embedding backfill runs newest first: until it has every planned row, older years are missing
    emb = (RELEASE or {}).get("embeddings") or {}
    plan = json.loads(EMBEDDING_PLAN.read_text()) if EMBEDDING_PLAN.exists() else {}
    growing = emb.get("rows") is None or emb["rows"] < plan.get("expected_rows", 0)
    for name, e in SEMANTIC.items():
        m = e.manifest
        what = "every paper" if name == "all" else "papers ScienceMadeReadable can open (collection=smr)"
        lines.append(f"- Search by meaning, {what}: {m['papers']:,} papers published from {m['min_date']}"
                     + (" (older years are being added)." if growing else "."))
    if KEYWORDS:
        lines.append(f"- Search by exact words and paper lookup (/works): {KEYWORDS.num_docs:,} papers, "
                     "all years.")
    if RELEASE:
        lines.append(f"- Data release {RELEASE['id']}, published {RELEASE.get('published_utc') or RELEASE.get('built_utc')}"
                     + ("; a new one most nights while older years are added." if growing else "."))
    lines.append(f"- OpenAlex snapshot of {CORPUS.name.split('-', 1)[-1]} (titles, abstracts, citations, labels).")
    if UPDATES_INFO:
        lines.append(f"- Plus {UPDATES_INFO['works']:,} papers OpenAlex added since, fetched daily (last fetch "
                     f"{UPDATES_INFO['created_utc'][:10]}). Their citation counts and labels are as of that fetch; "
                     "the citation walk does not include them yet.")
    return "\n".join(lines)


@app.get("/openapi.json", include_in_schema=False)
def openapi_public():
    if app.openapi_schema is None:
        spec = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        spec["paths"] = {p: v for p, v in spec["paths"].items() if p in PUBLIC_ROUTES}
        # first: relative to wherever this spec was fetched from (here, or behind the public door)
        spec["servers"] = [{"url": ".", "description": "where this spec was loaded from"},
                           {"url": PUBLIC_BASE, "description": "public (30 requests/minute per address)"}]
        app.openapi_schema = spec
    return JSONResponse(app.openapi_schema)


@app.get("/docs", include_in_schema=False)
def docs():
    return get_swagger_ui_html(openapi_url="openapi.json", title=f"{app.title} · reference")


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=2000,
                          description="A question, a description, keywords, or keyword syntax: \"phrase\", OR, "
                                      "NOT/-word, brackets, word*, \"a b\"~5, title:/abstract:/keywords:"),
           k: int = Query(10, ge=1, le=100, description=D_K),
           mode: str = Query("auto", pattern="^(auto|semantic|keyword)$",
                             description="semantic = by meaning; keyword = exact words (English stemming); auto = "
                                         "keyword when q uses search syntax, else semantic"),
           collection: str = Query("all", description=D_COLLECTION),
           year_from: int | None = Query(None, description=D_YEAR_FROM),
           year_to: int | None = Query(None, description=D_YEAR_TO),
           type: str | None = Query(None, description=D_TYPE),
           open_access: bool = Query(False, description=D_OA),
           license: str | None = Query(None, description=D_LICENSE),
           pmc_open_access: bool = Query(False, description=D_PMC),
           abstracts: bool = Query(True, description=D_ABSTRACTS),
           nprobe: int = Query(NPROBE, ge=1, le=1024, description=D_NPROBE),
           depth: int = Query(DEPTH, ge=10, le=5000, description=D_DEPTH),
           map: bool = Query(False, description="Add map-of-science positions (x, y in 0-1) to results and the question."),
           junk_penalty: float | None = Query(None, ge=0, le=10, description=D_JUNK)):
    """Search papers by meaning or by exact words. Retracted papers are always left out. Answers: mode, results,
    filters, timing; by meaning also ranking; by exact words also matches (exact count) and query (as understood)."""
    if junk_penalty is None:
        junk_penalty = DEFAULT_JUNK_PENALTY.get(collection, 0.0)
    f = filters(year_from, year_to, type, open_access, license, pmc_open_access)
    auto = mode == "auto"
    if auto:
        mode = "keyword" if KEYWORDS and looks_boolean(q) else "semantic"
    if mode == "keyword":
        if not KEYWORDS:
            raise HTTPException(503, "keyword search is not built yet")
        if collection not in KEYWORD_COLLECTIONS:
            raise HTTPException(404, f"unknown collection {collection!r}; have {sorted(KEYWORD_COLLECTIONS)}")
        t0 = time.perf_counter()
        f = {x: v for x, v in f.items() if v is not None}
        hits, matches, understood = KEYWORDS.search(q, k * 3 + 20, f, collection)
        t1 = time.perf_counter()
        results = engine.shape(hits, k, f, abstracts)
        if len(results) < k and "type" not in f and len(hits) >= k * 3 + 20:
            # the best matches were mostly non-publications (datasets, figures...): ask the index
            # for publications only (slower, so only here); its count is then exact for them
            hits, matches, understood = KEYWORDS.search(q, k * 3 + 20, {**f, "publications_only": True}, collection)
            t1 = time.perf_counter()
            results = engine.shape(hits, k, f, abstracts)
        t2 = time.perf_counter()
        out = {"mode": "keyword", "matches": matches, "query": understood, "results": results, "filters": f,
               "timing": {"keyword_ms": round((t1 - t0) * 1000, 1), "details_ms": round((t2 - t1) * 1000, 1),
                          "total_ms": round((t2 - t0) * 1000, 1)}}
        if results or not auto:
            return add_positions(out, None) if map else out
        # Auto mode guessed search syntax (brackets, quotes, a hyphen...) but nothing matches every word:
        # a sentence pasted with "(CBD)" in it should not come back empty. Search by meaning instead.
        fallback = {"from": "keyword", "query": understood, "matches": matches}
    else:
        fallback = None
    out = semantic_engine(collection).search(q, k, nprobe, depth, f, abstracts, junk_penalty=junk_penalty)
    out["mode"] = "semantic"
    if fallback:
        out["fallback"] = fallback
    return add_positions(out, q) if map else out


SITE = "https://sciencemadereadable.com"


@app.get("/search/{q:path}", response_class=PlainTextResponse)
def search_text(q: str,
                k: int = Query(10, ge=1, le=100, description=D_K),
                mode: str = Query("auto", pattern="^(auto|semantic|keyword)$",
                                  description="As in /search: auto, semantic or keyword."),
                collection: str = Query("all", description=D_COLLECTION),
                year_from: int | None = Query(None, description=D_YEAR_FROM),
                year_to: int | None = Query(None, description=D_YEAR_TO),
                type: str | None = Query(None, description=D_TYPE),
                open_access: bool = Query(False, description=D_OA),
                license: str | None = Query(None, description=D_LICENSE),
                pmc_open_access: bool = Query(False, description=D_PMC),
                abstracts: bool = Query(True, description=D_ABSTRACTS)):
    """The same search with the question in the address and the answer as plain text (Markdown), for AI tools
    that can only open a web address: /search/why do cats purr. Spaces may be written %20 or +."""
    # a form-style address (why+do+cats+purr) has no spaces at all; with real spaces a + is the search's own
    if " " not in q:
        q = q.replace("+", " ")
    q = q.strip()
    if not 2 <= len(q) <= 2000:
        raise HTTPException(422, "the question must be 2-2000 characters: /search/why do cats purr")
    out = search(q=q, k=k, mode=mode, collection=collection, year_from=year_from, year_to=year_to, type=type,
                 open_access=open_access, license=license, pmc_open_access=pmc_open_access, abstracts=abstracts,
                 nprobe=NPROBE, depth=DEPTH, map=False, junk_penalty=None)
    return PlainTextResponse(as_text(q, out, k), headers={"Cache-Control": "no-store"})


def as_text(q: str, out: dict, k: int) -> str:
    """A search answer as Markdown: every result with what a reader (or a model) needs to judge and cite it."""
    enc = urllib.parse.quote
    if out["mode"] == "keyword":
        how = f"Exact words, understood as `{out['query']}`: {out['matches']:,} papers match"
    else:
        how = "By meaning: the closest papers in what they are about, even in other words"
        if out.get("fallback"):
            how = (f"No paper contains all of `{out['fallback']['query']}`, so these are by meaning instead "
                   "(closest in what they are about)")
    lines = [f"# Studies about: {q}", "",
             f"Learning about a field or a topic, rather than looking for one fact? Open its map instead: "
             f"https://ask.sciencemadereadable.com/map/{enc(q)} gives the research groups inside it, each with its "
             "key papers, rising work, reviews to start with and experts; then read the groups that matter in full "
             "(every paper with its abstract), as that page explains. A list like the one below shows only the "
             "closest papers, not how the field divides.", "",
             f"{how}. Best first. Titles and abstracts from OpenAlex, not full texts.", "",
             f"Options after `?`, e.g. `?k=30&year_from=2020` (k up to 100; also year_to, type=review, open_access=true, "
             f"abstracts=false). The same search as JSON: {PUBLIC_BASE}/search?q={enc(q)} · "
             f"guide: {PUBLIC_BASE}/llms.txt"]
    if not out["results"]:
        lines.append("\nNothing found. Try other words, fewer words, or a question in plain language.")
    for i, r in enumerate(out["results"], 1):
        names = [a.strip() for a in (r.get("authors") or "").split("|") if a.strip()]
        n = r.get("authors_count") or len(names)
        who = "; ".join(names[:3]) + (f"; and {n - 3} more" if n > 3 else "") if names else ""   # names may be "Last, First"
        facts = [str(r["year"]) if r.get("year") else "", r.get("venue") or "", who,
                 f"cited {r['citations']:,} times" if r.get("citations") else "", r.get("type") or ""]
        if r.get("is_oa"):
            facts.append("open access")
        lines += ["", f"## {i}. {r['title'] or '(no title)'}", " · ".join(f for f in facts if f)]
        ref = r.get("doi") or r["id"]
        lines.append(f"Link: {'https://doi.org/' + r['doi'] if r.get('doi') else r.get('url') or r['openalex']}"
                     f" · on the map of science, with its citations: {SITE}/?work={enc(ref)}")
        if r.get("smr"):
            lines.append(f"Read in plain words: {SITE}/?paper={enc(r.get('doi') or r.get('pmcid') or ref)}")
        if r.get("abstract"):
            lines += ["", r["abstract"]]
    if len(out["results"]) == k:
        lines += ["", f"More results: ask for up to 100 with `k`, e.g. `?k={min(100, max(30, k * 3))}`."]
    return "\n".join(lines) + "\n"


@app.get("/similar/{ref:path}")
def similar(ref: str, k: int = Query(10, ge=1, le=100, description=D_K),
            year_from: int | None = Query(None, description=D_YEAR_FROM),
            year_to: int | None = Query(None, description=D_YEAR_TO),
            type: str | None = Query(None, description=D_TYPE),
            open_access: bool = Query(False, description=D_OA),
            license: str | None = Query(None, description=D_LICENSE),
            pmc_open_access: bool = Query(False, description=D_PMC),
            abstracts: bool = Query(True, description=D_ABSTRACTS),
            nprobe: int = Query(NPROBE, ge=1, le=1024, description=D_NPROBE),
            depth: int = Query(DEPTH, ge=10, le=5000, description=D_DEPTH),
            collection: str = Query("all", description=D_COLLECTION),
            junk_penalty: float | None = Query(None, ge=0, le=10, description=D_JUNK)):
    """Papers closest in meaning to one paper. ref: W-number (W2782456798), DOI or PMCID. 404 when the paper is
    not in the index searched by meaning (older years are still being added)."""
    if junk_penalty is None:
        junk_penalty = DEFAULT_JUNK_PENALTY.get(collection, 0.0)
    e = semantic_engine(collection)
    w = e.resolve(ref)
    out = e.similar(w, k, nprobe, depth,
                    filters(year_from, year_to, type, open_access, license, pmc_open_access),
                    abstracts, junk_penalty=junk_penalty) if w is not None else None
    if out is None:
        raise HTTPException(404, f"{ref} is not in the semantic collection {collection!r} (yet)")
    out["mode"] = "semantic"
    return out


@app.get("/map/place")
def map_place(ids: str = "", q: str | None = Query(None, max_length=2000)):
    """Map-of-science positions (x, y in 0-1) of up to 200 papers (ids=W1,W2 or DOIs) and/or a question (q)."""
    if ATLAS is None:
        raise HTTPException(503, "no atlas build")
    out = {"atlas": ATLAS.id, "papers": {}, "missing": []}
    refs = [x for x in ids.split(",") if x.strip()][:200]
    for ref in refs:
        w = engine.resolve(ref)
        p = place_work(w) if w is not None else None
        if p is None:
            out["missing"].append(ref)
            continue
        out["papers"][ref] = {"x": round(p[0], 5), "y": round(p[1], 5)}
    if q:
        x, y = ATLAS.place(engine.encoder.encode(q))[0]
        out["question"] = {"x": round(float(x), 5), "y": round(float(y), 5)}
    return out


@app.get("/map/regions")
def map_regions():
    """Map-of-science field names, colours and label positions, and the release id used in tile URLs."""
    if ATLAS is None:
        raise HTTPException(503, "no atlas build")
    # "release": the id the tile URLs need (/map/tiles/<release>/...); the site reads it here
    return {"atlas": ATLAS.id, "release": RELEASE["id"] if RELEASE else None, "fields": ATLAS.regions()}


@app.get("/works/{ref:path}")
def work(ref: str):
    """One paper's full record. ref: W2782456798, https://openalex.org/W..., a DOI (10.1038/..., https://doi.org/...,
    doi:...) or a PMCID (PMC1234567). Covers every paper in the corpus, all years."""
    w = engine.resolve(ref)
    d = engine.details([w]).get(w) if w is not None else None
    if d is None:
        raise HTTPException(404, f"{ref} is not in the corpus")
    d["id"] = f"W{w}"
    d["openalex"] = f"https://openalex.org/W{w}"
    return d


@app.get("/map/tiles/{release}/{path:path}")
def map_tiles(release: str, path: str):
    """Static map tiles of the current release. The release id in the URL makes them immutable."""
    if not TILES or not RELEASE:
        raise HTTPException(503, "no data release")
    if release != RELEASE["id"]:
        raise HTTPException(404, f"release {release!r} is not served; current is {RELEASE['id']!r}")
    f = (TILES / path).resolve()
    if TILES.resolve() not in f.parents or not f.is_file() or f.suffix not in (".json", ".pts", ".ids", ".bin") \
            or f.name == "label-candidates.json":
        raise HTTPException(404, "no such tile")
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if path.startswith(("v2/d/", "v2/p/")):   # stored gzipped: the browser inflates them off its main thread
        headers["Content-Encoding"] = "gzip"
    return FileResponse(f, media_type="application/json" if f.suffix == ".json" else "application/octet-stream",
                        headers=headers)


@app.get("/map/counts")
def map_counts():
    """Exact publications per year and subfield over the whole corpus (for the map's charts)."""
    f = P.get("rows", Path("/nonexistent")) / "counts.json"
    if not f.exists():
        raise HTTPException(503, "no data release")
    return FileResponse(f, media_type="application/json", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/walk")
def walk(id: str = Query(..., min_length=2, max_length=300, description="W-number or DOI of the starting paper."),
         k: int = Query(30, ge=1, le=200, description="How many papers sharing its references to walk through."),
         max_refs: int = Query(300, ge=0, description="Papers listing more references count as outliers (0 = no limit)."),
         max_cites: int = Query(500, ge=0, description="Papers cited more often count as outliers (0 = no limit).")):
    """The citation walk from one paper: the k papers citing most of the same references, and what they cite that
    the starting paper does not. answers = the strongest "should also read / should have cited" candidates
    (W-numbers); nodes = the papers (title, year, citations, topic, map position, readable on the site); links =
    citation edges. 0.5-5 s. Returns an error when the paper has no references in the snapshot."""
    params = urllib.parse.urlencode({"id": id, "k": k, "max_refs": max_refs, "max_cites": max_cites})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{CITEWALK}/api/analyze?{params}", timeout=60) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        return JSONResponse(json.loads(e.read() or b"{}") or {"error": str(e)}, status_code=e.code)
    except OSError as e:
        raise HTTPException(503, f"citation walk unavailable: {e}")
    t1 = time.perf_counter()
    out.pop("atlas", None)      # the citewalk page's own local map; positions below use the atlas
    out.pop("regions", None)
    placed = 0
    for wid, n in out.get("nodes", {}).items():
        w = int(wid[1:])
        p = place_work(w)
        if p is not None:
            n["x"], n["y"] = round(p[0], 5), round(p[1], 5)
            placed += 1
        n["readable"] = readable(w)
    out["map"] = {"atlas": ATLAS.id if ATLAS else None, "placed": placed, "papers": len(out.get("nodes", {}))}
    out["timing"] = {"walk_ms": round((t1 - t) * 1000, 1), "map_ms": round((time.perf_counter() - t1) * 1000, 1)}
    return out


# ------------------------------------------------------------------------------------------------ topic maps
# The citation index of this release, memory-mapped read-only (the citation walk's server maps the same files:
# shared page cache, no extra memory), and the topic maps over it (srl_search/topics.py, topic_service.py).
CITATIONS = Citations(P["citations"]) if "citations" in P and (P["citations"] / "meta.json").exists() else None
# (the dev API, search/dev.sh, keeps its own cache: answers from unpromoted code must never reach the public)
TOPICS = TopicService(sys.modules[__name__], Path(os.environ.get("SRL_TOPICS_CACHE", ROOT / "topics-cache")),
                      CITATIONS) if CITATIONS is not None else None
D_TOPIC_Q = ("A question or topic in plain words (by meaning), or exact-words syntax (quotes, AND/OR/NOT, word*), "
             "as in /search.")
D_PAPERS = "How many papers to map: the closest in meaning, or the best exact-word matches. More = broader, slower."
D_GROUP = "Split this group of the map (its id, from a previous answer) into its own groups."
D_READ = ("Read this group of the map in depth, with abstracts: its reading lists (part=lists) or all of its papers, "
          "most central first (part=papers), in pages of ~64k characters (page=N). The map must exist (409 if not: "
          "call /topics first). read=0: the whole topic's reading lists (lists only). Counts as 1 request, not 5.")
D_PART = "With read: lists (its reading lists) or papers (all its papers, page by page)."
D_SCOPE = ("closest (default): the papers closest to the question, mapped in seconds. field: the whole research field "
           "(every paper on topic: by meaning, an AI judge, the exact phrase and the studies the field cites), mapped "
           "by a background job in ~15-40 minutes the first time, then kept for everyone; while it is being made the "
           "answer is HTTP 202 with its progress. Groups of 300+ papers are split into subgroups (group=N).")
D_DEFINITION = "With scope=field: what the topic covers, in a sentence (optional; part of the map's identity)."
D_PAGE = ("Which page: of the map itself (as text), or with read of the group's lists or papers. Each answer says "
          "how many pages there are and what to call next.")


def topics_service() -> TopicService:
    if TOPICS is None:
        raise HTTPException(503, "topic maps need the citation index, which this release does not have")
    return TOPICS


@app.get("/topics")
def topics(q: str = Query(..., min_length=2, max_length=2000, description=D_TOPIC_Q),
           papers: int = Query(1000, ge=200, le=2000, description=D_PAPERS),
           mode: str = Query("auto", pattern="^(auto|semantic|keyword)$", description="As in /search."),
           year_from: int | None = Query(None, description=D_YEAR_FROM),
           year_to: int | None = Query(None, description=D_YEAR_TO),
           group: int | None = Query(None, ge=1, le=50, description=D_GROUP),
           read: int | None = Query(None, ge=0, le=50, description=D_READ),
           part: str = Query("lists", pattern="^(lists|papers)$", description=D_PART),
           page: int = Query(1, ge=1, le=100, description=D_PAGE),
           abstracts: str = Query("full", pattern="^(full|short|none)$",
                                  description="With read: full abstracts, their first sentences, or none."),
           scope: str = Query("closest", pattern="^(closest|field)$", description=D_SCOPE),
           definition: str | None = Query(None, max_length=1000, description=D_DEFINITION)):
    """The research groups inside a topic, and what to read in each. The papers (closest in meaning, or exact-word
    matches) are grouped by the works they cite and the words they use; each group gets a name (the phrases most
    over-represented in it) and reading lists computed from its own papers' citations: core (most cited by the
    group), distinctive (cited mostly by this group), rising (recent works gathering citations fast), start_here
    (recent papers citing most of the group's key works, often reviews), bridges (papers linking to another group),
    newest, foundations (whole-field maps only: works cited most from outside the field; their other lists keep to
    papers of the field), and experts (authors with the highest position-weighted h-index inside the group); "overall" has the
    same lists for the whole topic. Word clouds: each group's "cloud" holds the phrases it uses more than the other
    groups, and overall's the phrases the topic uses more than the rest of science (share = of its papers, lift = how
    many times more often, score = LDAvis relevance with lambda 0.6, what the order follows). Every paper comes
    with its group and map position. 2-6 s the first time (up to ~30 s when the disk cache is cold), then instant
    (cached for the data release). Counts as 5 requests at the public door. With read=N: one group in depth, with
    abstracts (part=lists: its reading lists; part=papers&page=N: all of its papers, most central first)."""
    svc = topics_service()
    try:
        out = svc.get(q, papers, mode, year_from, year_to, group, cached_only=read is not None, scope=scope,
                      definition=definition)
    except Pending as e:
        return JSONResponse({"status": e.status.get("status"), "progress": e.status, "question": q, "scope": "field",
                             "message": "The whole-field map is being made; ask again later (the same request)."},
                            status_code=202, headers={"Retry-After": "60"})
    return svc.read(out, read, part, page, abstracts) if read is not None else out


@app.get("/fields/queue")
def fields_queue():
    """Whole-field maps being made and waiting, in order, with when each should be ready. Two lines: maps paid for
    on sciencemadereadable.com go first; the free line follows (about `free_per_day` maps a day). Topics in the
    line are public, as their maps will be."""
    return JSONResponse(topics_service().field_store().queue_view(), headers={"Cache-Control": "no-store"})


_FIELD_PAPERS: dict[str, tuple[float, frozenset]] = {}


def _field_papers(store, key: str) -> frozenset:
    """The work numbers of one whole-field map (kept in memory; reread when its field.json changes)."""
    f = store.root / key / "field.json"
    m = f.stat().st_mtime
    if key not in _FIELD_PAPERS or _FIELD_PAPERS[key][0] != m:
        _FIELD_PAPERS[key] = (m, frozenset(int(w) for w in json.loads(f.read_text())["papers"]))
    return _FIELD_PAPERS[key][1]


@app.get("/fields/suggest")
def fields_suggest(q: str = Query(..., min_length=2, max_length=300, description="The topic as someone typed it."),
                   definition: str | None = Query(None, max_length=600, description="What they said it covers (optional).")):
    """Five well-phrased versions of a topic before asking for its whole-field map, each with a one-sentence
    definition that sets its scope and settles ambiguous words ("LLM": large language models, or the law degree):
    the closest to what was typed first, then narrower, broader or other readings (`note` says which). Written by
    an AI (Kimi K3), cached; a few seconds the first time. 503 when unavailable: the topic as typed still works.
    Counts as 5 requests at the public door."""
    from srl_search import suggest
    try:
        return JSONResponse(suggest.rewrites(q, definition), headers={"Cache-Control": "no-store"})
    except suggest.Unavailable as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "3600"})
    except ValueError:
        raise HTTPException(503, "no usable suggestions this time; the topic as typed works")


@app.get("/fields")
def fields_made(q: str | None = Query(None, max_length=500, description="Words or a question: the maps that hold "
                                      "most of the 50 studies closest to it first (score: that share, 0-1)."),
                limit: int = Query(50, ge=1, le=500)):
    """Every whole-field map made so far (public: anyone can open them with /topics?scope=field and the same topic
    and definition), newest first, or the maps that cover q: of the 50 studies closest in meaning to q, the share
    each map holds (`score`; "direct preference optimization" -> LLM post-training 0.6+, an unrelated subject 0)."""
    store = topics_service().field_store()
    rows = store.made()
    if q and rows:
        near = [int(r["id"][1:]) for r in engine.search(q, k=50, abstracts=False)["results"]]
        for r in rows:
            papers = _field_papers(store, r["key"])
            r["score"] = round(sum(w in papers for w in near) / max(1, len(near)), 3)
        rows.sort(key=lambda r: -r["score"])
    return JSONResponse({"maps": rows[:limit], "total": len(rows)}, headers={"Cache-Control": "no-store"})


@app.get("/topics/{q:path}", response_class=PlainTextResponse)
def topics_text(q: str,
                papers: int = Query(1000, ge=200, le=2000, description=D_PAPERS),
                mode: str = Query("auto", pattern="^(auto|semantic|keyword)$", description="As in /search."),
                year_from: int | None = Query(None, description=D_YEAR_FROM),
                year_to: int | None = Query(None, description=D_YEAR_TO),
                group: int | None = Query(None, ge=1, le=50, description=D_GROUP),
                read: int | None = Query(None, ge=0, le=50, description=D_READ),
                part: str = Query("lists", pattern="^(lists|papers)$", description=D_PART),
                page: int = Query(1, ge=1, le=100, description=D_PAGE),
                abstracts: str = Query("full", pattern="^(full|short|none)$",
                                       description="With read: full abstracts, their first sentences, or none."),
                scope: str = Query("closest", pattern="^(closest|field)$", description=D_SCOPE),
                definition: str | None = Query(None, max_length=1000, description=D_DEFINITION)):
    """The same topic map with the question in the address and the answer as plain text (Markdown): every group,
    every reading list, every expert. /topics/intermittent fasting. With read=N: one group in depth, with abstracts
    (?read=2&part=lists, then ?read=2&part=papers&page=1, 2, ...)."""
    if " " not in q:
        q = q.replace("+", " ")
    q = q.strip()
    if not 2 <= len(q) <= 2000:
        raise HTTPException(422, "the question must be 2-2000 characters: /topics/intermittent fasting")
    svc = topics_service()
    try:
        out = svc.get(q, papers, mode, year_from, year_to, group, cached_only=read is not None, scope=scope,
                      definition=definition)
    except Pending as e:
        return PlainTextResponse(pending_text(e.status, q, PUBLIC_BASE), status_code=202,
                                 headers={"Cache-Control": "no-store", "Retry-After": "60"})
    text = topic_read_text(svc.read(out, read, part, page, abstracts), PUBLIC_BASE) if read is not None else topic_text(out, PUBLIC_BASE, page)
    return PlainTextResponse(text, headers={"Cache-Control": "no-store"})


# Last, once every function above exists: the MCP tools call them.
from srl_search import mcp_server  # noqa: E402

MCP = mcp_server.mount(app, sys.modules[__name__])

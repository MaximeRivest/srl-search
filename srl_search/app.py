"""HTTP API: one search service for every project (the test page, ScienceMadeReadable, the MCP server).

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
Without an explicit type=, search leaves out records that are not publications (datasets,
figures, covers...; srl_search/scope.py), in both modes, and retracted papers (flag or title).
In auto mode, a query that looked like search syntax but matched no exact words is searched by
meaning instead ("fallback" says so).
  GET /health                       what each mode and collection covers
  GET /                             a small test page (web/index.html)

Run: see README.md. With a published data release (pipeline/release, DATA.md) every path comes
from SRL_RELEASE (default .../release/current); otherwise SRL_CORPUS, SRL_INDEX, SRL_SMR, SRL_ATLAS.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import numpy as np

from srl_search.atlas import Atlas, question_spot
from srl_search.engine import JUNK_PENALTY, Engine, ExactEngine
from srl_search.keyword import KeywordIndex, looks_boolean
from srl_search.quality import Quality

ROOT = Path("/mnt/fast/scholarsreadinglist/search")
RELEASE_LINK = Path(os.environ.get("SRL_RELEASE", "/mnt/fast/scholarsreadinglist/release/current"))
RELEASE = json.loads((RELEASE_LINK / "manifest.json").read_text()) if (RELEASE_LINK / "manifest.json").exists() else None
P = {k: Path(v) for k, v in RELEASE["paths"].items()} if RELEASE else {}
CORPUS = P.get("corpus") or Path(os.environ.get("SRL_CORPUS", ROOT / "corpus-2026-10-05"))
INDEX = P.get("index") or Path(os.environ.get("SRL_INDEX", ROOT / "index-2024plus"))
TILES = P.get("tiles")
CITEWALK = os.environ.get("SRL_CITEWALK", "http://127.0.0.1:8798")
WORKS = CORPUS / "works.sqlite"
META = WORKS if WORKS.exists() else Path(os.environ.get("SRL_META", ROOT / "meta-2024plus.sqlite"))
NPROBE = int(os.environ.get("SRL_NPROBE", "128"))  # eval 2026-10-06: recall@10 0.89, ~37 ms search
DEPTH = int(os.environ.get("SRL_DEPTH", "300"))
# Record-quality signals and the junk model (build_quality.py, fit_quality.py); per OpenAlex snapshot, like CORPUS.
QUALITY_DIR = P.get("quality") or Path(os.environ.get("SRL_QUALITY", ROOT / "quality-2026-10-05"))

app = FastAPI(title="Scholars Reading List search", version="0.2")
# Read-only GETs; lets the test page work when opened from another address (e.g. a preview panel).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"

QUALITY = Quality(QUALITY_DIR) if (QUALITY_DIR / "model.npz").exists() else None
# smr holds PMC research articles under CC BY only; on the 60 evaluation questions it had no junk in the top 10
# and the penalty changed nothing measurable (useful nDCG -0.007, 95% interval -0.017 … +0.004), so it is off.
DEFAULT_JUNK_PENALTY = {"all": JUNK_PENALTY, "smr": 0.0}
engine = Engine(INDEX, META, threads=int(os.environ.get("SRL_THREADS", "4")), quality=QUALITY)
SEMANTIC = {"all": engine}
SMR = str(P["smr"]) if "smr" in P else os.environ.get("SRL_SMR")
if SMR:
    # 512 of 4,096 groups: 99% of the exact top 15 for plain questions, ~45 ms (experiments/smr_hnsw.log)
    SEMANTIC["smr"] = ExactEngine(Path(SMR), threads=4, encoder=engine.encoder,
                                  nprobe=int(os.environ.get("SRL_SMR_NPROBE", "512")),
                                  meta_path=WORKS if WORKS.exists() else None, quality=QUALITY)
KEYWORDS = KeywordIndex(CORPUS / "keywords") if (CORPUS / "keywords" / "DONE.json").exists() else None
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
        out["keyword"] = {"papers": KEYWORDS.searcher.num_docs, "collections": sorted(KEYWORD_COLLECTIONS),
                          "searched_text": "title, abstract, OpenAlex keywords + MeSH (English stemming)"}
    # older clients read these top-level fields
    out.update({"papers": engine.manifest["papers"], "published_from": engine.manifest["min_date"],
                "collections": sorted(SEMANTIC)})
    if RELEASE:
        out["release"] = {k: RELEASE.get(k) for k in ("id", "built_utc", "published_utc", "counts", "embeddings")}
    out["ranking"] = ({"junk_model": str(QUALITY_DIR), "default_junk_penalty": DEFAULT_JUNK_PENALTY,
                       "records": QUALITY.manifest["works"], "built_utc": QUALITY.manifest["built_utc"]}
                      if QUALITY else {"junk_model": None})
    return out


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=2000), k: int = Query(10, ge=1, le=100),
           mode: str = Query("auto", pattern="^(auto|semantic|keyword)$"), collection: str = "all",
           year_from: int | None = None, year_to: int | None = None, type: str | None = None,
           open_access: bool = False, license: str | None = None, pmc_open_access: bool = False,
           abstracts: bool = True, nprobe: int = Query(NPROBE, ge=1, le=1024),
           depth: int = Query(DEPTH, ge=10, le=5000), map: bool = False,
           junk_penalty: float | None = Query(None, ge=0, le=10)):
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


@app.get("/similar/{ref:path}")
def similar(ref: str, k: int = Query(10, ge=1, le=100), year_from: int | None = None,
            year_to: int | None = None, type: str | None = None, open_access: bool = False,
            license: str | None = None, pmc_open_access: bool = False, abstracts: bool = True,
            nprobe: int = Query(NPROBE, ge=1, le=1024), depth: int = Query(DEPTH, ge=10, le=5000),
            collection: str = "all", junk_penalty: float | None = Query(None, ge=0, le=10)):
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
    if ATLAS is None:
        raise HTTPException(503, "no atlas build")
    # "release": the id the tile URLs need (/map/tiles/<release>/...); the site reads it here
    return {"atlas": ATLAS.id, "release": RELEASE["id"] if RELEASE else None, "fields": ATLAS.regions()}


@app.get("/works/{ref:path}")
def work(ref: str):
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
def walk(id: str = Query(..., min_length=2, max_length=300), k: int = Query(30, ge=1, le=200),
         max_refs: int = Query(300, ge=0), max_cites: int = Query(500, ge=0)):
    """The citation walk from one paper, with each paper's map position and whether the site can open it."""
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

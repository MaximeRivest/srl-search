"""HTTP API: one search service for every project (the test page, ScienceMadeReadable, the MCP server).

  GET /search?q=...&k=10
       &mode=auto|semantic|keyword   auto (default): keyword when the query uses search syntax
                                     (quotes, AND/OR/NOT, brackets, word*, -word, title:), else semantic
       &collection=all|smr           smr: papers ScienceMadeReadable can open (PMC full text, CC BY,
                                     research article)
       &year_from=&year_to=&type=article&open_access=true&license=cc-by&pmc_open_access=true
       &abstracts=true&nprobe=&depth=
       &map=true                         add atlas positions: "x","y" on each result we hold a vector
                                         for, and "map": {"x","y"} where the question lands (semantic)
  GET /similar/{ref}?k=10           papers closest in meaning to a paper (W123, DOI or PMCID)
  GET /works/{ref}                  one paper's details
  GET /map/place?ids=W1,W2&q=...    atlas positions of papers and/or a question
  GET /map/regions                  atlas field names, colours and label positions
  GET /health                       what each mode and collection covers
  GET /                             a small test page (web/index.html)

Run: see README.md (SRL_CORPUS, SRL_INDEX, SRL_SMR).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import numpy as np

from srl_search.atlas import Atlas
from srl_search.engine import Engine, ExactEngine
from srl_search.keyword import KeywordIndex, looks_boolean

ROOT = Path("/mnt/fast/scholarsreadinglist/search")
CORPUS = Path(os.environ.get("SRL_CORPUS", ROOT / "corpus-2026-10-05"))
INDEX = Path(os.environ.get("SRL_INDEX", ROOT / "index-2024plus"))
WORKS = CORPUS / "works.sqlite"
META = WORKS if WORKS.exists() else Path(os.environ.get("SRL_META", ROOT / "meta-2024plus.sqlite"))
NPROBE = int(os.environ.get("SRL_NPROBE", "128"))  # eval 2026-10-06: recall@10 0.89, ~37 ms search
DEPTH = int(os.environ.get("SRL_DEPTH", "300"))

app = FastAPI(title="Scholars Reading List search", version="0.2")
# Read-only GETs; lets the test page work when opened from another address (e.g. a preview panel).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"

engine = Engine(INDEX, META, threads=int(os.environ.get("SRL_THREADS", "4")))
SEMANTIC = {"all": engine}
SMR = os.environ.get("SRL_SMR")
if SMR:
    # 512 of 4,096 groups: 99% of the exact top 15 for plain questions, ~45 ms (experiments/smr_hnsw.log)
    SEMANTIC["smr"] = ExactEngine(Path(SMR), threads=4, encoder=engine.encoder,
                                  nprobe=int(os.environ.get("SRL_SMR_NPROBE", "512")),
                                  meta_path=WORKS if WORKS.exists() else None)
KEYWORDS = KeywordIndex(CORPUS / "keywords") if (CORPUS / "keywords" / "DONE.json").exists() else None
KEYWORD_COLLECTIONS = {"all", "smr"}
ATLAS_BUILD = Path(os.environ.get("SRL_ATLAS", "/mnt/fast/science-atlas/build/20261008T131030-tsne"))
ATLAS = Atlas(ATLAS_BUILD, cache_dir=ROOT) if (ATLAS_BUILD / "reference.npz").exists() else None


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
    found = [(r, vector_of(int(r["id"][1:]))) for r in out["results"]]
    found = [(r, v) for r, v in found if v is not None]
    if found:
        xy = ATLAS.place(np.stack([v for _, v in found]))
        for (r, _), (x, y) in zip(found, xy):
            r["x"], r["y"] = round(float(x), 5), round(float(y), 5)
    if q and out.get("mode") == "semantic":
        x, y = ATLAS.place(engine.encoder.encode(q))[0]
        out["map"] = {"x": round(float(x), 5), "y": round(float(y), 5)}
    out["map_placed"] = len(found)
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
    return out


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=2000), k: int = Query(10, ge=1, le=100),
           mode: str = Query("auto", pattern="^(auto|semantic|keyword)$"), collection: str = "all",
           year_from: int | None = None, year_to: int | None = None, type: str | None = None,
           open_access: bool = False, license: str | None = None, pmc_open_access: bool = False,
           abstracts: bool = True, nprobe: int = Query(NPROBE, ge=1, le=1024),
           depth: int = Query(DEPTH, ge=10, le=5000), map: bool = False):
    f = filters(year_from, year_to, type, open_access, license, pmc_open_access)
    if mode == "auto":
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
        t2 = time.perf_counter()
        out = {"mode": "keyword", "matches": matches, "query": understood, "results": results, "filters": f,
               "timing": {"keyword_ms": round((t1 - t0) * 1000, 1), "details_ms": round((t2 - t1) * 1000, 1),
                          "total_ms": round((t2 - t0) * 1000, 1)}}
        return add_positions(out, None) if map else out
    out = semantic_engine(collection).search(q, k, nprobe, depth, f, abstracts)
    out["mode"] = "semantic"
    return add_positions(out, q) if map else out


@app.get("/similar/{ref:path}")
def similar(ref: str, k: int = Query(10, ge=1, le=100), year_from: int | None = None,
            year_to: int | None = None, type: str | None = None, open_access: bool = False,
            license: str | None = None, pmc_open_access: bool = False, abstracts: bool = True,
            nprobe: int = Query(NPROBE, ge=1, le=1024), depth: int = Query(DEPTH, ge=10, le=5000),
            collection: str = "all"):
    e = semantic_engine(collection)
    w = e.resolve(ref)
    out = e.similar(w, k, nprobe, depth,
                    filters(year_from, year_to, type, open_access, license, pmc_open_access),
                    abstracts) if w is not None else None
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
        v = vector_of(w) if w is not None else None
        if v is None:
            out["missing"].append(ref)
            continue
        x, y = ATLAS.place(v[None])[0]
        out["papers"][ref] = {"x": round(float(x), 5), "y": round(float(y), 5)}
    if q:
        x, y = ATLAS.place(engine.encoder.encode(q))[0]
        out["question"] = {"x": round(float(x), 5), "y": round(float(y), 5)}
    return out


@app.get("/map/regions")
def map_regions():
    if ATLAS is None:
        raise HTTPException(503, "no atlas build")
    return {"atlas": ATLAS.id, "fields": ATLAS.regions()}


@app.get("/works/{ref:path}")
def work(ref: str):
    w = engine.resolve(ref)
    d = engine.details([w]).get(w) if w is not None else None
    if d is None:
        raise HTTPException(404, f"{ref} is not in the corpus")
    d["id"] = f"W{w}"
    d["openalex"] = f"https://openalex.org/W{w}"
    return d

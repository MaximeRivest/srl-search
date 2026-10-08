"""HTTP API.

  GET /search?q=...&k=10            semantic search (question or description in plain words)
       &year_from=&year_to=&type=article&open_access=true&license=cc-by&pmc_open_access=true
       &abstracts=true&nprobe=&depth=
  GET /similar/{ref}?k=10           papers closest to a paper (W123, DOI or PMCID)
  GET /works/{ref}                  one paper's details
  GET /health                       what this index covers
  GET /                             a small test page to search and browse (web/index.html)

Run: SRL_INDEX=... SRL_META=... uvicorn srl_search.app:app --host 127.0.0.1 --port 8810
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from srl_search.engine import Engine, ExactEngine

INDEX = Path(os.environ.get("SRL_INDEX", "/mnt/fast/scholarsreadinglist/search/index-2024plus"))
META = Path(os.environ.get("SRL_META", "/mnt/fast/scholarsreadinglist/search/meta-2024plus.sqlite"))
NPROBE = int(os.environ.get("SRL_NPROBE", "128"))  # eval 2026-10-06: recall@10 0.89, ~37 ms search
DEPTH = int(os.environ.get("SRL_DEPTH", "300"))

app = FastAPI(title="Scholars Reading List search", version="0.1")
# Read-only GETs; lets the test page work when opened from another address (e.g. a preview panel).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(PAGE, headers={"Cache-Control": "no-cache"})
engine = Engine(INDEX, META, threads=int(os.environ.get("SRL_THREADS", "4")))
# Collections searched exactly, by name: ?collection=smr (ScienceMadeReadable: papers it can open).
COLLECTIONS = {"all": engine}
SMR = os.environ.get("SRL_SMR")
if SMR:
    # 512 of 4,096 groups: 99% of the exact top 15 for plain questions, ~45 ms (experiments/smr_hnsw.log)
    COLLECTIONS["smr"] = ExactEngine(Path(SMR), threads=4, encoder=engine.encoder,
                                     nprobe=int(os.environ.get("SRL_SMR_NPROBE", "512")))


def pick(collection: str):
    e = COLLECTIONS.get(collection)
    if e is None:
        raise HTTPException(404, f"unknown collection {collection!r}; have {sorted(COLLECTIONS)}")
    return e


def filters(year_from, year_to, type, open_access, license, pmc_open_access):
    return {"year_from": year_from, "year_to": year_to, "type": type,
            "open_access": open_access or None, "license": license,
            "pmc_open_access": pmc_open_access or None}


@app.get("/health")
def health(collection: str = "all"):
    if collection != "all":
        m = pick(collection).manifest
        return {"collection": collection, "papers": m["papers"], "published_from": m["min_date"],
                "rule": m.get("rule"), "built_utc": m["created_utc"], "collections": sorted(COLLECTIONS)}
    m = engine.manifest
    return {"collections": sorted(COLLECTIONS), "papers": m["papers"], "published_from": m["min_date"], "index": m["index"]["factory"],
            "model": "lightonai/DenseOn", "searched_text": "English title + abstract (first 512 tokens)",
            "built_utc": m["created_utc"]}


@app.get("/search")
def search(q: str = Query(..., min_length=2, max_length=2000), k: int = Query(10, ge=1, le=100),
           year_from: int | None = None, year_to: int | None = None, type: str | None = None,
           open_access: bool = False, license: str | None = None, pmc_open_access: bool = False,
           abstracts: bool = True, nprobe: int = Query(NPROBE, ge=1, le=1024),
           depth: int = Query(DEPTH, ge=10, le=5000), collection: str = "all"):
    return pick(collection).search(q, k, nprobe, depth,
                         filters(year_from, year_to, type, open_access, license, pmc_open_access), abstracts)


@app.get("/similar/{ref:path}")
def similar(ref: str, k: int = Query(10, ge=1, le=100), year_from: int | None = None,
            year_to: int | None = None, type: str | None = None, open_access: bool = False,
            license: str | None = None, pmc_open_access: bool = False, abstracts: bool = True,
            nprobe: int = Query(NPROBE, ge=1, le=1024), depth: int = Query(DEPTH, ge=10, le=5000),
            collection: str = "all"):
    e = pick(collection)
    w = e.resolve(ref)
    out = e.similar(w, k, nprobe, depth,
                         filters(year_from, year_to, type, open_access, license, pmc_open_access),
                         abstracts) if w is not None else None
    if out is None:
        raise HTTPException(404, f"{ref} is not in collection {collection!r}")
    return out


@app.get("/works/{ref:path}")
def work(ref: str, collection: str = "all"):
    e = pick(collection)
    w = e.resolve(ref)
    d = e.details([w]).get(w) if w is not None else None
    if d is None:
        raise HTTPException(404, f"{ref} is not in this index")
    d["id"] = f"W{w}"
    d["openalex"] = f"https://openalex.org/W{w}"
    return d

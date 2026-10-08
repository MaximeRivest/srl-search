"""The shared paper universe: every work the embedding plan selected (English, non-empty abstract),
with details, PMC flags and a keyword index. Built once per content snapshot; used by every
collection (all, smr, ...), the semantic indexes, the API and the MCP server.

  uv run python build_corpus.py parts    --out DIR     # 1. selection + PMC flags -> DIR/parts/*.parquet
  uv run python build_corpus.py sqlite   --out DIR     # 2. DIR/works.sqlite  (details by id / DOI / PMCID)
  uv run python build_corpus.py keywords --out DIR     # 3. DIR/keywords/     (tantivy: boolean, phrases, BM25)

Steps 2 and 3 read the parts and can run at the same time. Each step refuses to redo finished work.

PMC flags: pmcid from NCBI's PMC-ids map (any PMC article); pmc_oa = full text in our Open Access
mirror; pmc_license / pmc_type from the mirror (NCBI licence code, JATS article-type);
smr = 1 when ScienceMadeReadable can open it (pmc_oa, "CC BY", "research-article").
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

EMBEDDING_MANIFEST = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/manifest.json")
SNAPSHOT = Path("/mnt/bulk/openalex/embedding-inputs/denseon-english-abstracts")
PMC_ARTICLES = Path("/mnt/bulk/pmc/oa-2026-10-05/derived/articles.parquet")
PMC_IDS = Path("/mnt/bulk/pmc/PMC-ids-2026-10-06.csv.gz")
MAX_AUTHORS = 12
CHUNK_ROW_GROUPS = 10

COLUMNS = ["id", "doi", "title", "date", "year", "type", "authors", "venue", "publisher", "is_open_access",
           "open_access_status", "license", "citations", "is_retracted", "url", "pdf_url", "topic", "field",
           "keywords", "mesh_terms", "abstract", "language"]

SCHEMA = """
CREATE TABLE works (
  id INTEGER PRIMARY KEY,          -- OpenAlex W-number
  doi TEXT, title TEXT, date TEXT, year INTEGER, type TEXT,
  authors TEXT, authors_count INTEGER, venue TEXT, publisher TEXT,
  is_oa INTEGER, oa_status TEXT, license TEXT, citations INTEGER, is_retracted INTEGER,
  url TEXT, pdf_url TEXT, topic TEXT, field TEXT, keywords TEXT, mesh TEXT,
  pmcid TEXT, pmc_oa INTEGER, pmc_license TEXT, pmc_type TEXT, smr INTEGER,
  abstract TEXT
);
"""
OUT_COLUMNS = ["id", "doi", "title", "date", "year", "type", "authors", "authors_count", "venue", "publisher",
               "is_oa", "oa_status", "license", "citations", "is_retracted", "url", "pdf_url", "topic", "field",
               "keywords", "mesh", "pmcid", "pmc_oa", "pmc_license", "pmc_type", "smr", "abstract"]


# ---------------------------------------------------------------- 1. parts

def selection_mask(t: pa.Table) -> pa.ChunkedArray:
    """Identical to pipeline/embeddings/embed_content.py, policy has_english_abstract."""
    has_abstract = pc.fill_null(pc.not_equal(pc.utf8_trim_whitespace(t["abstract"]), ""), False)
    return pc.and_(has_abstract, pc.fill_null(pc.equal(t["language"], "en"), False))


def pmc_table(out: Path) -> Path:
    """doi (lowercase) -> pmcid, pmc_oa, pmc_license, pmc_type; one row per DOI, mirror rows first."""
    path = out / "pmc-by-doi.parquet"
    if path.exists():
        return path
    a = pq.read_table(PMC_ARTICLES, columns=["pmcid", "doi", "license_code", "article_type", "error", "xml_bytes"])
    ok = pc.and_(pc.is_null(a["error"]), pc.greater(a["xml_bytes"], 0))
    a = a.filter(pc.and_(ok, pc.invert(pc.is_null(a["doi"]))))
    mirror = pa.table({"doi_l": a["doi"], "pmcid": a["pmcid"], "pmc_oa": pa.array(np.ones(a.num_rows, np.int8)),
                       "pmc_license": a["license_code"], "pmc_type": a["article_type"]})
    ids = pcsv.read_csv(PMC_IDS, convert_options=pcsv.ConvertOptions(
        include_columns=["DOI", "PMCID"], column_types={"DOI": pa.string(), "PMCID": pa.string()}))
    ids = ids.filter(pc.and_(pc.invert(pc.is_null(ids["DOI"])), pc.invert(pc.is_null(ids["PMCID"]))))
    rest = pa.table({"doi_l": pc.utf8_lower(ids["DOI"]), "pmcid": ids["PMCID"],
                     "pmc_oa": pa.array(np.zeros(ids.num_rows, np.int8)),
                     "pmc_license": pa.nulls(ids.num_rows, pa.string()), "pmc_type": pa.nulls(ids.num_rows, pa.string())})
    both = pa.concat_tables([mirror, rest])
    # first row per DOI wins: mirror rows come first
    idx = pa.table({"doi_l": both["doi_l"], "i": pa.array(np.arange(both.num_rows))}) \
        .group_by("doi_l").aggregate([("i", "min")])["i_min"]
    pq.write_table(both.take(pc.take(idx, pc.sort_indices(idx))), str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    return path


def dates_as_text(col: pa.ChunkedArray) -> pa.Array:
    try:
        return pc.cast(col, pa.string())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return pa.array([d.isoformat() if d else None for d in col.to_pylist()], pa.string())


def one_part(args: tuple[str, list[int], str, str]) -> dict:
    src, row_groups, dst, pmc_path = args
    t = pq.ParquetFile(src).read_row_groups(row_groups, columns=COLUMNS)
    t = t.filter(selection_mask(t))
    if t.num_rows == 0:
        pq.write_table(pa.table({c: pa.array([], pa.string()) for c in ["empty"]}), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        return {"file": src, "rows": 0}
    doi = pc.replace_substring_regex(t["doi"], "^https://doi.org/", "")
    names = pc.split_pattern(pc.fill_null(t["authors"], ""), " | ")
    count = pc.if_else(pc.equal(pc.fill_null(t["authors"], ""), ""), 0, pc.list_value_length(names))
    base = pa.table({
        "id": pc.cast(pc.replace_substring(t["id"], "https://openalex.org/W", ""), pa.int64()),
        "doi": doi, "doi_l": pc.cast(pc.utf8_lower(doi), pa.string()), "title": t["title"], "date": dates_as_text(t["date"]),
        "year": t["year"], "type": t["type"],
        "authors": pc.binary_join(pc.list_slice(names, 0, MAX_AUTHORS), pa.scalar(" | ", pa.large_string())),
        "authors_count": pc.cast(count, pa.int32()), "venue": t["venue"], "publisher": t["publisher"],
        "is_oa": pc.cast(pc.fill_null(t["is_open_access"], False), pa.int8()),
        "oa_status": t["open_access_status"], "license": t["license"], "citations": t["citations"],
        "is_retracted": pc.cast(pc.fill_null(t["is_retracted"], False), pa.int8()),
        "url": t["url"], "pdf_url": t["pdf_url"], "topic": t["topic"], "field": t["field"],
        "keywords": t["keywords"], "mesh": t["mesh_terms"], "abstract": t["abstract"],
    })
    pmc = pq.read_table(pmc_path)
    j = base.join(pmc, "doi_l", join_type="left outer", use_threads=False)
    smr = pc.and_(pc.and_(pc.equal(pc.fill_null(j["pmc_oa"], 0), 1), pc.equal(j["pmc_license"], "CC BY")),
                  pc.equal(j["pmc_type"], "research-article"))
    j = j.append_column("smr", pc.cast(pc.fill_null(smr, False), pa.int8()))
    j = j.select(OUT_COLUMNS).sort_by("id")
    pq.write_table(j, dst + ".tmp", compression="zstd", row_group_size=100_000)
    os.replace(dst + ".tmp", dst)
    return {"file": src, "rows": j.num_rows}


def cmd_parts(out: Path, processes: int) -> None:
    (out / "parts").mkdir(parents=True, exist_ok=True)
    plan = json.loads(EMBEDDING_MANIFEST.read_text())
    pmc_path = str(pmc_table(out))
    files = sorted(SNAPSHOT.glob("year=*/data.parquet"))
    todo = []
    for f in files:  # ~1M-row chunks of row groups: bounded memory per worker
        n = pq.ParquetFile(f).metadata.num_row_groups
        for c, start in enumerate(range(0, n, CHUNK_ROW_GROUPS)):
            dst = out / "parts" / f"{f.parent.name}-{c:04d}.parquet"
            if not dst.exists():
                todo.append((str(f), list(range(start, min(n, start + CHUNK_ROW_GROUPS))), str(dst), pmc_path))
    print(f"{len(files)} source files, {len(todo)} chunks to do", flush=True)
    t0 = time.time()
    with Pool(processes) as pool:
        for i, r in enumerate(pool.imap_unordered(one_part, todo)):
            if i % 100 == 0:
                print(f"  {i+1}/{len(todo)} chunks ({time.time()-t0:.0f}s)", flush=True)
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in (out / "parts").glob("*.parquet")
                if "empty" not in pq.ParquetFile(p).schema_arrow.names)
    if total != plan["expected_rows"]:
        raise SystemExit(f"{total:,} rows selected; the embedding plan selected {plan['expected_rows']:,}")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "works": total,
                "selection": plan["selection"], "embedding_plan_id": plan["plan_id"],
                "snapshot": str(SNAPSHOT), "pmc_articles": str(PMC_ARTICLES), "pmc_ids": str(PMC_IDS)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"parts done: {total:,} works = embedding plan ({time.time()-t0:.0f}s)", flush=True)


def parts(out: Path) -> list[str]:
    return sorted(str(p) for p in (out / "parts").glob("*.parquet")
                  if "empty" not in pq.ParquetFile(p).schema_arrow.names)


# ---------------------------------------------------------------- 2. sqlite

def cmd_sqlite(out: Path, threads: int, memory: str) -> None:
    import duckdb
    final = out / "works.sqlite"
    if final.exists():
        raise SystemExit(f"{final} exists")
    files = parts(out)
    t0 = time.time()
    tmp = out / "works.sqlite.tmp"
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA page_size=8192;" + SCHEMA)
    db.close()
    tmpdir = out / "duckdb-tmp"
    tmpdir.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory}'; SET temp_directory='{tmpdir}';"
                "SET preserve_insertion_order=false;")
    dup = con.execute(f"SELECT count(*) - count(DISTINCT id) FROM read_parquet({files!r})").fetchone()[0]
    if dup:
        raise SystemExit(f"{dup:,} duplicate work ids")
    con.execute(f"ATTACH '{tmp}' AS s (TYPE sqlite)")
    con.execute(f"INSERT INTO s.works SELECT {', '.join(OUT_COLUMNS)} FROM read_parquet({files!r}) ORDER BY id")
    con.execute("DETACH s")
    print(f"  rows written ({time.time()-t0:.0f}s)", flush=True)
    db = sqlite3.connect(tmp)
    n = db.execute("SELECT count(*) FROM works").fetchone()[0]
    db.execute("CREATE INDEX works_doi ON works(doi COLLATE NOCASE)")
    db.execute("CREATE INDEX works_pmcid ON works(pmcid)")
    db.commit()
    db.close()
    os.replace(tmp, final)
    print(f"works.sqlite: {n:,} rows, {final.stat().st_size/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 3. keywords

def cmd_keywords(out: Path, threads: int, heap_gb: int) -> None:
    import tantivy
    from srl_search.keyword import schema
    final = out / "keywords"
    if (final / "DONE.json").exists():
        raise SystemExit(f"{final} exists")
    tmp = out / "keywords.tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir()
    index = tantivy.Index(schema(), path=str(tmp))
    writer = index.writer(heap_size=heap_gb * 1_000_000_000, num_threads=threads)
    t0 = time.time()
    n = 0
    cols = ["id", "title", "abstract", "keywords", "mesh", "year", "type", "license", "is_oa",
            "is_retracted", "pmc_oa", "smr"]
    for f in parts(out):
        for b in pq.ParquetFile(f).iter_batches(batch_size=50_000, columns=cols):
            d = b.to_pydict()
            for i in range(b.num_rows):
                kw = " ; ".join(x for x in (d["keywords"][i], d["mesh"][i]) if x)
                writer.add_document(tantivy.Document(
                    id=d["id"][i], title=d["title"][i] or "", abstract=d["abstract"][i] or "", keywords=kw,
                    year=d["year"][i] if d["year"][i] is not None else -9999,
                    type=d["type"][i] or "", license=d["license"][i] or "",
                    oa=int(d["is_oa"][i] or 0), retracted=int(d["is_retracted"][i] or 0),
                    pmc_oa=int(d["pmc_oa"][i] or 0), smr=int(d["smr"][i] or 0)))
            n += b.num_rows
            if n % 5_000_000 < 50_000:
                print(f"  {n:,} documents ({n/(time.time()-t0):,.0f}/s)", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    got = index.searcher().num_docs
    if got != n:
        raise SystemExit(f"keyword index holds {got:,}, added {n:,}")
    (tmp / "DONE.json").write_text(json.dumps({"documents": n, "seconds": round(time.time() - t0)}))
    os.replace(tmp, final)
    size = sum(p.stat().st_size for p in final.iterdir())
    print(f"keywords: {n:,} documents, {size/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["parts", "sqlite", "keywords"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="30GB")
    ap.add_argument("--heap-gb", type=int, default=8)
    a = ap.parse_args()
    if a.step == "parts":
        cmd_parts(a.out, a.processes)
    elif a.step == "sqlite":
        cmd_sqlite(a.out, a.threads, a.memory)
    else:
        cmd_keywords(a.out, a.threads, a.heap_gb)


if __name__ == "__main__":
    main()

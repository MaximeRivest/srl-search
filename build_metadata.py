"""Paper details for an index's papers, in one SQLite file keyed by OpenAlex W-number.

  uv run python build_metadata.py --min-date 2024-01-01 --out /mnt/fast/scholarsreadinglist/search/meta-2024plus.sqlite

Rows come from the content table snapshot the embeddings were made from, limited to the
works the index covers. PMCID comes from NCBI's PMC-ids map (by DOI); `pmc_oa` says whether
the article is in the PMC Open Access files we mirror (so ScienceMadeReadable can open it).

DuckDB does the selection, joins and the sort by id in parallel; rows then go into SQLite in
id order, so the table is written append-only (random-order inserts of 64M rows were ~4 h).
SQLite is the serving format: millisecond lookups by id, read-only, no server.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from srl_search.vectors import select_shards, SHARDS

CONTENT = Path("/mnt/bulk/openalex/embedding-inputs/denseon-complete")
PMC_IDS = Path("/mnt/bulk/pmc/PMC-ids-2026-10-06.csv.gz")
PMC_OA = Path("/mnt/bulk/pmc/inventory-2026-10-05/latest.txt")
MAX_AUTHORS = 12

SCHEMA = """
CREATE TABLE works (
  id INTEGER PRIMARY KEY,          -- OpenAlex W-number
  doi TEXT, title TEXT, date TEXT, year INTEGER, type TEXT,
  authors TEXT, authors_count INTEGER, venue TEXT, publisher TEXT,
  is_oa INTEGER, oa_status TEXT, license TEXT, citations INTEGER, is_retracted INTEGER,
  url TEXT, pdf_url TEXT, topic TEXT, field TEXT, keywords TEXT,
  pmcid TEXT, pmc_oa INTEGER,
  abstract TEXT
);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="30GB")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    t0 = time.time()

    shards = select_shards(a.min_date)
    wanted = np.unique(np.concatenate([np.load(SHARDS / s["dir"] / "work_ids.npy") for s in shards])).astype(np.int64)
    folders = sorted({s["dir"].split("/")[0] for s in shards})
    print(f"{len(wanted):,} works from {len(folders)} source folders ({time.time()-t0:.0f}s)", flush=True)

    tmpdir = a.out.parent / (a.out.name + ".duckdb-tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='{a.memory}'; SET temp_directory='{tmpdir}';"
                "SET preserve_insertion_order=false;")
    con.register("wanted", pa.table({"id": wanted}))
    oa = pa.table({"pmcid": [l.split(".")[0] for l in PMC_OA.read_text().split()]})
    con.register("pmc_oa_ids", oa)
    files = [str(CONTENT / f / "data.parquet") for f in folders]
    sorted_parquet = tmpdir / "works.parquet"
    con.execute(f"""
      COPY (
        WITH c AS (
          SELECT CAST(replace(id, 'https://openalex.org/W', '') AS BIGINT) AS id, * EXCLUDE (id)
          FROM read_parquet({files!r})
        ),
        pmc AS (
          SELECT lower(DOI) AS doi_l, any_value(PMCID) AS pmcid
          FROM read_csv('{PMC_IDS}', header=true, all_varchar=true, delim=',', quote='"', escape='"')
          WHERE DOI IS NOT NULL AND PMCID IS NOT NULL GROUP BY 1
        ),
        sel AS (SELECT c.* FROM c SEMI JOIN wanted w ON c.id = w.id)
        SELECT s.id,
               regexp_replace(s.doi, '^https://doi.org/', '') AS doi,
               s.title, CAST(s.date AS VARCHAR) AS date, s.year, s.type,
               array_to_string(list_slice(string_split(s.authors, ' | '), 1, {MAX_AUTHORS}), ' | ') AS authors,
               CASE WHEN s.authors IS NULL OR s.authors = '' THEN 0 ELSE len(string_split(s.authors, ' | ')) END AS authors_count,
               s.venue, s.publisher, CAST(s.is_open_access AS INTEGER) AS is_oa, s.open_access_status AS oa_status,
               s.license, s.citations, CAST(s.is_retracted AS INTEGER) AS is_retracted,
               s.url, s.pdf_url, s.topic, s.field, s.keywords,
               p.pmcid, CASE WHEN p.pmcid IS NULL THEN NULL WHEN o.pmcid IS NULL THEN 0 ELSE 1 END AS pmc_oa,
               s.abstract
        FROM sel s
        LEFT JOIN pmc p ON p.doi_l = lower(regexp_replace(s.doi, '^https://doi.org/', ''))
        LEFT JOIN pmc_oa_ids o ON o.pmcid = p.pmcid
        ORDER BY s.id
      ) TO '{sorted_parquet}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 200000)
    """)
    n = con.execute(f"SELECT count(*) FROM '{sorted_parquet}'").fetchone()[0]
    dup = con.execute(f"SELECT count(*) - count(DISTINCT id) FROM '{sorted_parquet}'").fetchone()[0]
    if dup:  # the content table has one row per current work; anything else is a data error
        raise SystemExit(f"{dup:,} duplicate work ids in the content rows")
    print(f"selected, joined and sorted {n:,} rows ({time.time()-t0:.0f}s)", flush=True)

    tmp = a.out.with_suffix(".sqlite.tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript("PRAGMA page_size=8192;" + SCHEMA)
    db.close()
    # DuckDB writes the rows in id order through its SQLite extension (C++; ~5x faster than
    # Python executemany), so the table is built append-only.
    con.execute(f"ATTACH '{tmp}' AS s (TYPE sqlite)")
    con.execute(f"INSERT INTO s.works SELECT * FROM '{sorted_parquet}' ORDER BY id")
    con.execute("DETACH s")
    print(f"  sqlite rows written ({time.time()-t0:.0f}s)", flush=True)
    db = sqlite3.connect(tmp)
    done = db.execute("SELECT count(*) FROM works").fetchone()[0]
    if done != n:
        raise SystemExit(f"sqlite holds {done:,} rows, expected {n:,}")
    db.commit()
    db.execute("CREATE INDEX works_doi ON works(doi COLLATE NOCASE)")
    db.execute("CREATE INDEX works_pmcid ON works(pmcid)")
    db.commit()
    db.close()
    os.replace(tmp, a.out)
    sorted_parquet.unlink()
    tmpdir.rmdir() if not any(tmpdir.iterdir()) else None
    print(f"done: {done:,} works ({len(wanted) - done:,} index works without a content row); "
          f"{a.out.stat().st_size/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

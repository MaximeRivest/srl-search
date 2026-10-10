"""Per-paper record-quality signals from the raw OpenAlex works snapshot, for ranking and filtering.

    uv run python build_quality.py --out /mnt/fast/scholarsreadinglist/search/quality-2026-10-05

The details store (`works.sqlite`) is built from the flattened content table, which leaves out the
fields that say how trustworthy a *record* is. They are read here from the raw works files instead,
newest version of each work, for the works of the shared corpus only (`corpus/parts`):

  is_xpac          OpenAlex's "expansion pack": bulk-added repository/DataCite records that OpenAlex itself
                   hides from its API unless asked (include_xpac=true)
  is_paratext      covers, tables of contents, front matter
  source_type      type of the primary location's source (journal, repository, ebook platform, ...)
  source_is_core   the source is on OpenAlex's curated list of established venues (CWTS "core")
  source_in_doaj   the source is in the Directory of Open Access Journals
  indexed_in       crossref / pubmed / doaj / arxiv / datacite, as bits
  references, authors, citations, locations, fwci, citation_pct (field- and year-normalised percentile)
  source           the primary location's source (S-number), to learn per-venue priors

Output, one row per corpus work, sorted by W-number:
  features.parquet   the columns above
  ids.npy + one .npy per column, memory-mapped by the search service (srl_search/quality.py)
  manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

RAW = Path("/mnt/bulk/openalex/works")
CORPUS = Path("/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05")

SOURCE_TYPES = ["journal", "repository", "ebook platform", "book series", "conference", "other", "metadata"]
INDEXES = ["crossref", "pubmed", "doaj", "arxiv", "datacite"]


def build(out: Path, threads: int, memory: str, passes: int, raw_glob: str) -> None:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "tmp"
    tmp.mkdir()
    con = duckdb.connect()
    con.execute(f"SET threads={threads}; SET memory_limit='{memory}'; SET temp_directory='{tmp}';"
                "SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    files = sorted(glob.glob(raw_glob))
    stype = "CASE primary_location.source.type " + " ".join(
        f"WHEN '{t}' THEN {i + 1}" for i, t in enumerate(SOURCE_TYPES)) + " ELSE 0 END"
    idx = " + ".join(f"CASE WHEN list_contains(indexed_in, '{n}') THEN {1 << i} ELSE 0 END"
                     for i, n in enumerate(INDEXES))
    # 1. one read of the raw files, only the needed leaves, every version of every corpus work
    con.execute(f"CREATE VIEW corpus AS SELECT id FROM read_parquet('{CORPUS}/parts/*.parquet', union_by_name=true) WHERE id IS NOT NULL")
    staged = tmp / "staged.parquet"
    con.execute(f"""
      COPY (
        SELECT CAST(replace(id, 'https://openalex.org/W', '') AS BIGINT) AS id,
               updated_date,
               coalesce(is_xpac, false) AS is_xpac,
               coalesce(is_paratext, false) AS is_paratext,
               coalesce(is_retracted, false) AS is_retracted,
               CAST({stype} AS UTINYINT) AS source_type,
               coalesce(primary_location.source.is_core, false) AS source_is_core,
               coalesce(primary_location.source.is_in_doaj, false) AS source_in_doaj,
               CAST(replace(primary_location.source.id, 'https://openalex.org/S', '') AS BIGINT) AS source,
               CAST({idx} AS UTINYINT) AS indexed_in,
               CAST(least(coalesce(referenced_works_count, 0), 32767) AS SMALLINT) AS refs,
               CAST(least(coalesce(authors_count, 0), 32767) AS SMALLINT) AS authors,
               CAST(coalesce(cited_by_count, 0) AS INTEGER) AS citations,
               CAST(least(coalesce(locations_count, 0), 32767) AS SMALLINT) AS locations,
               CAST(CASE WHEN isfinite(fwci) THEN fwci END AS FLOAT) AS fwci,
               CAST(citation_normalized_percentile.value AS FLOAT) AS citation_pct
        FROM read_parquet({files!r})
        WHERE CAST(replace(id, 'https://openalex.org/W', '') AS BIGINT) IN (SELECT id FROM corpus)
      ) TO '{staged}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)
    """)
    n_staged = con.execute(f"SELECT count(*) FROM '{staged}'").fetchone()[0]
    print(f"staged {n_staged:,} versions from {len(files):,} files ({time.time()-t0:.0f}s)", flush=True)

    # 2. newest version per work, in id-range passes so the aggregation fits in memory
    lo, hi = con.execute(f"SELECT min(id), max(id) FROM '{staged}'").fetchone()
    edges = np.linspace(lo, hi + 1, passes + 1).astype(np.int64)
    parts = []
    for i in range(passes):
        p = tmp / f"latest-{i:02d}.parquet"
        con.execute(f"""
          COPY (
            SELECT * EXCLUDE (rn) FROM (
              SELECT *, row_number() OVER (PARTITION BY id ORDER BY updated_date DESC) AS rn
              FROM '{staged}' WHERE id >= {edges[i]} AND id < {edges[i + 1]})
            WHERE rn = 1 ORDER BY id
          ) TO '{p}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 500000)
        """)
        parts.append(str(p))
        print(f"  pass {i + 1}/{passes} ({time.time()-t0:.0f}s)", flush=True)
    feats = out / "features.parquet"
    con.execute(f"COPY (SELECT * EXCLUDE (updated_date) FROM read_parquet({parts!r}) ORDER BY id) "
                f"TO '{feats}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)")
    n = con.execute(f"SELECT count(*) FROM '{feats}'").fetchone()[0]
    n_corpus = con.execute("SELECT count(*) FROM corpus").fetchone()[0]
    con.close()

    # 3. memory-mappable columns
    t = pq.read_table(feats)
    ids = t.column("id").to_numpy()
    if not np.all(ids[1:] > ids[:-1]):
        raise SystemExit("ids are not strictly increasing")
    np.save(out / "ids.npy", ids)
    for name in t.column_names:
        if name == "id":
            continue
        col = t.column(name)
        if name == "source":
            arr = col.fill_null(0).to_numpy()
        elif name in ("fwci", "citation_pct"):
            arr = col.to_numpy(zero_copy_only=False).astype(np.float32)  # nulls -> nan
        else:
            arr = col.to_numpy(zero_copy_only=False)
        np.save(out / f"{name}.npy", arr)
    manifest = {"works": int(n), "corpus_works": int(n_corpus), "versions_read": int(n_staged),
                "raw_files": len(files), "source_types": SOURCE_TYPES, "indexed_in_bits": INDEXES,
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.time() - t0)}
    manifest["sources"] = build_sources(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(json.dumps(manifest), flush=True)
    if n != n_corpus:
        print(f"WARNING: {n_corpus - n:,} corpus works have no raw record", flush=True)


def build_sources(out: Path) -> int:
    """Per-source aggregates over every corpus work: a venue's habits, learned without any labels.
    sources.npz: source (sorted S-numbers), works, and the shares of its works with no references,
    with one author or none, never cited, and in the expansion pack."""
    con = duckdb.connect()
    t = con.execute(f"""
      SELECT source, count(*) AS works,
             avg((refs = 0)::INT) AS no_refs, avg((authors <= 1)::INT) AS single_author,
             avg((citations = 0)::INT) AS uncited, avg(is_xpac::INT) AS xpac
      FROM '{out / "features.parquet"}' WHERE source IS NOT NULL GROUP BY source ORDER BY source
    """).fetchnumpy()
    np.savez(out / "sources.npz", source=t["source"].astype(np.int64), works=t["works"].astype(np.int64),
             **{k: t[k].astype(np.float32) for k in ("no_refs", "single_author", "uncited", "xpac")})
    return int(len(t["source"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--memory", default="40GB")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--sources-only", action="store_true", help="(re)compute sources.npz of an existing build")
    ap.add_argument("--raw-glob", default=str(RAW / "updated_date=*" / "*.parquet"), help="a subset, for tests")
    a = ap.parse_args()
    if a.sources_only:
        print(build_sources(a.out), "sources")
    else:
        build(a.out, a.threads, a.memory, a.passes, a.raw_glob)

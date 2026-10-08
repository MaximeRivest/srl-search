"""The ScienceMadeReadable collection: papers the site can open, searched exactly.

  uv run python build_smr.py --out /mnt/fast/scholarsreadinglist/search/smr-YYYY-MM-DD

A paper is in it when:
  - its full text is in our PubMed Central mirror, with NCBI licence code "CC BY"
    (the site rewrites and shows CC BY only) and JATS article-type "research-article";
  - its DOI matches an OpenAlex work whose English title + abstract is already embedded.
One OpenAlex work per DOI (the smallest W-number when OpenAlex has duplicates).

Writes into --out:
  index.faiss     IVF (4,096 groups) over the original fp16 vectors, ids = OpenAlex W-numbers.
                  Scores are exact; searching 512 groups finds 99% of the exact top 15 for plain
                  questions in ~45 ms (comparing all 3M papers took 360 ms; experiments/smr_hnsw.log)
  meta.sqlite     paper details (same table as the big index's)
  manifest.json   counts, sources, the date range covered
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import build_metadata
from srl_search.vectors import SHARDS, select_shards

ARTICLES = Path("/mnt/bulk/pmc/oa-2026-10-05/derived/articles.parquet")
CONTENT = build_metadata.CONTENT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out: Path = a.out
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} is already built")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    art = pq.read_table(ARTICLES)
    keep = pc.and_(pc.equal(art["license_code"], "CC BY"), pc.equal(art["article_type"], "research-article"))
    keep = pc.and_(keep, pc.and_(pc.is_null(art["error"]), pc.greater(art["xml_bytes"], 0)))
    keep = pc.and_(keep, pc.invert(pc.is_null(art["doi"])))
    smr = art.filter(keep).select(["pmcid", "doi"])
    print(f"PMC: {art.num_rows:,} articles, {smr.num_rows:,} CC BY research articles with a DOI", flush=True)

    shards = select_shards("0000-01-01")
    folders = sorted({s["dir"].split("/")[0] for s in shards})
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}; SET memory_limit='20GB'")
    con.register("smr", smr)
    files = [str(CONTENT / f / "data.parquet") for f in folders]
    hits = con.execute(f"""
      SELECT min(CAST(replace(id, 'https://openalex.org/W', '') AS BIGINT)) AS id, s.pmcid
      FROM read_parquet({files!r}) c
      JOIN smr s ON s.doi = lower(regexp_replace(c.doi, '^https://doi.org/', ''))
      GROUP BY s.pmcid
    """).fetch_arrow_table()
    target = np.unique(np.asarray(hits["id"].to_numpy(), dtype=np.uint64))  # sorted, for searchsorted
    print(f"OpenAlex works for those DOIs: {len(target):,} ({time.time()-t0:.0f}s)", flush=True)

    ids_out, vecs_out, dates = [], [], []
    for n, s in enumerate(shards):
        w = np.load(SHARDS / s["dir"] / "work_ids.npy")
        i = np.minimum(np.searchsorted(target, w), len(target) - 1)  # np.isin re-sorts target every call
        m = target[i] == w
        if m.any():
            v = np.load(SHARDS / s["dir"] / "vectors.npy", mmap_mode="r")
            rows = np.nonzero(m)[0]
            ids_out.append(w[rows])
            vecs_out.append(np.asarray(v[rows]))
            dates.append(s["date"])
        if n % 2000 == 0:
            print(f"  shards {n:,}/{len(shards):,}", flush=True)
    ids = np.concatenate(ids_out).astype(np.int64)
    vecs = np.concatenate(vecs_out).astype(np.float32)
    ids, first = np.unique(ids, return_index=True)  # a work embedded twice counts once
    vecs = vecs[first]
    print(f"embedded so far: {len(ids):,} ({time.time()-t0:.0f}s)", flush=True)

    faiss.omp_set_num_threads(16)
    index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(768), 768, 4096, faiss.ScalarQuantizer.QT_fp16,
                                          faiss.METRIC_INNER_PRODUCT)
    index.cp.niter = 15
    sample = np.random.default_rng(0).choice(len(vecs), min(len(vecs), 80 * 4096), replace=False)
    index.train(vecs[sample])
    index.add_with_ids(vecs, ids)
    faiss.write_index(index, str(out / "index.faiss.tmp"))
    os.replace(out / "index.faiss.tmp", out / "index.faiss")

    meta = out / "meta.sqlite"
    meta.unlink(missing_ok=True)
    rows = build_metadata.build(ids, folders, meta, threads=a.threads, memory="20GB")

    embedding = json.loads((SHARDS.parent / "status.json").read_text()) if (SHARDS.parent / "status.json").exists() else {}
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "papers": int(len(ids)), "meta_rows": int(rows),
        "rule": "PMC mirror licence 'CC BY' + article-type 'research-article' + embedded OpenAlex title/abstract",
        "pmc_candidates": int(smr.num_rows), "openalex_matches": int(len(target)),
        "oldest_shard_date": min(dates), "embedding_in_progress_at": embedding.get("current_publication_date"),
        "index": {"type": "IVF4096 over fp16 vectors (exact scores), searched with nprobe=512",
                  "file_bytes": (out / "index.faiss").stat().st_size},
        "seconds": round(time.time() - t0),
    }
    (out / "manifest.json.tmp").write_text(json.dumps(manifest, indent=1))
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

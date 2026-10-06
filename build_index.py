"""Build a search index over the embedded papers published on or after a date.

  uv run python build_index.py --min-date 2024-01-01 --out /mnt/fast/scholarsreadinglist/search/index-2024plus

Writes, into --out (a new directory; refuses to overwrite a finished one):
  manifest.json     shards covered (in position order), rows, settings, timings
  ivf.faiss         IVF (k-means lists) + RaBitQ codes (~106 bytes/paper); ids = positions
  work_ids.npy      uint64 OpenAlex W-number per position
  by_work.npy       positions sorted by work id (for lookups by OpenAlex id)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np

from srl_search.vectors import Vectors, select_shards

EMBEDDING_PLAN = "/mnt/fast/openalex-embeddings/denseon-english-abstracts/manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lists", type=int, default=16384)
    ap.add_argument("--kmeans-iterations", type=int, default=10)
    ap.add_argument("--train-rows", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    out: Path = a.out
    if (out / "manifest.json").exists():
        raise SystemExit(f"{out} already holds a finished index")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shards = select_shards(a.min_date)
    vec = Vectors(shards)
    print(f"{len(shards)} shards, {vec.total:,} papers", flush=True)

    # training sample: the same number of random rows from every shard
    rng = np.random.default_rng(0)
    per = max(1, a.train_rows // len(shards))
    sample = []
    for i, s in enumerate(shards):
        rows = np.sort(rng.choice(s["rows"], min(per, s["rows"]), replace=False))
        sample.append(np.asarray(vec.shard_vectors(i)[rows], dtype=np.float32))
    train = np.concatenate(sample)
    t_sample = time.time() - t0
    print(f"training sample {len(train):,} in {t_sample:.0f}s", flush=True)

    # HNSW over the list centres: assigning a paper (or a question) to its lists takes a graph
    # walk instead of comparing with all centres (exact assignment of 64M papers was ~1.6e15 flops).
    factory = f"IVF{a.lists}_HNSW32,RaBitQ"
    trained = out / "trained.faiss"
    t = time.time()
    if trained.exists():  # resume: training is the expensive step
        index = faiss.read_index(str(trained))
        t_train = 0.0
        print("loaded trained index", flush=True)
    else:
        index = faiss.index_factory(768, factory, faiss.METRIC_INNER_PRODUCT)
        faiss.extract_index_ivf(index).cp.niter = a.kmeans_iterations
        index.train(train)
        t_train = time.time() - t
        faiss.write_index(index, str(trained) + ".tmp")
        os.replace(str(trained) + ".tmp", trained)
        print(f"trained in {t_train:.0f}s", flush=True)
    del train, sample

    work_ids = np.empty(vec.total, dtype=np.uint64)
    t = time.time()
    batch_x, batch_p = [], []
    pending = 0

    def flush() -> None:
        nonlocal batch_x, batch_p, pending
        if pending:
            index.add_with_ids(np.concatenate(batch_x), np.concatenate(batch_p))
        batch_x, batch_p, pending = [], [], 0

    for first, x, ids in vec.iter_shards():
        n = len(x)
        if n != len(ids):
            raise SystemExit(f"shard at {first}: {n} vectors but {len(ids)} ids")
        work_ids[first:first + n] = ids
        batch_x.append(x.astype(np.float32))
        batch_p.append(np.arange(first, first + n, dtype=np.int64))
        pending += n
        if pending >= 500_000:
            flush()
            done = first + n
            el = time.time() - t
            print(f"  added {done:,} / {vec.total:,} ({done/el:,.0f}/s)", flush=True)
    flush()
    t_add = time.time() - t
    if index.ntotal != vec.total:
        raise SystemExit(f"index holds {index.ntotal}, expected {vec.total}")

    tmp = out / "ivf.faiss.tmp"
    faiss.write_index(index, str(tmp))
    tmp.rename(out / "ivf.faiss")
    trained.unlink()
    np.save(out / "work_ids.npy", work_ids)
    np.save(out / "by_work.npy", np.argsort(work_ids, kind="stable").astype(np.int64))
    sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)])
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_date": a.min_date,
        "papers": int(vec.total),
        "shards": shards,
        "embedding_plan_id": json.loads(Path(EMBEDDING_PLAN).read_text())["plan_id"],
        "index": {"factory": factory, "kmeans_iterations": a.kmeans_iterations, "metric": "inner_product",
                  "train_rows": int(per * len(shards)), "list_size_max": int(sizes.max()),
                  "list_size_mean": float(sizes.mean()), "empty_lists": int((sizes == 0).sum()),
                  "file_bytes": (out / "ivf.faiss").stat().st_size},
        "seconds": {"sample": round(t_sample), "train": round(t_train), "add": round(t_add),
                    "total": round(time.time() - t0)},
        "threads": a.threads,
    }
    (out / "manifest.json.tmp").write_text(json.dumps(manifest, indent=1))
    os.replace(out / "manifest.json.tmp", out / "manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=1))


if __name__ == "__main__":
    main()

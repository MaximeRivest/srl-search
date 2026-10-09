"""How much does the fast index miss, and how fast is it? Compared with exact search over every paper.

  uv run python eval_index.py --index DIR --meta FILE [--out results.json]

Queries: 400 BEIR questions (SciFact, NFCorpus, SciDocs, TREC-COVID; 100 each, embedded by the
benchmark with the same model and prompt) + the plain-language questions below, encoded here.
Recall@10 = share of the exact top 10 that the index returns in its top 10.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from srl_search.engine import Engine

BEIR = Path("/mnt/fast/scholarsreadinglist/retrieval-bench/emb/denseon")

QUESTIONS = [
    "effects of microplastics on freshwater fish reproduction",
    "large language models for automated theorem proving",
    "how does sleep deprivation affect working memory in adolescents",
    "perovskite solar cell stability under humidity",
    "minimum wage increases and employment in small businesses",
    "CRISPR base editing for sickle cell disease",
    "dark matter detection with liquid xenon",
    "remote work and urban housing prices after the pandemic",
    "graph neural networks for molecular property prediction",
    "glacier retreat and downstream water availability in the Himalaya",
    "social media use and depression in teenagers",
    "battery recycling of lithium iron phosphate cathodes",
    "medieval trade routes in the Baltic Sea",
    "deep learning detection of diabetic retinopathy from fundus images",
    "soil carbon sequestration with cover crops",
    "quantum error correction with surface codes",
    "monetary policy transmission in emerging markets",
    "bilingual children vocabulary development",
    "antibiotic resistance genes in hospital wastewater",
    "reinforcement learning for robot grasping",
    "coral reef bleaching and marine heatwaves",
    "gut microbiome and response to cancer immunotherapy",
    "earthquake early warning systems using smartphones",
    "ethics of facial recognition in policing",
    "hydrogen embrittlement of high strength steels",
    "teacher feedback and student motivation in secondary school",
    "wildfire smoke exposure and respiratory hospital admissions",
    "transformer models for time series forecasting",
    "pollinator decline and agricultural yields",
    "long COVID fatigue and exercise intolerance",
    "carbon pricing and industrial competitiveness in Europe",
    "electric vehicle charging infrastructure planning",
    "ancient DNA and human migration into Europe",
    "superconductivity in twisted bilayer graphene",
    "misinformation spread on social networks during elections",
    "maternal obesity and offspring metabolic outcomes",
    "urban heat islands and tree canopy cover",
    "federated learning privacy attacks",
    "drug repurposing for Alzheimer's disease",
    "gender wage gap among physicians",
]


def exact_top(engine: Engine, Q: np.ndarray, k: int = 10, threads: int = 24) -> np.ndarray:
    faiss.omp_set_num_threads(threads)
    best_s = np.full((len(Q), k), -np.inf, dtype=np.float32)
    best_p = np.full((len(Q), k), -1, dtype=np.int64)
    t = time.time()
    keep_file = Path(engine.manifest.get("index_dir", "")) / "keep.npy"
    keep = np.load(keep_file, mmap_mode="r") if keep_file.exists() else None   # rows the index holds
    for n, (first, x, _) in enumerate(engine.vectors.iter_shards()):
        s = Q @ x.astype(np.float32).T
        if keep is not None:
            s[:, ~np.asarray(keep[first:first + len(x)])] = -np.inf
        if s.shape[1] <= k:  # small shard: every row is a candidate
            idx = np.tile(np.arange(s.shape[1]), (len(Q), 1))
        else:
            idx = np.argpartition(-s, k, axis=1)[:, :k]
        cs = np.take_along_axis(s, idx, axis=1)
        allp = np.concatenate([best_p, idx + first], axis=1)
        alls = np.concatenate([best_s, cs], axis=1)
        o = np.argsort(-alls, axis=1)[:, :k]
        best_p = np.take_along_axis(allp, o, axis=1)
        best_s = np.take_along_axis(alls, o, axis=1)
        if n % 1000 == 0:
            print(f"  exact: shard {n}/{len(engine.vectors.dirs)} ({time.time()-t:.0f}s)", flush=True)
    return best_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    engine = Engine(a.index, a.meta, threads=4, encoder=True)

    sets = {}
    for d in ["scifact", "nfcorpus", "scidocs", "trec-covid"]:
        q = np.load(BEIR / d / "queries.npy")[:100].astype(np.float32)
        sets[d] = q / np.linalg.norm(q, axis=1, keepdims=True)
    enc = [engine.encoder.encode(t)[0] for t in QUESTIONS]
    t = time.perf_counter()
    for txt in QUESTIONS[:20]:
        engine.encoder.encode(txt)
    encode_ms = (time.perf_counter() - t) / 20 * 1000
    sets["plain questions"] = np.stack(enc)
    names = list(sets)
    Q = np.concatenate([sets[n] for n in names])
    owner = np.concatenate([[n] * len(sets[n]) for n in names])

    import hashlib
    key = hashlib.sha256(Q.tobytes()).hexdigest()[:16]
    cache = a.index / f"eval-exact-top10-{key}.npy"
    t = time.time()
    if cache.exists():
        gt = np.load(cache)
        print("exact answers loaded from", cache, flush=True)
    else:
        print("exact search over every paper...", flush=True)
        gt = exact_top(engine, Q)
        np.save(cache, gt)
    exact_s = time.time() - t
    faiss.omp_set_num_threads(4)

    rows = []
    for nprobe in (16, 32, 64, 128, 256):
        for depth in (100, 300, 1000):
            rec, lat = [], []
            for i in range(len(Q)):
                t = time.perf_counter()
                hits, _ = engine.search_vector(Q[i], 10, nprobe, depth)
                lat.append(time.perf_counter() - t)
                got = {w for w, _ in hits[:10]}
                want = set(np.asarray(engine.work_ids[gt[i]], dtype=np.int64).tolist())
                rec.append(len(got & want) / 10)
            rec, lat = np.array(rec), np.array(lat) * 1000
            row = {"nprobe": nprobe, "depth": depth, "recall@10": round(float(rec.mean()), 3),
                   "by_set": {n: round(float(rec[owner == n].mean()), 3) for n in names},
                   "p50_ms": round(float(np.median(lat)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = {"papers": engine.manifest["papers"], "queries": {n: int((owner == n).sum()) for n in names},
           "exact_search_seconds_all_queries": round(exact_s), "query_encoding_ms_cpu4": round(encode_ms, 1),
           "threads": 4, "results": rows}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

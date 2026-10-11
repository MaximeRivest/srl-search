"""The search engine: question -> vector -> candidate papers (IVF + RaBitQ) -> exact rerank
from the original float16 vectors -> paper details from SQLite.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import faiss
import numpy as np

from srl_search.quality import Quality
from srl_search.scope import SCHOLARLY_TYPES
from srl_search.vectors import Vectors

EMBEDDING_PLAN = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/manifest.json")

# Ranking: similarity - JUNK_PENALTY * P(junk) (srl_search/quality.py). 0.3 was chosen on the training questions
# and then measured once on the 60 evaluation questions (experiments/quality/README.md, 2026-10-09).
JUNK_PENALTY = 0.3
# Some retracted papers, and retraction notices, are only marked in their title (OpenAlex's is_retracted is
# not set): "[RETRACTED] ...", "RETRACTED: ...", "Retraction Note: ...". A bracket or colon is required, so a
# study *about* retractions ("Retracted papers in medicine") is kept.
RETRACTED_TITLE = re.compile(r"^\W*\[?\s*(retracted(\s+article)?|retraction(\s+(note|notice|statement))?)\s*[\]:]", re.I)

FIELDS = ["id", "doi", "title", "date", "year", "type", "authors", "authors_count", "venue", "publisher",
          "is_oa", "oa_status", "license", "citations", "is_retracted", "url", "pdf_url", "topic",
          "field", "keywords", "pmcid", "pmc_oa", "abstract"]


class Encoder:
    """DenseOn query encoder on CPU, exactly as the documents were embedded (pinned snapshot)."""

    def __init__(self, threads: int = 4):
        from concurrent.futures import ThreadPoolExecutor
        import torch
        from sentence_transformers import SentenceTransformer
        cfg = json.loads(EMBEDDING_PLAN.read_text())["embedding"]
        cfg = eval(cfg) if isinstance(cfg, str) else cfg  # manifest stores a Python-repr dict
        self.model = SentenceTransformer(cfg["model_snapshot"], device="cpu")
        if self.model.prompts.get("query") != cfg["query_prompt"]:
            raise ValueError("query prompt differs from the embedding manifest")
        self.model.max_seq_length = cfg["max_seq_length"]
        # One dedicated thread runs the model. OpenMP's thread count is per calling thread, so
        # calling from the web server's many worker threads gave each one 48 threads (2x slower).
        self.pool = ThreadPoolExecutor(1, initializer=torch.set_num_threads, initargs=(threads,))
        self._cache: dict[str, np.ndarray] = {}  # same question again (e.g. placing it on the map): no re-encoding

    def _encode(self, text: str) -> np.ndarray:
        v = self.model.encode([text], prompt_name="query", normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return v.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        hit = self._cache.get(text)
        if hit is not None:
            return hit.copy()
        v = self.pool.submit(self._encode, text).result()
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[text] = v
        return v.copy()


class Engine:
    def __init__(self, index_dir: Path, meta_path: Path, threads: int = 4, encoder: bool = True,
                 quality: Quality | None = None, more_meta: list[Path] | None = None):
        self.threads = threads
        self.quality = quality if quality is not None and quality.model is not None else None
        faiss.omp_set_num_threads(threads)
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        self.index = faiss.read_index(str(index_dir / "ivf.faiss"))
        # Exact routing: compare the question with every list centre (16k x 768, ~1-3 ms).
        # The HNSW router stored in the index (used to build it fast) missed the single best
        # list for 17-20% of questions at its default efSearch=16 (measured 2026-10-06).
        ivf = faiss.extract_index_ivf(self.index)
        centres = faiss.downcast_index(ivf.quantizer).storage.reconstruct_n(0, ivf.nlist)
        self._router = faiss.IndexFlatIP(centres.shape[1])
        self._router.add(centres)
        self._hnsw_router = ivf.quantizer  # keep the original alive; the index does not own the new one
        ivf.own_fields = False
        ivf.quantizer = self._router
        self.vectors = Vectors(self.manifest["shards"])
        self.vectors.open_all()  # first-touch opens cost ~1.5 s per cold query otherwise
        self.work_ids = np.load(index_dir / "work_ids.npy", mmap_mode="r")
        self.by_work = np.load(index_dir / "by_work.npy", mmap_mode="r")
        self.meta_path = meta_path
        self.more_meta = list(more_meta or [])   # the daily updates corpus (works the base does not hold)
        self.local = threading.local()
        self.encoder = Encoder(threads) if encoder else None

    # -------------------------------------------------------------- details
    def db(self) -> sqlite3.Connection:
        return self.dbs()[0]

    def dbs(self) -> list[sqlite3.Connection]:
        """The base details, then the daily updates' (one connection each per thread)."""
        cs = getattr(self.local, "dbs", None)
        if cs is None:
            cs = []
            for path in [self.meta_path, *self.more_meta]:
                c = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
                c.row_factory = sqlite3.Row
                cs.append(c)
            self.local.dbs = cs
        return cs

    def details(self, work_ids: list[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        left = list(work_ids)
        for c in self.dbs():
            if not left:
                break
            q = f"SELECT * FROM works WHERE id IN ({','.join('?' * len(left))})"
            out.update({r["id"]: dict(r) for r in c.execute(q, left)})
            left = [w for w in left if w not in out]
        return out

    def lookup(self, sql: str, arg) -> int | None:
        for c in self.dbs():
            row = c.execute(sql, (arg,)).fetchone()
            if row:
                return row[0]
        return None

    def position_of(self, work_id: int) -> int | None:
        lo, hi = 0, len(self.by_work)
        while lo < hi:  # binary search over the sorted permutation (mmap, no full load)
            mid = (lo + hi) // 2
            if self.work_ids[self.by_work[mid]] < work_id:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.by_work) and self.work_ids[self.by_work[lo]] == work_id:
            return int(self.by_work[lo])
        return None

    def resolve(self, ref: str) -> int | None:
        """OpenAlex id (W123, URL), DOI or PMCID -> W-number, if it is in this index."""
        r = ref.strip()
        low = r.lower()
        if low.startswith("https://openalex.org/"):
            r = r.rsplit("/", 1)[-1]
        if r[:1] in "Ww" and r[1:].isdigit():
            return int(r[1:])
        if low.startswith("pmc") and low[3:].isdigit():
            return self.lookup("SELECT id FROM works WHERE pmcid=?", r.upper())
        doi = low.removeprefix("https://doi.org/").removeprefix("doi:")
        if doi.startswith("10."):
            # DOIs are case-insensitive; the index is built with COLLATE NOCASE
            return self.lookup("SELECT id FROM works WHERE doi=? COLLATE NOCASE", doi)
        return None

    # -------------------------------------------------------------- search
    def search_vector(self, q: np.ndarray, k: int, nprobe: int, depth: int, keep=None, exclude: int | None = None,
                      junk_penalty: float = 0.0) -> tuple[list[tuple], dict]:
        """Candidates best first, as (W-number, ranking score, similarity, P(junk) or None)."""
        faiss.omp_set_num_threads(self.threads)  # OpenMP thread count is per calling thread
        t = time.perf_counter()
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        _, cand = self.index.search(q.reshape(1, -1), depth, params=params)
        cand = cand[0][cand[0] >= 0]
        t_ivf = time.perf_counter() - t
        t = time.perf_counter()
        vecs = self.vectors.get(cand)
        exact = vecs @ q.reshape(-1)
        wids = np.asarray(self.work_ids[cand], dtype=np.int64)
        t_rerank = time.perf_counter() - t
        hits, t_junk = rank(wids, exact, vecs, self.quality, junk_penalty, exclude)
        timing = {"candidates": int(len(cand)), "ivf_ms": round(t_ivf * 1000, 1),
                  "rerank_ms": round(t_rerank * 1000, 1)}
        if t_junk is not None:
            timing["junk_ms"] = round(t_junk * 1000, 1)
        return hits, timing

    def search(self, text: str, k: int = 10, nprobe: int = 64, depth: int = 300,
               filters: dict | None = None, abstracts: bool = True, junk_penalty: float = JUNK_PENALTY) -> dict:
        t0 = time.perf_counter()
        q = self.encoder.encode(text)[0]
        t_enc = time.perf_counter() - t0
        return self._finish(q, k, nprobe, depth, filters, abstracts, {"encode_ms": round(t_enc * 1000, 1)}, t0,
                            junk_penalty=junk_penalty)

    def similar(self, work_id: int, k: int = 10, nprobe: int = 64, depth: int = 300,
                filters: dict | None = None, abstracts: bool = True, junk_penalty: float = JUNK_PENALTY) -> dict | None:
        t0 = time.perf_counter()
        pos = self.position_of(work_id)
        if pos is None:
            return None
        q = self.vectors.get(np.array([pos]))[0]
        return self._finish(q, k, nprobe, depth, filters, abstracts, {}, t0, exclude=work_id,
                            junk_penalty=junk_penalty)

    def shape(self, hits: list[tuple], k: int, filters: dict, abstracts: bool) -> list[dict]:
        """Details for scored ids, best first: filters applied, identical records once, k at most.
        Hits are (W-number, score) or (W-number, ranking score, similarity, P(junk) or None); "score" stays
        what it always was (similarity, or the keyword score), "junk" is added when known."""
        results, seen = [], set()
        step = 200 if filters else max(k, 20)
        for start in range(0, len(hits), step):
            page = hits[start:start + step]
            det = self.details([h[0] for h in page])
            for h in page:
                w, s = h[0], (h[2] if len(h) > 2 else h[1])
                d = det.get(w)
                if d is None or not passes(d, filters):
                    continue
                key = ((d["title"] or "").strip().lower(), (d["abstract"] or "")[:300])
                if key in seen:
                    continue
                seen.add(key)
                if not abstracts:
                    d.pop("abstract", None)
                d["score"] = round(s, 4)
                if len(h) > 3 and h[3] is not None:
                    d["junk"] = round(h[3], 3)
                d["openalex"] = f"https://openalex.org/W{w}"
                d["id"] = f"W{w}"
                results.append(d)
                if len(results) == k:
                    return results
        return results

    def _finish(self, q, k, nprobe, depth, filters, abstracts, timing, t0, exclude=None,
                junk_penalty: float = JUNK_PENALTY) -> dict:
        filters = {f: v for f, v in (filters or {}).items() if v is not None}
        if filters:
            depth = max(depth, 2000)
        hits, t = self.search_vector(q, k, nprobe, depth, exclude=exclude, junk_penalty=junk_penalty)
        timing.update(t)
        t = time.perf_counter()
        results = self.shape(hits, k, filters, abstracts)
        timing["details_ms"] = round((time.perf_counter() - t) * 1000, 1)
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        on = self.quality is not None and junk_penalty > 0
        ranking = {"by": "similarity - junk_penalty * junk" if on else "similarity",
                   "junk_penalty": junk_penalty if on else 0}
        return {"results": results, "timing": timing, "filters": filters, "ranking": ranking}


class ExactEngine(Engine):
    """A small collection whose stored vectors are the original fp16 ones (scores are exact); ids are
    OpenAlex W-numbers. Flat (every paper compared) or IVF (the `nprobe` closest groups compared)."""

    def __init__(self, collection_dir: Path, threads: int = 4, encoder: "Encoder | None" = None, nprobe: int = 512,
                 meta_path: Path | None = None, quality: Quality | None = None):
        self.threads = threads
        self.quality = quality if quality is not None and quality.model is not None else None
        self.manifest = json.loads((collection_dir / "manifest.json").read_text())
        self.manifest.setdefault("min_date", self.manifest.get("oldest_shard_date"))
        self.index = faiss.read_index(str(collection_dir / "index.faiss"))
        ivf = faiss.try_extract_index_ivf(self.index)
        if ivf is not None:
            ivf.nprobe = nprobe
            ivf.set_direct_map_type(faiss.DirectMap.Hashtable)  # reconstruct(W-number) for "similar"
        self.meta_path = meta_path or collection_dir / "meta.sqlite"
        self.more_meta = []          # the smr collection holds base works only
        self.local = threading.local()
        self.encoder = encoder

    def search_vector(self, q, k, nprobe, depth, keep=None, exclude=None, junk_penalty: float = 0.0):
        faiss.omp_set_num_threads(self.threads)
        t = time.perf_counter()
        scores, ids = self.index.search(q.reshape(1, -1).astype(np.float32), depth)
        m = ids[0] >= 0
        wids, sims = ids[0][m].astype(np.int64), scores[0][m]
        timing = {"candidates": int(m.sum()), "exact_ms": round((time.perf_counter() - t) * 1000, 1)}
        vecs = None
        if self.quality is not None and len(wids):
            vecs = np.vstack([self.index.reconstruct(int(w)) for w in wids])
        hits, t_junk = rank(wids, sims, vecs, self.quality, junk_penalty, exclude)
        if t_junk is not None:
            timing["junk_ms"] = round(t_junk * 1000, 1)
        return hits, timing

    def similar(self, work_id, k=10, nprobe=0, depth=300, filters=None, abstracts=True, junk_penalty=JUNK_PENALTY):
        t0 = time.perf_counter()
        try:
            q = self.index.reconstruct(int(work_id))
        except RuntimeError:  # not in this collection
            return None
        return self._finish(q, k, nprobe, depth, filters, abstracts, {}, t0, exclude=work_id,
                            junk_penalty=junk_penalty)


def rank(wids: np.ndarray, sims: np.ndarray, vecs: np.ndarray | None, quality: Quality | None,
         junk_penalty: float, exclude: int | None) -> tuple[list[tuple], float | None]:
    """Order candidates by similarity - junk_penalty * P(junk). P(junk) is computed (and reported) whenever a
    model is loaded, even with the penalty at 0, so clients can see it; ties keep the similarity order."""
    t = time.perf_counter()
    junk = quality.junk(wids, vecs) if quality is not None and len(wids) else None
    t_junk = time.perf_counter() - t if junk is not None else None
    score = sims - junk_penalty * junk if junk is not None and junk_penalty else sims
    order = np.lexsort((-sims, -score))
    hits = [(int(wids[i]), float(score[i]), float(sims[i]), None if junk is None else float(junk[i]))
            for i in order if int(wids[i]) != exclude]
    return hits, t_junk


def passes(d: dict, f: dict) -> bool:
    if "year_from" in f and (d["year"] is None or d["year"] < f["year_from"]):
        return False
    if "year_to" in f and (d["year"] is None or d["year"] > f["year_to"]):
        return False
    if "type" in f and d["type"] != f["type"]:
        return False
    if "type" not in f and d["type"] not in SCHOLARLY_TYPES:  # datasets, covers, figures, logs...
        return False
    if f.get("open_access") and not d["is_oa"]:
        return False
    if "license" in f and (d["license"] or "") != f["license"]:
        return False
    if f.get("pmc_open_access") and not d["pmc_oa"]:
        return False
    if f.get("exclude_retracted", True) and (d["is_retracted"] or RETRACTED_TITLE.match(d.get("title") or "")):
        return False
    return True

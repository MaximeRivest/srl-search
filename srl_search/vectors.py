"""The embedding shards, read in place: which shards an index covers, and exact vector access.

An index's ids are *positions*: the row's place in the concatenation of the manifest's
shards, in manifest order. No vector is copied; reranking reads the float16 rows straight
from the shard files with many small reads issued in parallel (an NVMe SSD serves them
concurrently; one-by-one page faults through mmap cost ~1 s for 300 cold rows).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SHARDS = Path("/mnt/fast/openalex-embeddings/denseon-english-abstracts/shards")
DIM = 768
ROW_BYTES = DIM * 2  # float16


def select_shards(min_date: str, root: Path = SHARDS) -> list[dict]:
    """Committed shards whose publication date is >= min_date (ISO), oldest key first."""
    out = []
    for c in root.glob("year=*/rg-*/part-*/commit.json"):
        rec = json.loads(c.read_text())
        task = rec["task"]
        date = task.get("publication_date")
        if not date or date < min_date:
            continue
        out.append({"dir": str(c.parent.relative_to(root)), "rows": int(task["rows"]), "date": date})
    out.sort(key=lambda s: s["dir"])
    return out


def npy_data_offset(path: Path) -> int:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0(f))
        if dtype != np.float16 or fortran or len(shape) != 2 or shape[1] != DIM:
            raise ValueError(f"{path}: unexpected array {shape} {dtype}")
        return f.tell()


class Vectors:
    """Positions -> float16 vectors and OpenAlex work ids, across the manifest's shards."""

    def __init__(self, shards: list[dict], root: Path = SHARDS, readers: int = 32):
        self.root = root
        self.dirs = [s["dir"] for s in shards]
        rows = np.array([s["rows"] for s in shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(rows)])
        self.total = int(self.offsets[-1])
        self._fd: dict[int, tuple[int, int]] = {}
        self._ids: dict[int, np.ndarray] = {}
        self._pool = ThreadPoolExecutor(readers)

    def _file(self, i: int) -> tuple[int, int]:
        h = self._fd.get(i)
        if h is None:
            p = self.root / self.dirs[i] / "vectors.npy"
            h = (os.open(p, os.O_RDONLY), npy_data_offset(p))
            os.posix_fadvise(h[0], 0, 0, os.POSIX_FADV_RANDOM)
            self._fd[i] = h
        return h

    def open_all(self) -> None:
        """Open every shard up front (one file descriptor per shard; set LimitNOFILE accordingly)."""
        for i in range(len(self.dirs)):
            self._file(i)

    def locate(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.searchsorted(self.offsets, pos, side="right") - 1
        return s, pos - self.offsets[s]

    def get(self, pos: np.ndarray) -> np.ndarray:
        """float32 [len(pos), 768] in the order of pos."""
        pos = np.asarray(pos, dtype=np.int64)
        s, r = self.locate(pos)
        handles = [self._file(int(x)) for x in s]

        def read(k: int) -> bytes:
            fd, off = handles[k]
            return os.pread(fd, ROW_BYTES, off + int(r[k]) * ROW_BYTES)

        buf = b"".join(self._pool.map(read, range(len(pos))))
        if len(buf) != len(pos) * ROW_BYTES:
            raise IOError("short read from a vector shard")
        return np.frombuffer(buf, dtype=np.float16).reshape(len(pos), DIM).astype(np.float32)

    def get_many(self, pos: np.ndarray, dense: int = 8) -> np.ndarray:
        """float16 [len(pos), 768] in the order of pos, for many rows (hundreds of thousands to millions): one task
        per shard; a shard whose wanted rows are at least 1 in `dense` of their span is read in one piece, the
        others row by row. ~10x faster than get() for a million rows (get() pays a thread-pool task per row)."""
        pos = np.asarray(pos, dtype=np.int64)
        out = np.empty((len(pos), DIM), np.float16)
        if not len(pos):
            return out
        s, r = self.locate(pos)
        order = np.lexsort((r, s))
        bounds = np.flatnonzero(np.diff(s[order])) + 1
        flat = out.reshape(-1)

        def one(g: np.ndarray) -> None:
            fd, off = self._file(int(s[g[0]]))
            rr = r[g]
            lo, hi = int(rr[0]), int(rr[-1]) + 1
            if (hi - lo) <= dense * len(g):
                buf = os.pread(fd, (hi - lo) * ROW_BYTES, off + lo * ROW_BYTES)
                if len(buf) != (hi - lo) * ROW_BYTES:
                    raise IOError("short read from a vector shard")
                out[g] = np.frombuffer(buf, dtype=np.float16).reshape(hi - lo, DIM)[rr - lo]
            else:
                for i, x in zip(g.tolist(), rr.tolist()):
                    b = os.pread(fd, ROW_BYTES, off + x * ROW_BYTES)
                    if len(b) != ROW_BYTES:
                        raise IOError("short read from a vector shard")
                    flat[i * DIM:(i + 1) * DIM] = np.frombuffer(b, dtype=np.float16)

        list(self._pool.map(one, np.split(order, bounds)))
        return out

    def work_ids(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.int64)
        out = np.empty(len(pos), dtype=np.uint64)
        s, r = self.locate(pos)
        for sh in np.unique(s):
            m = s == sh
            a = self._ids.get(int(sh))
            if a is None:
                a = np.load(self.root / self.dirs[int(sh)] / "work_ids.npy", mmap_mode="r")
                self._ids[int(sh)] = a
            out[m] = a[r[m]]
        return out

    def shard_vectors(self, i: int) -> np.ndarray:
        return np.load(self.root / self.dirs[i] / "vectors.npy", mmap_mode="r")

    def iter_shards(self):
        """(first position, float16 vectors, uint64 work ids) per shard, in order (full reads)."""
        for i, d in enumerate(self.dirs):
            yield int(self.offsets[i]), np.load(self.root / d / "vectors.npy"), np.load(self.root / d / "work_ids.npy")

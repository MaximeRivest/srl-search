"""Positions on the science atlas (citewalk/science) for any embedded paper or any question.

Same frozen map function as citewalk/science/placement.py: a vector lands at the
softmax(similarity / tau)-weighted mean of the map positions of its k most similar reference
papers. k, tau, references and their coordinates come from the atlas build, so positions match
the atlas. Coordinates are in [0, 1] x [0, 1], like the atlas's points.f32.

The k nearest references are found with an HNSW graph (M=32, efConstruction=200, efSearch=512)
instead of comparing with all 291,556 (0.9 GB read per call, 110-400 ms). Measured 2026-10-09 on
5,000 non-reference atlas papers: 99.5% land exactly on their atlas position, none more than
0.05 of the map width away (max 0.044); 15 papers in ~18 ms.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class Atlas:
    def __init__(self, build: Path, cache_dir: Path | None = None):
        ref = np.load(build / "reference.npz")
        self.ref_X = np.ascontiguousarray(ref["vectors"], dtype=np.float32)
        self.meta = json.loads((build / "meta.json").read_text())
        cache = (cache_dir or build) / f"atlas-{self.meta['id']}-hnsw32.faiss"
        if cache.exists():
            self.index = faiss.read_index(str(cache))
            if self.index.ntotal != len(self.ref_X):
                raise ValueError(f"{cache} does not match the build's references")
        else:
            self.index = faiss.IndexHNSWFlat(self.ref_X.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.add(self.ref_X)
            faiss.write_index(self.index, str(cache) + ".tmp")
            Path(str(cache) + ".tmp").rename(cache)
        self.index.hnsw.efSearch = 512
        self.ref_xy = ref["coords"].astype(np.float32)
        self.k = int(ref["k"])
        self.tau = float(ref["tau"])
        self.id = self.meta["id"]

    def place(self, vectors: np.ndarray) -> np.ndarray:
        """float32 [n, 768] normalized -> [n, 2] map coordinates."""
        q = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.omp_set_num_threads(4)  # per calling thread
        s, nn = self.index.search(q, self.k)  # sorted, best first
        w = np.exp((s - s[:, :1]) / self.tau)
        w /= w.sum(1, keepdims=True)
        return (w[:, :, None] * self.ref_xy[nn]).sum(1).astype(np.float32)

    def regions(self) -> list[dict]:
        """Field names, colours and label anchors (for drawing the map's labels)."""
        return [{"name": f["name"], "colour": f["colour"], "domain": f.get("domain", ""),
                 "x": f["anchor"]["x"], "y": f["anchor"]["y"]}
                for f in self.meta["fields"] if f["name"] != "Unclassified"]

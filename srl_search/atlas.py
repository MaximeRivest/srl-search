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
        # the build's placement rule (citewalk/science/placement.py): "mean" (first build) or "mode"
        self.rule = str(ref["rule"]) if "rule" in ref.files else "mean"
        if self.rule == "mode":
            self.h, self.jitter = float(ref["h"]), float(ref["jitter"])
            self.spacing = np.asarray(ref["spacing"], np.float64)

    def place(self, vectors: np.ndarray, works: np.ndarray | None = None) -> np.ndarray:
        """float32 [n, 768] normalized -> [n, 2] map coordinates. works: W-numbers (the "mode" rule's
        fixed per-paper offset; a question has none)."""
        q = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.omp_set_num_threads(4)  # per calling thread
        s, nn = self.index.search(q, self.k)  # sorted, best first
        if self.rule == "mode":
            return mode_place(s, nn, self.ref_xy, self.spacing, self.tau, self.h, self.jitter, works)
        w = np.exp((s - s[:, :1]) / self.tau)
        w /= w.sum(1, keepdims=True)
        return (w[:, :, None] * self.ref_xy[nn]).sum(1).astype(np.float32)

    def regions(self) -> list[dict]:
        """Field names, colours and label anchors (for drawing the map's labels)."""
        return [{"name": f["name"], "colour": f["colour"], "domain": f.get("domain", ""),
                 "x": f["anchor"]["x"], "y": f["anchor"]["y"]}
                for f in self.meta["fields"] if f["name"] != "Unclassified"]


# Same as citewalk/science/placement.py (mode_place, offsets); tests/test_atlas_rule.py keeps them equal.
SALT_R, SALT_A = np.uint64(0xA24BAED4963EE407), np.uint64(0x9FB21C651E98DF25)


def splitmix_u(ids):
    with np.errstate(over="ignore"):
        z = np.asarray(ids).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def offsets(works):
    w = np.asarray(works).astype(np.uint64)
    u1 = np.maximum(splitmix_u(w ^ SALT_R), 1e-300)
    u2 = splitmix_u(w ^ SALT_A)
    r = np.sqrt(-2.0 * np.log(u1))
    return np.stack([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)], 1)


def mode_place(sims, nn, Yref, spacing, tau, h, jitter, works=None, iters=3):
    sims = np.asarray(sims, np.float64)
    w = np.exp((sims - sims[:, :1]) / tau)
    w /= w.sum(1, keepdims=True)
    Y = np.asarray(Yref, np.float64)[nn]
    d2 = ((Y[:, :, None, :] - Y[:, None, :, :]) ** 2).sum(-1)
    j = (w[:, :, None] * np.exp(-d2 / (2 * h * h))).sum(1).argmax(1)
    rows = np.arange(len(Y))
    c = Y[rows, j]
    for _ in range(iters):
        kk = w * np.exp(-((Y - c[:, None]) ** 2).sum(-1) / (2 * h * h))
        c = (kk[:, :, None] * Y).sum(1) / kk.sum(1, keepdims=True)
    if works is not None and jitter > 0:
        near = ((Y - c[:, None]) ** 2).sum(-1).argmin(1)
        c = c + offsets(works) * (jitter * np.asarray(spacing, np.float64)[nn[rows, near]])[:, None]
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def question_spot(pts: np.ndarray, h: float = 0.04, top: int = 20) -> np.ndarray:
    """Where a question "lands" on the map: the middle of the densest group of its top results (earlier results
    weigh more; a few mean-shift steps, Gaussian of width h map units). Placing the question itself through the
    291,556 references put short questions far from every result ("coral reefs": 0.41 map widths away, 0 of 20
    results within 0.05). On 15 questions (2026-10-10), share of the top 20 within 0.05 of the spot: 0.82 on average
    and never 0 (the atlas rule: 0.74, worst 0; the old mean rule: 0.62)."""
    p = np.asarray(pts, np.float64)[:top]
    w = 1.0 / np.sqrt(1.0 + np.arange(len(p)))
    d2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    c = p[(w[None] * np.exp(-d2 / (2 * h * h))).sum(1).argmax()]
    for _ in range(3):
        k = w * np.exp(-((p - c) ** 2).sum(1) / (2 * h * h))
        c = (k[:, None] * p).sum(0) / k.sum()
    return c

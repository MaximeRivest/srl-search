"""How likely a record is junk (spam, advertising, or not scholarship at all), for ranking.

Signals per paper come from `build_quality.py` (raw OpenAlex fields the details store leaves out, plus each
venue's habits over the whole corpus); the paper's own embedding adds what the text looks like. A logistic
model over both (fitted by `fit_quality.py` on judged search results, never on the evaluation questions)
gives P(junk). Ranking subtracts a multiple of it from the similarity, so junk sinks below real papers
without being hidden: a genuine paper that merely looks odd can still surface when nothing better matches.

Citations are deliberately NOT a junk signal: new papers have none yet, and junk is about what a record is,
not how famous it is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLUMNS = ["is_xpac", "is_paratext", "source_type", "source_is_core", "source_in_doaj", "indexed_in", "refs",
           "authors", "locations", "citation_pct", "source"]
N_SOURCE_TYPES = 8   # 0 = none/unknown, then build_quality.SOURCE_TYPES
N_INDEXES = 5
FEATURES = (["is_xpac", "is_paratext"] + [f"source_type={i}" for i in range(N_SOURCE_TYPES)]
            + ["source_is_core", "source_in_doaj"] + [f"indexed_in_bit{i}" for i in range(N_INDEXES)]
            + ["log_refs", "no_refs", "log_authors", "single_author", "log_locations", "no_citation_pct",
               "no_source", "src_log_works", "src_no_refs", "src_single_author", "src_uncited", "src_xpac",
               "unknown_record"])


class Quality:
    def __init__(self, root: Path, model: Path | None = None, more: list[Path] | None = None):
        self.root = Path(root)
        # the daily updates corpus' facts (pipeline/updates: same columns, works the base does not hold)
        self.more = []
        for m in more or []:
            if (Path(m) / "ids.npy").exists() and len(np.load(Path(m) / "ids.npy", mmap_mode="r")):
                self.more.append((np.load(Path(m) / "ids.npy"), {c: np.load(Path(m) / f"{c}.npy") for c in COLUMNS}))
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.ids = np.load(self.root / "ids.npy", mmap_mode="r")
        self.col = {c: np.load(self.root / f"{c}.npy", mmap_mode="r") for c in COLUMNS}
        s = np.load(self.root / "sources.npz")
        self.src = {k: s[k] for k in s.files}
        self.model = None
        model = model or (self.root / "model.npz")
        if Path(model).exists():
            m = np.load(model, allow_pickle=False)
            self.model = {k: m[k] for k in m.files}
            if list(self.model["features"]) != FEATURES:
                raise ValueError(f"{model}: fitted on other features")

    def rows(self, work_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row of each W-number in the arrays, and whether it was found."""
        w = np.asarray(work_ids, dtype=np.int64)
        r = np.searchsorted(self.ids, w)
        r = np.minimum(r, len(self.ids) - 1)
        found = np.asarray(self.ids[r]) == w
        return r, found

    def raw(self, work_ids) -> dict[str, np.ndarray]:
        r, found = self.rows(work_ids)
        out = {c: np.array(a[r]) for c, a in self.col.items()}
        w = np.asarray(work_ids, dtype=np.int64)
        for ids, col in self.more:
            miss = np.flatnonzero(~found)
            if not len(miss):
                break
            i = np.minimum(np.searchsorted(ids, w[miss]), len(ids) - 1)
            hit = ids[i] == w[miss]
            for c in COLUMNS:
                out[c][miss[hit]] = col[c][i[hit]]
            found[miss[hit]] = True
        out["found"] = found
        return out

    def features(self, work_ids) -> np.ndarray:
        """One row of FEATURES per paper (all zeros but `unknown_record` when we hold no record)."""
        v = self.raw(work_ids)
        n = len(v["found"])
        f = np.zeros((n, len(FEATURES)), dtype=np.float32)
        f[:, 0] = v["is_xpac"]
        f[:, 1] = v["is_paratext"]
        st = np.clip(v["source_type"].astype(np.int64), 0, N_SOURCE_TYPES - 1)
        f[np.arange(n), 2 + st] = 1
        j = 2 + N_SOURCE_TYPES
        f[:, j] = v["source_is_core"]
        f[:, j + 1] = v["source_in_doaj"]
        for b in range(N_INDEXES):
            f[:, j + 2 + b] = (v["indexed_in"].astype(np.int64) >> b) & 1
        j += 2 + N_INDEXES
        refs = v["refs"].astype(np.float32)
        authors = v["authors"].astype(np.float32)
        f[:, j] = np.log1p(refs)
        f[:, j + 1] = refs == 0
        f[:, j + 2] = np.log1p(authors)
        f[:, j + 3] = authors <= 1
        f[:, j + 4] = np.log1p(v["locations"].astype(np.float32))
        f[:, j + 5] = np.isnan(v["citation_pct"])
        src = v["source"].astype(np.int64)
        k = np.searchsorted(self.src["source"], src)
        k = np.minimum(k, len(self.src["source"]) - 1)
        has = (src > 0) & (self.src["source"][k] == src)
        f[:, j + 6] = ~has
        f[:, j + 7] = np.where(has, np.log1p(self.src["works"][k]) / 10, 0)
        for i, name in enumerate(("no_refs", "single_author", "uncited", "xpac")):
            f[:, j + 8 + i] = np.where(has, self.src[name][k], 0)
        f[:, j + 12] = ~v["found"]
        f[~v["found"], : j + 12] = 0
        return f

    def junk(self, work_ids, vectors: np.ndarray | None = None) -> np.ndarray | None:
        """P(junk) per paper; the record-only model when no vectors are given. None without a model."""
        if self.model is None or len(work_ids) == 0:
            return None if self.model is None else np.zeros(0, np.float32)
        x = (self.features(work_ids) - self.model["mean"]) / self.model["scale"]
        if vectors is not None:
            z = x @ self.model["w_meta"] + np.asarray(vectors, np.float32) @ self.model["w_vec"] + self.model["b"]
        else:
            z = x @ self.model["w_meta_only"] + self.model["b_meta_only"]
        return (1 / (1 + np.exp(-z))).astype(np.float32)

"""Read-only access to the citation index (citewalk/build_index.py) and its authorship index
(citewalk/build_authors.py), memory-mapped: nothing is loaded up front, and pages already cached for the citation
walk's server are shared through the page cache.

Nodes are positions in wn.npy (sorted OpenAlex work numbers). Only citations whose target is in the snapshot are
indexed; papers added by the daily updates after the snapshot have no references here yet.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class Authors:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.off = np.load(self.root / "auth_off.npy", mmap_mode="r")
        self.auth = np.load(self.root / "auth.npy", mmap_mode="r")
        self.pos = np.load(self.root / "auth_pos.npy", mmap_mode="r")
        self.nauth = np.load(self.root / "nauth.npy", mmap_mode="r")
        self.an = np.load(self.root / "author_an.npy", mmap_mode="r")
        self.name_off = np.load(self.root / "name_off.npy", mmap_mode="r")
        self._names = open(self.root / "names.bin", "rb")

    def of(self, node: int) -> tuple[np.ndarray, np.ndarray, int]:
        """(author rows, their positions, how many authors the paper lists)."""
        a, b = int(self.off[node]), int(self.off[node + 1])
        return np.asarray(self.auth[a:b]), np.asarray(self.pos[a:b]), int(self.nauth[node])

    def name(self, row: int) -> str:
        a, b = int(self.name_off[row]), int(self.name_off[row + 1])
        return self._read(a, b)

    def _read(self, a: int, b: int) -> str:
        return os.pread(self._names.fileno(), b - a, a).decode("utf-8", "replace") if b > a else ""

    def openalex_id(self, row: int) -> str:
        return f"A{int(self.an[row])}"


class Citations:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.generation = self.meta["generation"]
        self.wn = np.load(self.root / "wn.npy", mmap_mode="r")
        self.fwd_off = np.load(self.root / "fwd_off.npy", mmap_mode="r")
        self.fwd = np.load(self.root / "fwd.npy", mmap_mode="r")
        self.year = np.load(self.root / "year.npy", mmap_mode="r")
        self.cites = np.load(self.root / "cites.npy", mmap_mode="r")
        self.kind = np.load(self.root / "kind.npy", mmap_mode="r")
        self.types = self.meta["types"]
        self.title_off = np.load(self.root / "title_off.npy", mmap_mode="r")
        self.title_row = np.load(self.root / "title_row.npy", mmap_mode="r")
        self._titles = open(self.root / "titles.bin", "rb")
        self.authors = Authors(self.root / "authors") if (self.root / "authors" / "meta.json").exists() else None

    def nodes(self, work_numbers) -> np.ndarray:
        """Node of each work number, -1 when the work is not in the snapshot."""
        w = np.asarray(work_numbers, dtype=np.int64)
        if len(w) == 0:
            return np.zeros(0, np.int64)
        i = np.searchsorted(self.wn, w)
        i_c = np.minimum(i, len(self.wn) - 1)
        return np.where(np.asarray(self.wn[i_c]) == w, i_c, -1).astype(np.int64)

    def refs(self, node: int) -> np.ndarray:
        return np.asarray(self.fwd[int(self.fwd_off[node]):int(self.fwd_off[node + 1])])

    def work(self, node: int) -> int:
        return int(self.wn[node])

    def title(self, node: int) -> str:
        r = int(self.title_row[node])
        a, b = int(self.title_off[r]), int(self.title_off[r + 1])
        return os.pread(self._titles.fileno(), b - a, a).decode("utf-8", "replace") if b > a else ""

    def type(self, node: int) -> str | None:
        k = int(self.kind[node])
        return self.types[k] if k < len(self.types) else None

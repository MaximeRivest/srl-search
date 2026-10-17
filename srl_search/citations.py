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
        # the other way round (citewalk/build_author_works.py): each author's papers, and per-author figures
        self.by = (self.root / "meta_works.json").exists()
        if self.by:
            self.by_off = np.load(self.root / "by_off.npy", mmap_mode="r")
            self.by_node = np.load(self.root / "by_node.npy", mmap_mode="r")
            self.by_pos = np.load(self.root / "by_pos.npy", mmap_mode="r")
            self.works = np.load(self.root / "a_works.npy", mmap_mode="r")
            self.cites = np.load(self.root / "a_cites.npy", mmap_mode="r")
            self.h = np.load(self.root / "a_h.npy", mmap_mode="r")
            self.year0 = np.load(self.root / "a_year0.npy", mmap_mode="r")
            self.year1 = np.load(self.root / "a_year1.npy", mmap_mode="r")
            self.topic = np.load(self.root / "a_topic.npy", mmap_mode="r")

    def row_of(self, an: int) -> int | None:
        """The row of an OpenAlex author number, if known."""
        i = int(np.searchsorted(self.an, an))
        return i if i < len(self.an) and int(self.an[i]) == an else None

    def papers(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, positions) of the author's papers, oldest first."""
        a, b = int(self.by_off[row]), int(self.by_off[row + 1])
        return np.asarray(self.by_node[a:b]), np.asarray(self.by_pos[a:b])

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


class Institutions:
    """The institution index (citewalk/build_institutions.py): a paper's organisations, an organisation's papers,
    and where each author has published from."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.kinds = self.meta["kinds"]
        self.inst_in = np.load(self.root / "inst_in.npy", mmap_mode="r")
        self.w_off = np.load(self.root / "w_off.npy", mmap_mode="r")
        self.w_inst = np.load(self.root / "w_inst.npy", mmap_mode="r")
        self.i_off = np.load(self.root / "i_off.npy", mmap_mode="r")
        self.i_node = np.load(self.root / "i_node.npy", mmap_mode="r")
        self.ai_off = np.load(self.root / "ai_off.npy", mmap_mode="r")
        self.ai_inst = np.load(self.root / "ai_inst.npy", mmap_mode="r")
        self.ai_n = np.load(self.root / "ai_n.npy", mmap_mode="r")
        self.ai_year0 = np.load(self.root / "ai_year0.npy", mmap_mode="r")
        self.ai_year1 = np.load(self.root / "ai_year1.npy", mmap_mode="r")
        self.country = np.load(self.root / "country.npy", mmap_mode="r")
        self.kind = np.load(self.root / "kind.npy", mmap_mode="r")
        self.name_off = np.load(self.root / "name_off.npy", mmap_mode="r")
        self._names = open(self.root / "names.bin", "rb")
        self.ror_off = np.load(self.root / "ror_off.npy", mmap_mode="r")
        self._rors = open(self.root / "rors.bin", "rb")

    def _read(self, fh, off, row: int) -> str:
        a, b = int(off[row]), int(off[row + 1])
        return os.pread(fh.fileno(), b - a, a).decode("utf-8", "replace") if b > a else ""

    def name(self, row: int) -> str:
        return self._read(self._names, self.name_off, row)

    def ror(self, row: int) -> str:
        return self._read(self._rors, self.ror_off, row)

    def openalex_id(self, row: int) -> str:
        return f"I{int(self.inst_in[row])}"

    def row_of(self, inum: int) -> int | None:
        i = int(np.searchsorted(self.inst_in, inum))
        return i if i < len(self.inst_in) and int(self.inst_in[i]) == inum else None

    def of_paper(self, node: int) -> np.ndarray:
        return np.asarray(self.w_inst[int(self.w_off[node]):int(self.w_off[node + 1])])

    def papers(self, row: int) -> np.ndarray:
        """Nodes of the institution's papers, oldest first."""
        return np.asarray(self.i_node[int(self.i_off[row]):int(self.i_off[row + 1])])

    def of_author(self, author_row: int) -> list[tuple[int, int, int, int]]:
        """(institution row, papers from there, first year, last year), most papers first."""
        a, b = int(self.ai_off[author_row]), int(self.ai_off[author_row + 1])
        rows = list(zip(self.ai_inst[a:b].tolist(), self.ai_n[a:b].tolist(), self.ai_year0[a:b].tolist(), self.ai_year1[a:b].tolist()))
        rows.sort(key=lambda r: (-r[1], -r[3]))
        return rows

    def describe(self, row: int) -> dict:
        c = self.country[row].decode() if hasattr(self.country[row], "decode") else str(self.country[row])
        k = self.kinds[int(self.kind[row])] if int(self.kind[row]) < len(self.kinds) else ""
        return {"id": self.openalex_id(row), "name": self.name(row), "country": c or None, "kind": k or None,
                "ror": self.ror(row) or None}


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
        self.institutions = Institutions(self.root / "institutions") \
            if (self.root / "institutions" / "meta.json").exists() else None

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

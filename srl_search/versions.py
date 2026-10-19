"""Paper versions (pipeline/identity/build_versions.py in the data repository): OpenAlex records that are one paper
(a preprint and its journal article, mirror copies, repository copies), so that a paper is listed once.

Data: <dir>/work.npy (sorted W-numbers of grouped records), cluster.npy, rank.npy (0 = the group's canonical record:
the version of record), versions.sqlite (members of each group: type, year, DOI, venue).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np

MAX_LISTED = 20


class Versions:
    def __init__(self, root: Path):
        self.root = root
        self.work = np.load(root / "work.npy", mmap_mode="r")
        self.cluster = np.load(root / "cluster.npy", mmap_mode="r")
        self.rank = np.load(root / "rank.npy", mmap_mode="r")
        self._db = str(root / "versions.sqlite")
        self._local = threading.local()

    def _con(self) -> sqlite3.Connection:
        c = getattr(self._local, "con", None)
        if c is None:
            c = self._local.con = sqlite3.connect(f"file:{self._db}?mode=ro", uri=True, check_same_thread=False)
        return c

    def group(self, w: int) -> int | None:
        i = int(np.searchsorted(self.work, np.uint64(w)))
        return int(self.cluster[i]) if i < len(self.work) and int(self.work[i]) == w else None

    def groups(self, ws: list[int]) -> list[int | None]:
        a = np.asarray(ws, np.uint64)
        i = np.minimum(np.searchsorted(self.work, a), len(self.work) - 1)
        hit = self.work[i] == a
        return [int(self.cluster[j]) if h else None for j, h in zip(i, hit)]

    def members(self, g: int) -> list[dict]:
        rows = self._con().execute("SELECT rank, work, type, year, doi, venue FROM members WHERE cluster = ? ORDER BY rank",
                                   (g,)).fetchall()
        return [{"id": f"W{w}", "type": t, "year": y, "doi": d, "venue": v, "canonical": r == 0}
                for r, w, t, y, d, v in rows]

    @staticmethod
    def listed(members: list[dict], shown: int) -> dict:
        """What a result carries: the other versions (up to MAX_LISTED) and how many there are."""
        others = [m for m in members if m["id"] != f"W{shown}"]
        return {"versions": others[:MAX_LISTED], "versions_count": len(members)}

"""The source index (citewalk/build_sources.py): where each paper was published (journal, conference, repository,
book series), each source's papers and figures, and a search by name or ISSN. Memory-mapped; the names are read
into memory on the first search (~245,000 of them, ~30 MB).

Kept apart from citations.py on purpose: that file is hashed into the topic maps' version.
"""
from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

WORD = re.compile(r"[^\W_]+", re.UNICODE)
ISSN = re.compile(r"^\s*(\d{4})-?(\d{3}[\dxX])\s*$")
KINDS = {"journal": "journal", "repository": "repository", "conference": "conference", "book series": "book series",
         "ebook platform": "book platform", "other": "source", "metadata": "source"}


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


class Sources:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        L = lambda n: np.load(self.root / f"{n}.npy", mmap_mode="r")   # noqa: E731
        self.s_in, self.w_src, self.s_off, self.s_node, self.oa = L("src_in"), L("w_src"), L("s_off"), L("s_node"), L("oa")
        self.works, self.cites, self.h = L("s_works"), L("s_cites"), L("s_h")
        self.year0, self.year1, self.n_oa = L("s_year0"), L("s_year1"), L("s_oa")
        self.kind, self.pub = L("kind"), L("pub")
        self.name_off, self.issn_off, self.pub_name_off = L("name_off"), L("issn_off"), L("pub_name_off")
        self.pub_ids = np.load(self.root / "pub_ids.npy")
        self._names = open(self.root / "names.bin", "rb")
        self._issns = open(self.root / "issns.bin", "rb")
        self._pubs = open(self.root / "pub_names.bin", "rb")
        self.kinds = self.meta["kinds"]
        self._folded: list[str] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _read(fh, off, row: int) -> str:
        a, b = int(off[row]), int(off[row + 1])
        return os.pread(fh.fileno(), b - a, a).decode("utf-8", "replace") if b > a else ""

    def name(self, row: int) -> str:
        return self._read(self._names, self.name_off, row)

    def issn(self, row: int) -> str:
        return self._read(self._issns, self.issn_off, row)

    def openalex_id(self, row: int) -> str:
        return f"S{int(self.s_in[row])}"

    def row_of(self, snum: int) -> int | None:
        i = int(np.searchsorted(self.s_in, snum))
        return i if i < len(self.s_in) and int(self.s_in[i]) == snum else None

    def row(self, ref: str) -> int | None:
        r = ref.strip()
        if r.lower().startswith("https://openalex.org/"):
            r = r.rsplit("/", 1)[-1]
        if r[:1] in "Ss" and r[1:].isdigit():
            return self.row_of(int(r[1:]))
        m = ISSN.match(r)
        if m:
            hits = self.search(f"{m.group(1)}-{m.group(2)}", 1)
            return hits[0] if hits else None
        return None

    def of_paper(self, node: int) -> int | None:
        r = int(self.w_src[node])
        return r if r >= 0 else None

    def papers(self, row: int) -> np.ndarray:
        return np.asarray(self.s_node[int(self.s_off[row]):int(self.s_off[row + 1])]).astype(np.int64)

    def publisher(self, row: int) -> dict | None:
        p = int(self.pub[row])
        if p < 0:
            return None
        return {"id": str(self.pub_ids[p]), "name": self._read(self._pubs, self.pub_name_off, p) or None}

    def kind_of(self, row: int) -> str:
        k = self.kinds[int(self.kind[row])] if int(self.kind[row]) < len(self.kinds) else ""
        return KINDS.get(k, k or "source")

    def describe(self, row: int) -> dict:
        y0, y1 = int(self.year0[row]), int(self.year1[row])
        w = int(self.works[row])
        return {"id": self.openalex_id(row), "name": self.name(row), "type": self.kind_of(row),
                "issn": self.issn(row) or None, "publisher": self.publisher(row), "works": w,
                "citations": int(self.cites[row]), "h": int(self.h[row]), "years": [y0, y1] if y0 else None,
                "open_access": round(int(self.n_oa[row]) / w, 3) if w else None}

    # ------------------------------------------------------------------ search
    def _all(self) -> tuple[list[str], dict[str, np.ndarray], list[str]]:
        """Folded names, an index word -> rows, and the sorted vocabulary (for a last word typed in part)."""
        with self._lock:
            if self._folded is None:
                data = (self.root / "names.bin").read_bytes()
                off = np.asarray(self.name_off)
                names, idx = [], {}
                for i in range(len(off) - 1):
                    nm = " ".join(WORD.findall(fold(data[int(off[i]):int(off[i + 1])].decode("utf-8", "replace"))))
                    names.append(nm)
                    for w in set(nm.split()):
                        idx.setdefault(w, []).append(i)
                self._index = {w: np.array(v, np.int64) for w, v in idx.items()}
                self._vocab = sorted(self._index)
                self._folded = names
            return self._folded, self._index, self._vocab

    def search(self, q: str, k: int) -> list[int]:
        """Rows whose name holds every word of q (accents and case ignored; the last word may be a prefix), or whose
        ISSN is q; full-name matches first, then the most published."""
        m = ISSN.match(q)
        if m:
            want = f"{m.group(1)}-{m.group(2)}".upper()
            data = (self.root / "issns.bin").read_bytes()
            at = data.find(want.encode())
            if at < 0:
                return []
            row = int(np.searchsorted(np.asarray(self.issn_off), at, side="right")) - 1
            return [row]
        words = WORD.findall(fold(q))
        if not words:
            return []
        import bisect
        names, index, vocab = self._all()
        rows = None
        for w in words[:-1]:
            r = index.get(w)
            if r is None:
                return []
            rows = r if rows is None else np.intersect1d(rows, r)
        last = words[-1]                                    # whole word, or the start of one
        at = bisect.bisect_left(vocab, last)
        pre = []
        while at < len(vocab) and vocab[at].startswith(last) and len(pre) < 5000:
            pre.append(index[vocab[at]])
            at += 1
        if not pre:
            return []
        r = np.unique(np.concatenate(pre))
        rows = r if rows is None else np.intersect1d(rows, r)
        hits = rows.tolist()
        target = " ".join(words)
        works = np.asarray(self.works)
        hits.sort(key=lambda i: (names[i] != target, -int(works[i])))
        return hits[:k]

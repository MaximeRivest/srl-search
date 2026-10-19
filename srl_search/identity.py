"""Author identity overlay (data repository: pipeline/identity/authors.py): corrections to OpenAlex's author records
that rest on ORCIDs, applied when pages are made. Nothing is guessed from names: names only veto.

  * records carrying the same ORCID, with compatible names, are one person: their pages open as one (the record
    with most papers first), like /authors/{id}?with= but without the reader having to choose;
  * an authorship whose printed ORCID belongs to another person moves to that person (it leaves the record OpenAlex
    gave it to); an authorship OpenAlex gave no record joins the person its ORCID names;
  * records that are not people (organisations, placeholders such as ':unav' or 'Anonymous', lists of names) are
    marked, and left out of people search.

Data: <dir>/records.parquet (author, person, merged_by_orcid, not_person, not_person_why), moves.parquet (work, pos,
author, person). Loaded once: ~1.5M merged records, ~7M moves, ~1M flags (~200 MB).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


class Identity:
    def __init__(self, root: Path, cit):
        self.root = Path(root)
        au = cit.authors
        rec = pq.read_table(root / "records.parquet", columns=["author", "person", "merged_by_orcid", "not_person", "not_person_why", "orcid"])
        merged = rec.filter(pc.field("merged_by_orcid"))
        a = merged["author"].to_numpy().astype(np.int64)
        p = merged["person"].to_numpy().astype(np.int64)
        # person -> its records (the person's own record first), as citation-index author rows
        persons = np.unique(p)
        self._person_an = persons
        rows_a, rows_p = self._rows(au, a), self._rows(au, p)
        ok = (rows_a >= 0) & (rows_p >= 0)
        a, p, rows_a, rows_p = a[ok], p[ok], rows_a[ok], rows_p[ok]
        self._m_row = rows_a                    # merged record row -> its person's row
        self._m_person = rows_p
        o = np.argsort(self._m_row)
        self._m_row, self._m_person = self._m_row[o], self._m_person[o]
        o = np.argsort(rows_p, kind="stable")
        self._g_person, self._g_member = rows_p[o], rows_a[o]
        orc = merged["orcid"].to_pylist()
        self._orcid_of_person = {int(r): orc[i] for i, r in zip(range(len(orc)), rows_p.tolist())}
        np_ = rec.filter(pc.field("not_person"))
        r = self._rows(au, np_["author"].to_numpy().astype(np.int64))
        why = np.array(np_["not_person_why"].to_pylist(), object)
        o = np.argsort(r)
        self._np_row, self._np_why = r[o], why[o]
        keep = self._np_row >= 0
        self._np_row, self._np_why = self._np_row[keep], self._np_why[keep]
        # moves, as citation-index nodes and author rows (the person's row; the record it leaves, -1 for none)
        mv = pq.read_table(root / "moves.parquet", columns=["work", "pos", "author", "person"])
        nodes = np.asarray(cit.nodes(mv["work"].to_numpy().astype(np.int64)), np.int64)
        to = self._rows(au, mv["person"].to_numpy().astype(np.int64))
        frm_an = mv["author"].to_numpy(zero_copy_only=False)
        frm = self._rows(au, np.nan_to_num(frm_an.astype(float), nan=-1).astype(np.int64))
        ok = (nodes >= 0) & (to >= 0)
        nodes, to, frm, pos = nodes[ok], to[ok], frm[ok], mv["pos"].to_numpy()[ok].astype(np.int64) + 1
        o = np.lexsort((nodes, to))
        self._in_row, self._in_node, self._in_pos = to[o], nodes[o], pos[o]
        o = np.lexsort((nodes, frm))
        self._out_row, self._out_node = frm[o], nodes[o]
        o = np.argsort(nodes, kind="stable")
        self._w_node, self._w_pos, self._w_to = nodes[o], pos[o], to[o]
        self.counts = {"merged_records": int(len(self._m_row)), "persons": int(len(np.unique(self._g_person))),
                       "moves": int(len(nodes)), "not_people": int(len(self._np_row))}

    @staticmethod
    def _rows(au, an: np.ndarray) -> np.ndarray:
        i = np.minimum(np.searchsorted(au.an, an), len(au.an) - 1)
        return np.where(np.asarray(au.an[i]) == an, i, -1).astype(np.int64)

    @staticmethod
    def _range(keys: np.ndarray, k: int) -> slice:
        return slice(int(np.searchsorted(keys, k, "left")), int(np.searchsorted(keys, k, "right")))

    # ---------------------------------------------------------------- persons
    def person_row(self, row: int) -> int:
        s = self._range(self._m_row, row)
        return int(self._m_person[s.start]) if s.stop > s.start else row

    def records(self, row: int) -> tuple[int, ...]:
        """All records of the person this record belongs to, the person's own first ((row,) when it is alone)."""
        p = self.person_row(row)
        s = self._range(self._g_person, p)
        return (p, *sorted(int(x) for x in self._g_member[s])) if s.stop > s.start else (row,)

    def orcid(self, person_row: int) -> str | None:
        return self._orcid_of_person.get(int(person_row))

    def not_person(self, row: int) -> str | None:
        s = self._range(self._np_row, row)
        return str(self._np_why[s.start]) if s.stop > s.start else None

    # ---------------------------------------------------------------- authorships
    def moved_in(self, rows) -> tuple[np.ndarray, np.ndarray]:
        """(nodes, positions) of authorships moved to these records."""
        parts = [self._range(self._in_row, int(r)) for r in rows]
        if not parts:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return (np.concatenate([self._in_node[s] for s in parts]), np.concatenate([self._in_pos[s] for s in parts]))

    def moved_out(self, rows) -> np.ndarray:
        """Nodes of authorships that left these records (for another person)."""
        parts = [self._range(self._out_row, int(r)) for r in rows]
        return np.concatenate([self._out_node[s] for s in parts]) if parts else np.zeros(0, np.int64)

    def on_paper(self, node: int) -> list[tuple[int, int]]:
        """(position, person row) of the moves on one paper."""
        s = self._range(self._w_node, node)
        return list(zip(self._w_pos[s].tolist(), self._w_to[s].tolist()))

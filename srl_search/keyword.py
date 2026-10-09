"""Keyword / boolean search over the corpus (tantivy): exact words, phrases, AND / OR / NOT, BM25 ranking.

Query language (what users type):
  "gut microbiome" autism            both (plain terms are joined with AND, like PubMed/Europe PMC)
  sleep OR insomnia                  either
  adolescents NOT caffeine           exclude (also: AND NOT, -caffeine)
  (sleep OR insomnia) AND teens      grouping
  microplastic*                      word starts with
  "BRCA1 ovarian"~5                  words within 5 positions
  title:(sleep AND adolescents)      in the title only; also abstract:, keywords: (OpenAlex keywords + MeSH)
Words are stemmed (English): "cells" also finds "cell".
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import tantivy

from srl_search.scope import NON_PUBLICATION_TYPES

FIELDS = ["title", "abstract", "keywords"]
BOOSTS = {"title": 2.0, "abstract": 1.0, "keywords": 1.5}


def schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("id", stored=True, indexed=True, fast=True)
    for f in FIELDS:
        sb.add_text_field(f, stored=False, tokenizer_name="en_stem", index_option="position")
    sb.add_integer_field("year", indexed=True, fast=True)
    sb.add_text_field("type", stored=False, tokenizer_name="raw", index_option="basic")
    sb.add_text_field("license", stored=False, tokenizer_name="raw", index_option="basic")
    for f in ("oa", "retracted", "pmc_oa", "smr"):
        sb.add_integer_field(f, indexed=True, fast=True)  # 0/1 (Python ints arrive as i64)
    return sb.build()


# ---------------------------------------------------------------- query text

TOKEN = re.compile(r'\s*("(?:[^"\\]|\\.)*"(?:~\d+)?\*?|\(|\)|[^\s()"]+)')
OPS = {"AND", "OR", "NOT"}
BOOLEAN_HINT = re.compile(r'"|\b(AND|OR|NOT)\b|[()*]|(^|\s)[-+]\w|\b(title|abstract|keywords):')


def looks_boolean(q: str) -> bool:
    """Did the user write search syntax (quotes, AND/OR/NOT in capitals, brackets, *, -word, field:)?"""
    return bool(BOOLEAN_HINT.search(q))


def normalize(q: str) -> str:
    """Make implicit AND explicit, turn 'AND NOT' into NOT, balance quotes and brackets."""
    if q.count('"') % 2:
        q = q.replace('"', " ")
    out: list[str] = []
    depth = 0
    glue = False  # previous token was "field:" and this one belongs to it
    for t in (m.group(1) for m in TOKEN.finditer(q)):
        if t == ")":
            if depth == 0:
                continue
            depth -= 1
            out.append(t)
            continue
        ends_operand = bool(out) and (out[-1] not in OPS and not out[-1].endswith(("(", ":")))
        if t in OPS:
            if t == "NOT" and out and out[-1] == "AND":
                out.pop()          # tantivy wants "a NOT b", not "a AND NOT b"
            if not out or out[-1] in OPS or out[-1].endswith("("):
                if t != "NOT":
                    continue       # dangling AND/OR
            out.append(t)
            continue
        if ends_operand and not glue:
            out.append("AND")
        if t == "(":
            depth += 1
        if glue:
            out[-1] += t
        else:
            out.append(t)
        glue = t.endswith(":")
    while out and out[-1] in OPS:
        out.pop()
    out += [")"] * depth
    return " ".join(out)


# ---------------------------------------------------------------- engine

class KeywordIndex:
    def __init__(self, path: Path):
        self.index = tantivy.Index.open(str(path))
        self.schema = self.index.schema
        self.lock = threading.Lock()
        self.searcher = self.index.searcher()

    def filters(self, f: dict, collection: str) -> list[tuple]:
        S, O, Q = self.schema, tantivy.Occur, tantivy.Query
        out = []
        if collection == "smr":
            out.append((O.Must, Q.term_query(S, "smr", 1)))
        if f.get("year_from") is not None or f.get("year_to") is not None:
            out.append((O.Must, Q.range_query(S, "year", tantivy.FieldType.Integer,
                                             f.get("year_from", -9999), f.get("year_to", 9999))))
        if f.get("type"):
            out.append((O.Must, Q.term_query(S, "type", f["type"])))
        # Covers/tables of contents and retracted papers are dropped from the returned page by
        # Engine.shape (they are rare); as index clauses they cost 30-100 ms on broad queries.
        if f.get("publications_only"):   # costs 100-500 ms on broad queries: used only as a fallback
            out.append((O.MustNot, Q.term_set_query(S, "type", list(NON_PUBLICATION_TYPES))))
        if f.get("license"):
            out.append((O.Must, Q.term_query(S, "license", f["license"])))
        if f.get("open_access"):
            out.append((O.Must, Q.term_query(S, "oa", 1)))
        if f.get("pmc_open_access"):
            out.append((O.Must, Q.term_query(S, "pmc_oa", 1)))
        return out

    def search(self, q: str, k: int, filters: dict, collection: str = "all") -> tuple[list[tuple[int, float]], int, str]:
        """(id, bm25) best first, number of matching papers, the query as understood."""
        text = normalize(q)
        query, errors = self.index.parse_query_lenient(text, FIELDS, field_boosts=BOOSTS)
        clauses = [(tantivy.Occur.Must, query)] + self.filters(filters, collection)
        full = tantivy.Query.boolean_query(clauses)
        res = self.searcher.search(full, k, count=True)
        hits = []
        for score, addr in res.hits:
            hits.append((int(self.searcher.doc(addr)["id"][0]), float(score)))
        return hits, int(res.count or 0), text

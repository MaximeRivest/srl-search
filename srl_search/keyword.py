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
TOKEN_WORD = re.compile(r"[A-Za-z0-9]+")
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
    """One tantivy index, or several with the same schema searched as one: the base corpus' and the daily updates'
    (pipeline/updates). Hits from several are merged by score. Each index ranks with its own word statistics; the
    updates index is a recent sample of the same literature, so its BM25 scores are close to what the base would
    give, not identical."""

    def __init__(self, paths: Path | list[Path]):
        paths = [paths] if isinstance(paths, Path) else list(paths)
        self.indexes = [tantivy.Index.open(str(p)) for p in paths]
        self.index = self.indexes[0]
        self.schema = self.index.schema
        self.lock = threading.Lock()
        self.searchers = [i.searcher() for i in self.indexes]
        self.searcher = self.searchers[0]

    @property
    def num_docs(self) -> int:
        return sum(s.num_docs for s in self.searchers)

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
        hits, count = [], 0
        for index, searcher in zip(self.indexes, self.searchers):
            query, errors = index.parse_query_lenient(text, FIELDS, field_boosts=BOOSTS)
            clauses = [(tantivy.Occur.Must, query)] + self.filters(filters, collection)
            res = searcher.search(tantivy.Query.boolean_query(clauses), k, count=True)
            hits += [(int(searcher.doc(addr)["id"][0]), float(score)) for score, addr in res.hits]
            count += int(res.count or 0)
        hits.sort(key=lambda h: -h[1])
        return hits[:k], count, text

    def phrase_counts(self, phrases: list[str], workers: int = 8) -> tuple[dict[str, int], int]:
        """How many papers use each phrase in their title or abstract (stemmed, as searches match), and how many
        papers there are: the background for a topic map's word cloud. 50-300 ms a phrase, so they run side by side
        (tantivy releases the GIL) and are remembered for the life of the process (the same words recur across maps)."""
        from concurrent.futures import ThreadPoolExecutor
        cache = self.__dict__.setdefault("_phrase_counts", {})
        todo = [ph for ph in dict.fromkeys(phrases) if ph.lower() not in cache]

        def one(ph: str) -> int:
            n = 0
            words = TOKEN_WORD.findall(ph)
            if not words:
                return 0
            text = '"' + " ".join(words) + '"' if len(words) > 1 else words[0]
            for index, searcher in zip(self.indexes, self.searchers):
                query, _ = index.parse_query_lenient(text, ["title", "abstract"])
                n += int(searcher.search(query, 1, count=True).count or 0)
            return n
        if todo:
            with ThreadPoolExecutor(min(workers, len(todo))) as ex:
                for ph, n in zip(todo, ex.map(one, todo)):
                    cache[ph.lower()] = n
        return {ph: cache[ph.lower()] for ph in phrases}, self.num_docs

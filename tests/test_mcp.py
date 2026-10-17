"""The MCP server (srl_search/mcp_server.py) over a stand-in search service: no indexes are loaded.

In process for what the tools answer; over real HTTP (uvicorn on a free port) for the mounting, the lifespan and
both protocol eras (the initialize handshake and the per-request 2026-07-28 one)."""
import socket
import threading
import time
from types import SimpleNamespace

import anyio
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from mcp import Client

from srl_search import mcp_server

PAPER = {"id": "W2782456798", "doi": "10.1126/science.aan8048", "title": "Spatial and temporal patterns of <i>mass</i> bleaching",
         "year": 2018, "date": "2018-01-05", "type": "article", "venue": "Science", "publisher": "AAAS",
         "authors": "Terry P. Hughes | Kristen D. Anderson | Sean R. Connolly | Scott F. Heron", "authors_count": 25,
         "citations": 2281, "is_oa": 1, "oa_status": "green", "license": None, "is_retracted": 0,
         "url": "https://doi.org/10.1126/science.aan8048", "pdf_url": None, "keywords": "Coral bleaching | Reef",
         "mesh": "Coral Reefs", "pmcid": None, "smr": 0, "openalex": "https://openalex.org/W2782456798",
         "abstract": "Tropical reef systems are transitioning to a new era. " * 20, "junk": 0.01}
JUNK = {**PAPER, "id": "W9", "doi": None, "title": "Buy cheap pills", "openalex": "https://openalex.org/W9", "junk": 0.9,
        "abstract": None, "smr": 1, "pmcid": "PMC1"}


def fake_api():
    calls = []

    def search(**kw):
        calls.append(kw)
        if kw["q"] == "nothing at all":
            return {"mode": "semantic", "results": []}
        if '"' in kw["q"]:
            return {"mode": "keyword", "matches": 1234, "query": kw["q"], "results": [PAPER]}
        return {"mode": "semantic", "results": [PAPER, JUNK][:kw["k"]]}

    def similar(ref, **kw):
        if ref == "W404":
            raise HTTPException(404, "not in the index")
        return {"results": [JUNK]}

    def work(ref):
        if ref in ("W2782456798", "10.1126/science.aan8048"):
            return dict(PAPER)
        raise HTTPException(404, f"{ref} is not in the corpus")

    def walk(id, k, max_refs, max_cites):
        if id == "W1":
            return JSONResponse({"error": "Not found in the snapshot: 'W1'"}, status_code=404)
        node = {"title": "T", "year": 2014, "cited_by": 345, "type": "article"}
        return {"focus": "W2782456798", "answers": ["W11"], "stats": {"k": k, "refs_indexed": 22, "papers_sharing_a_reference": 1571},
                "columns": {"similar": ["W10"], "candidates": ["W11", "W12"], "candidates_out": ["W13"]},
                "nodes": {"W2782456798": {**node, "title": "Focus", "refs_reported": 22},
                          "W10": {**node, "shared": 7}, "W11": {**node, "support": 5}, "W12": {**node, "support": 3},
                          "W13": {**node, "support": 9, "cited_by": 729}}}

    engine = SimpleNamespace(details=lambda ws: {w: {"doi": f"10.1/{w}"} for w in ws if w != 12})

    def topics_get(q, papers, mode, year_from, year_to, group, cached_only=False, scope="closest", definition=None):
        if scope == "field":
            from srl_search.field import Pending
            raise Pending({"status": "running", "stage": "citations", "stage_index": 5, "eta_s": 900})
        calls.append({"topics": q, "papers": papers, "group": group, "year_from": year_from})
        if q == "nothing":
            raise HTTPException(404, "Only 3 papers found for this; a map needs at least 30.")
        return {"question": q, "mode": "semantic"}

    def topic_text(out, base, page=1):
        return f"# Research groups in: {out['question']} · {base}" + (f" · page {page}" if page > 1 else "")

    def topics_read(out, group, part, page, abstracts="full"):
        calls.append({"read": group, "part": part, "page": page, "of": out["question"]})
        return {"question": out["question"], "group": group, "part": part, "abstracts": abstracts}

    def topic_read_text(r, base):
        return f"# Group {r['group']} of “{r['question']}” · {r['part']} · abstracts {r['abstracts']}"
    PERSON = {"id": "A1", "name": "Ann Reef", "works": 4, "citations": 100, "h": 3, "years": [2010, 2024],
              "topic": {"name": "Coral Bleaching", "subfield": "Marine Biology", "field": "Agricultural and Biological Sciences"},
              "institutions": [{"id": "I7", "name": "Reef University", "papers": 3, "years": [2010, 2024]}],
              "positions": {"first": 3, "second": 1, "other": 0}}
    TP = {"id": "W1000", "title": "Paper 0", "year": 2010, "authors": ["Ann Reef", "Bo Fish"], "authors_count": 2, "citations": 40}
    CO = {"author": "A2", "name": "Bo Fish", "n": 2, "years": [2010, 2015], "works": 3, "top": TP}
    unit = {"size": 4, "years": [2010, 2015, 2024], "lenses": {"cited": [TP], "newest": [TP], "first": [TP], "coauthors": [CO], "_total": {"cited": 40}}}
    profile = {"kind": "author", "person": PERSON, "level": "subfields", "mode": "author", "question": "Ann Reef",
               "groups": [dict(unit, id=1, name="Marine Biology", subgroups=[dict(unit, id=1, name="Coral Bleaching")])],
               "overall": unit, "coverage": {"papers": 4, "in_groups": 3, "other": 1}}
    people = SimpleNamespace(
        search=lambda q, k: {"q": q, "results": [dict(PERSON, match="full", institution={"id": "I7", "name": "Reef University"})][:k], "how": "exact"},
        search_institutions=lambda q, k: {"q": q, "results": [{"id": "I7", "name": "Reef University", "works": 4, "country": "AU", "kind": "education", "match": "full"}], "how": "exact"},
        row=lambda ref: 0 if ref == "A1" else None, inst_row=lambda ref: 0 if ref == "I7" else None,
        ident=lambda kind, row: "A1" if kind == "author" else "I7",
        profile=lambda kind, row: profile,
        more=lambda kind, row, at, name, offset, limit: {"items": [TP] * 2, "total": 40})
    api = SimpleNamespace(
        people_service=lambda: people,
        SEMANTIC={"all": SimpleNamespace(manifest={"papers": 124_892_915, "min_date": "1800-01-01"})},
        KEYWORDS=SimpleNamespace(num_docs=203_543_614), app=SimpleNamespace(version="0.3"),
        search=search, similar=similar, work=work, walk=walk, engine=engine, NPROBE=128, DEPTH=300,
        guide=lambda: PlainTextResponse("# guide"),
        topics_service=lambda: SimpleNamespace(get=topics_get, read=topics_read), topic_text=topic_text,
        topic_read_text=topic_read_text)
    return api, calls


def text(result) -> str:
    return "\n".join(c.text for c in result.content)


def test_tools_answer_in_markdown_the_model_can_cite():
    api, calls = fake_api()
    server = mcp_server.build(api)

    async def go():
        async with Client(server) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert set(tools) == {"search_papers", "get_papers", "find_similar_papers", "citation_walk", "map_topic", "read_topic_group",
                                  "find_people", "get_profile"}
            assert all(t.annotations.read_only_hint for t in tools.values())
            assert "review" in str(tools["search_papers"].input_schema)        # the type list reaches the model

            r = await c.call_tool("search_papers", {"query": "coral bleaching frequency", "limit": 2})
            out = text(r)
            assert not r.is_error and "Found by meaning" in out and "https://doi.org/10.1126/science.aan8048" in out
            assert "<i>" not in out and "et al. (25 authors)" in out and "probably not a scholarly paper" in out
            assert "plain-language version: https://sciencemadereadable.com/?paper=PMC1" in out
            assert len(out.split("Tropical")[1].split("(Abstracts")[0]) < 400   # short abstracts by default
            assert "map_topic" in out                                           # points to the field's map
            assert calls[-1]["collection"] == "all" and calls[-1]["abstracts"] is True

            await c.call_tool("search_papers", {"query": "x y", "readable_only": True, "abstracts": "none",
                                                "publication_type": "review"})
            assert calls[-1]["collection"] == "smr" and calls[-1]["abstracts"] is False and calls[-1]["type"] == "review"
            assert "1,234 papers match" in text(await c.call_tool("search_papers", {"query": '"gut microbiome"'}))
            assert "Nothing found" in text(await c.call_tool("search_papers", {"query": "nothing at all"}))
            bad = await c.call_tool("search_papers", {"query": "x", "limit": 500})
            assert bad.is_error

            r = await c.call_tool("get_papers", {"ids": ["10.1126/science.aan8048", "W0"]})
            assert "Coral bleaching, Reef; MeSH: Coral Reefs" in text(r) and "## W0\nNot found" in text(r)

            r = await c.call_tool("find_similar_papers", {"paper": "W404"})
            assert r.is_error and "not among the papers searchable by meaning" in text(r)

            r = await c.call_tool("citation_walk", {"paper": "W2782456798", "breadth": 10})
            out = text(r)
            assert "cited by 5 of the 10 neighbours · strongest candidate" in out and "shares 7 references" in out
            assert out.index("W11") < out.index("W12") < out.index("very widely cited") < out.index("W13")
            assert "https://doi.org/10.1/11" in out and "https://openalex.org/W12" in out
            r = await c.call_tool("citation_walk", {"paper": "W1"})
            assert r.is_error and "Not found in the snapshot" in text(r)

            r = await c.call_tool("map_topic", {"topic": "intermittent fasting", "group": 2, "year_from": 2015})
            assert not r.is_error and text(r).startswith("# Research groups in: intermittent fasting")
            assert calls[-1] == {"topics": "intermittent fasting", "papers": 1000, "group": 2, "year_from": 2015}
            r = await c.call_tool("map_topic", {"topic": "habitat fragmentation", "scope": "whole_field"})
            assert not r.is_error and "being made" in text(r) and "15 more minutes" in text(r)
            r = await c.call_tool("read_topic_group", {"topic": "habitat fragmentation", "group": 1, "scope": "whole_field"})
            assert not r.is_error and "being made" in text(r)
            r = await c.call_tool("map_topic", {"topic": "nothing"})
            assert r.is_error and "at least 30" in text(r)
            r = await c.call_tool("read_topic_group", {"topic": "intermittent fasting", "group": 3, "part": "papers", "page": 2})
            assert text(r) == "# Group 3 of “intermittent fasting” · papers · abstracts full"
            assert calls[-2]["group"] is None and calls[-1] == {"read": 3, "part": "papers", "page": 2, "of": "intermittent fasting"}
            await c.call_tool("read_topic_group", {"topic": "intermittent fasting", "group": 1, "within": 4})
            assert calls[-2]["group"] == 4                       # the split map, then its group 1
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert "map_topic first" in tools["map_topic"].description or "first" in tools["map_topic"].description

            r = await c.call_tool("find_people", {"name": "reef"})
            assert not r.is_error and "**Ann Reef** (A1): 4 papers · cited 100 times · h 3 · 2010–2024 · Coral Bleaching · Reef University" in text(r)
            r = await c.call_tool("find_people", {"name": "reef", "kind": "institution"})
            assert "**Reef University** (I7): 4 papers · education · AU" in text(r)
            r = await c.call_tool("get_profile", {"id": "A1"})
            out = text(r)
            assert out.startswith("# Ann Reef (A1)\n4 publications · cited 100 times · h-index 3 · 2010–2024")
            assert "Has published from: Reef University (3 papers, 2010–2024" in out
            assert "1. **Marine Biology**: 4 papers, mostly 2010–2024 — 1.1 Coral Bleaching (4)" in out
            assert "### Most cited (40 in all)" in out and "**Paper 0** · 2010 · Ann Reef; Bo Fish · cited 40 times · W1000" in out
            assert "- **Bo Fish** (A2, 3 papers in all): 2 papers together, 2010–2015; most cited together: Paper 0" in out
            assert "/authors/A1?read=1&list=cited&offset=1&limit=25" in out
            r = await c.call_tool("get_profile", {"id": "A1", "list": "cited", "group": "1", "offset": 8})
            assert text(r).startswith("# Most cited: items 9–10 of 40") and "read=1&list=cited&offset=10" in text(r)
            r = await c.call_tool("get_profile", {"id": "A1", "list": "partners"})
            assert r.is_error and "not a list of an author" in text(r)
            r = await c.call_tool("get_profile", {"id": "X9"})
            assert r.is_error and "find_people" in text(r)

            prompts = {p.name for p in (await c.list_prompts()).prompts}
            assert prompts == {"field_overview", "literature_review", "check_claim"}
            p = await c.get_prompt("field_overview", {"topic": "microplastics"})
            assert "read_topic_group" in p.messages[0].content.text
            p = await c.get_prompt("check_claim", {"claim": "coffee stunts growth"})
            assert "coffee stunts growth" in p.messages[0].content.text
            res = (await c.list_resources()).resources
            assert str(res[0].uri).endswith("/llms.txt")
            assert (await c.read_resource(res[0].uri)).contents[0].text == "# guide"
    anyio.run(go)


@pytest.fixture(scope="module")
def url():
    api, _ = fake_api()
    app = FastAPI()
    mcp_server.mount(app, api)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    t.join(5)


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28", "auto"])
def test_over_http_in_every_protocol_era(url, mode):
    async def go():
        async with Client(url, mode=mode) as c:
            assert c.protocol_version == ("2025-11-25" if mode == "legacy" else "2026-07-28")
            assert len((await c.list_tools()).tools) == 8
            if mode != "2026-07-28":   # (a pinned version skips discovery, so has no capabilities to read)
                caps = c.session.server_capabilities   # nothing to listen for: no open-ended subscriptions/listen
                assert caps.tools is not None and not caps.tools.list_changed and not caps.resources.subscribe
            r = await c.call_tool("search_papers", {"query": "coral bleaching", "limit": 1})
            assert not r.is_error and "W2782456798" in text(r)
    anyio.run(go)

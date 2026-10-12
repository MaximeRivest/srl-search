"""The search as an MCP server: AI apps (Claude, ChatGPT, Cursor, Codex, ...) add one address and get the tools.

    POST /mcp      Streamable HTTP, stateless, JSON answers. Every protocol revision the SDK speaks: the
                   initialize handshake (2024-11-05 ... 2025-11-25) and the per-request 2026-07-28 one.
                   Public: https://api.sciencemadereadable.com/mcp (the door, sciencemadereadable/app/server.py,
                   counts tool calls against the same 30 a minute as the JSON API).

Tools (read-only, Markdown answers written for a model to read and cite):
    search_papers        by meaning or exact words, with filters
    get_papers           full records of up to 20 papers (DOI, W-number, PMCID, URLs)
    find_similar_papers  closest in meaning to one paper
    citation_walk        what the papers sharing its references cite that it does not
Also: the guide (llms.txt) as a resource, and two prompts (literature review, check a claim).

Why Markdown and not structured JSON: the answers go into a model's context, and the spec asks a tool that
returns structured content to repeat it as JSON text, which costs about three times the tokens for the same
facts. Programs that want JSON have the HTTP API. The tools call the same functions as the HTTP routes
(srl_search/app.py), so both always answer the same.
"""
from __future__ import annotations

import html
import re
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anyio
from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import Field

from srl_search.public import PUBLIC_BASE
from srl_search.scope import SCHOLARLY_TYPES

SITE = "https://sciencemadereadable.com"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
PaperType = Literal[tuple(sorted(SCHOLARLY_TYPES))]  # type: ignore[valid-type]
Abstracts = Literal["short", "full", "none"]
SHORT = 320   # characters of abstract in "short" mode: about two sentences, enough to judge relevance


def clean(text: str | None) -> str:
    """Abstracts can hold HTML or JATS tags and entities; a model reads plain text better."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def shorten(text: str, n: int = SHORT) -> str:
    if len(text) <= n:
        return text
    cut = text[:n]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > n * 0.6 else cut[:cut.rfind(" ")] + " …"


def authors(r: dict, n: int = 3) -> str:
    names = [a.strip() for a in (r.get("authors") or "").split("|") if a.strip()]
    total = r.get("authors_count") or len(names)
    if not names:
        return ""
    return "; ".join(names[:n]) + (f"; et al. ({total} authors)" if total > n else "")


def link(r: dict) -> str:
    return f"https://doi.org/{r['doi']}" if r.get("doi") else r.get("openalex") or f"https://openalex.org/{r['id']}"


def readable_link(r: dict) -> str | None:
    ref = r.get("doi") or r.get("pmcid")
    return f"{SITE}/?paper={ref}" if r.get("smr") and ref else None


def result_block(i: int, r: dict, abstracts: str) -> str:
    """One search hit: what a model needs to judge it, cite it and ask for more."""
    facts = [r.get("venue") or "", authors(r), r.get("type") or "",
             f"cited {r['citations']:,} times" if r.get("citations") else "not cited yet",
             "open access" if r.get("is_oa") else ""]
    lines = [f"{i}. **{clean(r.get('title')) or '(no title)'}** ({r.get('year') or 'n.d.'})",
             "   " + " · ".join(f for f in facts if f),
             f"   {r['id']} · {link(r)}"]
    if readable_link(r):
        lines[-1] += f" · plain-language version: {readable_link(r)}"
    if (r.get("junk") or 0) >= 0.5:
        lines.append(f"   ⚠ probably not a scholarly paper (junk score {r['junk']:.2f})")
    text = clean(r.get("abstract"))
    if abstracts != "none":
        if text:
            lines.append("   " + (text if abstracts == "full" else shorten(text)))
        else:
            lines.append("   (no abstract in the record)")
    return "\n".join(lines)


def record_block(d: dict) -> str:
    """One paper in full, for get_papers."""
    head = f"## {clean(d.get('title')) or '(no title)'}"
    if d.get("is_retracted"):
        head += "  ⚠ RETRACTED"
    venue = " ".join(x for x in (d.get("venue"), f"({d['publisher']})" if d.get("publisher") else "") if x)
    facts = [str(d.get("date") or d.get("year") or ""), d.get("type") or "", venue,
             f"cited {d['citations']:,} times" if d.get("citations") is not None else ""]
    oa = ("open access" + (f" ({d['oa_status']})" if d.get("oa_status") else "")) if d.get("is_oa") else "not open access"
    if d.get("license"):
        oa += f", licence {d['license']}"
    ids = [d["id"], f"https://doi.org/{d['doi']}" if d.get("doi") else "", d.get("pmcid") or ""]
    lines = [head, " · ".join(f for f in facts if f) + " · " + oa,
             "Authors: " + (authors(d, 12) or "not listed"),
             "Identifiers: " + " · ".join(i for i in ids if i) + f" · {d['openalex']}"]
    if d.get("pdf_url") or d.get("url"):
        lines.append("Links: " + " · ".join(x for x in (d.get("url"), f"PDF {d['pdf_url']}" if d.get("pdf_url") else "") if x))
    lines.append(f"On ScienceMadeReadable (map of science, citations): {SITE}/?work={d.get('doi') or d['id']}"
                 + (f" · plain-language version: {readable_link(d)}" if readable_link(d) else ""))
    if d.get("keywords") or d.get("mesh"):
        lines.append("Subjects: " + "; ".join(x for x in (d.get("keywords", "").replace(" | ", ", "),
                                                          ("MeSH: " + d["mesh"].replace(" | ", ", ")) if d.get("mesh") else "") if x))
    text = clean(d.get("abstract"))
    lines += ["", text or "(no abstract in the record)"]
    return "\n".join(lines)


async def call(fn, *args, **kwargs):
    """Run one of the HTTP routes' functions off the event loop (they are blocking: 50 ms to 5 s), turning
    their HTTP errors into messages the model can act on."""
    try:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    except HTTPException as e:
        raise ToolError(str(e.detail)) from None


def build(api) -> MCPServer:
    """The MCP server over the search service in `api` (the srl_search.app module, already loaded)."""
    meaning = api.SEMANTIC["all"].manifest
    words = api.KEYWORDS.num_docs if api.KEYWORDS else None
    # (the index's earliest date is a stray record's, so the coverage is said in papers, not years)
    coverage = (f"{meaning['papers'] / 1e6:.1f} million papers searchable by meaning"
                + (f", {words / 1e6:.1f} million by exact words and by identifier" if words else ""))
    server = MCPServer(
        name="sciencemadereadable-science-search",
        title="Science search (ScienceMadeReadable)",
        version=api.app.version,
        website_url=f"{SITE}/?api",
        description="Search about 200 million scientific papers by meaning or exact words, read their records, "
                    "find similar papers and walk their citations. Free, read-only, no key.",
        log_level="WARNING",
        # The tools, prompts and guide never change while the server runs, so there is nothing to subscribe to.
        # Served, subscriptions/listen is an open-ended request: in JSON mode it would hold a connection through
        # the door and the relay until their timeouts, for every connected app (Claude Code opens one at start).
        subscriptions=False,
        instructions=f"""Scientific literature search over OpenAlex titles and abstracts: {coverage}. English abstracts, no full texts.

How to search well:
- Rewrite casual questions in the language of a paper before searching ("how to lose weight" -> "effectiveness of dietary and exercise interventions for weight loss in adults"). This is the biggest quality gain.
- Questions and descriptions: search by meaning (the default). Exact names, genes, drugs, acronyms, phrases: quotes or AND / OR / NOT, which search exact words. For a review, do both.
- Screen with abstracts="short" or "none", then get_papers for the ones you keep. Snowball from the best 2-3 with find_similar_papers and citation_walk.
- Judge papers from the record: reviews vs single studies, citations relative to age, venue. A record flagged as junk is probably not scholarship. Citation counts are not proof that a paper is right.
- Cite the DOI link when there is one, else the OpenAlex link, copied from these results. Never invent details a record lacks.
- Each tool call counts against 30 a minute per address. One search with a larger limit beats several small ones.""",
    )

    @server.tool(title="Search scientific papers", annotations=READ_ONLY, structured_output=False)
    async def search_papers(
        query: Annotated[str, Field(min_length=2, max_length=2000, description=(
            "What to find. A question or description in research language searches by meaning "
            "(\"effects of screen time on sleep in toddlers\"). Search syntax searches exact words: \"exact phrase\", "
            "AND, OR, NOT or -word, brackets, word*, \"a b\"~5 (within 5 words), title:, abstract:, keywords:."))],
        limit: Annotated[int, Field(ge=1, le=100, description="Number of results, best first. No paging: ask for up to 100 at once.")] = 10,
        mode: Annotated[Literal["auto", "semantic", "keyword"], Field(description=(
            "auto (default): exact words when the query uses search syntax, else by meaning. semantic: by meaning, "
            "finds papers about the idea in other words. keyword: exact words (English stemming), with an exact count."))] = "auto",
        year_from: Annotated[int | None, Field(ge=1500, le=2100, description="Earliest publication year.")] = None,
        year_to: Annotated[int | None, Field(ge=1500, le=2100, description="Latest publication year.")] = None,
        publication_type: Annotated[PaperType | None, Field(description="Only this kind of publication, e.g. review for overviews.")] = None,
        open_access_only: Annotated[bool, Field(description="Only papers free to read.")] = False,
        readable_only: Annotated[bool, Field(description=(
            "Only papers ScienceMadeReadable can rewrite in plain language (open PubMed Central research articles, CC BY)."))] = False,
        abstracts: Annotated[Abstracts, Field(description=(
            "short (default): the first sentences, enough to judge relevance. full: whole abstracts (long). none: titles and facts only."))] = "short",
    ) -> str:
        """Search scientific papers (titles and abstracts) by meaning or by exact words. Returns each paper's title,
        year, venue, authors, citation count, identifier (W-number), DOI link and abstract. Retracted papers are
        left out."""
        out = await call(api.search, q=query, k=limit, mode=mode, collection="smr" if readable_only else "all",
                         year_from=year_from, year_to=year_to, type=publication_type, open_access=open_access_only,
                         license=None, pmc_open_access=False, abstracts=abstracts != "none", nprobe=api.NPROBE,
                         depth=api.DEPTH, map=False, junk_penalty=None)
        if out["mode"] == "keyword":
            how = f"Exact words, understood as `{out['query']}`: {out['matches']:,} papers match."
        elif out.get("fallback"):
            how = (f"No paper contains all of `{out['fallback']['query']}`, so these were found by meaning "
                   "(closest in what they are about).")
        else:
            how = "Found by meaning: closest in what they are about, best first."
        if not out["results"]:
            return (f"{how}\nNothing found. Try other words, fewer filters, or a question phrased like a paper's "
                    "title; for exact words, fewer of them or OR between alternatives.")
        blocks = [result_block(i, r, abstracts) for i, r in enumerate(out["results"], 1)]
        tail = "\n\n(Abstracts shortened: get_papers gives them in full.)" if abstracts == "short" else ""
        return f"{how} {len(blocks)} results.\n\n" + "\n\n".join(blocks) + tail

    @server.tool(title="Get papers by identifier", annotations=READ_ONLY, structured_output=False)
    async def get_papers(
        ids: Annotated[list[Annotated[str, Field(min_length=2, max_length=300)]], Field(min_length=1, max_length=20, description=(
            "Up to 20 papers: W-numbers from search results (W2782456798), DOIs (10.1126/science.aan8048, "
            "https://doi.org/...), PMCIDs (PMC1234567) or OpenAlex URLs."))],
    ) -> str:
        """Full records of up to 20 papers: title, date, venue, all identifiers, authors, citations, open-access
        status and links, subjects, and the whole abstract. Use it for the papers you keep after a search, or to
        resolve a DOI. Covers every paper in the corpus, all years."""
        def one(ref: str) -> str:
            try:
                return record_block(api.work(ref))
            except HTTPException:
                return f"## {ref}\nNot found. Check the identifier, or search for the title with search_papers."
        blocks = [await anyio.to_thread.run_sync(one, ref.strip()) for ref in dict.fromkeys(ids)]
        return "\n\n".join(blocks)

    @server.tool(title="Find similar papers", annotations=READ_ONLY, structured_output=False)
    async def find_similar_papers(
        paper: Annotated[str, Field(min_length=2, max_length=300, description="The paper: W-number, DOI or PMCID.")],
        limit: Annotated[int, Field(ge=1, le=100, description="Number of results, most similar first.")] = 10,
        year_from: Annotated[int | None, Field(ge=1500, le=2100, description="Earliest publication year.")] = None,
        year_to: Annotated[int | None, Field(ge=1500, le=2100, description="Latest publication year.")] = None,
        publication_type: Annotated[PaperType | None, Field(description="Only this kind of publication.")] = None,
        open_access_only: Annotated[bool, Field(description="Only papers free to read.")] = False,
        abstracts: Annotated[Abstracts, Field(description="short (default), full or none, as in search_papers.")] = "short",
    ) -> str:
        """Papers closest in meaning to one paper (by their titles and abstracts), the paper itself left out.
        Good for "more like this" from a strong search result."""
        try:
            out = await call(api.similar, paper, k=limit, year_from=year_from, year_to=year_to, type=publication_type,
                             open_access=open_access_only, license=None, pmc_open_access=False,
                             abstracts=abstracts != "none", nprobe=api.NPROBE, depth=api.DEPTH, collection="all",
                             junk_penalty=None)
        except ToolError:
            raise ToolError(f"{paper} is not among the papers searchable by meaning: check the identifier with "
                            "get_papers. Papers without an English abstract cannot be compared by meaning.") from None
        blocks = [result_block(i, r, abstracts) for i, r in enumerate(out["results"], 1)]
        return f"Closest in meaning to {paper}, most similar first. {len(blocks)} results.\n\n" + "\n\n".join(blocks)

    @server.tool(title="Citation walk", annotations=READ_ONLY, structured_output=False)
    async def citation_walk(
        paper: Annotated[str, Field(min_length=2, max_length=300, description="The starting paper: W-number or DOI.")],
        breadth: Annotated[int, Field(ge=5, le=100, description=(
            "How many papers sharing its references to learn from (default 30). More finds more, takes longer."))] = 30,
        limit: Annotated[int, Field(ge=1, le=50, description="Papers listed per section.")] = 12,
    ) -> str:
        """Walk the citation graph from one paper: the papers that cite most of the same references (its closest
        neighbours in the literature), and what those neighbours cite that this paper does not: candidates for
        "what to read next" or "what it should also have cited". Takes 0.5-5 seconds."""
        out = await call(api.walk, id=paper, k=breadth, max_refs=300, max_cites=500)
        if not isinstance(out, dict):        # an error from the citation service, passed on as an HTTP answer
            body = getattr(out, "body", b"")
            raise ToolError(f"No citation walk for {paper}: " + (body.decode(errors="replace") if body else "unavailable")
                            + ". It needs a paper with references in the citation index.")
        if out.get("error"):
            raise ToolError(f"No citation walk for {paper}: {out['error']}")
        nodes, cols, stats = out["nodes"], out["columns"], out.get("stats", {})
        wanted = {w for c in ("similar", "candidates", "candidates_out") for w in cols.get(c, [])[:limit]} | {out["focus"]}
        details = await anyio.to_thread.run_sync(lambda: api.engine.details([int(w[1:]) for w in wanted]))

        def line(w: str, extra: str) -> str:
            n, d = nodes.get(w, {}), details.get(int(w[1:])) or {}
            r = {**d, "id": w, "openalex": f"https://openalex.org/{w}"}
            facts = [extra, f"cited {n['cited_by']:,} times" if n.get("cited_by") else "", n.get("type") or ""]
            return (f"- **{clean(n.get('title') or d.get('title')) or '(no title)'}** ({n.get('year') or 'n.d.'}) · "
                    + " · ".join(f for f in facts if f) + f" · {w} · {link(r)}")

        f, fd = nodes[out["focus"]], details.get(int(out["focus"][1:])) or {}
        k = stats.get("k", breadth)
        parts = [f"# Citation walk from: {clean(f.get('title'))} ({f.get('year')}) · {out['focus']} · "
                 f"{link({**fd, 'id': out['focus']})}",
                 f"It lists {f.get('refs_reported') or 0} references ({stats.get('refs_indexed', '?')} in the index). "
                 f"{stats.get('papers_sharing_a_reference', 0):,} papers share at least one of them; the {k} sharing "
                 "the most are its neighbours below."]
        answers = set(out.get("answers") or [])
        cand = [w for w in cols.get("candidates", [])][:limit]
        if cand:
            parts += ["", f"## Cited by its neighbours, not by this paper (most cited by the {k} neighbours first)"]
            parts += [line(w, f"cited by {nodes[w].get('support', '?')} of the {k} neighbours"
                           + (" · strongest candidate" if w in answers else "")) for w in cand]
        else:
            parts += ["", "No paper is cited by several of its neighbours that it does not cite itself."]
        classics = cols.get("candidates_out", [])[:limit]
        if classics:
            parts += ["", "## Also cited by its neighbours, but very widely cited (over 500 citations: classics that "
                          "much of the field cites)"]
            parts += [line(w, f"cited by {nodes[w].get('support', '?')} of the {k} neighbours") for w in classics]
        sim = cols.get("similar", [])[:limit]
        if sim:
            parts += ["", "## Its neighbours: papers citing most of the same references"]
            parts += [line(w, f"shares {nodes[w].get('shared', '?')} references") for w in sim]
        parts += ["", f"See it drawn on the map of science: {SITE}/?work={fd.get('doi') or out['focus']}"]
        return "\n".join(parts)

    @server.resource(f"{PUBLIC_BASE}/llms.txt", name="guide", title="Search API guide",
                     description="Every call, what is covered today, result fields, limits and how to search well.",
                     mime_type="text/markdown")
    def guide() -> str:
        return api.guide().body.decode()

    @server.prompt(title="Literature review")
    def literature_review(question: Annotated[str, Field(description="The research question.")]) -> str:
        """Find and summarise what the research says about a question, with citations."""
        return f"""Review the scientific literature on this question, with the science search tools: {question}

1. Restate the question in the language a paper would use; note 2-3 alternative phrasings and the key exact terms (names, acronyms).
2. Search by meaning with each phrasing (limit 30, abstracts "short"), and by exact words for the key terms. Add a search with publication_type "review".
3. Snowball from the 2-3 most relevant papers with find_similar_papers and citation_walk.
4. Read the most relevant 10-20 in full with get_papers.
5. Write the answer: what the evidence shows, how strong it is (reviews vs single studies, sample sizes and designs stated in the abstracts), where studies disagree, and what is not known. Number each claim's source [1] and end with a References list of DOI links (else OpenAlex links) copied from the results. Say plainly that this is based on titles and abstracts, not full texts."""

    @server.prompt(title="Check a claim")
    def check_claim(claim: Annotated[str, Field(description="The claim to check, e.g. 'coffee stunts growth'.")]) -> str:
        """Check what the scientific literature says about a claim."""
        return f"""Check this claim against the scientific literature, with the science search tools: {claim}

Search for evidence both for and against it, phrased as research questions (search by meaning), plus a search for reviews. Read the strongest papers with get_papers. Then give a verdict (supported, contradicted, mixed, or not studied), the strength of the evidence, and the key papers, citing each with its DOI link copied from the results. Do not stretch what an abstract says."""

    return server


def mount(app: FastAPI, api) -> MCPServer:
    """Serve the MCP endpoint at /mcp inside the search API's FastAPI app."""
    server = build(api)
    # Stateless JSON: every request stands alone (no session to keep, any worker or restart can answer it), and
    # a plain JSON answer passes through the public door, which buffers. The cost: no progress notifications
    # during a long call (a citation walk takes up to 5 s, so nothing to report). DNS-rebinding protection
    # guards servers that hold something private behind localhost; this one is public and read-only, and is
    # reached under several host names (127.0.0.1, the tailnet name, the door), so it is off.
    asgi = server.streamable_http_app(streamable_http_path="/mcp", json_response=True, stateless_http=True,
                                      transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    app.router.routes.extend(asgi.routes)
    before = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(a):
        async with before(a), server.session_manager.run():
            yield

    app.router.lifespan_context = lifespan
    return server

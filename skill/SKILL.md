---
name: scholars-search
description: Search ~200 million scientific papers (OpenAlex titles + abstracts) by meaning or exact words, look papers up by DOI / OpenAlex id / PMCID, find similar papers, walk citations, and map the research groups inside a topic with reading lists for each, through Maxime's own search API. Use when the task needs scientific literature - finding studies or evidence on a question, a literature review, checking what research says, citing papers, resolving a DOI to title/authors/abstract, building a reading list, getting an overview of a field, or code that calls this API (ScienceMadeReadable, Scholars Reading List).
---

# Scholars Reading List search API

Read-only JSON over HTTP, no key. Full guide (coverage, every parameter, field meanings, limits):
`GET <base>/llms.txt`. Interactive reference: `<base>/docs`, spec: `<base>/openapi.json`.

## Which address

| where your code runs | base | limit |
|---|---|---|
| on lambda (`hostname` = lambda) | `http://127.0.0.1:8810` | none (be reasonable) |
| another tailnet machine (XPSwhite, phone) | `https://lambda.tail69222b.ts.net:18810` | none |
| anywhere else, or code that ships to users | `https://api.sciencemadereadable.com/api/data` | 30 requests/min per address |

Same data and routes everywhere. Start with `curl -s <base>/llms.txt` when you need details.

## The six calls

```sh
B=http://127.0.0.1:8810
curl -sG "$B/search" --data-urlencode "q=effect of intermittent fasting on weight loss in adults" -d k=20 -d abstracts=false
curl -sG "$B/search" --data-urlencode 'q="gut microbiome" AND (autism OR ASD) NOT mice' -d k=20 -d abstracts=false
curl -s  "$B/works/10.1126/science.aan8048"          # DOI, W123..., PMC123..., or a doi.org / openalex.org URL
curl -s  "$B/similar/W2782456798?k=10&abstracts=false"
curl -s  "$B/walk?id=10.1126/science.aan8048"         # 'answers' = what similar papers cite and this one doesn't
curl -s  "$B/topics/intermittent fasting for weight loss"   # research groups + what to read in each (plain text)
```

**Start with the landscape.** `/topics/{question}` maps the 1,000 closest papers into named research groups and
gives every group's full reading lists (`start_here`, `core`, `rising`, `distinctive`, `bridges`, `newest`) and
experts; `?group=N` splits a group (5 requests; 3-30 s the first time, then cached). Then read each group that
bears on the question in depth (1 request each): `?read=N&part=lists` (the lists with full abstracts), then
`?read=N&part=papers&page=1, 2, ...` (all its papers with abstracts, most central first). Everything comes in pages of
~16-20k tokens, never cut: each answer says what to call next (the map too: `?page=2`). You can hold
hundreds of abstracts: read them, and answer organised by group, citing what you read.
For a thorough review of a whole field, add `scope=field`: every paper of the field (often 10-50k, including classics
without abstracts found through the field's citations), in groups and pre-split subgroups. The first request starts
a ~15-40 minute background job (HTTP 202 with progress); ask again later, then it is instant.

## As MCP tools instead

Apps that speak MCP can add `https://api.sciencemadereadable.com/mcp` (no key) and get the same search as tools:
`search_papers`, `get_papers`, `find_similar_papers`, `citation_walk`, `map_topic`, `read_topic_group` (and a
`field_overview` prompt). Claude Code:
`claude mcp add --transport http --scope user science https://api.sciencemadereadable.com/mcp`. If those tools are
already connected, use them rather than curl: same data, answers ready to cite.

## Doing it well

- Questions and descriptions → by meaning (default). Exact names, genes, drugs, phrases, "everything
  mentioning X" → quotes / AND / OR (switches to exact words automatically). Reviews: run both, merge by `id`.
- Rewrite casual questions into research language first ("how to loose weight" → "effectiveness of dietary
  and exercise interventions for weight loss in adults"): biggest quality gain.
- Screen with `abstracts=false`, then `/works/{id}` for the papers you keep. Snowball with `/similar` and `/walk`.
- A field, a topic, a reading list, "what does research say": map it first (`/topics`), read every relevant group in
  full (`read=N`), then search inside groups and walk citations. Never stop at a flat list of hits.
- Search by meaning does not reach the oldest years yet (llms.txt says how far); exact words and `/works`
  cover all 203.5M papers. English abstracts only, no full texts.
- `topic` / `field` labels are often wrong (stale OpenAlex snapshot): judge from title and abstract.
- Cite DOI when present, else the `openalex` URL. Never invent details a record lacks.
- 502/503 for a minute or two = nightly data update or restart: retry later.

## Citing in an answer

Chattering (and anything reading the same format) turns this into paper cards with the real title,
authors, summary and links, and flags a DOI that does not exist or a retracted paper:

```markdown
Semaglutide gave about 15% weight loss [1]; tirzepatide about 20% [2][3].

## References
- [1] https://doi.org/10.1056/nejmoa2032183 — Wilding 2021, NEJM
- [2] https://doi.org/10.1056/nejmoa2206038 — Jastreboff 2022, NEJM
- [3] https://doi.org/10.1056/nejmoa2416394 — Aronne 2025, NEJM
```

- The number goes right after the claim it supports. Number in order of first use; a paper keeps its number.
- One line per number, the DOI link first (else the OpenAlex link). The label after the dash is optional.
- Copy every identifier from a record you fetched in this conversation, never from memory.

Source and maintenance: `~/Projects/scholarsreadinglist/search` (README.md, `dev.sh` to change it safely).

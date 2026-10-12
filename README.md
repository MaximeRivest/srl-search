# Scholars Reading List · search

Search over ~203M English title+abstract records (OpenAlex; by meaning with DenseOn embeddings, by exact
words with tantivy), served as one HTTP API. First user: ScienceMadeReadable.

**Using the API** (people, programs, AI agents): [web/llms.txt](web/llms.txt), served live with current
coverage at **https://api.sciencemadereadable.com/api/data/llms.txt**; interactive reference at
`.../api/data/docs`, spec at `.../api/data/openapi.json` (public routes only). Coding agents on Maxime's
machines get it as the `scholars-search` skill ([skill/SKILL.md](skill/SKILL.md), linked into
`~/.pi/agent/skills`, `~/.claude/skills`, `~/.agents/skills`). An MCP server is not built (llms.txt covers
the same ground for any agent). When you change a route or parameter, update llms.txt and the skill:
`tests/test_public.py` checks that every documented route is passed by the public door.

- [../DATA.md](../DATA.md): **the data releases the site uses** (what, where, nightly update, publish,
  rollback). Since 2026-10-09 the service runs from a release, declared in
  `~/Projects/os/machines/hm/scholarsreadinglist-data.nix`; the run commands below are history.
- [PLAN.md](PLAN.md): steps, measurements, timeline, update pipeline.
- [hosting-lesson.md](hosting-lesson.md): how hosting on lambda works (systemd, cgroups, NixOS, tunnels).
- `experiments/`: measured design questions, each script with its log.

## The search API (running since 2026-10-06)

Addresses: public `https://api.sciencemadereadable.com/api/data` (30 requests/min per address for programs, through the
site's door: `sciencemadereadable/app/server.py` `DATA_ROUTES`), tailnet `https://lambda.tail69222b.ts.net:18810`,
local `http://127.0.0.1:8810`. What it covers changes with each data release: `/health` (tailnet/local) or the
top of `/llms.txt`. The first index (2026-10-06) covered 2024+ only (63.8M papers); the measurements below
are from it.

```sh
curl -G http://127.0.0.1:8810/search --data-urlencode "q=sleep apnea and atrial fibrillation" -d k=5
curl -G .../search --data-urlencode "q=CRISPR base editing sickle cell" -d pmc_open_access=true -d license=cc-by -d type=article
curl .../similar/10.1126/science.aan8048?k=5        # by DOI, W-id or PMCID
curl .../works/PMC5904187
curl .../health
```

Filters: `year_from`, `year_to`, `type`, `open_access`, `license`, `pmc_open_access`
(the article's full text is in our PMC mirror), `abstracts=false`. Retracted works and
paratext (covers, tables of contents) are left out unless asked for. Each result carries
OpenAlex id, DOI, PMCID, title, authors (first 12 + count), venue, year, type, licence,
open-access status, citations, links, abstract, a similarity score and `junk` (how likely the record is spam
or not scholarship). `timing` shows where the
time went.

How a search works: DenseOn encodes the question on CPU (`query:` prompt) → exact comparison
with the 16,384 list centres → the 64 closest lists are scanned with 1-bit RaBitQ codes
(7.2 GB in RAM) → the best 300 are re-scored exactly with the original float16 vectors, read in
parallel from the embedding shards (no copy) → details from SQLite (72.6 GB, by id/DOI/PMCID).
Default: 128 lists, rerank 300. Measured end to end: ~61 ms median, 68 ms worst of 8.

Accuracy (`experiments/eval-2024plus.json`): share of the exact top 10 (checking all 63.8M
papers) that the index returns, 440 questions:

| Lists searched | Rerank | All | SciFact | NFCorpus | SciDocs | TREC-COVID | Plain questions | Search p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 300 | 0.81 | 0.80 | 0.63 | 0.88 | 0.94 | 0.98 | 17 / 26 ms |
| 64 | 300 | 0.85 | 0.86 | 0.69 | 0.92 | 0.95 | 0.98 | 34 / 82 ms |
| **128** | **300** | **0.89** | 0.91 | 0.74 | 0.95 | 0.97 | 0.99 | 37 / 70 ms |
| 256 | 300 | 0.91 | 0.94 | 0.77 | 0.96 | 0.97 | 0.99 | 62 / 133 ms |

Search times exclude question encoding (~50-80 ms on 4 cores). NFCorpus (short, vague
nutrition queries) is the weak spot.

Running it (transient user unit: survives logout, not a reboot; to be declared in NixOS.
Until then a reboot leaves the site on its Europe PMC fallback):

```sh
systemd-run --user --unit=srl-search -p CPUWeight=1000 -p MemoryMax=24G -p LimitNOFILE=65536 \
  -p Restart=on-failure -p WorkingDirectory=$PWD -E TOKENIZERS_PARALLELISM=false \
  -E SRL_SMR=/mnt/fast/scholarsreadinglist/search/smr-2026-10-08 \
  -E SRL_CORPUS=/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05 \
  -p ExecStartPre="-/run/current-system/sw/bin/tailscale serve --bg --https=18810 http://127.0.0.1:8810" \
  $PWD/.venv/bin/uvicorn srl_search.app:app --host 127.0.0.1 --port 8810
```

Building (2024+ took 48 min for the index, 30 min for the details on a loaded machine):

```sh
uv run python build_index.py --min-date 2024-01-01 --out /mnt/fast/scholarsreadinglist/search/index-2024plus
uv run python build_metadata.py --min-date 2024-01-01 --out /mnt/fast/scholarsreadinglist/search/meta-2024plus.sqlite
uv run python eval_index.py --index ... --meta ... --out experiments/eval-2024plus.json
```

Lessons from the first build:

- Exact list assignment of 64M papers against 16k centres is ~1.6e15 flops: the build uses an
  HNSW router (57k papers/s). At search time that router, at its default efSearch=16, missed the
  single best list for 17-20% of questions; the service routes exactly instead (~1-3 ms).
- OpenMP's thread count is per calling thread: the model and FAISS must not be called from web
  worker threads without setting it (the encoder runs on one dedicated thread).
- Reading 300 scattered vectors through mmap page faults took ~1 s cold; parallel `pread`
  takes 10-60 ms.
- 2.9M of the 63.8M rows (4.6%) are exact duplicate texts (BacDive strain templates x65,210,
  DEPRECATED Zenodo records, journal covers). Results collapse identical records; the next
  build should index each distinct text once and leave boilerplate out.

## MCP server (since 2026-10-12)

`POST /mcp` (`srl_search/mcp_server.py`): the same search for AI apps, public at
`https://api.sciencemadereadable.com/mcp`, tailnet `https://lambda.tail69222b.ts.net:18810/mcp`, local
`http://127.0.0.1:8810/mcp`. Four read-only tools (`search_papers`, `get_papers`, `find_similar_papers`,
`citation_walk`) that call the HTTP routes' own functions, two prompts (`literature_review`, `check_claim`), the
guide as a resource, and instructions the app reads on connecting (how to search well, how to cite).

```sh
claude mcp add --transport http --scope user science https://api.sciencemadereadable.com/mcp
codex mcp add science --url https://api.sciencemadereadable.com/mcp
```

Choices, and what they cost:
- **Official SDK** (`mcp` 2.x, `MCPServer`), not a hand-written protocol: it speaks every revision (the initialize
  handshake 2024-11-05 ... 2025-11-25, and the per-request 2026-07-28) and keeps up with the spec. Cost: 17 more
  packages in the environment.
- **Stateless, JSON answers**: no session to keep, so a restart or the nightly release never breaks a connected app,
  and one request gets one complete answer through the door (which buffers). Cost: no progress messages during a
  call; the slowest, a citation walk, takes about 5 s. `subscriptions/listen` is not served (nothing here ever
  changes while running); served, it would hold a connection open per app until the relay's timeout.
- **Markdown answers, no structured output**: the reader is a model, and structured output must be repeated as
  JSON text, about three times the tokens for the same facts. Programs that want JSON use the HTTP API.
- **A small surface**: the tools leave out `nprobe`, `depth`, `junk_penalty`, `license`, `pmc_open_access` and map
  positions, which help programs more than models. Abstracts default to their first sentences (`abstracts`:
  short / full / none) so a 30-result screen stays around 6,000 tokens.
- **Limits at the door**: each `tools/call` counts against the programs' 30 a minute per address (the handshake,
  listing tools and reading the guide are free). Over it, the door answers with a tool error that says how long to
  wait, which the model reads and acts on; an HTTP 429 would break many apps' connection instead. The door logs
  the method, tool and app name of each MCP request (to see who uses it), never the arguments.

Tests: `tests/test_mcp.py` (no indexes: a stand-in search service; in process, and over HTTP in both protocol eras).
Checked live on 2026-10-12 with Claude Code (2026-07-28) and Codex (handshake) through a copy of the door.

## Ranking: junk sinks (since 2026-10-09)

Search by meaning ranks by similarity − `junk_penalty` × P(junk) (default 0.3 for `all`, 0 for the curated
`smr`; `junk_penalty=0` gives pure similarity, `ranking` in the answer says which). P(junk) comes from a
small model over record signals OpenAlex has but the details store lacked (expansion-pack flag, source type,
core/DOAJ lists, references, authors, the venue's habits) and the paper's own embedding:
`build_quality.py` (11 min per snapshot) → `fit_quality.py` → `srl_search/quality.py`. On 60 judged questions,
junk in the top 10 went from 27% to 1% (everyday-advice questions: 54% to 0%) and useful nDCG@10 from
0.57 to 0.78. Method, costs and what was tried instead (question rewriting): `experiments/quality/README.md`.
Exact-word results are not re-ranked. Titles marked "[RETRACTED]" / "Retraction Note:" are treated as
retracted. In auto mode, a query that looked like search syntax but matched nothing is searched by meaning.

## The shared corpus (since 2026-10-08)

One "paper universe" for every project (API, test page, ScienceMadeReadable, the MCP server):
the **203,486,035** works the embedding plan selected (English, non-empty abstract; same
snapshot, count checked equal). `/mnt/fast/scholarsreadinglist/search/corpus-2026-10-05/`:

| File | What | Size | Build |
|---|---|---:|---:|
| `parts/*.parquet` | selection + PMC flags, the input of the next two | 83 GB | 41 min |
| `works.sqlite` | details by W-number / DOI (NOCASE) / PMCID: title, authors (12 + count), venue, year, type, licence, OA, citations, links, topic, keywords, MeSH, abstract; `pmcid`, `pmc_oa` (full text in our mirror), `pmc_license`, `pmc_type`, `smr` (site can open it) | 330 GB | 64 min |
| `keywords/` | tantivy: title, abstract, keywords (OpenAlex keywords + MeSH), English stemming, positions; year/type/licence/OA/PMC/smr filters | 71 GB | 51 min |
| `vocabulary-keywords-mesh.parquet` | the 87,007 distinct OpenAlex keyword + MeSH terms with counts | small | 2 min |

`build_corpus.py parts|sqlite|keywords`. Semantic indexes and collections refer to the same
W-numbers, so every mode and collection shares one details store.

### Search modes

`/search?mode=auto` (default) picks **keyword** when the query uses search syntax, else
**semantic**. Keyword syntax (`srl_search/keyword.py`): plain words are ANDed; `"phrases"`,
`OR`, `NOT`/`-word`, brackets, `word*`, `"a b"~5`, `title:` / `abstract:` / `keywords:`. Answers
carry `matches` (exact count) and `query` (as understood). Keyword search covers all 203.5M
papers now; semantic `all` covers 2024+ until the full rebuild.

Measured warm over 203.5M (`all`) / the site's openable subset (`smr`): selective queries
4-140 ms; a single very common word 120-320 ms (`cancer`: 5.65M matches). The paratext and
retraction exclusions are applied to the returned page, not in the index (they cost 30-100 ms).
SQLite FTS5 was tried first on 3M papers: fine for selective queries, 250 ms for one common word
(it ranks every match); tantivy's top-k skipping and 36%-of-text index size scale better.

## Collection `smr`: what ScienceMadeReadable searches (live since 2026-10-08)

`/search?collection=smr&q=...` (also `/similar`, `/works`, `/health?collection=smr`).
Papers the site can open: in our PMC mirror with NCBI licence code **CC BY** and JATS
article-type **research-article**, DOI-matched to an embedded OpenAlex title+abstract.
2026-10-08 build (`/mnt/fast/scholarsreadinglist/search/smr-2026-10-08`): 3,944,966 PMC
candidates → 3,173,358 OpenAlex matches → **3,052,236 embedded** (publication dates from
2017-08; older ones arrive as the backfill continues).

IVF with 4,096 groups over the original fp16 vectors (scores exact), 512 groups searched:
99% of the exact top 15 for plain questions, 89% for NFCorpus-style terse queries; ~85 ms
end to end. Comparing all 3M papers took 360 ms (`experiments/smr_ivf.log`, `smr_hnsw.log`).

The site reaches it through its queue server's public door (`/api/search` in
`sciencemadereadable/app/server.py`: collection fixed to `smr`, `mode=auto`, 30 searches/min per
address, words not stored). Exact-word searches there cover openable papers of every year
(keyword index), searches by meaning those embedded so far and falls back to Europe PMC when it fails or takes over 7 s.

Refresh (after more embeddings): ~10 min build, then restart the service on the new folder.

```sh
uv run python build_smr.py --out /mnt/fast/scholarsreadinglist/search/smr-$(date +%F)
# then recreate srl-search with -E SRL_SMR=<new folder> (stopping a transient unit deletes it)
```

Known gap: on devices running Tailscale, the machine's name resolves to its private address,
and Chrome blocks/asks before a public page calls it, so those visitors get the Europe PMC
fallback (and "GPU offline"). The server answers the private-network preflight; the real fix
is a public hostname of our own (tunnel/relay, planned).

Data lives outside the repository:

- vectors: `/mnt/fast/openalex-embeddings/denseon-english-abstracts/` (`pipeline/embeddings`)
- metadata: `/mnt/bulk/openalex/embedding-inputs/denseon-complete/` (content table)
- full texts: `/mnt/bulk/pmc/oa-2026-10-05/` (`pipeline/pmc`)

## Experiments

### Topic tree vs automatic clusters as the index's groups (2026-10-06)

`experiments/topic_routing.py`, log alongside. 2.34M embedded 2024–2025 papers that carry
an OpenAlex topic label; 600 BEIR questions + 600 held-out papers as queries; exact
scoring inside the searched groups, so only the grouping differs.

| Grouping | Biggest groups | Questions: recall@10 / papers scanned | Papers as queries: recall@10 / scanned |
|---|---|---|---|
| k-means, 4,096 | top 1% hold 4% | 0.715 / 9.9k · 0.858 / 38k | 0.949 / 10.7k · 0.980 / 39k |
| OpenAlex topic centres (4,516), nearest centre | top 1% hold 39%, max 117k | 0.721 / 10.0k · 0.841 / 35k | 0.857 / 25.6k · 0.939 / 40k |
| Same centres, papers kept in labelled topic | max 209k | 0.543 / 9.3k · 0.661 / 35k | 0.386 / 17k · 0.620 / 73k |

A human-made tree routes biomedical questions about as well as k-means, but its groups
are very uneven (science is not evenly spread over topics), so typical searches scan
several times more papers for the same recall. Topic labels as group membership lose
a third of the results. LLM-written archetype papers would at best match the topic-centre
row. Decision: data-driven (k-means) groups for the index; a topic tree as a browsing,
filtering and evaluation layer on top.

### Aho-Corasick term counting (measured 2026-10-08, not built)

Vocabulary: the 87,007 OpenAlex keyword + MeSH terms. pyahocorasick on 200k abstracts:
~9,000 abstracts/s/core with whole-word checks (all 203.5M ≈ 12 min on 32 cores). Raw
substring matching is unusable ("ion", "men", "sed" top the counts); generic terms ("data",
"methods") dominate without weighting. Fits as a concept layer: per-paper tags, term × year
counts, autocomplete, phrase detection; not a replacement for the keyword index, which
matches any word and counts any phrase on demand.

## Changing the search API safely (dev → public)

Never edit `search/` itself: the public API runs it and restarts on its own (nightly release, crash, reboot).

```sh
search/dev.sh up            # a private copy on branch dev: ~/Projects/scholarsreadinglist/search-dev, :8811
# edit and commit in search-dev; search/dev.sh restart to load the change (~30 s)
sciencemadereadable/app/tools/dev.sh up --data dev   # the dev site on top of it
search/dev.sh promote       # tests, master := dev, restart the public API, smoke test, undo by itself on failure
search/dev.sh rollback      # the version before the last promote
search/dev.sh down          # frees its ~28 GB
```

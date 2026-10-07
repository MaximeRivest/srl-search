# Scholars Reading List · search

Semantic search over ~203M English title+abstract records (OpenAlex, DenseOn embeddings)
and the PubMed Central open-access full texts, served as an HTTP API and an MCP server.
First user: ScienceMadeReadable.

- [PLAN.md](PLAN.md): steps, measurements, timeline, update pipeline.
- [hosting-lesson.md](hosting-lesson.md): how hosting on lambda works (systemd, cgroups, NixOS, tunnels).
- `experiments/`: measured design questions, each script with its log.

## The search API (running since 2026-10-06)

**https://lambda.tail69222b.ts.net:18810** (tailnet only; local `http://127.0.0.1:8810`).
Covers every embedded English title+abstract published from 2024-01-01: **63,814,851 papers**.

```sh
curl -G https://lambda.tail69222b.ts.net:18810/search --data-urlencode "q=sleep apnea and atrial fibrillation" -d k=5
curl -G .../search --data-urlencode "q=CRISPR base editing sickle cell" -d pmc_open_access=true -d license=cc-by -d type=article
curl .../similar/10.1073/pnas.2532413123?k=5        # by DOI, W-id or PMCID
curl .../works/PMC13552904
curl .../health
```

Filters: `year_from`, `year_to`, `type`, `open_access`, `license`, `pmc_open_access`
(the article's full text is in our PMC mirror), `abstracts=false`. Retracted works and
paratext (covers, tables of contents) are left out unless asked for. Each result carries
OpenAlex id, DOI, PMCID, title, authors (first 12 + count), venue, year, type, licence,
open-access status, citations, links, abstract and a similarity score. `timing` shows where the
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

Running it (transient user unit: survives logout, not a reboot; to be declared in NixOS):

```sh
systemd-run --user --unit=srl-search -p CPUWeight=1000 -p MemoryMax=24G -p LimitNOFILE=65536 \
  -p Restart=on-failure -p WorkingDirectory=$PWD -E TOKENIZERS_PARALLELISM=false \
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

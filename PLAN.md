# From embeddings to a search service and MCP server

Revised 2026-10-06 after measurements on the real embeddings (61.4M of 203.5M done,
2026 → 2024-01, ~720 vectors/s ⇒ ~2.3 days left).

Priorities (Maxime, 2026-10-06): fast; light on RAM/disk/CPU so lambda keeps doing other
work; accuracy "pretty good", not maximal; growable as OpenAlex adds papers. First user:
the ScienceMadeReadable (SMR) site.

## What SMR needs (from `sciencemadereadable/app/web/src/paper.ts`)

Today the site searches Europe PMC from the browser:
`(q) AND OPEN_ACCESS:y AND HAS_FT:y AND LICENSE:"cc by" AND PUB_TYPE:"research-article"`,
15 results, each `{pmcid, doi, title, authors, journal, year}`. Only papers with a
PubMed Central full text and a CC BY licence can be opened; `server.py /api/check`
then checks licence/layout per PMCID. So our service must:

- return the same fields, **PMCID required**;
- search only the openable subset for SMR;
- leave `/api/check` as the final word (OpenAlex licence data can be wrong).

The content table has no PMCID column. OpenAlex `ids` in this snapshot has no `pmcid`
key either (only openalex/doi/mag/pmid); PMCIDs appear in `locations[].landing_page_url`.
Authoritative source: NCBI's PMC ID map (PMCID↔PMID↔DOI) and PMC open-access file list
(with licence), joined on PMID/DOI.

~~1–3M openable~~ (from OpenAlex PMC locations) was wrong: OpenAlex lists a PMC location
for only ~12% of PLOS ONE, which is 100% in PMC. Use NCBI's `PMC-ids.csv.gz` (DOI↔PMCID,
copy in `/mnt/bulk/pmc/`) + the OA inventory instead: **~5M CC BY/CC0 full texts** in the
PMC open-access files (9.23M × 54% CC BY sample).

## Measurements (2026-10-06, 16 CPU threads, nice 15)

| Step | Measured | Full-scale estimate |
|---|---|---|
| Read `locations` from raw works (PMCID) | ~0.5–0.8 s per 300k rows | 476M rows: ~25 min one process, minutes in parallel |
| Train IVF16384 + RaBitQ on 1M vectors | 459 s | IVF65536 on ~2.6M: ~1–1.5 h (unmeasured) |
| Add vectors (RaBitQ, HNSW router) | 190k/s | 203.5M: ~20 min |
| RaBitQ code size | 106 B/vector | 203.5M: ~21.5 GB |

Recall@10 vs exact search, 6.5M random embedded papers, 600 real BEIR questions
(SciFact/NFCorpus/SciDocs/TREC-COVID, 150 each), 4096 lists, exact rerank of candidates:

| Lists searched | Rerank 100 | Rerank 400 | p50 / p95, 1 thread |
|---|---:|---:|---|
| 0.39% | 0.654 | 0.665 | 1.0 / 2.5 ms |
| 0.78% | 0.710 | 0.723 | 1.8 / 3.1 ms |
| 1.56% | 0.744 | 0.762 | 3.1 / 5.0 ms |

Papers used as queries gave higher recall (0.82 at 0.39% of 16384 lists): real questions
are harder for IVF routing. The 1.2M benchmark (README) showed IVF-SQ8 losing only 2–5%
nDCG for comparable recall, because the missed papers are near-equivalents; not yet
re-checked at 203M.

## Design v1

Two indexes, one service.

1. **SMR index** (openable subset, ~1–3M): exact or HNSW over float16 vectors. A few GB,
   nearly exact results, a few ms. Built in minutes.
2. **Full index** (203.5M): IVF65536 (HNSW router) + RaBitQ codes (~21.5 GB, page cache,
   protected with `MemoryLow`), then exact rerank of the top ~200 from the float16 vectors
   already on `/mnt/fast` (read in place; no 312 GB copy, needs an id → shard/offset table).
3. **Metadata store**: id → OpenAlex ID, DOI, PMCID, title, authors (first few + count),
   venue, year, licence, citations. Embedded key-value/SQLite on `/mnt/fast`, size TBD
   (estimate 40–60 GB).
4. **Query encoder**: DenseOn with `query: ` prompt on CPU (~50 ms measured, float32);
   try ONNX/int8 and keep only if query vectors match closely.
5. **Service**: one HTTP API (`search`, `similar`, `works`) on 127.0.0.1, NixOS system
   service in `public.slice`, sandboxed (see `hosting-lesson.md`). MCP server as a thin
   layer over it. SMR calls `search?collection=smr`.

Runtime estimate (unmeasured at full scale): ~30 GB RAM mostly page cache, ~4 cores,
~50–100 ms per search with encoding dominating.

Trade-off taken: IVF instead of a disk graph (DiskANN-style). IVF builds in ~2 h, appends
new papers cheaply, and is simple; a graph index would find more of the exact top 10
(likely 0.9+) but takes ~a day to build and is harder to update. Upgrade path if quality
matters.

## Timeline

| Phase | Machine time | Gate |
|---|---|---|
| Embeddings finish + `verify` | ~2.3 days + ~1 h | running now |
| PMCID/licence table (NCBI lists + OpenAlex) | ~30–60 min | can start now |
| Metadata store | ~1 h | can start now (content table exists) |
| SMR index | minutes | PMCID table; old openable papers need their embeddings |
| Train + build full index | ~2 h | all embeddings (sample from every year) |
| Measure full index (recall, latency, RAM) | ~1–2 h | built index |
| Service, MCP, SMR hookup, NixOS units | development, in parallel with embedding | — |

Early preview possible on the 61M already embedded (2024–2026 only).

## Updates

1. Sync new OpenAlex works (snapshot or daily changes), rebuild content rows.
2. Compare `text_hashes`: embed only new/changed English abstracts (GPU, ~400–700/s).
3. Append to the full index with the existing centroids; tombstone changed/removed ids;
   upsert metadata; refresh the SMR subset.
4. Retrain centroids when the collection has grown ~30% or recall checks drift.
5. Each run is resumable and publishes a new index version atomically (old one keeps serving).

## Later (Maxime, 2026-10-06)

- Reuse existing paid services as relays/tunnels: a Cloudflare account and an OVH machine
  probably exist; find the order confirmations in his email via agent_browser. Decide
  whether double duty is wise then.
- Keyword/boolean search (Tantivy/BM25) and hybrid ranking.

## Full text we can host ourselves (checked 2026-10-06)

**OpenAlex**: has Grobid TEI XML for ~43M works (~20 TB) and PDFs for 50M+ (~250 TB),
flags `has_content.{pdf,grobid_xml}` in our snapshot (sample: ~13% of all works). Not
practical for us: downloads cost $10 per 1,000 files (free key ≈ 100/day; 43M ≈ $430k);
bulk sync is a paid add-on to an annual plan (price via sales). Most of it is not openly
licensed, so we could not republish it anyway.

**PubMed Central Open Access (NCBI, AWS open data, free, no login)**: the right source.
`s3://pmc-oa-opendata/PMCxxxx.v/` holds per-article `.xml` (JATS, what SMR already
reads), `.txt` (plain text with header), `.pdf`, images, and `metadata/PMC….json`
(pmid, doi, `license_code`, is_manuscript, is_historical_ocr). Daily S3 inventory under
`inventory-reports/` (copy of 2026-10-05 in `/mnt/bulk/pmc/inventory-2026-10-05/`).
The old FTP bulk tarballs were removed in Aug 2026.

- **9,230,596 articles** (latest versions; 9.39M version files).
- Licences, random 400: CC BY 54%, CC BY-NC-ND 14%, TDM-only 12%, CC BY-NC 10%,
  none 5%, CC BY-NC-SA 3%, CC0 1% ⇒ ~5M CC BY/CC0.
- Size: txt mean 46 KB (≈ 425 GB all, ≈ 230 GB CC BY); XML mean 106 KB (≈ 980 GB all).
  zstd on text ≈ 3× ⇒ ~140 GB all txt packed. HDD has 10 TB free.
- Download speed from lambda, 64 parallel: 282 files/s, 13 MB/s ⇒ ~9 h per file type
  (latency-bound; more parallelism likely faster).
- Updates: diff the daily inventory (LastModified/ETag) and fetch changed files.

Plan: mirror XML + metadata JSON for all 9.2M into packed Parquet/zstd shards on
`/mnt/bulk`, keyed by PMCID; derive text/sections from XML. Serve per licence:
CC BY/CC0 full text and rewrites; NC licences show/rewrite only if SMR counts as
non-commercial (donations: ask a lawyer); ND: verbatim only, no rewrite; TDM-only:
search/mining, never redistribute. Images keep coming from PMC's open S3.
Later option: chunk + embed full texts (~11k tokens/article ⇒ ~200M 512-token chunks,
≈ one more week of GPU, estimate).

### Coverage of PMC vs open-licence journals (2026-10-06)

CC BY/CC0 English articles of 2019 + 2023 in OpenAlex (2.73M), matched by DOI to
`PMC-ids.csv.gz`: **28% have PMC open-access full text**. Health sciences 51%, life
sciences 42%, physical sciences 16%, social sciences 9%. Journal-level: PLOS ONE,
Scientific Reports, Nature Communications, PeerJ, Frontiers in Immunology/Psychology,
BMC Public Health, Cureus, Molecules, Sensors: ~100%; eLife 94%. Zero or ~1%: MDPI
Sustainability/Applied Sciences/Remote Sensing/Energies, IEEE Access, Frontiers in Marine
Science/Physics, Physical Review *, JHEP. PMC takes only journals in biomedical/life
sciences scope. For the rest: arXiv (physics/CS/maths; LaTeX source and HTML), publishers'
own XML (MDPI, Frontiers publish JATS for CC BY articles), or CORE's aggregated full texts
(check licence/terms). Not yet measured.

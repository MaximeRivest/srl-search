# Scholars Reading List · search

Semantic search over ~203M English title+abstract records (OpenAlex, DenseOn embeddings)
and the PubMed Central open-access full texts, served as an HTTP API and an MCP server.
First user: ScienceMadeReadable.

- [PLAN.md](PLAN.md): steps, measurements, timeline, update pipeline.
- [hosting-lesson.md](hosting-lesson.md): how hosting on lambda works (systemd, cgroups, NixOS, tunnels).
- `experiments/`: measured design questions, each script with its log.

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

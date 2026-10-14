# Topic maps: Scimeetr, rebuilt on today's data

**Question → the papers about it → the research groups inside it → what to read in each group.**

Scimeetr (2016, R, github.com/MaximeRivest/scimeetr) did this from a Web of Science export of a few hundred
papers. Since 2026-10-13 it runs on the release: 125M papers searchable by meaning, a 3.0-billion-link citation
index and an 857-million-row authorship index.

| Where | What |
|---|---|
| `GET /topics?q=` (JSON), `GET /topics/{q}` (Markdown) | search API (`srl_search/app.py`) |
| MCP tool `map_topic` | `srl_search/mcp_server.py` |
| `https://ask.sciencemadereadable.com/map/<topic>` | plain text, through the public door |
| `https://sciencemadereadable.com/?topics=<topic>` | the site: "See the research groups in this topic" under any search |
| method / service / indexes | `srl_search/topics.py`, `topic_service.py`, `citations.py`; `citewalk/build_authors.py` |
| measurements | `eval.py` → `eval.json` (30 questions); `prototype.py` (the first sketch, superseded) |

## What a map is made of

1. **Papers**: the 1,000 closest in meaning (ranked as /search ranks: similarity − 0.3 × P(junk); records more
   likely junk than not are left out; nprobe 512 so the 1,000th is found reliably), or the best matches of an
   exact-words query. Publications only, retracted out. One per study: identical titles collapse, and so do a
   preprint and its journal version under another title (same first author, cosine ≥ 0.95; the journal version,
   then the one with references, is kept).
2. **Links**, half the total weight each:
   - shared references: cosine of the reference lists, each reference weighted by idf (log n/df), so two papers
     sharing an obscure source are closer than two papers citing the field's textbook;
   - meaning: each paper's 15 nearest neighbours by embedding, weighted as UMAP's fuzzy graph does [4].
3. **Groups**: Leiden (modularity) [3]. Resolution: of 0.5–1.5, the one whose runs give closest to 7 groups
   (≥ max(12, n/50) papers each), one step coarser if the consensus still has more than 9. Final groups: consensus
   of 20 runs [5] (links kept where both ends share a group in ≥ half the runs, weighted by that share). A paper
   that sits elsewhere in ≥ 40% of runs is marked *between*. Seeded: same input, same map.
4. **Names**: up to 4 phrases (title 1–3 words, abstract 1–2 words; no stopword or number at either end) most
   over-represented in the group against the rest (p·log(p/q), Scimeetr's relative frequency weighed by how common
   the phrase is); a phrase sharing a word with a chosen one, or naming mostly the same papers (Jaccard ≥ 0.6), is
   skipped; the question's own words are never used.
5. **Lenses**, from the citations the group's own papers make (Scimeetr's point: importance *to this group*):
   `core` (cited most), `distinctive` (≥ half of its citations in the map come from this group), `rising` (last
   5 years, cited per year since publication), `start_here` (recent group papers citing most of the group's 30
   most-cited works: reviews rise), `bridges` (participation coefficient × √strength × (1 + log(1+citations)):
   broad reviews in venues nobody cites are pushed down), `newest`, `experts` (harmonic local h-index: h works each
   cited by ≥ h group papers after position credit: first author 1, last ½, the k-th other 1/(k+1); over OpenAlex's
   disambiguated authors). Plus `cites_groups` (direct citations between groups) and each group's map centre.
   Non-reading references (datasets, software, errata, …) still link papers but never appear in a lens.

Scimeetr's `by_expert_LC` described this credit rule but its code computed `worth = position`, giving *more* credit
to later authors; the rule here is the described one.

## Measured

### Linking (30 questions, 14 biomedical; `eval.json`)

Accuracy from facts the grouping never sees, as in [1]: **MeSH lift** (same-group pairs share NLM's MeSH headings
how many times more than random pairs; biomedical only) and **venue lift** (same journal). **Seed ARI**: the groups
made with another random seed agree how well (1 = identical).

| Linking | Groups | In groups | MeSH lift | Venue lift | Seed ARI | Time |
|---|---:|---:|---:|---:|---:|---:|
| **hybrid, idf (chosen)** | 7.0 | 0.99 | 1.149 | 1.67 | **0.90** | 3.3 s |
| hybrid, no idf | 7.5 | 0.99 | 1.156 | 1.77 | 0.84 (−0.07, CI −0.12…−0.02) | 3.0 s |
| hybrid, 30 strongest reference links | 7.2 | 0.99 | 1.162 | 1.78 | 0.88 (−0.03, CI −0.07…+0.01) | 1.6 s |
| references only | 6.5 | **0.66** | 1.123 | 2.17 | 0.92 | 2.6 s |
| text only | 7.7 | 0.99 | 1.176 (+0.03, CI +0.01…+0.05) | 1.67 | 0.87 | 1.3 s |

Reading it: accuracy differences are ≤ 2.5% and mostly inside the noise. References alone leave a third of the
papers in no group (those without references in OpenAlex). Text alone is slightly better on MeSH, a measure built
from content as the embeddings are, so it favours text by construction; the hybrid keeps the citation logic the
lenses are made of and is the steadiest. Dropping idf makes groups measurably less reproducible. The 30-link
variant halves the time at a stability cost inside the noise: the one to switch to if speed matters more.

An earlier resolution rule (most stable of a few runs) made groups depend on the seed (coral reefs: ARI 0.50
between seeds); choosing by group count fixed it.

### Speed (dev API, 2026-10-13, after the first request of the day)

| | first time | cached |
|---|---:|---:|
| a map of 1,000 papers | 4.4–8.3 s (papers 0.9–1.7 s, grouping 1.6–4.3 s, experts up to 3 s when cold) | 20–30 ms |
| splitting a group | 0.4–0.6 s | |

### Examples

- *intermittent fasting for weight loss* → 8 groups: continuous energy restriction; mechanisms (inflammation,
  autophagy); type 2 diabetes; time-restricted eating; exercise and Ramadan; alternate-day fasting; fatty liver;
  gut microbiota. Start here: the Cochrane review on cardiovascular prevention (2021), the BMJ network
  meta-analysis (2025). Experts: Varady, Mattson, Panda, Peterson, Younossi, Cani.
- *coral reef bleaching and recovery* → 5 groups (temperature and warming; disturbances and coral cover;
  symbionts; restoration; regional studies). Experts: Hughes, Hoegh-Guldberg, Mumby, Rinkevich.
- *effects of remote work on productivity* → 7 groups; only half the papers have references (the answer says so).
  Experts in the economics group: Bloom, Barrero, Bartik.

## Choices rejected

- **Direct citation** links: least accurate for small recent sets [1]. **Co-citation**: only works for papers
  already cited. **Louvain**: can return disconnected groups [3].
- **One Leiden run**: groups changed between runs (ARI 0.51–0.87). **Choosing the resolution by stability**: the
  estimate is noisy and made the result seed-dependent (above).
- **Labels written by a language model**: friendlier, but needs a model running for every new question (the home
  9B is off), costs time, and can invent; the phrases are checkable. Kept as a possible later addition.
- **OpenAlex topics as labels**: often wrong (a remote-work group labelled "Magnetic confinement fusion"); shown
  only as `openalex_topics` with their share.
- **Precomputing maps**: any question works on request, and a cache makes repeats instant.

## Limits (also said to users)

- "The 1,000 closest papers" defines the topic: distant relatives are not mapped. A missing group means it was not
  among these papers. `papers` goes up to 2,000.
- References come from OpenAlex and only count links between works it holds (139M of 3.1B links point outside);
  papers added by the daily updates have none until the next snapshot. `references_thin` flags maps where under
  65% of papers have references.
- Names are keyword phrases, sometimes rough ("century · scenario · permanent").
- On the site, the small globe beside the list is too zoomed out to tell groups apart; "open the map" flies to
  the topic at a zoom where they are.
- igraph's random generator is global, so clustering runs one map at a time (1–4 s); at most 2 maps are computed
  at once; the public door counts a map as 5 requests (cached or not).

## References

- [1] https://doi.org/10.1002/asi.21419 · Boyack & Klavans 2010, JASIST: bibliographic coupling beats co-citation; direct citation least accurate; a citation–text hybrid best.
- [2] https://doi.org/10.1162/qss_a_00085 · Boyack et al. 2020, QSS: text–citation hybrids give the most accurate clusters; the two logics are complementary.
- [3] https://doi.org/10.1038/s41598-019-41695-z · Traag, Waltman & van Eck 2019, Sci. Rep.: Leiden guarantees connected communities.
- [4] https://doi.org/10.48550/arxiv.1802.03426 · McInnes, Healy & Melville 2018: UMAP (the fuzzy k-nearest-neighbour graph).
- [5] https://doi.org/10.1038/srep00336 · Lancichinetti & Fortunato 2012, Sci. Rep.: consensus clustering in complex networks.

## The whole field, not the closest 1,000 (experiments, 2026-10-13/14, habitat fragmentation)

- `calibrate_scope.py`: 8 archetypes → 300k closest by meaning → Jev (jev-1.13.0, one Noul) labels 850 papers
  stratified by distance (5 s, ~2¢). No single distance cut-off works: 98% on topic in the top 500, still 8-20%
  at ranks 20k-120k. ~24k on-topic papers in the 282k; about half never use a fragmentation phrase.
- `probe_scope.py`: logistic regression on meaning vectors trained on those labels ranks the far papers far better
  than distance (cross-validated AUC beyond rank 10k: 0.97 vs 0.67).
- `recall_reviews.py`: recall on the 589 on-topic studies cited by 19 meta-analyses and systematic reviews (not
  archetypes; on topic = Jev on title + abstract; 95% intervals by resampling reviews):

| Method | Papers kept | Recall | Recall, studies with an abstract vector |
|---|---:|---:|---:|
| exact phrases | 36.5k | 0.42 | 0.60 |
| 300k closest by meaning | 282k | 0.68 | 0.96 |
| classifier ≥ 0.5 | 31.9k | 0.52 | 0.73 |
| classifier ≥ 0.5 or phrases | 54.7k | 0.55 | 0.78 |
| … + works cited by ≥ 3 of those papers | +111k | **0.94** (0.88-0.99) | 0.94 |
| … + works cited by ≥ 5 | +58k | 0.91 (0.82-0.98) | 0.91 |

  30% of the reviews' on-topic studies have no English abstract in the corpus (older papers, books, chapters,
  theses): no text search can find them; the field's own citations do. Weak spots: dam-fragmented rivers
  (classifier 0.06: its training sample had almost no river papers), old island and patch-size studies.
  Limits: on-topic is Jev's judgment, so Jev's own errors are not measured (a person checking ~50 references
  would); reviews favour well-cited studies, so recall of recent work is overstated (2010+: 0.87 at ≥ 3).

## Whole-field maps, as built (srl_search/field.py, 2026-10-14)

Fully automatic (only the topic's words; archetypes chosen by Jev among the 300 closest papers):

| Topic | Papers | Groups | Time | Jev | Checked |
|---|---:|---:|---:|---:|---|
| habitat fragmentation | 35,123 | 11 (each split) | 13.5 min (map rebuild alone 9.5) | $1.56 | recall 0.895 (0.83-0.96) on 19 meta-analyses' studies; precision ~0.81 on 400 random papers (`validate_field.py`) |
| microplastics in freshwater ecosystems | 10,097 | 6 | 6.6 min | $0.37 | experts and groups read by eye only |

Recall by period on fragmentation: before 2000 0.94, 2000s 0.92, 2010 on 0.81; dam-fragmented rivers 0.52 (was 0.06
with the hand-made classifier). Weak point seen: the exact-phrase step uses the whole topic as one phrase, which finds
almost nothing for long topics (45 matches for the microplastics wording); meaning and citations carry it.

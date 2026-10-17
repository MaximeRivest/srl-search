# Whole-field maps of any size (2026-10-16/17)

Until selection v6 a whole-field map started from the 300,000 papers closest in meaning to the topic and refused
fields of more than 150,000 papers, after ~20 minutes of work. Three real requests failed that way on 2026-10-16
("reinforcement learning", "ai programming", a question about measurement). Selection v7 and the map changes in
srl_search/field.py and topics.py lift both limits; the live limit is SRL_FIELD_MAX_PAPERS (1,000,000).

## What changed

- **Gathering**: the teacher's first ring is the 300k closest in the whole index (every IVF list, one query, ~2 s).
  While the farthest stratum is still >30% on topic it judges 96 more papers in a ring 4x deeper, up to 20M deep.
  Stratum share on topic x stratum size estimates the field's size, so a field over the limit is refused within
  ~2 minutes. Candidates by meaning: the first classifier's direction, ~3x the estimate deep (300k-8M).
- **Reading many rows**: vectors per shard (Vectors.get_many, ~10x get()), details sorted by id on 16 threads
  (_rows, ~5x), positions vectorised (Ctx.positions), logistic scores in float32 chunks.
- **Big maps**: exact text neighbours on GPU 1 (judges.knn_on_gpu; 1.9M papers in 4.5 min, recall 0.995 vs fp32;
  CPU exact up to 200k, IVF beyond as a fallback, 0.97); coupling within a pair budget (the most cited references
  left out past it); 2 Leiden iterations a run past 50k papers (100k papers: ARI 0.97 vs converged at one seed,
  1/20 of the time; two converged seeds differ far more, ARI 0.54, which the consensus absorbs); groups found on a
  500k sample past that, the rest placed by their links; words from a 300k sample (100k in subgroups).
- **Levels**: groups split side by side (forked, up to 3, when the service's own memory allows), and subgroups of
  1,500+ papers split once more: the group's page shows them inside each subgroup.
- **Serving**: maps over 25k papers list a proportional sample in their JSON (papers_shown) and keep every paper in
  a .npz; read=N&part=papers pages through those, reading only what the page needs (0.3 s for a 118k group).
- **The line**: a free field estimated at 300k+ papers goes back in the line after the smaller free maps (Deferred),
  with an expected time that counts its size. A refused topic gets narrower suggestions (/fields/suggest?narrower=true).
- The general judge waits for 9 GB on GPU 1, not 10: the household services hold ~14.4 of 24 GB; at 10 maps waited
  for hours (measured: vLLM at 20% of the card judges 3,000 papers in 74 s with start-up).

## Results (one worker, 80 GB limit as the live service; Kimi K3 judged samples: check.py)

On topic: of 100 random papers of the map, and of 100 cited 50+ times (what reading lists show). Found: the share of
on-topic candidates the map kept (sampled by score band; papers never gathered as candidates are not counted, so it
is an upper bound on recall).

| topic | papers | on topic | cited 50+ | found | time |
|---|---|---|---|---|---|
| reinforcement learning | 190,154 | 98% | 87% | 91% | ~35 min |
| sleep | 284,687 | 97% | 90% | 93% | 42 min |
| CRISPR | 54,795 | 92% | 91% | 84% | 17 min |
| gut microbiome | 228,761 | 91% | 86% | 92% | 40 min |
| climate change | 723,409 | 91% | 84% | 84% | 84 min (swapping) |
| exoplanets | 64,477 | 89% | 89% | 87% | 15 min |
| AI-assisted programming (with a definition) | 15,880 | 86% | 80% | 58% | 10 min |
| How to make rubric, evaluations, metrics, survey... (a visitor's words) | 144,463 | 86% | 81% | 56% | 33 min |
| coral reefs | 66,718 | 81% | 95% | 68% | 14 min |
| biodiversity | 636,813 | 59% | 79% | 83% | ~65 min |
| ai programming (a visitor's words) | 77,452 | 46% | 29% | 38% | 21 min |

95% intervals at 100 papers are about +-6-10 points. Concrete topics score high at any size. Loose words score low:
"ai programming" drifted into applied AI in general; "biodiversity" takes in scanned natural-history archives on
Zenodo (Biodiversity Heritage Library copies: letters, floras), which use its words without studying it. Both are
better asked through the suggestions (the visitor's own "AI-assisted programming" option: 86%).

## Memory

A 723k-paper map with three groups split side by side reached the 80 GB limit: 31 GB anonymous, 49 GB file cache,
12 GB swapped (slower, not killed). The split guard now counts the service's anonymous memory and the pages a child
copies of the whole map. Deep learning (~2.2M papers) is the test of the next limit.

# Junk in search results: measured, then demoted (2026-10-09)

"how to loose weight" returned mostly supplement ads and clickbait uploaded to Zenodo. This folder holds
the measurement and the fix now in `srl_search/quality.py` + `engine.rank`.

## How it is measured

- **Questions**: `queries.jsonl`, 60 written by hand in four styles: everyday advice where spam lives (24),
  curiosity (12), research phrasing (14), terse keywords (10). Each has the information need in one sentence.
- **Judge**: every (question, paper) in the top results is graded by Claude Sonnet (`claude -p`, prompt in
  `eval_quality.py`), which sees what a reader sees (title, type, year, venue, start of the abstract), never
  citation counts or venue flags, so it cannot grade the rankers' own inputs. Grades: `rel` 0–3 and `kind`
  research / non_research / spam. 24,162 judgments in `judgments.jsonl`, reused by every run (pooling).
- **Metrics** over the top 10: spam share, junk share (spam + non-research), *useful nDCG* (relevance counted
  only for research; ideal = best judged papers for that question), useful results (rel ≥ 2 and research).
  Paired against the baseline per question with a bootstrap interval.
- Not done yet: a human check of the judge. On ~60 judgments read by hand it agreed with common sense; a
  person labelling 50 would tell how far to trust the absolute numbers. Comparisons between runs share the
  judge's biases, so they are more trustworthy than the absolute levels.

## Results (top 10, `results-2026-10-09.txt`)

| | spam | junk | useful nDCG | useful / 10 |
|---|---:|---:|---:|---:|
| before (public API) | 15% | 27% | 0.57 | 6.5 |
| **junk demotion, as served (dev API)** | **0%** | **1%** | **0.78** | **8.7** |
| … everyday-advice questions only: before → after | 35% → 0% | 54% → 0% | 0.29 → 0.68 | 3.8 → 8.5 |
| question rewritten by a model (Claude Haiku), no demotion | 1% | 7% | 0.72 | 8.2 |
| "ideal abstract" written by our 9B, searched (HyDE) | 3% | 9% | 0.75 | 8.3 |
| 9B HyDE + junk demotion | 0% | 0% | 0.81 | 8.9 |

Junk demotion vs before: junk −27 points (95% −35 … −19), useful nDCG +0.21 (+0.14 … +0.27); 44 questions
better, 9 worse, none with more junk. Research-phrased questions also gain (0.856 → 0.884).

In the readable collection (`smr`: PMC CC BY research articles) there is no junk to remove: the penalty
changed useful nDCG by −0.007 (−0.017 … +0.004), so it is off there by default.

## The junk model

- **Record signals** (`build_quality.py`, 11 min from the raw snapshot, `/mnt/fast/.../quality-2026-10-05`):
  OpenAlex's expansion-pack flag, paratext, source type / core list / DOAJ, which indexes list the record,
  references, authors, locations, and each venue's habits over the whole corpus (share of its works with no
  references, one author, never cited, expansion pack). Of the 203.5M corpus works, 34.7% are expansion-pack
  records, 58.7% have no references, 50.1% sit in a repository.
- **Text signal**: the paper's own DenseOn embedding (already read for the exact rerank, so free).
- **Model**: logistic regression on both (`fit_quality.py`), fitted on 14,087 judged results of 367
  *separate* training questions (`train_queries.jsonl`; 126 papers that also appear in evaluation results
  were dropped). Cross-validated, grouped by question: average precision 0.70 record only, 0.86 text only,
  **0.90 both** (ROC AUC 0.97). Weights: `model-2026-10-09.npz`.
- **Ranking**: similarity − 0.3 × P(junk). 0.3 was chosen on the training questions (0.1: 4% junk, 0.3: 1%,
  1.0: no better) before the evaluation questions were scored with it.
- **Citations are not a junk signal** on purpose: new papers have none. Junk is what a record is, not how
  famous it is.

## Costs and limits (stated, not absorbed)

- Demotion, not removal: junk is still there further down and with `junk_penalty=0`.
- Real but thinly described records get pushed down too. Example: the Great Barrier Reef Marine Park
  Authority's *Interim report: 2016 coral bleaching event* (no authors or references in its record, an
  expansion-pack repository entry) scores P(junk) 0.68 and drops out of the top 30 for "coral reef bleaching".
  9 evaluation questions lost useful nDCG: good real papers left the top 10 for other, mostly also good,
  ones. Biggest: "apple cider vinegar" −0.19, "transformer models for protein structure prediction" −0.12,
  "natural remedies for high blood pressure" −0.10, CBT for insomnia −0.08; the other five under −0.04.
- Exact-word (keyword) search is not re-ranked yet.
- Crank papers (e.g. viXra "proofs" of the Riemann hypothesis) count as research to the judge and to the
  model; that is a different problem (credibility, not junk).
- Cost per search: 1–20 ms (memory-mapped signals; cold pages are the slow case).
- The labels come from one judge model; the model can learn its blind spots.

## Question rewriting: measured, not shipped

Rewriting the question into research language (Haiku) or writing an ideal abstract and searching with it
(HyDE) removes most junk by itself, and HyDE adds +0.03 useful nDCG on top of junk demotion. Not shipped:

- our 9B is a plain-language paper rewriter; asked to rewrite questions it makes them *more* casual
  (`rewrite_9b`: no gain). Its HyDE works but takes ~2.4 s per search on the GPU that writes the site's
  rewrites. A short "ideal title" (0.3 s) helped much less.
- a hosted model (Haiku) would send every visitor's search to a third party, cost money per search, and add
  a network dependency; the site currently promises words are not stored.
- HyDE hurt research-phrased questions (0.856 → 0.791) unless fused with the original search.

Revisit when a small instruction model can run next to the embedding server (GPU 1 frees up when the backfill
ends): rewrite + junk demotion, fused with the original question.

## Re-running

```sh
P=../search/.venv/bin/python
$P eval_quality.py run --name NAME --api http://127.0.0.1:8812 [--param junk_penalty=0]
$P eval_quality.py judge NAME            # only papers nobody judged yet
$P eval_quality.py report baseline NAME
SRL_QSET=train $P eval_quality.py ...    # the training questions
$P fit_quality.py --quality /mnt/fast/scholarsreadinglist/search/quality-2026-10-05   # refit
```

Runs (with abstracts, ~20 MB each) live in `/mnt/fast/scholarsreadinglist/search/quality-eval/runs`.

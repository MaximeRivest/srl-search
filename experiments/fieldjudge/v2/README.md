# Round 2: Sol + Opus data for a general field judge

**Status:** data preparation and independent benchmark labeling started. No new
model is trained or deployed. Full-scale labeling is gated; the first production
cohort stops at 2,000 pairs for audit. Source and data live separately:

- Code: `experiments/fieldjudge/v2/`
- Data/progress: `/mnt/fast/scholarsreadinglist/fieldjudge/v2/`
- Search/corpus scripts: `search-dev/.venv/bin/python`
- Subscription/FunctAI scripts: `sciencemadereadable/.venv/bin/python`

## Scope and accounting

The user requested Sol 6.1 and Opus 5.5 through their existing subscriptions,
25 papers per request, for approximately 100,000 topic–paper pairs. A paper can
appear under different topics: learning the topic boundary is the point. This is
not necessarily 100,000 unique papers.

The frozen `recipe.json` targets 500 families with five scopes each (2,500 topic
scopes), 200 pairs per family:

- 100 coverage pairs: near/far retrieval, cited papers (including title-only), and
  random candidate-pool papers. Missing citation coverage is recorded as fallback.
- 50 cross-scope pairs: a real paper judged under a related scope. A sibling is
  **not automatically labeled negative**.
- 50 mined pairs, selected from 100 candidates using the old local 0.8B/2B models:
  half near the 2B boundary and half where the models disagree most. Neither
  teacher sees these scores, sample route, or the other teacher's answers.

Training and validation topics are grouped before splitting. In addition to the
five-scope families, families sharing at least half their nearest 50 papers are
connected and kept together. Validation families are collected first. Work IDs
and normalized titles cannot cross train/validation/test boundaries. The old
fine-tunes are only miners; round-two training must start from base weights to
avoid inheriting their evaluation contamination.

## Frozen benchmark

The original fragmentation and LLM post-training sets are now development
benchmarks: many decisions have been guided by them. Eight new holdouts in
`recipe.json` cover medicine, batteries, causal inference, manuscripts, glaciers,
retrieval-augmented generation, electrical transformers, and bat disease.
Generated families are rejected if either teacher flags substantial overlap.

For each new topic, 500 rank-stratified papers from the eligible nearest-300k
retrieval pool plus up to 100 cited title-only papers are frozen **before** any
training collection. Sampling denominators and weights are retained. The two
populations must be reported separately; neither is all of science. Kimi labels
these as an independent silver reference, with a $20 run budget. No model's
answer is human-reviewed ground truth. Human/expert review of disagreements and
shared mistakes is still needed before claiming accuracy or publishing weights.

## Batch pilot: findings, not assumptions

The initial mixed, often undefined-topic pilot failed the operational 95/100
consistency gate. Sol matched single-paper calls on 94/100 and reordered batches
on 91/100; even repeated single-paper calls matched only 95/100. Five-paper
batches did not cure this (95/100 and 90/100). Opus sometimes paraphrased evidence,
reordered outputs, and once omitted one of 25 papers. These failures are retained
under `pilot/`, not silently erased.

A second pilot uses a fresh seed and **defined topics**, matching the new recipe,
with 20% title-only cases and short request-local IDs. Results (`pilot-defined/`):

| Teacher | Same as singleton (n=100) | Same after reordering (n=100) |
|---|---:|---:|
| Sol | 95 | 94 |
| Opus | 94 | 98 |

Neither teacher individually passed both 95/100 gates. A stricter protocol was
then preregistered in `acceptance.py`: both teachers must agree, retain that label
under a changed ordering, give non-abstaining labels, and supply verified evidence.
At least 60/100 cases must survive; 95% of survivors must match BOTH singletons.
**72 survived; 69/72 (95.8%) matched both singletons.** This passes that operational
pilot gate, but is a small development result, not an accuracy or confidence claim.

Production therefore asks each teacher twice with different grouping/order.
This doubles the planned classification calls: approximately 8,000 requests per
teacher at 25 pairs/request, before repairs/audits. The user was told this tradeoff.
Keep all 100k pairs and all votes; unstable, conflicting, abstaining, refused or
unverified-evidence cases go to a review pool. Do not turn them into negatives,
or call a two-vote average a calibrated probability. Agreements are provisional
supervision, not truth. The hard review pool must not be permanently ignored:
it contains precisely the boundaries we want the students to learn.

## Reliability and allowance safeguards

- FunctAI typed functions, persistent reply cache and raw call logs.
- Durable wave checkpoints; IDs checked as sets and matched by ID, never position.
- Quotes verified against the actual supplied text (allow typography/whitespace
  changes, not rewritten words). No invented quotations in accepted supervision.
- Two in-flight calls per provider; fail closed if usage cannot be read. Usage
  observations shared for at most 60 seconds because the usage endpoint is itself
  rate limited. Pause at 85% of any reported window. Persist the starting Codex
  credit balance; stop if it drops. No billed-API fallback.
- These checks cannot guarantee zero extra-usage spend if reporting is delayed or
  another process spends the shared allowance. Disable account extra-usage billing
  for a hard limit. Refused requests are retained, not repeatedly retried.
- Mining waits for GPU memory and live maps between bounded chunks. No services
  are stopped. A miner chunk may delay a new map briefly; it does not preempt it.
- No automatic release of weights; provider/model provenance is retained and
  publication/training-use permissions need a separate review.

## Commands and checkpoints

```sh
# From this folder; use the interpreters described above.
python pilot.py sol                  # then opus; saved replies resume
python acceptance.py                # refuses incomplete pilot data
python topics.py                     # generation + dual-teacher overlap screening
python collect.py benchmark
python collect.py benchmark-titles
python group_split.py
python collect.py families           # validation ownership first
python mine.py                       # GPU; writes exactly 100k pairs or refuses
python label.py sol train --limit-pairs 2000
python label.py opus train --limit-pairs 2000
# Audit first cohort, then explicitly run remaining train + validation labeling.
python export.py                     # provisional labels + unresolved review files
python -m unittest test_contract -v
```

`launch.sh` starts three independent systemd user services. They are not children
of Chattering: closing the browser/conversation or disconnecting does not stop
them (user lingering is enabled). These are transient services, so after a machine
reboot run `launch.sh` again. `launch.sh --restart` reloads changed code and resumes
from checkpoints. Logs: `journalctl --user -u fj-v2-prepare -f`.

`run_stage.py` distinguishes temporary network/provider failures (exit 75, retry
after five minutes) from data, account, allowance and quality failures (exit 78,
pause for inspection). Retry is capped at eight starts/hour. A real systemd test
simulated one dropped connection and verified one restart and successful recovery.
The labelers wait through preparation's auto-restart intervals. Completed services
remain `active/exited`, distinguishable from `active/running`.

`prepare.sh` pins the corpus release. It explicitly allows deferred topic sections:
Opus refused all four immunology screening proposals, so those 19 families are
recorded as missing, not retried or counted as covered. Other subjects proceed.
`topics.json` records `deferred` and `complete`; `dataset-status.json` records the
actual pair count against the 100,000 target and `training_approved: false`. No
final training/release approval is implied by collecting a partial dataset.

`first_cohort.sh MODEL` waits for preparation and deliberately stops after 2,000
production pairs. Neither starts fine-tuning. Next gates are:
first-cohort audit (including correlated false positives), resolution of the hard
review pool, training-data leakage audit, matched-data 0.8B/2B training and ablations,
then the new benchmark and actual reading lists. No live-site changes in this work.

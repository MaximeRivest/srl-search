# Round 2: Sol + Opus data for a general field judge

**Current status: accelerated streaming run.** The user explicitly prioritized
labels over order checks and heavyweight preparation. Sol and Opus label in
25-paper requests with short ID + label outputs. The remaining work is now split
between them, not double-judged (see below). Labeling is now finished; experimental
0.8B training is running, with 2B queued afterward. Nothing is deployed. The old
staged jobs are stopped; the accelerated labelers did NOT stop at
2,000 pairs. Source and data live separately:

- Code: `experiments/fieldjudge/v2/`
- Data/progress: `/mnt/fast/scholarsreadinglist/fieldjudge/v2/`
- Search/corpus scripts: `search-dev/.venv/bin/python`
- Subscription/FunctAI scripts: `sciencemadereadable/.venv/bin/python`

## Training and live dashboard (2026-10-15 evening)

- Dashboard: https://lambda.tail69222b.ts.net:18943/ (private tailnet; also opened as
  a Chattering artifact). `dashboard_server.py` serves metrics only, not papers or credentials.
- `training_data.py` freezes topic-family connected components based on retrieval
  overlap, excludes held-out topics/papers and normalized titles, and fixes validation
  before the first gradient step. Teacher agreement is silver supervision, not truth.
- First pass: 84,377 usable training pairs. Second pass: 85,303, including 926 late
  arrivals. Validation stays fixed at 3,921 pairs; late validation labels are not added.
- Both sizes use exactly the same initial-then-final curriculum, from their public
  base weights, not the old fine-tunes. No guessed labels for uncertainty/refusals/conflicts.
- `train_incremental.py`: LoRA rank 32, batch 32, lr 1e-4; tokenization preserves
  the question and answer prefix. Checkpoints include optimizer, row cursor/order,
  RNG, configuration and data hashes. Export is the final state so it has seen all
  usable late examples; best-validation checkpoints are retained separately.
- `run_models.sh` runs 0.8B then 2B, yielding to live maps. `finalize_labels.py`
  automatically freezes the final snapshot when both labelers finish. If that is
  delayed, training checkpoints at the epoch boundary and releases GPU while waiting.
- `evaluate_finished.py` waits for both models, then evaluates on all eight new
  held-out topics, with retrieval-pool and title-only results reported separately.
  This uses no provider calls. It does not deploy either model.
- Services: `fj-v2-models`, `fj-v2-final-labels`, `fj-v2-evaluate`, `fj-v2-dashboard`.
  They are independent of Chattering; jobs are transient, not reboot-persistent.

## Remaining work split between teachers (2026-10-15)

At the user's request, both workers were stopped before freezing `fast/assignment.json`.
34,319 pairs already had an answer from at least one teacher (including uncertainty).
They are not requested again. The remaining 65,681 pairs were split disjointly:
32,840 to Sol and 32,841 to Opus. Whole batches are used where possible, with family
and overall balancing; prior refusals go to the other teacher, never retried with
the same one. If both refused, the pair stays for review. Assignment is immutable
across restarts. `partition.py` verifies source checksums and locks both worker files
while creating the plan; tests prove no overlap and exact remaining coverage.

Earlier double judgments are retained, including disagreements. `export_fast.py`
now distinguishes `provisional_single_teacher` from `provisional_agreement`; a
missing second opinion is expected for assigned work, not a pending task. It still
flags uncertainty, refusals, Kimi conflicts where old labels exist, and leakage.
This saves about half the remaining labeling work but removes the second-opinion
check for these pairs. No automatic training or deployment.

## Active fast path (supersedes the staged commands below)

```sh
./launch_fast.sh                     # independent processes; resumes checkpoints
python fast.py status               # actual queue and per-teacher counts
python export_fast.py --partial     # joined provisional labels + disagreements
journalctl --user -u fj-fast-sol -f
journalctl --user -u fj-fast-opus -f
journalctl --user -u fj-fast-collect -f
```

- Immediately enrich 34,273 existing pairs with both teachers (exclude the old
  validation topics, reserved topic terms and all frozen test IDs/normalized titles).
- Add 367 already dual-screened new families (1,835 scopes) as each CPU collection
  finishes. The six remaining ungenerated areas and refused immunology section do
  not block work. Existing-paper enrichment provides additional field diversity.
- Stop collection at 100,000 queued pairs total. Distinct papers are fewer because
  cross-topic judgments are intentional. Files live under `fast/`, separate from
  the original double-pass protocol.
- Cut repeated order checks, quotations, rationales, GPU hard-example mining and
  the all-data-ready barrier. The extra retrieval examples are random, explicitly
  not model-mined. This trades some annotation consistency/targeting for throughput.
- Keep both teacher implementations (disjoint remaining assignments as above), exactly matched IDs (output ordering is irrelevant),
  uncertainty, refusals, disagreements, held-out paper protection, quota checks,
  cached replies, atomic checkpoints and five-minute retries on temporary failures.
- Initial observed throughput: Sol ~300 pairs/min; Opus ~500 pairs/min. Refusals
  are counted separately, not disguised as successful labels. No completion-time
  guarantee: shared quotas and provider stalls can interrupt this rate.
- Existing Kimi labels are available as free checks on legacy enrichment. They are
  never shown to the teachers. New independent Kimi benchmark cost: $6.22.
- `export_fast.py` removes cross-split work/title overlaps and flags disagreement
  with old Kimi labels. Outputs remain provisional: whole-family overlap and
  disputed cases still need auditing BEFORE training, not before producing labels.

The sections below preserve the original recipe and pilot findings; their
order-check and first-cohort gates are **not** the active accelerated workflow.

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

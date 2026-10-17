# Round 3: one classifier for any option list, at any level, and yes/no

Goal: give the model a paper and ANY list of options (a real scheme, one of our maps, an invented one; any level of
a hierarchy) and get a probability for every option, with no generated text. Yes/no for one topic is the same task
with two options.

## Design (fmt.py)
- Paper first, question after: many questions about one paper reuse its cached computation.
- Options carry two-letter codes that are single tokens for Qwen3.5 (codes.json: 562). The answer is one token; the
  distribution is the softmax of the offered codes' logits. Yes/no reads "yes"/"no". Up to 562 options a question;
  bigger schemes go level by level (eval3 "tree": P(child) = P(parent) x P(child | parent)).
- Rejected: a pointer head over option hidden states (custom serving); one yes/no pass per option (176 passes).

## Data
- Schemes (build_schemes.py, invent.py): OpenAlex (4 levels), arXiv, ACM CCS, MSC 2020, MeSH, ANZSRC FoR, OECD FORD,
  28 of our whole-field maps (AI group names and descriptions), 284 invented by Sol/Opus (16 ways of dividing).
  Held out: Science-Metrix (expert benchmark), JEL, 18 invented schemes, and 2,496 papers.
- Teachers (teacher.py, label.py): Sol 6.1 and Opus 5.5 on subscriptions, 20 papers per request, 1-3 options with
  the share of experts who would pick each. Pilot on the expert benchmark: Sol 66/69%, Opus 65/69% (Jev 64-68/61-67,
  one human expert 63/57). Sol and Opus see different schemes and different papers; eval units get both
  (they agree 90-92%).
- Training rows (make_data.py): 46,179 pick questions + 40,000 round-2 yes/no examples (replay).

## Training (train3.py, run3.sh)
From the round-2 judge; LoRA rank 32 + the answer tokens' output rows; soft cross-entropy over the offered codes.
Every presentation reshuffles options and codes. Yields to live maps on GPU 1.
Step 400/2,694 (0.8B): validation agreement with the teacher 75.5% (start 23%), yes/no 94%.

## Tests (eval3.py)
expert flat / shuffled / field / tree; yes/no on the 8 held-out topics (round 2: AP 0.684 / 0.754);
unseen schemes vs teachers; speed. Round-2 0.8B start: expert 0%, unseen 25-32%, yes/no AP 0.684.
Results: evaluation/<name>.json under /mnt/fast/scholarsreadinglist/fieldjudge/v3/.

## Acceptance
2B >= 55% on the expert benchmark with one question; yes/no AP within 0.02 of round 2; shuffled options keep the
pick >= 90%; ECE <= 0.05. Nothing is deployed.

# A general field judge: one small model for "is this paper about topic X?", any topic

Why: whole-field maps (srl_search/field.py) teach a 0.8B "decider" per topic from 400 Kimi K3 answers (~95 s of GPU
and $0.48 per map), and it is still clearly behind Jev, which only checks (its terms forbid training on its answers).
A model trained once on many topics should know the judgment itself ("studies the topic" vs "a neighbouring subject")
and need no per-topic training. Rerankers are made this way (small models trained on large models' judgments across
many queries: InPars, doi:10.1145/3477495.3531863).

## Data (2026-10-15)

- `topics.py`: Kimi K3 proposed 18 topics per OpenAlex field (26 fields): 436 topics, 203 with a definition, a few
  deliberately ambiguous ("transformers" the electrical device vs the network). `topics.json`.
- `sample.py`: for each topic, 90 papers chosen as the pipeline chooses its teacher's: 60 spread over 16 buckets from
  nearest to farthest of the 300,000 closest in meaning, 30 works the 5,000 nearest cite (20 title only). Junk and
  non-publications out. 39,113 papers, 0.5 s per topic.
- `label.py`: Kimi K3 answers with the pipeline's own question: $46.98, 0 failures, 47% on topic (10th-90th
  percentile of topics: 18-76%; cited works with a vector 26%).
- Held out: the two test topics (habitat fragmentation, LLM post-training) and every paper of their test sets (4,706
  papers never sampled); 3 topics overlapping LLM post-training dropped by hand (RLHF, "LLM", reinforcement learning;
  the meaning-distance guard at 0.55 let them through). 40 more topics (seeded) choose the best checkpoint.

## Training (`train.py`, `run_train.sh`)

LoRA r=32 on all linear layers + a yes/no head (the decider's recipe), the topic's question in every prompt, 2 passes
over 35,269 papers (2,204 steps of 32), lr 1e-4 cosine. It steps aside when a whole-field map needs GPU 1.

| Model | Time | Validation AUC (40 unseen topics) | Agreement with Kimi |
|---|---:|---:|---:|
| Qwen3.5-0.8B, no training (zero-shot, habitat fragmentation only) | - | 0.69 | - |
| Qwen3.5-0.8B general | 1 h 40 (14 papers/s) | 0.966 | 90.3% |
| Qwen3.5-2B general | 3 h 20 (7-9 papers/s) | 0.976 | 91.5% |

## Test: the two held-out topics, Kimi as the yardstick (`build_tests.py`, `eval.py`, `results.json`)

Population = stratified samples of each topic's whole-field candidates (1,585 and 970 papers), weighted: precision and
recall if the judge decided every candidate alone, at 0.5.

| Judge | Topic | AUC | Agreement | P | R | F1 | Cited 50+ P/R | Meta-analyses' studies (title only) / landmarks |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| per-topic decider (400 Kimi answers on the topic) | fragmentation | 0.980 | 96.5% | 0.79 | 0.73 | 0.76 | 0.75/0.80 | 0.78 (0.76) |
| general 0.8B (never saw the topic) | fragmentation | 0.980 | 95.6% | 0.67 | 0.83 | 0.74 | 0.58/0.90 | 0.88 (0.90) |
| Jev | fragmentation | 0.990 | 97.7% | 0.89 | 0.80 | 0.84 | 0.84/0.84 | 0.89 (0.89) |
| per-topic decider | LLM post-training | 0.985 | 96.4% | 0.58 | 0.82 | 0.68 | - | 42 of 44 → 0.89 |
| general 0.8B | LLM post-training | 0.983 | 97.1% | 0.73 | 0.63 | 0.67 | - | 0.82 |
| Jev | LLM post-training | 0.995 | 98.4% | 0.84 | 0.80 | 0.82 | - | 0.95 |
| general 2B | fragmentation | 0.987 | 97.2% | 0.81 | 0.82 | 0.82 | 0.83/0.86 (AUC 0.994; Jev 0.993) | 0.87 (0.93) |
| general 2B | LLM post-training | 0.982 | 96.0% | 0.55 | 0.80 | 0.65 | - | 0.91 |

The general 0.8B ranks as well as the per-topic deciders without any per-topic training, and finds the field's
classics far better (88% of the meta-analyses' studies vs 78%; title-only 90% vs 76%). Its 0.5 line is not right per
topic (too lenient on fragmentation, too strict on LLM post-training): `calibrate.py` sets it from the 400 answers each
run already has. Jev is still ahead (AUC 0.990/0.995).

The 2B is close to Jev on habitat fragmentation (AUC 0.987 vs 0.990, F1 0.82 vs 0.84; better on papers cited 50+
times and on title-only classics), but no better than the 0.8B on LLM post-training. Its mistakes there: any
fine-tuning of a language model counted as LLM post-training (intermediate-task training, adapters, BERT-era
tuning; Kimi and Jev say no to 93 of its 126 wrong yeses), and still "LLM" the law degree once (0.94 for a civil
procedure paper, despite the definition); it misses reward-model landmarks (Christiano 2017, overoptimization). The
LLM precision estimate is noisy: 4 papers of the sample stand for ~1,100 candidates each.

Setting the line per topic from the run's 400 Kimi answers (`calibrate.py`, `calibration_*.json`): F1 +0.01 to +0.03
(0.8B LLM 0.67 -> 0.70; 2B LLM 0.65 -> 0.68; 2B fragmentation already right at 0.5). It does make the expected number
of on-topic papers right (8.6% vs Kimi's 7.6%; 5.5% vs 4.7%), which counts need. The ranking is the limit, not the line.

Cost: Kimi $46.98 for the training answers; GPU 1 for ~5 hours (shared, stepping aside for maps).

## Inside the real pipeline (`replay.py`, `replay.json`, 2026-10-15)

Both test maps replayed with today's code against the release they were made from (20261014T073821Z: last night's
release changed the pool, so the teacher's sample would not have matched), the runs' own cached Kimi answers, and only
the judge that labels 10,000 papers for the students changed. Students, citation combiner, cut-off and Jev's check
ran for real. The per-topic arm reproduces the real run (38,557 papers vs 38,655; same P/R).

| Topic | Judge | Papers | Population P / R / F1 | Cited 50+ P / R | Meta-analyses' studies (title only) / landmarks | Judge time | Jev |
|---|---|---:|---:|---:|---:|---:|---:|
| fragmentation | per-topic decider | 38,557 | 0.82 / 0.69 / 0.75 | 0.93 / 0.81 | 0.82 (0.80) | ~245 s (train 95 + label 150) | $0.43 |
| fragmentation | general 2B | 47,122 | 0.77 / 0.72 / 0.75 | 0.80 / 0.85 | 0.87 (0.88) | 270 s | $0.60 (cap) |
| fragmentation | general 0.8B | 61,848 | 0.62 / 0.74 / 0.67 | 0.67 / 0.89 | 0.90 (0.90) | 148 s | $0.60 (cap) |
| LLM post-training | per-topic decider | 28,099 | 0.55 / 0.79 / 0.65 | - | 0.84 | ~245 s | $0.04 |
| LLM post-training | general 2B | 25,784 | 0.57 / 0.73 / 0.64 | - | 0.82 | 263 s | $0.03 |
| LLM post-training | general 0.8B | 18,130 | 0.75 / 0.65 / 0.70 | - | 0.82 | 147 s | $0.03 |

No switch. The general 2B gives the same maps' balance (F1 0.75 / 0.64 vs 0.75 / 0.65) and finds more classics
(0.87 vs 0.82), but it makes fragmentation's map more lenient (47k papers; well-cited precision 0.80 vs 0.93, the
part reading lists show), uses Jev's whole cap, and saves no time (a 2B labels 10k papers about as slowly as the
per-topic decider trains and labels). The 0.8B is faster but mixed (worse on fragmentation, better on LLM
post-training, whose estimates are noisy: single papers stand for ~1,100 candidates). Why the judge's better
classics do not carry through: the per-topic decider is already ~0.98 AUC on its own topic, and the final answer comes
from the students, the combiner and Jev; the general judge's leniency moves the whole map, its classics only a few.

## The reading lists (`readinglists.py`, `readinglists.json`)

Each replay arm's selection mapped into groups (main map), the top 10 of every paper list of every group (what readers
see first), every distinct one judged by Kimi for its topic ($2.16 for 1,595 papers). "Core" lists are the works a
group cites most (foundations, often methods), so they are partly off topic by design in every arm.

| Topic | Judge | Groups | Reading-list papers on topic | "Start here" on topic | Same papers as per-topic |
|---|---|---:|---:|---:|---:|
| fragmentation | per-topic | 11 | 80% | 97% | - |
| fragmentation | general 2B | 12 | 79% | 94% | 40% |
| fragmentation | general 0.8B | 14 | 67% | 75% | 28% |
| LLM post-training | per-topic | 11 | 49% | 46% | - |
| LLM post-training | general 2B | 10 | 61% | 67% | 34% |
| LLM post-training | general 0.8B | 9 | 65% | 76% | 30% |

Only 30-40% of reading-list papers are the same between arms: groups split differently (fragmentation 2B: amphibians
and bats as groups, no farmland-consolidation group). On fragmentation the 2B's lists are as good as the per-topic
ones; the 0.8B adds mostly-off-topic groups (elephants and protected areas, caribou). On LLM post-training the
per-topic map has 4 mostly-off-topic groups (BERT-era machine translation, vision-language, text-to-image diffusion,
speech recognition: their lists 10-38% on topic); the general models' maps have none, and their "start here" lists go
from 46% to 67-76% on topic. So where readers look, the general judges help on the topic the per-topic decider gets
wrong, and the 2B costs nothing on the other. One run per arm, two topics.

## Ensemble, and the 2B alone (no Kimi, no Jev)

Two more replay arms: "ensemble" (Kimi and Jev as now; the 10k labels are the mean logit of the per-topic decider
and the 2B) and "2B-only" (the 2B is the teacher of the 400 papers, labels the 10k, and does Jev's final check: no
paid AI at all). Reading lists judged as above (+$0.42 for the new papers).

| Topic | Arm | Papers | Map P / R / F1 | Cited 50+ P / landmarks | Lists on topic | "Start here" | Mostly off-topic groups | AI $ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fragmentation | per-topic (now) | 38,557 | 0.82 / 0.69 / 0.75 | 0.93 | 80% | 97% | 0 of 11 | 0.91 |
| fragmentation | general 2B | 47,122 | 0.77 / 0.72 / 0.75 | 0.80 | 79% | 94% | 0 of 12 | 1.08 |
| fragmentation | ensemble | 41,083 | 0.80 / 0.68 / 0.74 | 0.81 | 80% | 95% | 0 of 12 | 0.96 |
| fragmentation | 2B only | 49,978 | 0.75 / 0.73 / 0.74 | 0.84 | 77% | 92% | 0 of 11 | 0 |
| LLM post-training | per-topic (now) | 28,099 | 0.55 / 0.79 / 0.65 | 0.84 | 48% | 46% | 4 of 11 | 0.56 |
| LLM post-training | general 2B | 25,784 | 0.57 / 0.73 / 0.64 | 0.82 | 61% | 67% | 0 of 10 | 0.55 |
| LLM post-training | ensemble | 25,297 | 0.63 / 0.75 / 0.69 | 0.84 | 62% | 70% | 2 of 10 | 0.56 |
| LLM post-training | 2B only | 29,426 | 0.48 / 0.81 / 0.60 | 0.86 | 55% | 61% | 1 of 9 | 0 |

The ensemble is between its two parts, not better. The 2B alone, with no paid AI, makes maps about as good as today's
on these two topics: a little worse on fragmentation (lists 77% vs 80%, "start here" 92% vs 97%, well-cited precision
0.84 vs 0.93), better where today's map fails (LLM post-training lists 55% vs 48%, 1 off-topic group vs 4), worse in
map-wide precision there (0.48 vs 0.55). It costs GPU time instead: ~17 min of selection in the replay vs ~8 min for a
real run today (two extra vLLM passes: the teacher's 400 and ~19k checked papers), so a map takes ~25 min.

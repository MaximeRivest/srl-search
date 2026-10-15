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

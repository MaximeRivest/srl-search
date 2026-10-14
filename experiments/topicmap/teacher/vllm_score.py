"""Score papers with the merged 0.8B decider in vLLM: P(on topic) = sigmoid(logprob(yes) - logprob(no)) of the first
answer token, all prompts sent at once (vLLM batches them; the shared system prompt is cached).

    /nix/store/...-vllm-runtime-0.29.0/bin/python vllm_score.py   (the rewriter service's vLLM) MODEL_DIR SPLIT [SPLIT ...] [--mem 0.45]

Writes preds/<split>__vllm__<model>.json and prints AUC (when the split has labels) and papers per second.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = Path.home() / "Projects/tiny-classifiers/data/tasks/frag"
args = [x for x in sys.argv[1:] if not x.startswith("--")]
mem = float(sys.argv[sys.argv.index("--mem") + 1]) if "--mem" in sys.argv else 0.45
mdir, splits = Path(args[0]), [s for s in args[1:] if not s.replace(".", "").isdigit()]
from vllm import LLM, SamplingParams  # noqa: E402

cfg = json.load(open(mdir / "decider.json"))
t = time.time()
llm = LLM(model=str(mdir), dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=1024,
          enable_prefix_caching=True, max_num_seqs=512, max_num_batched_tokens=32768,
          enforce_eager="--eager" in sys.argv)
load_s = time.time() - t
tok = llm.get_tokenizer()
sp = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=[cfg["yes"], cfg["no"]])


def prompt(text):
    ids = tok(text, add_special_tokens=False)["input_ids"][:700]       # the paper, at most 700 tokens
    text = tok.decode(ids)
    msgs = [{"role": "system", "content": cfg["system"]}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


print(json.dumps({"load_s": round(load_s, 1)}), flush=True)
for split in splits:
    rows = [json.loads(l) for l in open(T / f"{split}.jsonl")]
    t = time.time()
    ps = [prompt(r["text"]) for r in rows]
    prep_s = time.time() - t
    t = time.time()
    outs = llm.generate(ps, sp, use_tqdm=False)
    gen_s = time.time() - t
    p = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        p.append(1 / (1 + np.exp(-(lp[cfg["yes"]].logprob - lp[cfg["no"]].logprob))))
    p = np.array(p)
    res = {"split": split, "rows": len(rows), "prompt_s": round(prep_s, 1), "score_s": round(gen_s, 1),
           "papers_per_s": round(len(rows) / gen_s)}
    if "label" in rows[0]:
        y = np.array([r["label"] == "on topic" for r in rows])
        rk = p.argsort().argsort() + 1                       # AUC = Mann-Whitney U / (n_pos n_neg)
        res["auc"] = round(float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())), 4)
    (HERE / "preds").mkdir(exist_ok=True)
    (HERE / "preds" / f"{split}__vllm__{mdir.name}.json").write_text(json.dumps({r["id"]: round(float(x), 4) for r, x in zip(rows, p)}))
    print(json.dumps(res), flush=True)

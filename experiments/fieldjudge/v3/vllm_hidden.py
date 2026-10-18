"""The round-3 classifier, fast path: vLLM returns the last position's hidden state (pooling runner, no sampler, one
request per question however many options); the answer scores are computed here with the model's own output rows:
logit(code) = h . W[code]. Same arithmetic as training (train3.py: h @ head.T), so no 128-token limit.

    $SRL_VLLM_PYTHON vllm_hidden.py MODEL rows.jsonl out.json GPU [--repeat N] [--seqs S] [--tokens T] [--eager]
    $SRL_VLLM_PYTHON vllm_hidden.py MODEL rows.jsonl out.json GPU --sweep        (speed only: batch sizes, one at a time)

rows: {"user": question (fmt.py), "answers": ["AA", ...] or ["yes", "no"]}. out: probabilities per row.
"""
import json, math, os, subprocess, sys, time


def main():
    import numpy as np
    import torch
    from safetensors import safe_open
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs import TokensPrompt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fmt
    a = sys.argv
    mdir, src, dst, gpu = a[1:5]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d
    free, total = [int(x) for x in subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True).stdout.split(",")]
    util = min(float(os.environ.get("FJ3_UTIL", "0.85")), (free - 2048) / total)
    llm = LLM(model=mdir, dtype="bfloat16", runner="pooling", convert="embed",
              pooler_config=PoolerConfig(pooling_type="LAST", use_activation=False),
              gpu_memory_utilization=util, max_model_len=4096, enable_prefix_caching=False,
              max_num_seqs=opt("--seqs", 256), max_num_batched_tokens=opt("--tokens", 32768),
              enforce_eager="--eager" in a, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    with safe_open(f"{mdir}/model.safetensors", "pt") as f:
        key = next(k for k in f.keys() if k.endswith("lm_head.weight"))
        W = f.get_tensor(key)
    rows = [json.loads(l) for l in open(src)]
    ids = {}
    def tid(s):
        if s not in ids:
            t = tok.encode(s, add_special_tokens=False); assert len(t) == 1, s; ids[s] = t[0]
        return ids[s]
    prompts = []
    for r in rows:
        p = tok.apply_chat_template(fmt.messages(r["user"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False)["input_ids"][-4000:]))
    need = sorted({tid(x) for r in rows for x in r["answers"]})
    col = {t: i for i, t in enumerate(need)}
    Wn = W[need].float().numpy()

    def run(ps):
        outs = llm.embed(ps, use_tqdm=False)
        return np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs])

    if "--sweep" in a:                                   # throughput is set by the engine's limits: report the run
        res = {"questions": len(rows), "prompt_tokens": sum(len(p["prompt_token_ids"]) for p in prompts), "one_at_a_time": []}
        run(prompts[:32])                                 # warm-up
        for p in prompts[:40]:
            t = time.perf_counter(); run([p]); res["one_at_a_time"].append(time.perf_counter() - t)
        t = time.perf_counter(); run(prompts); res["all_at_once_s"] = time.perf_counter() - t
        json.dump(res, open(dst, "w"))
        return
    times = []
    for _ in range(opt("--repeat", 1)):
        t = time.perf_counter(); H = run(prompts); times.append(time.perf_counter() - t)
    L = H @ Wn.T
    res = []
    for r, l in zip(rows, L):
        v = np.array([l[col[tid(x)]] for x in r["answers"]], dtype=np.float64)
        e = np.exp(v - v.max()); res.append([round(float(x), 6) for x in e / e.sum()])
    json.dump(res, open(dst, "w"))
    json.dump({"questions": len(rows), "requests": len(rows), "prompt_tokens": sum(len(p["prompt_token_ids"]) for p in prompts),
               "seconds": times}, open(dst + ".speed.json", "w"))


if __name__ == "__main__":
    main()

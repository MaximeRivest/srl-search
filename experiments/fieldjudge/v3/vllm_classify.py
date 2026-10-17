"""The round-3 classifier in vLLM: one forward pass per question, no generated text.

    $SRL_VLLM_PYTHON vllm_classify.py MODEL rows.jsonl out.json GPU [--repeat N]

rows: {"user": the question (fmt.pick_prompt / fmt.yesno_prompt), "answers": ["AA", "AB", ...] or ["yes", "no"]}
out:  one list per row: the probabilities of its answers (softmax of their logits, i.e. renormalised over the
      offered answers only). vLLM reads at most 128 given tokens per request: longer lists are sent as several
      requests with the same prompt (prefix cache: the extra ones cost almost nothing).
--repeat N runs the whole set N times and reports the steady-state speed (the first pass includes warm-up).
"""
import json, math, subprocess, sys, time


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import fmt
    mdir, src, dst, gpu = sys.argv[1:5]
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    free, total = [int(x) for x in subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True).stdout.split(",")]
    util = min(0.4, (free - 4096) / total)
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=4096, enable_prefix_caching=True,
              max_num_seqs=256, max_num_batched_tokens=16384, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    rows = [json.loads(l) for l in open(src)]
    ids = {}
    def tid(s):
        if s not in ids:
            t = tok.encode(s, add_special_tokens=False)
            assert len(t) == 1, s
            ids[s] = t[0]
        return ids[s]
    prompts, params, owner = [], [], []
    for i, r in enumerate(rows):
        p = tok.apply_chat_template(fmt.messages(r["user"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        pid = tok(p, add_special_tokens=False)["input_ids"][-4000:]
        a = [tid(x) for x in r["answers"]]
        for j in range(0, len(a), 128):
            prompts.append(TokensPrompt(prompt_token_ids=pid)); owner.append((i, a[j:j + 128]))
            params.append(SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=a[j:j + 128]))
    times = []
    for _ in range(repeat):
        t = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        times.append(time.time() - t)
    lp = [dict() for _ in rows]
    for (i, a), o in zip(owner, outs):
        got = o.outputs[0].logprobs[0]
        lp[i].update({t: got[t].logprob for t in a})
    res = []
    for r, d in zip(rows, lp):
        v = [d[tid(x)] for x in r["answers"]]
        m = max(v); z = sum(math.exp(x - m) for x in v)
        res.append([round(math.exp(x - m) / z, 6) for x in v])
    json.dump(res, open(dst, "w"))
    tokens = sum(len(p["prompt_token_ids"]) for p in prompts)
    json.dump({"questions": len(rows), "requests": len(prompts), "prompt_tokens": tokens, "seconds": times},
              open(dst + ".speed.json", "w"))


if __name__ == "__main__":
    main()

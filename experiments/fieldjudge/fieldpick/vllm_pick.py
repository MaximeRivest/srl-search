"""Runs in vLLM's own Python. Two ways to ask a model which of the 176 Science-Metrix subfields a paper belongs to:

  pick   MODEL pick  prompts.jsonl names.json out.json GPU
         One chat prompt per paper listing every subfield (the jevarticle Qwen/Jev prompt). Every name is scored as a
         complete answer (its tokens, then <|im_end|>) through a token trie: one 1-token request per branching trie node,
         prefix-cached. Writes per paper: the constrained distribution over the 176 names (renormalised at each node,
         = what choice-constrained decoding samples from) and "coverage": the model's unconstrained probability that its
         answer starts a listed name (first token), a direct measure of whether it still wants to answer this way.
  yesno  MODEL yesno rows.jsonl - out.json GPU
         rows {"system","text"}: P(yes) = sigmoid(logprob yes - logprob no) of the first answer token, as vllm_decider.py.
"""
import json, math, subprocess, sys
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    global tok

    mdir, mode, src, names_f, dst, gpu = sys.argv[1:7]
    free, total = [int(x) for x in subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True).stdout.split(",")]
    util = min(0.4, (free - 4096) / total)
    llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=6144 if mode == "pick" else 1024,
              enable_prefix_caching=True, max_num_seqs=128, max_num_batched_tokens=16384, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    YES, NO = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    END = tok.convert_tokens_to_ids("<|im_end|>")
    rows = [json.loads(l) for l in open(src)]


    def chat(system, user):
        s = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok(s, add_special_tokens=False)["input_ids"]


    if mode == "yesno":
        prompts = [TokensPrompt(prompt_token_ids=chat(r["system"], tok.decode(tok(r["text"], add_special_tokens=False)["input_ids"][:700])))
                   for r in rows]
        outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=[YES, NO]), use_tqdm=False)
        p = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            p.append(round(1 / (1 + math.exp(-(lp[YES].logprob - lp[NO].logprob))), 5))
        json.dump(p, open(dst, "w"))
        return

    names = json.load(open(names_f))
    paths = [tok(n, add_special_tokens=False)["input_ids"] + [END] for n in names]
    children = {}
    for p in paths:
        for i in range(len(p)):
            children.setdefault(tuple(p[:i]), set()).add(p[i])
    nodes = [k for k, v in children.items() if len(v) > 1]           # single-child nodes: probability 1 once constrained
    assert () in nodes
    reqs = []                                                       # (paper, node, token ids asked)
    prompts = []
    for i, r in enumerate(rows):
        base = chat(r["system"], r["user"])
        for k in nodes:
            ch = sorted(children[k])
            for j in range(0, len(ch), 128):                        # vLLM caps the ids asked per request at 128
                reqs.append((i, k, ch[j:j + 128]))
                prompts.append(TokensPrompt(prompt_token_ids=base + list(k)))
    outs = llm.generate(prompts, [SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=c) for _, _, c in reqs],
                        use_tqdm=False)
    raw = [dict() for _ in rows]                                    # paper -> node -> {token: logprob}
    for (i, k, c), o in zip(reqs, outs):
        lp = o.outputs[0].logprobs[0]
        raw[i].setdefault(k, {}).update({t: lp[t].logprob for t in c})
    res = []
    for i in range(len(rows)):
        logp = []
        for p in paths:
            s = 0.0
            for d in range(len(p)):
                k = tuple(p[:d])
                if k in raw[i]:
                    lp = raw[i][k]
                    m = max(lp.values())
                    s += lp[p[d]] - (m + math.log(sum(math.exp(v - m) for v in lp.values())))
            logp.append(s)
        m = max(logp); z = sum(math.exp(v - m) for v in logp)
        cov = sum(math.exp(v) for v in raw[i][()].values())
        res.append({"p": [round(math.exp(v - m) / z, 6) for v in logp], "coverage": round(cov, 5)})
    json.dump(res, open(dst, "w"))


if __name__ == "__main__":      # vLLM may spawn its engine: the module must import cleanly
    main()

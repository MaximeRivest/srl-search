"""Score texts with a merged decider (srl_search/judges.py) in vLLM: P(on topic) = sigmoid(logprob(yes) - logprob(no))
of the first answer token. Runs in vLLM's own Python (SRL_VLLM_PYTHON): imports only vLLM and the standard library.

    $SRL_VLLM_PYTHON vllm_decider.py MERGED_DIR texts.jsonl scores.json GPU_INDEX

Eager mode: starts in ~20 s instead of ~110 s for ~13% fewer papers a second (measured; better for ~10k papers).
Takes at most 40% of the card, or what is free minus 4 GB, whichever is less (SRL_VLLM_UTIL caps it lower): vLLM
uses ~1.5 GB beyond its share, and the household services on the card grow while it runs (dictation: +1.4 GB seen).
"""
import json
import math
import os
import subprocess
import sys

from vllm import LLM, SamplingParams

mdir, src, dst, gpu = sys.argv[1:5]
cfg = json.load(open(f"{mdir}/decider.json"))
free, total = [int(x) for x in subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits", "-i", gpu],
    capture_output=True, text=True).stdout.split(",")]        # nvidia-smi ignores CUDA_VISIBLE_DEVICES: the real index
util = min(float(os.environ.get("SRL_VLLM_UTIL", "0.4")), (free - 4096) / total)
llm = LLM(model=mdir, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=1024,
          enable_prefix_caching=True, max_num_seqs=256, max_num_batched_tokens=16384, enforce_eager=True)
tok = llm.get_tokenizer()


def prompt(text):
    text = tok.decode(tok(text, add_special_tokens=False)["input_ids"][:700])
    return tok.apply_chat_template([{"role": "system", "content": cfg["system"]}, {"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


texts = [json.loads(l)["text"] for l in open(src)]
outs = llm.generate([prompt(t) for t in texts], SamplingParams(max_tokens=1, temperature=0.0,
                                                               logprob_token_ids=[cfg["yes"], cfg["no"]]), use_tqdm=False)
p = []
for o in outs:
    lp = o.outputs[0].logprobs[0]
    p.append(round(1 / (1 + math.exp(-(lp[cfg["yes"]].logprob - lp[cfg["no"]].logprob))), 4))
json.dump(p, open(dst, "w"))

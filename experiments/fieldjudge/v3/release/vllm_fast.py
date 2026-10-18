"""Fast bulk scoring with vLLM (tested with vLLM 0.29): the pooling runner returns the hidden state at the answer
position, and the answer scores are h . W[code] with the model's own output rows. One request per question, any
number of options (no 128-logprob limit), no generated text. On one RTX 3090: about 60,000 papers an hour with the
0.8B and 28,000 with the 2B for a 176-option question, 0.065 s / 0.14 s for one question alone.

    python vllm_fast.py maximerivest/paper-classifier-0.8B
"""
import sys

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from vllm import LLM
from vllm.config import PoolerConfig

from paper_classifier import SYSTEM, paper_text, pick_prompt

model = snapshot_download(sys.argv[1] if len(sys.argv) > 1 else "maximerivest/paper-classifier-0.8B")
llm = LLM(model=model, runner="pooling", convert="embed", dtype="bfloat16",
          pooler_config=PoolerConfig(pooling_type="LAST", use_activation=False),   # raw last hidden state
          max_model_len=4096, max_num_seqs=64, max_num_batched_tokens=32768, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()
with safe_open(f"{model}/model.safetensors", "pt") as f:
    W = f.get_tensor(next(k for k in f.keys() if k.endswith("lm_head.weight"))).float().numpy()

options = ["Ecology", "Ornithology", "Particle physics", "Economics"]
codes = ["AA", "AB", "AC", "AD"]                                 # classifier.json: codes, in order
ids = [tok.encode(c, add_special_tokens=False)[0] for c in codes]
papers = [paper_text("Habitat fragmentation reduces bird diversity", "We surveyed birds in 40 forest fragments."),
          paper_text("Search for dark matter with a liquid xenon detector", "We report limits on WIMP scattering.")]
prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": pick_prompt(p, "Fields of science",
                                     "Which option best describes the research content of this paper?",
                                     [{"name": o} for o in options], codes)}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in papers]
H = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in llm.embed(prompts)])
L = H @ W[ids].T
P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
for p, row in zip(papers, P):
    print(p.splitlines()[0], {o: round(float(q), 3) for o, q in zip(options, row)})

"""Zero-shot small language models as judges of "is this paper about habitat fragmentation?", against Kimi K3.

    cd ~/Projects/tiny-classifiers && uv run --with scikit-learn python .../zeroshot.py MODEL [MODEL ...]

The same instruction Kimi had (task.json), as the system message; the paper as the user message; the model must
answer "yes" or "no" (thinking off). P(yes) = softmax over the first answer token's logits for yes / no.
Scored on test.jsonl (1,414 papers) and refs_labelled.jsonl (the meta-analyses' references), Kimi's labels as truth.
Appends to zeroshot_results.jsonl beside this file.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
T = Path.home() / "Projects/tiny-classifiers/data/tasks/frag"
task = json.load(open(T / "task.json"))
SYSTEM = task["instruction"].replace("Is habitat", "Is habitat") + ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
meta = json.load(open(T / "meta.json"))
load = lambda n: [json.loads(l) for l in open(T / f"{n}.jsonl")]

for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    ids = lambda w: [tok.encode(v, add_special_tokens=False)[0] for v in (w, w.capitalize(), " " + w, " " + w.capitalize())]
    yes, no = sorted(set(ids("yes"))), sorted(set(ids("no")))

    def prompt(text):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:  # templates without a system role
            msgs = [{"role": "user", "content": SYSTEM + "\n\n" + text}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    res = {"model": name}
    for split, file in (("test", "test"), ("refs", "refs_labelled")):
        rows = load(file)
        ps = np.zeros(len(rows))
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
        t = time.time()
        with torch.no_grad():
            for s in range(0, len(rows), 32):
                b = order[s:s + 32]
                enc = tok([prompt(rows[i]["text"]) for i in b], return_tensors="pt", padding=True, truncation=True,
                          max_length=1024, add_special_tokens=False).to("cuda")
                lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()   # the last position only
                ly, ln = torch.logsumexp(lg[:, yes], 1), torch.logsumexp(lg[:, no], 1)
                ps[b] = torch.sigmoid(ly - ln).cpu().numpy()
        speed = len(rows) / (time.time() - t)
        y = np.array([r["label"] == "on topic" for r in rows])
        ab = np.array([meta[r["id"]]["abstract"] for r in rows])
        # the best threshold is unknown without labels; report AUC (threshold-free) and accuracy at 0.5
        k = ps >= 0.5
        res[split] = {"auc": round(float(roc_auc_score(y, ps)), 4), "acc": round(float((k == y).mean()), 4),
                      "recall": round(float(k[y].mean()), 4), "precision": round(float(y[k].mean()), 4) if k.any() else 0,
                      "recall_title_only": round(float(k[y & ~ab].mean()), 4) if (y & ~ab).any() else None}
        res["papers_per_s"] = round(speed)
    with open(HERE / "zeroshot_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()

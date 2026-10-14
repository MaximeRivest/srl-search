"""A small language model turned into a decision model (Perplexity's pplx-decider recipe, at 0.8B), fine-tuned on a few
of Kimi K3's labels: can it get close to Kimi with very few examples, and so be a cheap teacher?

    cd ~/Projects/tiny-classifiers
    uv run --with peft --with scikit-learn python .../fewshot_llm.py --model Qwen/Qwen3.5-0.8B --n 250 [--bidirectional]

- Decision head = the model's own output rows for "yes" and "no" (it starts from what the model already knows);
  score = logit(yes) - logit(no) at the last position. The pplx readout does the same over its candidate tokens.
- --bidirectional: the full-attention layers see the whole input both ways (the causal mask lifted, as in
  pplx-decider-v1.1); the linear-attention layers keep their native behaviour.
- LoRA (r 16, all linear layers) + the head, so it trains beside other GPU work; binary cross-entropy;
  same prompt as the zero-shot test. Training papers: the first N of Kimi's labels, shuffled by the seed (curve.py's).
Appends to fewshot_results.jsonl beside this file.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
T = Path.home() / "Projects/tiny-classifiers/data/tasks/frag"
L = Path.home() / "Projects/tiny-classifiers/labels"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bidirectional", action="store_true")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--predict", help="also score NAME.jsonl and save P(on topic)")
ap.add_argument("--save", help="save a merged model for vLLM: LoRA merged, the yes/no head in an untied output layer")
ap.add_argument("--eval-at", default="", help="also measure the test AUC after these steps, e.g. 50,100")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--acc", type=int, default=4)
ap.add_argument("--no-gc", action="store_true", help="no gradient checkpointing (faster, more memory)")
a = ap.parse_args()
torch.manual_seed(a.seed)
random.seed(a.seed)
task = json.load(open(T / "task.json"))
SYSTEM = task["instruction"] + ' Answer with one word: "yes" (on topic) or "no" (not on topic).'
meta = json.load(open(T / "meta.json"))
load = lambda n: [json.loads(l) for l in open(T / f"{n}.jsonl")]
kimi = {}
for f in ("frag-all.jsonl", "frag-more.jsonl"):
    for l in open(L / f):
        r = json.loads(l)
        kimi[r["id"]] = r["pred"] == "on topic"
pool = [r for r in load("train") + load("more") if r["id"] in kimi]
random.Random(a.seed).shuffle(pool)
train = pool[:a.n]

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
if a.bidirectional:
    import transformers.masking_utils as mu
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm

    def bidirectional(config, inputs_embeds, attention_mask, past_key_values=None, position_ids=None, **kw):
        return mu.create_bidirectional_mask(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    qm.create_causal_mask = bidirectional
    for mod in model.modules():
        if getattr(mod, "is_causal", False) and hasattr(mod, "layer_type") and mod.layer_type == "full_attention":
            mod.is_causal = False
yes = tok.encode("yes", add_special_tokens=False)[0]
no = tok.encode("no", add_special_tokens=False)[0]
lm_head = model.get_output_embeddings()
head = torch.nn.Parameter(lm_head.weight[[yes, no]].detach().clone().float())      # [2, hidden], trainable
if not a.no_gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
                                         task_type="CAUSAL_LM"))


def prompt(text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def margin(texts):
    enc = tok([prompt(t) for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=640,
              add_special_tokens=False).to("cuda")
    h = model.get_base_model().model(**enc).last_hidden_state[:, -1, :].float()   # after the final norm
    lg = h @ head.T
    return lg[:, 0] - lg[:, 1]


def score(rows):
    p = np.zeros(len(rows))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(rows), 32):
            b = order[s:s + 32]
            p[b] = torch.sigmoid(margin([rows[i]["text"] for i in b]).float()).cpu().numpy()
    return p, len(rows) / (time.time() - t)


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + [head], lr=a.lr, weight_decay=0.0)
BATCH, ACC = a.batch, a.acc
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
model.train()
t0 = time.time()
eval_s = 0.0
order = []
for step in range(a.steps):
    loss_sum = 0.0
    for _ in range(ACC):
        if len(order) < BATCH:
            order += random.sample(range(len(train)), len(train))
        b, order = order[:BATCH], order[BATCH:]
        y = torch.tensor([float(kimi[train[i]["id"]]) for i in b], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = margin([train[i]["text"] for i in b])
        loss = F.binary_cross_entropy_with_logits(m, y) / ACC
        loss.backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + [head], 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if str(step + 1) in a.eval_at.split(","):
        model.eval()
        t_eval = time.time()
        rows_t = load("test")
        pp, _ = score(rows_t)
        yy = np.array([r["label"] == "on topic" for r in rows_t])
        print(json.dumps({"at_step": step + 1, "train_s": round(t_eval - t0 - eval_s), "test_auc": round(float(roc_auc_score(yy, pp)), 4)}), flush=True)
        eval_s += time.time() - t_eval
        model.train()
train_s = time.time() - t0 - eval_s
model.eval()


res = {"model": a.model, "bidirectional": a.bidirectional, "n": a.n, "seed": a.seed, "steps": a.steps,
       "train_s": round(train_s)}
for name, file in (("test", "test"), ("refs", "refs_labelled")):
    rows = load(file)
    p, speed = score(rows)
    y = np.array([r["label"] == "on topic" for r in rows])
    ab = np.array([meta[r["id"]]["abstract"] for r in rows])
    k = p >= 0.5
    res[name] = {"auc": round(float(roc_auc_score(y, p)), 4), "acc": round(float((k == y).mean()), 4),
                 "recall": round(float(k[y].mean()), 4), "precision": round(float(y[k].mean()), 4) if k.any() else 0,
                 "recall_title_only": round(float(k[y & ~ab].mean()), 4) if (y & ~ab).any() else None}
    res["papers_per_s"] = round(speed)
if a.predict:
    rows = load(a.predict)
    p, speed = score(rows)
    out = HERE / "preds" / f"{a.predict}__{a.model.split('/')[-1]}{'-bi' if a.bidirectional else ''}__n{a.n}__s{a.seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({r["id"]: round(float(x), 4) for r, x in zip(rows, p)}))
    res["predicted"] = {"file": out.name, "rows": len(rows), "papers_per_s": round(speed)}
if a.save:
    merged = model.merge_and_unload()
    W = merged.get_input_embeddings().weight.detach().clone()
    W[[yes, no]] = head.detach().to(W.dtype)
    merged.lm_head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype, device=W.device)
    merged.lm_head.weight = torch.nn.Parameter(W)
    merged.config.tie_word_embeddings = False
    if getattr(merged.config, "text_config", None) is not None:
        merged.config.text_config.tie_word_embeddings = False
    merged.save_pretrained(a.save, safe_serialization=True)
    tok.save_pretrained(a.save)
    json.dump({"system": SYSTEM, "yes": yes, "no": no}, open(Path(a.save) / "decider.json", "w"))
    res["saved"] = a.save
with open(HERE / "fewshot_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)

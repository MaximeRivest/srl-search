"""Step 4: one judge for every topic. Qwen3.5 (0.8B or 2B) learns Kimi K3's answers on ~430 topics: LoRA on all
linear layers plus a yes/no head started from the model's own "yes"/"no" rows (the per-topic decider's recipe,
srl_search/judges.py, with the topic's question in each prompt instead of one fixed question). Saved merged, with the
head in an untied output layer, so srl_search/vllm_decider.py scores with it (a row's own "system").

Topics: 3 held out for overlapping the test topics (README.md); 40 others (seeded) are the validation topics that
choose the best checkpoint (AUC). The test topics are never seen here.

Shares GPU 1 politely: every 20 steps it asks the whole-field line (srl_search.field.FieldStore.plan) whether a map
needs the GPU within 5 minutes, or Figurer reserved it; then it saves everything and exits with code 3, and
run_train.sh starts it again (--resume) once the map is done.

    ../../.venv/bin/python train.py --base Qwen/Qwen3.5-0.8B --out /mnt/fast/scholarsreadinglist/fieldjudge/m08 [--resume]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from srl_search import judges as J  # noqa: E402
from srl_search.field import FieldStore, ROOT  # noqa: E402

D = Path("/mnt/fast/scholarsreadinglist/fieldjudge")
HELD_OUT = {113, 117, 137}          # RLHF, LLM, reinforcement learning: they overlap the LLM post-training test topic
ANSWER = ' Answer with one word: "yes" (on topic) or "no" (not on topic).'

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--out", required=True)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--batch", type=int, default=32, help="papers a step")
ap.add_argument("--val-every", type=int, default=200)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
log = open(out / "log.jsonl", "a")


def say(**kw):
    kw["t"] = round(time.time())
    log.write(json.dumps(kw) + "\n")
    log.flush()
    print(json.dumps(kw), flush=True)


# ------------------------------------------------------------------ data
topics = {t["id"]: t for t in json.loads((D / "topics.json").read_text())["topics"]}
text = {(s["topic_id"], s["id"]): s["text"] for s in map(json.loads, open(D / "samples.jsonl"))}
rows = [{"topic_id": r["topic_id"], "text": text[(r["topic_id"], r["id"])], "y": r["y"]}
        for r in map(json.loads, open(D / "labels.jsonl")) if r["topic_id"] not in HELD_OUT]
ids = sorted({r["topic_id"] for r in rows})
val_topics = set(random.Random(0).sample(ids, 40))
train = [r for r in rows if r["topic_id"] not in val_topics]
val = [r for r in rows if r["topic_id"] in val_topics]
system = {i: J.instruction(t["topic"], t["definition"]) + ANSWER for i, t in topics.items()}

# ------------------------------------------------------------------ model
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
yes, no = tok.encode("yes", add_special_tokens=False)[0], tok.encode("no", add_special_tokens=False)[0]
head = torch.nn.Parameter(model.get_output_embeddings().weight[[yes, no]].detach().clone().float())
free = torch.cuda.mem_get_info()[0]
small = "0.8B" in a.base
bs = (8 if free >= 14 * 2**30 else 4) if small else (4 if free >= 14 * 2**30 else 2)
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules="all-linear",
                                         task_type="CAUSAL_LM"))
params = [p for p in model.parameters() if p.requires_grad] + [head]
steps = int(a.epochs * len(train) / a.batch)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
warm = 50
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


def prompt(r):
    return tok.apply_chat_template([{"role": "system", "content": system[r["topic_id"]]}, {"role": "user", "content": r["text"]}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def margin(batch):
    enc = tok([prompt(r) for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=768,
              pad_to_multiple_of=128, add_special_tokens=False).to("cuda")
    h = model.get_base_model().model(**enc).last_hidden_state[:, -1, :].float()
    lg = h @ head.T
    return lg[:, 0] - lg[:, 1]


def validate() -> float:
    model.eval()
    order = sorted(range(len(val)), key=lambda i: len(val[i]["text"]))
    p = np.zeros(len(val))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(order), 32):
            b = order[s:s + 32]
            p[b] = torch.sigmoid(margin([val[i] for i in b])).float().cpu().numpy()
    model.train()
    y = np.array([r["y"] >= 0.5 for r in val])
    per_topic = [roc_auc_score(y[m], p[m]) for t in val_topics
                 if 0 < y[m := np.array([r["topic_id"] == t for r in val])].sum() < m.sum()]
    return float(roc_auc_score(y, p)), float(np.mean(per_topic)), float(((p >= 0.5) == y).mean())


def gpu_wanted() -> bool:
    """A whole-field map needs GPU 1 within 5 minutes (it is running, or next and starting), or Figurer reserved it."""
    if J.RESERVED.exists():
        return True
    pl = FieldStore(ROOT).plan()
    if pl["running"] and pl["running"]["st"].get("kind") == "full":
        return True
    return any(w["st"].get("kind") == "full" and w["start"] - pl["now"] < 300 for w in pl["waiting"][:1])


# ------------------------------------------------------------------ resume
state = {"step": 0, "order": [], "best": -1.0}
ck = out / "checkpoint.pt"
if a.resume and ck.exists():
    c = torch.load(ck, map_location="cuda", weights_only=False)
    set_peft_model_state_dict(model, c["lora"])
    head.data.copy_(c["head"])
    opt.load_state_dict(c["opt"])
    sched.load_state_dict(c["sched"])
    state = c["state"]
    say(event="resumed", step=state["step"])
else:
    say(event="start", base=a.base, train=len(train), val=len(val), topics=len(ids), val_topics=len(val_topics),
        steps=steps, bs=bs, rank=a.rank, lr=a.lr)


def save_checkpoint():
    torch.save({"lora": get_peft_model_state_dict(model), "head": head.detach(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "state": state}, ck)


rng = random.Random(1 + state["step"])
model.train()
t0 = time.time()
acc = a.batch // bs
while state["step"] < steps:
    if state["step"] % 20 == 0 and state["step"] and gpu_wanted():
        save_checkpoint()
        say(event="paused for a whole-field map", step=state["step"])
        sys.exit(3)
    loss_sum = 0.0
    for _ in range(acc):
        if len(state["order"]) < bs:
            state["order"] += rng.sample(range(len(train)), len(train))
        b, state["order"] = state["order"][:bs], state["order"][bs:]
        y = torch.tensor([float(train[i]["y"] >= 0.5) for i in b], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = margin([train[i] for i in b])
        loss = F.binary_cross_entropy_with_logits(m.float(), y) / acc
        loss.backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    state["step"] += 1
    if state["step"] % 20 == 0:
        say(step=state["step"], loss=round(loss_sum, 4), papers_s=round(20 * a.batch / (time.time() - t0), 1))
        t0 = time.time()
    if state["step"] % a.val_every == 0 or state["step"] == steps:
        auc, auc_topic, acc_ = validate()
        say(step=state["step"], val_auc=round(auc, 4), val_auc_per_topic=round(auc_topic, 4), val_acc=round(acc_, 4))
        if auc > state["best"]:
            state["best"] = auc
            torch.save({"lora": get_peft_model_state_dict(model), "head": head.detach(), "step": state["step"]}, out / "best.pt")
        save_checkpoint()
        t0 = time.time()

# ------------------------------------------------------------------ the best checkpoint, merged for vLLM
best = torch.load(out / "best.pt", map_location="cuda", weights_only=False)
set_peft_model_state_dict(model, best["lora"])
merged = model.merge_and_unload()
W = merged.get_input_embeddings().weight.detach().clone()
W[[yes, no]] = best["head"].to(W.dtype)
merged.lm_head = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype, device=W.device)
merged.lm_head.weight = torch.nn.Parameter(W)
merged.config.tie_word_embeddings = False
if getattr(merged.config, "text_config", None) is not None:
    merged.config.text_config.tie_word_embeddings = False
merged.save_pretrained(out / "merged", safe_serialization=True)
tok.save_pretrained(out / "merged")
json.dump({"system": None, "yes": yes, "no": no, "base": a.base, "answer": ANSWER}, open(out / "merged" / "decider.json", "w"))
say(event="done", best_step=best["step"], best_val_auc=state["best"])
